# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import json
import sys
import types

import pytest

import topic_classifier as topics


def encoder(texts):
    vectors = {
        "NOVA architecture": (1, 0, 0),
        "NOVA design": (1, 0, 0),
        "NOVA migration": (1, 0, 0),
        "MCP research": (0, 1, 0),
        "MCP tooling": (0, 1, 0),
        "Unrelated": (0, 0, 1),
    }
    return [vectors[text] for text in texts]


def test_discovery_known_topics_overrides_and_stability(tmp_path):
    path = tmp_path / "topics.json"
    catalog = topics.empty_catalog()
    catalog["topics"].append({
        "id": "mcp", "name": "MCP", "examples": ["MCP tooling"],
        "keywords": [], "kind": "user",
    })
    topics.save_catalog(path, catalog)
    sessions = [
        ("user", "1", "NOVA architecture"),
        ("user", "2", "NOVA design"),
        ("user", "3", "MCP research"),
        ("user", "4", "Unrelated"),
    ]
    assignments, catalog = topics.classify(sessions, path, encoder=encoder)
    discovered_id = assignments[topics.session_key("user", "1")]
    assert discovered_id != "other"
    assert assignments[topics.session_key("user", "2")] == discovered_id
    assert assignments[topics.session_key("user", "3")] == "mcp"
    assert assignments[topics.session_key("user", "4")] == "other"
    assert next(t for t in catalog["topics"] if t["id"] == discovered_id)["name"] == "NOVA"

    topics.apply_edit(catalog, {"action": "override", "user": "user", "session_id": "2", "topic_id": "mcp"})
    topics.apply_edit(catalog, {"action": "rename", "topic_id": "mcp", "name": "MCP / tooling"})
    topics.save_catalog(path, catalog)
    assignments, catalog = topics.classify(sessions + [("user", "5", "NOVA migration")], path, encoder=encoder)
    assert assignments[topics.session_key("user", "1")] == discovered_id
    assert assignments[topics.session_key("user", "2")] == "mcp"
    assert assignments[topics.session_key("user", "5")] == discovered_id
    assert next(t for t in catalog["topics"] if t["id"] == "mcp")["name"] == "MCP / tooling"


def test_auto_topics_with_same_short_label_get_distinct_names(tmp_path):
    path = tmp_path / "topics.json"
    sessions = [
        ("u", "1", "Operations review"),
        ("u", "2", "Operations audit"),
        ("u", "3", "Operations routing"),
        ("u", "4", "Operations migration"),
    ]
    vectors = {
        "Operations review": (1, 0),
        "Operations audit": (1, 0),
        "Operations routing": (0, 1),
        "Operations migration": (0, 1),
    }
    assignments, catalog = topics.classify(
        sessions, path, encoder=lambda texts: [vectors[text] for text in texts],
    )
    assert len({assignments[topics.session_key("u", sid)] for sid in ("1", "2", "3", "4")}) == 2
    assert [topic["name"] for topic in catalog["topics"]] == [
        "Operations", "Operations routing",
    ]


def test_merge_preserves_existing_assignments_and_overrides():
    catalog = topics.empty_catalog()
    topics.apply_edit(catalog, {"action": "create", "name": "First", "examples": ["NOVA design"]})
    topics.apply_edit(catalog, {"action": "create", "name": "Second", "examples": ["MCP tooling"]})
    first, second = (t["id"] for t in catalog["topics"])
    key = topics.session_key("user", "1")
    catalog["assignments"][key] = first
    catalog["overrides"][key] = first
    topics.apply_edit(catalog, {"action": "create_family", "name": "Developer tooling"})
    family_id = catalog["families"][0]["id"]
    catalog["family_assignments"][first] = family_id
    topics.apply_edit(catalog, {"action": "merge", "source_id": first, "target_id": second})
    assert catalog["assignments"][key] == second
    assert catalog["overrides"][key] == second
    assert len(catalog["topics"]) == 1
    assert catalog["family_assignments"] == {second: family_id}


def test_uncategorized_is_reserved_family_name():
    catalog = topics.empty_catalog()
    with pytest.raises(ValueError, match="invalid family name"):
        topics.apply_edit(catalog, {"action": "create_family", "name": "Uncategorized"})
    assert catalog["families"] == []


def test_family_assignments_survive_rebuilds_edits_and_merge(tmp_path):
    path = tmp_path / "topics.json"
    catalog = topics.empty_catalog()
    for name in ("Cairn worktree", "Cairn inbox", "Invoice"):
        topics.apply_edit(catalog, {"action": "create", "name": name})
    worktree, inbox, invoice = (topic["id"] for topic in catalog["topics"])
    topics.apply_edit(catalog, {"action": "create_family", "name": "Developer tooling"})
    family = catalog["families"][0]["id"]
    catalog["family_assignments"] = {worktree: family, inbox: family}
    topics.apply_edit(catalog, {
        "action": "assign_family", "topic_id": inbox, "family_id": "uncategorized",
    })
    topics.save_catalog(path, catalog)
    _, rebuilt = topics.classify([], path)
    assert rebuilt["family_assignments"][worktree] == family
    assert rebuilt["family_overrides"][inbox] == "uncategorized"
    topics.apply_edit(rebuilt, {
        "action": "rename_family", "family_id": family, "name": "Copilot tooling",
    })
    topics.apply_edit(rebuilt, {"action": "merge", "source_id": inbox, "target_id": worktree})
    assert inbox not in rebuilt["family_assignments"]
    assert inbox not in rebuilt["family_overrides"]
    assert rebuilt["families"][0]["name"] == "Copilot tooling"
    with pytest.raises(ValueError, match="unknown family"):
        topics.apply_edit(rebuilt, {
            "action": "assign_family", "topic_id": invoice, "family_id": "missing",
        })
    assert invoice not in rebuilt["family_overrides"]


def test_family_model_results_are_validated_and_manual_edits_win(tmp_path):
    catalog = topics.empty_catalog()
    for name in ("Worktree", "Inbox", "Unclear"):
        topics.apply_edit(catalog, {"action": "create", "name": name})
    worktree, inbox, unclear = (topic["id"] for topic in catalog["topics"])
    topics.apply_edit(catalog, {"action": "create_family", "name": "Developer tooling"})
    family = catalog["families"][0]["id"]
    topics.apply_edit(catalog, {
        "action": "assign_family", "topic_id": inbox, "family_id": "uncategorized",
    })
    with pytest.raises(ValueError, match="family model response"):
        topics.assign_family_batch(catalog, [worktree, unclear], [family])
    assert catalog["family_assignments"] == {}
    with pytest.raises(ValueError, match="unknown family"):
        topics.assign_family_batch(catalog, [worktree, unclear], [family, "bad"])
    assert catalog["family_assignments"] == {}
    topics.assign_family_batch(catalog, [worktree, inbox, unclear],
                               [family, family, "uncategorized"])
    assert catalog["family_assignments"] == {
        worktree: family, unclear: "uncategorized",
    }
    assert catalog["family_overrides"][inbox] == "uncategorized"
    path = tmp_path / "families.json"
    topics.save_catalog(path, catalog)
    assert topics.load_catalog(path)["family_assignments"] == catalog["family_assignments"]


def test_invalid_topic_file_fails_instead_of_resetting(tmp_path):
    path = tmp_path / "topics.json"
    path.write_text('{"version": 1, "topics": [{"id": "x"}]}')
    with pytest.raises(ValueError, match="invalid topic"):
        topics.load_catalog(path)
    assert json.loads(path.read_text())["topics"] == [{"id": "x"}]


def test_invalid_override_does_not_change_catalog():
    catalog = topics.empty_catalog()
    with pytest.raises(ValueError, match="unknown topic"):
        topics.apply_edit(catalog, {"action": "override", "user": "u", "session_id": "1", "topic_id": "missing"})
    assert catalog["overrides"] == {}


def test_user_topic_can_be_created_from_name_alone():
    catalog = topics.empty_catalog()
    topics.apply_edit(catalog, {"action": "create", "name": "Authentication"})
    assert catalog["topics"][0]["examples"] == ["Authentication"]


def test_user_defined_example_beats_similar_discovered_topic(tmp_path):
    path = tmp_path / "topics.json"
    catalog = topics.empty_catalog()
    catalog["topics"] = [
        {"id": "user-topic", "name": "User", "examples": ["User example"],
         "keywords": [], "kind": "user"},
        {"id": "auto-topic", "name": "Auto", "examples": ["Auto example"],
         "keywords": [], "kind": "auto"},
    ]
    topics.save_catalog(path, catalog)
    vectors = {
        "User example": (0, 0.9, 0.1),
        "Auto example": (0, 1, 0),
        "MCP research": (0, 1, 0),
    }
    assignments, _ = topics.classify(
        [("u", "s", "MCP research")], path,
        encoder=lambda texts: [vectors[text] for text in texts],
    )
    assert assignments[topics.session_key("u", "s")] == "user-topic"


def test_long_summaries_can_be_discovered_and_saved(tmp_path):
    path = tmp_path / "topics.json"
    first = "NOVA architecture " + "design " * 110
    second = "NOVA architecture " + "planning " * 110
    assignments, catalog = topics.classify(
        [("u", "1", first), ("u", "2", second)], path,
        encoder=lambda texts: [(1, 0) for _ in texts],
    )
    assert assignments[topics.session_key("u", "1")] == assignments[topics.session_key("u", "2")]
    assert len(catalog["topics"][0]["examples"][0]) <= 500


def test_topic_examples_and_keyword_rules_can_be_edited(tmp_path):
    path = tmp_path / "topics.json"
    catalog = topics.empty_catalog()
    topics.apply_edit(catalog, {"action": "create", "name": "AEON"})
    topic_id = catalog["topics"][0]["id"]
    topics.apply_edit(catalog, {
        "action": "update", "topic_id": topic_id,
        "examples": ["AEON architecture"], "keywords": ["AEON"],
    })
    topics.save_catalog(path, catalog)
    result, _ = topics.classify([("u", "1", "AEON deployment")], path)
    assert result[topics.session_key("u", "1")] == topic_id


def test_discovered_topic_name_uses_shared_phrase_not_generic_single_word(tmp_path):
    first = "Roster correctness verification for sparse records"
    second = "Audit roster correctness for batch records"
    assignments, catalog = topics.classify(
        [("u", "1", first), ("u", "2", second)],
        tmp_path / "topics.json", encoder=lambda texts: [(1, 0) for _ in texts],
    )
    assert assignments[topics.session_key("u", "1")] == assignments[topics.session_key("u", "2")]
    assert catalog["topics"][0]["name"] == "Roster correctness records"


def test_generic_common_words_do_not_group_unrelated_tasks(tmp_path):
    path = tmp_path / "topics.json"
    sessions = [
        ("u", "1", "Verify final authentication flow"),
        ("u", "2", "Verify final deployment pipeline"),
        ("u", "3", "Roster correctness for sparse records"),
        ("u", "4", "Audit roster correctness for batch records"),
        ("u", "5", "Unknown"),
        ("u", "6", "Unknown"),
        ("u", "7", "Architecture perimeter reputation"),
        ("u", "8", "Architecture post breach review"),
    ]
    assignments, catalog = topics.classify(
        sessions, path, encoder=lambda texts: [(1, 0) for _ in texts],
    )
    assert assignments[topics.session_key("u", "1")] == topics.OTHER
    assert assignments[topics.session_key("u", "2")] == topics.OTHER
    assert assignments[topics.session_key("u", "5")] == topics.OTHER
    assert assignments[topics.session_key("u", "6")] == topics.OTHER
    assert assignments[topics.session_key("u", "7")] == topics.OTHER
    assert assignments[topics.session_key("u", "8")] == topics.OTHER
    assert assignments[topics.session_key("u", "3")] == assignments[topics.session_key("u", "4")]
    assert len(catalog["topics"]) == 1


def test_unknown_and_path_only_slm_labels_remain_other(tmp_path):
    samples = [
        ("u", "1", "Unknown."),
        ("u", "2", "Unknown."),
        ("u", "3", "`files/wiki-page/MDO/Current/Services/Protection.md`"),
        ("u", "4", "`Content-Store#dated-mdo-adoption-state`"),
    ]
    assignments, catalog = topics.classify(
        samples, tmp_path / "topics.json",
        encoder=lambda texts: [(1, 0) for _ in texts],
    )
    assert all(value == topics.OTHER for value in assignments.values())
    assert catalog["topics"] == []


def test_cairn_launcher_summaries_do_not_become_topics(tmp_path):
    sessions = [
        ("u", "1", "First read the cairn protocol context at /Users/test/.cairn/work/project/.copilot/session-state/one"),
        ("u", "2", r"First read the cairn protocol context at C:\Users\test\.cairn\work\project\.copilot"),
        ("u", "3", "First read the cairn protocol context at /Users/test/.cairn/work/another/.copilot"),
    ]
    assignments, catalog = topics.classify(
        sessions, tmp_path / "topics.json",
        encoder=lambda texts: [(1, 0) for _ in texts],
    )
    assert set(assignments.values()) == {topics.OTHER}
    assert catalog["topics"] == []
    assert not topics._uninformative_summary("First read the project architecture review, then implement the API")


def test_reclassify_changed_additional_subject_label(tmp_path):
    path = tmp_path / "topics.json"
    catalog = topics.empty_catalog()
    catalog["topics"] = [
        {"id": "passkeys", "name": "Passkeys", "kind": "user",
         "examples": [], "keywords": ["passkey"]},
        {"id": "rollout", "name": "Rollout", "kind": "user",
         "examples": [], "keywords": ["rollout"]},
        {"id": "identity", "name": "Identity", "kind": "user",
         "examples": [], "keywords": ["identity"]},
    ]
    topics.save_catalog(path, catalog)
    sessions = [("u", "s", "Passkey adoption", ["Rollout planning"])]
    _, tags, _ = topics.classify_tagged(sessions, path)
    assert tags[topics.session_key("u", "s")] == ["passkeys", "rollout"]
    sessions = [("u", "s", "Passkey adoption", ["Identity migration"])]
    _, tags, _ = topics.classify_tagged(sessions, path)
    assert tags[topics.session_key("u", "s")] == ["passkeys", "identity"]


def test_multiple_subjects_keep_primary_cost_identity_and_manual_extra_override(tmp_path):
    path = tmp_path / "topics.json"
    catalog = topics.empty_catalog()
    catalog["topics"] = [
        {"id": "passkeys", "name": "Passkeys", "kind": "user",
         "examples": [], "keywords": ["passkey"]},
        {"id": "rollout", "name": "Rollout", "kind": "user",
         "examples": [], "keywords": ["rollout"]},
        {"id": "identity", "name": "Identity", "kind": "user",
         "examples": [], "keywords": ["identity"]},
    ]
    topics.save_catalog(path, catalog)
    sessions = [
        ("u", "s", "Passkey adoption", ["Rollout planning", "Identity migration"]),
        ("u", "t", "Identity migration", ["Passkey adoption"]),
    ]
    primary, tags, _ = topics.classify_tagged(sessions, path)
    key = topics.session_key("u", "s")
    assert primary[key] == "passkeys"
    assert tags[key] == ["passkeys", "rollout", "identity"]
    assert tags[topics.session_key("u", "t")] == ["identity", "passkeys"]
    catalog = topics.load_catalog(path)
    topics.apply_edit(catalog, {
        "action": "override_tags", "user": "u", "session_id": "s", "topic_ids": ["identity"],
    })
    topics.apply_edit(catalog, {
        "action": "override_deliverables", "user": "u", "session_id": "s",
        "deliverables": ["deck", "doc"],
    })
    topics.save_catalog(path, catalog)
    primary, tags, _ = topics.classify_tagged(sessions, path)
    assert primary[key] == "passkeys"
    assert tags[key] == ["passkeys", "identity"]
    assert topics.load_catalog(path)["deliverable_overrides"][key] == ["deck", "doc"]


def test_local_encoder_uses_single_onnx_worker(monkeypatch):
    created = []

    class FakeEmbedding:
        def __init__(self, **options):
            created.append(options)

        def embed(self, texts):
            return [(1, 0) for _ in texts]

    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=FakeEmbedding))
    assert topics._local_encoder(["synthetic input"]) == [(1, 0)]
    assert created == [{"model_name": topics.MODEL, "threads": 1}]
