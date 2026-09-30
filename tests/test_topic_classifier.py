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


def test_merge_preserves_existing_assignments_and_overrides():
    catalog = topics.empty_catalog()
    topics.apply_edit(catalog, {"action": "create", "name": "First", "examples": ["NOVA design"]})
    topics.apply_edit(catalog, {"action": "create", "name": "Second", "examples": ["MCP tooling"]})
    first, second = (t["id"] for t in catalog["topics"])
    key = topics.session_key("user", "1")
    catalog["assignments"][key] = first
    catalog["overrides"][key] = first
    topics.apply_edit(catalog, {"action": "merge", "source_id": first, "target_id": second})
    assert catalog["assignments"][key] == second
    assert catalog["overrides"][key] == second
    assert len(catalog["topics"]) == 1


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
    first = "Architecture " + "design " * 110
    second = "Architecture " + "planning " * 110
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
