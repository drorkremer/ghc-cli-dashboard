# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import pytest

import topic_classifier
import topic_families


def test_family_inference_keeps_edits_made_while_the_model_runs(tmp_path):
    path = tmp_path / "topics.json"
    catalog = topic_classifier.empty_catalog()
    topic_classifier.apply_edit(catalog, {"action": "create", "name": "Worktree"})
    topic_id = catalog["topics"][0]["id"]
    topic_classifier.save_catalog(path, catalog)

    def infer(batch):
        edited = topic_classifier.load_catalog(path)
        edited["topics"][0]["name"] = "Renamed worktree"
        edited["family_overrides"][topic_id] = "uncategorized"
        topic_classifier.save_catalog(path, edited)
        return [next(iter(topic_families.FAMILY_NAMES))]

    topic_families.classify_catalog(path, "test-model", infer=infer)
    result = topic_classifier.load_catalog(path)
    assert result["topics"][0]["name"] == "Renamed worktree"
    assert result["family_overrides"][topic_id] == "uncategorized"


def test_family_inference_does_not_resurrect_removed_topics(tmp_path):
    path = tmp_path / "topics.json"
    catalog = topic_classifier.empty_catalog()
    for name in ("Worktree", "Inbox"):
        topic_classifier.apply_edit(catalog, {"action": "create", "name": name})
    removed, remaining = (topic["id"] for topic in catalog["topics"])
    topic_classifier.save_catalog(path, catalog)
    family_id = next(iter(topic_families.FAMILY_NAMES))

    def infer(batch):
        edited = topic_classifier.load_catalog(path)
        edited["topics"] = [topic for topic in edited["topics"] if topic["id"] != removed]
        topic_classifier.save_catalog(path, edited)
        return [family_id] * len(batch)

    topic_families.classify_catalog(path, "test-model", infer=infer)
    result = topic_classifier.load_catalog(path)
    assert [topic["id"] for topic in result["topics"]] == [remaining]
    assert result["family_assignments"] == {remaining: family_id}


def test_families_resume_new_topics_and_refresh_without_losing_overrides(tmp_path):
    path = tmp_path / "topics.json"
    catalog = topic_classifier.empty_catalog()
    for name in ("Worktree", "Inbox", "Invoice"):
        topic_classifier.apply_edit(catalog, {"action": "create", "name": name})
    first, second, third = (topic["id"] for topic in catalog["topics"])
    topic_classifier.save_catalog(path, catalog)
    ids = list(topic_families.FAMILY_NAMES)
    calls = []

    def infer(batch):
        calls.append([topic["id"] for topic in batch])
        return [ids[0]] * len(batch)

    assert topic_families.classify_catalog(path, "test-model", infer=infer, limit=2) == 2
    assert calls == [[first, second]]
    catalog = topic_classifier.load_catalog(path)
    assert catalog["family_assignments"] == {first: ids[0], second: ids[0]}
    topic_classifier.apply_edit(catalog, {
        "action": "assign_family", "topic_id": second, "family_id": "uncategorized",
    })
    topic_classifier.save_catalog(path, catalog)
    calls.clear()
    assert topic_families.classify_catalog(path, "test-model", infer=infer) == 1
    assert calls == [[third]]
    calls.clear()
    assert topic_families.classify_catalog(
        path, "test-model", refresh=True, infer=lambda batch: [ids[1]] * len(batch),
    ) == 2
    catalog = topic_classifier.load_catalog(path)
    assert catalog["family_assignments"][first] == ids[1]
    assert catalog["family_assignments"][third] == ids[1]
    assert catalog["family_overrides"][second] == "uncategorized"
    assert len(catalog["families"]) == len(topic_families.FAMILY_NAMES)


def test_malformed_family_model_batch_is_not_saved(tmp_path):
    path = tmp_path / "topics.json"
    catalog = topic_classifier.empty_catalog()
    for name in ("First", "Second"):
        topic_classifier.apply_edit(catalog, {"action": "create", "name": name})
    topic_classifier.save_catalog(path, catalog)
    with pytest.raises(ValueError, match="family model response"):
        topic_families.classify_catalog(
            path, "test-model", infer=lambda _: ["uncategorized"], batch_size=2,
        )
    assert topic_classifier.load_catalog(path)["family_assignments"] == {}


def test_family_inference_rejects_invalid_model_output(monkeypatch):
    import io
    import json

    def response(request, **__):
        payload = json.loads(request.data)
        assert "Cairn inbox" in payload["messages"][0]["content"]
        assert "roster" in payload["messages"][0]["content"]
        return io.BytesIO(json.dumps({"message": {"content": '{"families":["bad"]}'}}).encode())

    monkeypatch.setattr(topic_families.urllib.request, "urlopen", response)
    with pytest.raises(ValueError, match="unknown family"):
        topic_families._infer([{"name": "First", "examples": []}], "gpt-oss:20b")
