# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import json
import threading
import urllib.error
import urllib.request

import pytest

import topic_classifier
import topic_server


@pytest.fixture
def server(tmp_path):
    html = tmp_path / "usage_dashboard.html"
    html.write_text("<script>const TOPIC_EDIT_TOKEN = null;</script>", encoding="utf-8")
    catalog = tmp_path / "topics.json"
    topic_classifier.save_catalog(catalog, topic_classifier.empty_catalog())
    instance = topic_server.make_server(html, catalog)
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    try:
        yield instance, html, catalog
    finally:
        instance.shutdown()
        instance.server_close()
        thread.join(timeout=5)


def request(server, method="GET", path="/", body=None, token=None, origin=None):
    url = f"http://127.0.0.1:{server.server_port}{path}"
    headers = {}
    if token:
        headers["X-Topic-Token"] = token
    if origin:
        headers["Origin"] = origin
    if body is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(body).encode()
    return urllib.request.urlopen(
        urllib.request.Request(url, data=body, headers=headers, method=method), timeout=5
    )


def test_localhost_editor_persists_session_override(server):
    instance, html, catalog_path = server
    origin = f"http://127.0.0.1:{instance.server_port}"
    with request(instance) as response:
        assert f'const TOPIC_EDIT_TOKEN = "{instance.topic_token}";' in response.read().decode()
    assert "const TOPIC_EDIT_TOKEN = null;" in html.read_text()
    with request(instance, "POST", "/api/topics", {
        "action": "create", "name": "Architecture", "examples": ["System design"]
    }, instance.topic_token, origin) as response:
        topic = json.load(response)["topics"][0]
    with request(instance, "POST", "/api/topics", {
        "action": "override", "user": "user", "session_id": "session-1", "topic_id": topic["id"]
    }, instance.topic_token, origin):
        pass
    with request(instance, "POST", "/api/topics", {
        "action": "update", "topic_id": topic["id"],
        "examples": ["Architecture review"], "keywords": ["design"],
    }, instance.topic_token, origin):
        pass
    assert topic_classifier.load_catalog(catalog_path)["overrides"][
        topic_classifier.session_key("user", "session-1")
    ] == topic["id"]
    assert topic_classifier.load_catalog(catalog_path)["topics"][0]["keywords"] == ["design"]


def test_localhost_editor_rejects_cross_origin_and_missing_token(server):
    instance, _, catalog_path = server
    action = {"action": "create", "name": "Secret", "examples": ["text"]}
    with pytest.raises(urllib.error.HTTPError) as exc:
        request(instance, "POST", "/api/topics", action, instance.topic_token, "https://evil.example")
    assert exc.value.code == 403
    with pytest.raises(urllib.error.HTTPError) as exc:
        request(instance, "POST", "/api/topics", action, origin=f"http://127.0.0.1:{instance.server_port}")
    assert exc.value.code == 403
    with pytest.raises(urllib.error.HTTPError) as exc:
        request(instance, "GET", "/api/topics")
    assert exc.value.code == 403
    assert topic_classifier.load_catalog(catalog_path)["topics"] == []


def test_localhost_editor_rejects_invalid_topic_save(server):
    instance, _, catalog_path = server
    origin = f"http://127.0.0.1:{instance.server_port}"
    with pytest.raises(urllib.error.HTTPError) as exc:
        request(instance, "POST", "/api/topics", {
            "action": "override", "user": "u", "session_id": "s", "topic_id": "missing"
        }, instance.topic_token, origin)
    assert exc.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as exc:
        request(instance, "POST", "/api/topics", {
            "action": "override", "user": "u", "session_id": "s", "topic_id": ["bad"]
        }, instance.topic_token, origin)
    assert exc.value.code == 400
    assert topic_classifier.load_catalog(catalog_path)["overrides"] == {}


def test_localhost_editor_persists_subject_and_deliverable_overrides(server):
    instance, _, catalog_path = server
    origin = f"http://127.0.0.1:{instance.server_port}"
    with request(instance, "POST", "/api/topics", {
        "action": "create", "name": "Passkeys",
    }, instance.topic_token, origin) as response:
        topic_id = json.load(response)["topics"][0]["id"]
    key = topic_classifier.session_key("u", "s")
    with request(instance, "POST", "/api/topics", {
        "action": "override_tags", "user": "u", "session_id": "s", "topic_ids": [topic_id],
    }, instance.topic_token, origin):
        pass
    with request(instance, "POST", "/api/topics", {
        "action": "override_deliverables", "user": "u", "session_id": "s",
        "deliverables": ["deck", "doc"],
    }, instance.topic_token, origin):
        pass
    catalog = topic_classifier.load_catalog(catalog_path)
    assert catalog["tag_overrides"][key] == [topic_id]
    assert catalog["deliverable_overrides"][key] == ["deck", "doc"]


def test_localhost_editor_persists_family_correction(server):
    instance, _, catalog_path = server
    origin = f"http://127.0.0.1:{instance.server_port}"
    with request(instance, "POST", "/api/topics", {
        "action": "create", "name": "Worktree",
    }, instance.topic_token, origin) as response:
        topic_id = json.load(response)["topics"][0]["id"]
    with request(instance, "POST", "/api/topics", {
        "action": "create_family", "name": "Developer tooling",
    }, instance.topic_token, origin) as response:
        family_id = json.load(response)["families"][0]["id"]
    with request(instance, "POST", "/api/topics", {
        "action": "assign_family", "topic_id": topic_id, "family_id": family_id,
    }, instance.topic_token, origin):
        pass
    assert topic_classifier.load_catalog(catalog_path)["family_overrides"][topic_id] == family_id
