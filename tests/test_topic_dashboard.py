# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import json
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

import dashboard
import topic_classifier


def test_topic_totals_count_sessions_not_model_rows(tmp_path, monkeypatch):
    path = tmp_path / "topics.json"
    catalog = topic_classifier.empty_catalog()
    catalog["topics"] = [{
        "id": "nova", "name": "NOVA", "examples": ["NOVA architecture"],
        "keywords": ["NOVA"], "kind": "user",
    }]
    topic_classifier.save_catalog(path, catalog)
    rows = [
        dict(user="u", session_id="a", task_summary="NOVA architecture",
             project="repo", model=model, date="2026-01-01", calls=1,
             cost_data_calls=1, total_tokens=tokens, total_nano_aiu=cost)
        for model, tokens, cost in (("gpt-4o", 100, 1e10), ("gpt-5", 200, 2e10))
    ]
    rows.append(dict(user="u", session_id="b", task_summary="NOVA design",
                     project="repo", model="gpt-4o", date="2026-01-02",
                     calls=1, cost_data_calls=1, total_tokens=50, total_nano_aiu=5e9))
    out = tmp_path / "dashboard.html"
    dashboard.build_dashboard(pd.DataFrame(rows), str(out), "Topics", [], [], "topics", topic_file=path)
    html = out.read_text(encoding="utf-8")
    assert 'id="fig_topic"' in html and 'id="topic-table"' in html
    if shutil.which("node"):
        result = subprocess.run(
            ["node", str(Path(__file__).with_name("dom_harness.js")), str(out)],
            text=True, capture_output=True, check=True,
        )
        rendered = json.loads(result.stdout)
        assert re.search(
            r"<tr><td>NOVA</td><td>2</td>\s*<td>350</td><td>\$0.35</td>",
            rendered["elements"]["topic-table"]["innerHTML"],
        )
        assert "<th" in rendered["elements"]["table-wrap"]["innerHTML"]
        assert "Topic" in rendered["elements"]["table-wrap"]["innerHTML"]
        assert "NOVA" in rendered["elements"]["table-wrap"]["innerHTML"]


def test_local_slm_summary_is_used_for_topics_but_not_embedded_in_html(tmp_path):
    path = tmp_path / "topics.json"
    catalog = topic_classifier.empty_catalog()
    catalog["topics"] = [{
        "id": "auth", "name": "Authentication", "examples": [],
        "keywords": ["sign-in"], "kind": "user",
    }]
    topic_classifier.save_catalog(path, catalog)
    data = pd.DataFrame([dict(
        user="u", session_id="a", task_summary="Generic launcher boilerplate",
        topic_summary="Sign-in flow refinement PRIVATE-MARKER",
        project="repo", model="gpt-4o", date="2026-01-01",
        calls=1, total_tokens=1, total_nano_aiu=1e9,
    )])
    out = tmp_path / "dashboard.html"
    dashboard.build_dashboard(data, str(out), "Topics", [], [], "slm-topics", topic_file=path)
    html = out.read_text(encoding="utf-8")
    assert "Authentication" in html
    assert "PRIVATE-MARKER" not in html


def test_topic_classification_rejects_summary_redaction_to_prevent_leaks(tmp_path):
    path = tmp_path / "topics.json"
    out = tmp_path / "dashboard.html"
    data = pd.DataFrame([dict(
        user="u", session_id="a", task_summary="Secret project design",
        project="secret", model="gpt-4o", date="2026-01-01", calls=1,
        total_tokens=1, total_nano_aiu=1e9,
    )])
    with pytest.raises(ValueError, match="cannot be combined"):
        dashboard.build_dashboard(
            data, str(out), "Topics", [], [], "topics", topic_file=path,
            omit_task_summaries=True,
        )
    assert not out.exists() and not path.exists()


def test_topic_filter_combines_with_model_filter(tmp_path):
    path = tmp_path / "topics.json"
    catalog = topic_classifier.empty_catalog()
    catalog["topics"].append({
        "id": "mcp", "name": "MCP", "examples": ["MCP tools"],
        "keywords": ["MCP"], "kind": "user",
    })
    catalog["overrides"][topic_classifier.session_key("u", "b")] = "other"
    topic_classifier.save_catalog(path, catalog)
    rows = [
        dict(user="u", session_id=sid, task_summary=summary, project="repo",
             model=model, date="2026-01-01", calls=1, cost_data_calls=1,
             total_tokens=100, total_nano_aiu=1e10)
        for sid, summary, model in (
            ("a", "MCP tools", "gpt-5"), ("a", "MCP tools", "gpt-4o"),
            ("b", "Unrelated", "gpt-4o"),
        )
    ]
    out = tmp_path / "dashboard.html"
    dashboard.build_dashboard(pd.DataFrame(rows), str(out), "Topics", [], [], "topic-filter", topic_file=path)
    if not shutil.which("node"):
        return
    runner = Path(__file__).with_name("dom_harness.js")
    result = subprocess.run(
        ["node", str(runner), str(out), "{}", json.dumps([
            {"name": "setTopicFilter", "args": ["mcp"]},
            {"name": "setDateFilter", "args": ["all"]},
        ])],
        text=True, capture_output=True, check=True,
    )
    rendered = json.loads(result.stdout)
    assert rendered["filteredCount"] == 2
    assert "<td>MCP</td><td>1</td>" in rendered["elements"]["topic-table"]["innerHTML"]
    assert "$0.20" in rendered["elements"]["topic-table"]["innerHTML"]
    result = subprocess.run(
        ["node", str(runner), str(out),
         json.dumps({"copilot_usage_excluded_models::topic-filter": '["gpt-5"]'}),
         json.dumps([{"name": "setTopicFilter", "args": ["mcp"]}])],
        text=True, capture_output=True, check=True,
    )
    rendered = json.loads(result.stdout)
    assert rendered["filteredCount"] == 1
    assert "<td>MCP</td><td>1</td>" in rendered["elements"]["topic-table"]["innerHTML"]
    assert "$0.10" in rendered["elements"]["topic-table"]["innerHTML"]


def test_excluded_project_topic_name_not_embedded(tmp_path):
    path = tmp_path / "topics.json"
    catalog = topic_classifier.empty_catalog()
    catalog["topics"] = [
        {"id": "secret", "name": "Private Project Topic", "kind": "user",
         "keywords": ["Secret"], "examples": ["Secret details"]},
        {"id": "public", "name": "Public Topic", "kind": "user",
         "keywords": ["Public"], "examples": ["Public details"]},
    ]
    topic_classifier.save_catalog(path, catalog)
    data = pd.DataFrame([
        dict(user="u", session_id=sid, task_summary=summary, project=project,
             model="gpt-4o", date="2026-01-01", calls=1, total_tokens=10,
             total_nano_aiu=1e9)
        for sid, project, summary in (
            ("1", "private", "Secret details"), ("2", "public", "Public details")
        )
    ])
    out = tmp_path / "dashboard.html"
    dashboard.build_dashboard(
        data, str(out), "Topics", [], [], "topic-privacy",
        topic_file=path, exclude_projects=["private"],
    )
    html = out.read_text(encoding="utf-8")
    assert "Private Project Topic" not in html
    assert "Public Topic" in html


def test_topic_name_is_escaped_in_script_and_table(tmp_path):
    path = tmp_path / "topics.json"
    name = '</script><script>alert("unsafe")</script>'
    catalog = topic_classifier.empty_catalog()
    catalog["topics"].append({
        "id": "xss", "name": name, "kind": "user",
        "examples": ["Safe topic"], "keywords": ["Safe"],
    })
    topic_classifier.save_catalog(path, catalog)
    out = tmp_path / "dashboard.html"
    dashboard.build_dashboard(pd.DataFrame([dict(
        user="u", session_id="s", task_summary="Safe topic", project="repo",
        model="gpt-4o", date="2026-01-01", calls=1, total_tokens=10,
        total_nano_aiu=1e9,
    )]), str(out), "Topics", [], [], "topic-xss", topic_file=path)
    html = out.read_text(encoding="utf-8")
    start = html.index("let TOPIC_CATALOG = ")
    end = html.index(";\nconst TOPIC_EDIT_TOKEN", start)
    assert "</script>" not in html[start:end]
    if shutil.which("node"):
        result = subprocess.run(
            ["node", str(Path(__file__).with_name("dom_harness.js")), str(out)],
            text=True, capture_output=True, check=True,
        )
        table = json.loads(result.stdout)["elements"]["topic-table"]["innerHTML"]
        assert "&lt;/script&gt;" in table
        assert "</script>" not in table
