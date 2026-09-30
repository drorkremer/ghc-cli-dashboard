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
            r'<tr><td><button class="session-open topic-drill" data-topic-id="nova">NOVA</button></td><td>2</td>\s*<td>350</td><td>\$0.35</td>',
            rendered["elements"]["topic-table"]["innerHTML"],
        )
        assert "<th" in rendered["elements"]["table-wrap"]["innerHTML"]
        assert "Topic" in rendered["elements"]["table-wrap"]["innerHTML"]
        assert "NOVA" in rendered["elements"]["table-wrap"]["innerHTML"]


def test_local_slm_summary_is_visible_for_review_and_used_for_topics(tmp_path):
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
    assert "PRIVATE-MARKER" in html
    if shutil.which("node"):
        result = subprocess.run(
            ["node", str(Path(__file__).with_name("dom_harness.js")), str(out)],
            text=True, capture_output=True, check=True,
        )
        task_table = json.loads(result.stdout)["elements"]["table-wrap"]["innerHTML"]
        assert "Sign-in flow refinement PRIVATE-MARKER" in task_table
        assert "Generic launcher boilerplate" not in task_table


def test_unclassified_multi_topic_does_not_inherit_generic_task_topic(tmp_path):
    path = tmp_path / "topics.json"
    catalog = topic_classifier.empty_catalog()
    catalog["topics"] = [{
        "id": "mcp", "name": "MCP", "examples": ["MCP tools"],
        "keywords": ["MCP"], "kind": "user",
    }]
    topic_classifier.save_catalog(path, catalog)
    data = pd.DataFrame([dict(
        user="u", session_id="a", task_summary="MCP tools",
        topic_subjects="[]", topic_deliverables="[]", topic_status="error",
        topic_review="[]", topic_summary="",
        project="repo", model="gpt-4o", date="2026-01-01",
        calls=1, total_tokens=1, total_nano_aiu=1e9,
    )])
    out = tmp_path / "dashboard.html"
    dashboard.build_dashboard(data, str(out), "Topics", [], [], "multi-topics", topic_file=path)
    html = out.read_text(encoding="utf-8")
    assert '"topic_id": "other"' in html
    assert '"topic_status": "error"' in html


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
    assert 'data-topic-id="mcp">MCP</button></td><td>1</td>' in rendered["elements"]["topic-table"]["innerHTML"]
    assert "$0.20" in rendered["elements"]["topic-table"]["innerHTML"]
    result = subprocess.run(
        ["node", str(runner), str(out),
         json.dumps({"copilot_usage_excluded_models::topic-filter": '["gpt-5"]'}),
         json.dumps([{"name": "setTopicFilter", "args": ["mcp"]}])],
        text=True, capture_output=True, check=True,
    )
    rendered = json.loads(result.stdout)
    assert rendered["filteredCount"] == 1
    assert 'data-topic-id="mcp">MCP</button></td><td>1</td>' in rendered["elements"]["topic-table"]["innerHTML"]
    assert "$0.10" in rendered["elements"]["topic-table"]["innerHTML"]
    result = subprocess.run(
        ["node", str(runner), str(out), "{}", json.dumps([
            {"name": "setTopicFilter", "args": [["mcp", "other"]]},
        ])],
        text=True, capture_output=True, check=True,
    )
    rendered = json.loads(result.stdout)
    assert rendered["filteredCount"] == 3
    assert rendered["storage"]["copilot_usage_topic::topic-filter"] == '["mcp","other"]'
    assert 'multiple' in out.read_text(encoding="utf-8").split('id="topic-filter"')[1][:120]
    assert "2 selected" in rendered["elements"]["topic-selection-summary"]["textContent"]
    result = subprocess.run(
        ["node", str(runner), str(out),
         json.dumps({"copilot_usage_topic::topic-filter": "mcp"})],
        text=True, capture_output=True, check=True,
    )
    assert json.loads(result.stdout)["filteredCount"] == 2
    result = subprocess.run(
        ["node", str(runner), str(out), "{}", json.dumps([
            {"name": "setTopicFilter", "args": [["mcp", "other"]]},
            {"name": "setTopicFilter", "args": [[]]},
        ])],
        text=True, capture_output=True, check=True,
    )
    assert json.loads(result.stdout)["filteredCount"] == 3


def test_topic_choices_follow_deliverable_and_date_without_stale_selection(tmp_path):
    path = tmp_path / "topics.json"
    catalog = topic_classifier.empty_catalog()
    catalog["topics"] = [
        {"id": name, "name": name.title(), "kind": "user", "keywords": [name], "examples": []}
        for name in ("passkey", "rollout", "security", "api")
    ]
    catalog["families"] = [
        {"id": "access", "name": "Access"},
        {"id": "audit", "name": "Audit"},
        {"id": "platform", "name": "Platform"},
    ]
    catalog["family_assignments"] = {
        "passkey": "access", "rollout": "access",
        "security": "audit", "api": "platform",
    }
    topic_classifier.save_catalog(path, catalog)
    data = pd.DataFrame([
        dict(user="u", session_id=sid, project=project, model=model,
             date=date, task_summary="Launcher", topic_subjects=json.dumps(subjects),
             topic_summary=subjects[0], topic_deliverables=json.dumps(deliverables),
             calls=1, cost_data_calls=1, total_tokens=100, total_nano_aiu=cost)
        for sid, date, project, model, subjects, deliverables, cost in (
            ("s1", "2026-02-20", "repo", "gpt-5", ["Passkey rollout", "Rollout planning"], ["doc"], 1e10),
            ("s2", "2026-01-01", "repo", "gpt-5", ["Security audit"], ["doc"], 2e10),
            ("s3", "2026-01-01", "other", "gpt-4o", ["API migration"], ["deck"], 3e10),
        )
    ])
    out = tmp_path / "dashboard.html"
    dashboard.build_dashboard(data, str(out), "Topics", [], [], "cascading", topic_file=path)
    if not shutil.which("node"):
        return

    def render(actions):
        result = subprocess.run(
            ["node", str(Path(__file__).with_name("dom_harness.js")),
             str(out), "{}", json.dumps(actions)],
            text=True, capture_output=True, check=True,
        )
        return json.loads(result.stdout)

    by_deliverable = render([{"name": "setDeliverableFilter", "args": ["doc"]}])
    choices = by_deliverable["elements"]["topic-filter"]["innerHTML"]
    assert 'value="passkey"' in choices and 'value="rollout"' in choices
    assert 'value="security"' in choices and 'value="api"' not in choices
    by_date = render([
        {"name": "setDeliverableFilter", "args": ["doc"]},
        {"name": "setDateFilter", "args": ["7"]},
    ])
    choices = by_date["elements"]["topic-filter"]["innerHTML"]
    assert 'value="passkey"' in choices and 'value="rollout"' in choices
    assert 'value="security"' not in choices and 'value="api"' not in choices
    assert by_date["filteredCount"] == 1
    assert 'value="audit"' not in by_date["elements"]["family-filter"]["innerHTML"]
    assert 'value="access"' in by_date["elements"]["family-filter"]["innerHTML"]
    by_family = render([{"name": "setFamilyFilter", "args": ["audit"]}])
    assert by_family["filteredCount"] == 1
    assert 'value="security"' in by_family["elements"]["topic-filter"]["innerHTML"]
    assert 'value="passkey"' not in by_family["elements"]["topic-filter"]["innerHTML"]
    by_topic = render([{"name": "setTopicFilter", "args": [["passkey"]]}])
    assert 'value="access"' in by_topic["elements"]["family-filter"]["innerHTML"]
    assert 'value="audit"' not in by_topic["elements"]["family-filter"]["innerHTML"]
    assert 'value="platform"' not in by_topic["elements"]["family-filter"]["innerHTML"]
    by_two_topics = render([{"name": "setTopicFilter", "args": [["passkey", "api"]]}])
    assert 'value="access"' in by_two_topics["elements"]["family-filter"]["innerHTML"]
    assert 'value="platform"' in by_two_topics["elements"]["family-filter"]["innerHTML"]
    assert 'value="audit"' not in by_two_topics["elements"]["family-filter"]["innerHTML"]
    by_two_families = render([{"name": "setFamilyFilter", "args": [["access", "platform"]]}])
    assert by_two_families["filteredCount"] == 2
    assert 'value="security"' not in by_two_families["elements"]["topic-filter"]["innerHTML"]
    assert by_two_families["storage"]["copilot_usage_family::cascading"] == '["access","platform"]'
    by_two_deliverables = render([{"name": "setDeliverableFilter", "args": [["doc", "deck"]]}])
    assert by_two_deliverables["filteredCount"] == 3
    assert by_two_deliverables["storage"]["copilot_usage_deliverable::cascading"] == '["doc","deck"]'
    assert 'multiple' in out.read_text(encoding="utf-8").split('id="deliverable-filter"')[1][:110]
    narrowed_families = render([
        {"name": "setFamilyFilter", "args": [["access", "platform"]]},
        {"name": "setDateFilter", "args": ["7"]},
    ])
    assert narrowed_families["filteredCount"] == 1
    assert narrowed_families["storage"]["copilot_usage_family::cascading"] == '["access"]'
    assert "1 selection(s) cleared" in narrowed_families["elements"]["family-selection-summary"]["textContent"]
    narrowed_deliverables = render([
        {"name": "setDeliverableFilter", "args": [["doc", "deck"]]},
        {"name": "setDateFilter", "args": ["7"]},
    ])
    assert narrowed_deliverables["storage"]["copilot_usage_deliverable::cascading"] == '["doc"]'
    narrowed_topics = render([
        {"name": "setTopicFilter", "args": [["passkey", "api"]]},
        {"name": "setDateFilter", "args": ["7"]},
    ])
    assert narrowed_topics["storage"]["copilot_usage_topic::cascading"] == '["passkey"]'
    assert 'value="access"' in narrowed_topics["elements"]["family-filter"]["innerHTML"]
    legacy = subprocess.run(
        ["node", str(Path(__file__).with_name("dom_harness.js")), str(out),
         json.dumps({"copilot_usage_family::cascading": "access",
                     "copilot_usage_deliverable::cascading": "doc"})],
        text=True, capture_output=True, check=True,
    )
    assert json.loads(legacy.stdout)["filteredCount"] == 1
    cleared = render([
        {"name": "setFamilyFilter", "args": [["access", "platform"]]},
        {"name": "setDeliverableFilter", "args": [["doc", "deck"]]},
        {"name": "resetFilters"},
    ])
    assert cleared["filteredCount"] == 3
    assert not any(key.startswith("copilot_usage_family::") or
                   key.startswith("copilot_usage_deliverable::")
                   for key in cleared["storage"])
    expired_family = render([
        {"name": "setFamilyFilter", "args": ["audit"]},
        {"name": "setDateFilter", "args": ["7"]},
    ])
    assert expired_family["filteredCount"] == 1
    assert "cleared" in expired_family["elements"]["family-selection-summary"]["textContent"]
    assert expired_family["storage"].get("copilot_usage_family::cascading") is None
    incompatible = render([
        {"name": "setTopicFilter", "args": [["api"]]},
        {"name": "setDeliverableFilter", "args": ["doc"]},
    ])
    assert incompatible["filteredCount"] == 2
    assert 'value="api"' not in incompatible["elements"]["topic-filter"]["innerHTML"]
    assert "cleared" in incompatible["elements"]["topic-selection-summary"]["textContent"]
    assert incompatible["storage"].get("copilot_usage_topic::cascading") is None
    zoomed = render([{"name": "zoomChart", "args": ["fig_trend", {
        "xaxis.range[0]": "2026-01-01 00:00:00",
        "xaxis.range[1]": "2026-01-01 23:59:59",
    }]}])
    assert zoomed["filteredCount"] == 2
    assert 'value="passkey"' not in zoomed["elements"]["topic-filter"]["innerHTML"]
    assert 'value="security"' in zoomed["elements"]["topic-filter"]["innerHTML"]
    assert 'value="access"' not in zoomed["elements"]["family-filter"]["innerHTML"]
    assert 'value="audit"' in zoomed["elements"]["family-filter"]["innerHTML"]
    assert "doc" in zoomed["elements"]["deliverable-filter"]["innerHTML"]
    assert "deck" in zoomed["elements"]["deliverable-filter"]["innerHTML"]
    assert "$0.50" in zoomed["elements"]["session-status"]["textContent"]
    assert "2026-01-01" in zoomed["elements"]["date-range-hint"]["textContent"]
    reset = render([
        {"name": "zoomChart", "args": ["fig_trend", {
            "xaxis.range": ["2026-01-01", "2026-01-01"],
        }]},
        {"name": "resetChartZoom", "args": ["fig_trend"]},
    ])
    assert reset["filteredCount"] == 3
    assert reset["elements"]["date-all"]["attributes"]["aria-pressed"] == "true"
    by_project = render([{"name": "selectOnly", "args": ["project", 0]}])
    assert 'value="api"' not in by_project["elements"]["topic-filter"]["innerHTML"]
    by_model = subprocess.run(
        ["node", str(Path(__file__).with_name("dom_harness.js")), str(out),
         json.dumps({"copilot_usage_excluded_models::cascading": '["gpt-4o"]'})],
        text=True, capture_output=True, check=True,
    )
    assert 'value="api"' not in json.loads(by_model.stdout)["elements"]["topic-filter"]["innerHTML"]


def test_excluded_project_topic_name_not_embedded(tmp_path):
    path = tmp_path / "topics.json"
    catalog = topic_classifier.empty_catalog()
    catalog["topics"] = [
        {"id": "secret", "name": "Private Project Topic", "kind": "user",
         "keywords": ["Secret"], "examples": ["Secret details"]},
        {"id": "public", "name": "Public Topic", "kind": "user",
         "keywords": ["Public"], "examples": ["Public details"]},
    ]
    catalog["families"] = [
        {"id": "private-family", "name": "Private Project Family"},
        {"id": "public-family", "name": "Public Family"},
    ]
    catalog["family_assignments"] = {
        "secret": "private-family", "public": "public-family",
    }
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
    assert "Private Project Family" not in html
    assert "Public Topic" in html
    assert "Public Family" in html


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


def test_subject_and_deliverable_filters_do_not_double_count_primary_cost(tmp_path):
    path = tmp_path / "topics.json"
    catalog = topic_classifier.empty_catalog()
    catalog["topics"] = [
        {"id": "passkey", "name": "Passkeys", "kind": "user",
         "keywords": ["passkey"], "examples": []},
        {"id": "rollout", "name": "Rollout", "kind": "user",
         "keywords": ["rollout"], "examples": []},
    ]
    catalog["families"] = [
        {"id": "access", "name": "Identity & access"},
        {"id": "planning", "name": "Project planning"},
    ]
    catalog["family_assignments"] = {"passkey": "access", "rollout": "planning"}
    topic_classifier.save_catalog(path, catalog)
    data = pd.DataFrame([
        dict(user="u", session_id=sid, project="repo", model="gpt-5",
             date="2026-01-01", task_summary="First read generic launcher",
             topic_subjects=json.dumps(subjects), topic_summary=subjects[0],
             topic_deliverables=json.dumps(deliverables), calls=1,
             cost_data_calls=1, total_tokens=100, total_nano_aiu=cost)
        for sid, subjects, deliverables, cost in (
            ("s1", ["Passkey adoption", "Rollout planning"], ["deck", "doc"], 1e10),
            ("s2", ["Rollout planning", "Passkey adoption"], ["code"], 2e10),
        )
    ])
    out = tmp_path / "dashboard.html"
    dashboard.build_dashboard(data, str(out), "Multi-topic", [], [], "multi", topic_file=path)
    assert "\x00" not in out.read_text(encoding="utf-8")
    if not shutil.which("node"):
        return

    def render(actions=()):
        result = subprocess.run(
            ["node", str(Path(__file__).with_name("dom_harness.js")),
             str(out), "{}", json.dumps(actions)],
            text=True, capture_output=True, check=True,
        )
        return json.loads(result.stdout)

    initial = render()
    assert "Filter by family" in out.read_text(encoding="utf-8")
    assert 'value="access"' in initial["elements"]["family-filter"]["innerHTML"]
    assert 'value="planning"' in initial["elements"]["family-filter"]["innerHTML"]
    table = initial["elements"]["topic-table"]["innerHTML"]
    assert 'data-topic-id="passkey">Passkeys</button></td><td>1</td>' in table
    assert 'data-topic-id="rollout">Rollout</button></td><td>1</td>' in table
    assert "$0.30" in initial["elements"]["session-status"]["textContent"]
    session_table = initial["elements"]["session-table"]["innerHTML"]
    assert "Passkeys" in session_table and "Rollout" in session_table
    assert "deck" in session_table and "doc" in session_table
    by_subject = render([{"name": "setTopicFilter", "args": ["rollout"]}])
    assert by_subject["filteredCount"] == 2
    assert "$0.30" in by_subject["elements"]["session-status"]["textContent"]
    assert 'value="access"' not in by_subject["elements"]["family-filter"]["innerHTML"]
    assert 'value="planning"' in by_subject["elements"]["family-filter"]["innerHTML"]
    by_both = render([{"name": "setTopicFilter", "args": [["passkey", "rollout"]]}])
    assert by_both["filteredCount"] == 2
    assert "$0.30" in by_both["elements"]["session-status"]["textContent"]
    by_family = render([{"name": "setFamilyFilter", "args": ["access"]}])
    assert by_family["filteredCount"] == 2  # matches primary or additional subject
    assert "$0.30" in by_family["elements"]["session-status"]["textContent"]
    assert 'value="planning"' not in by_family["elements"]["topic-filter"]["innerHTML"]
    by_families = render([{"name": "setFamilyFilter", "args": [["access", "planning"]]}])
    assert by_families["filteredCount"] == 2
    assert by_families["storage"]["copilot_usage_family::multi"] == '["access","planning"]'
    narrowed = render([
        {"name": "setFamilyFilter", "args": ["access"]},
        {"name": "setDeliverableFilter", "args": ["deck"]},
    ])
    assert narrowed["filteredCount"] == 1
    assert "$0.10" in narrowed["elements"]["session-status"]["textContent"]
    assert 'value="rollout"' not in narrowed["elements"]["topic-filter"]["innerHTML"]
    assert narrowed["storage"]["copilot_usage_family::multi"] == '["access"]'
    timed = render([{"name": "setDateFilter", "args": ["7"]}])
    assert timed["elements"]["family-filter"]["innerHTML"]
    assert by_both["elements"]["topic-table"]["innerHTML"].count('class="session-open topic-drill"') == 2
    assert by_both["storage"]["copilot_usage_topic::multi"] == '["passkey","rollout"]'
    primary_chart = render([{"name": "clickChart", "args": ["fig_topic", {"customdata": "rollout"}]}])
    assert primary_chart["elements"]["session-table"]["innerHTML"].count('data-session-key=') == 1
    assert "$0.20" in primary_chart["elements"]["session-status"]["textContent"]
    by_deliverable = render([{"name": "setDeliverableFilter", "args": ["deck"]}])
    assert by_deliverable["filteredCount"] == 1
    assert "$0.10" in by_deliverable["elements"]["session-status"]["textContent"]
    grouped = render([{"name": "setSessionGroup", "args": ["deliverable", "deck"]}])
    assert grouped["elements"]["session-table"]["innerHTML"].count('data-session-key=') == 1
    assert "$0.10" in grouped["elements"]["session-status"]["textContent"]
    merged_catalog = topic_classifier.load_catalog(path)
    topic_classifier.apply_edit(merged_catalog, {
        "action": "merge", "source_id": "passkey", "target_id": "rollout",
    })
    merged = render([{"name": "applyTopicCatalog", "args": [merged_catalog]}])
    assert "Passkeys" not in merged["elements"]["session-table"]["innerHTML"]
    assert "Rollout" in merged["elements"]["session-table"]["innerHTML"]


def test_invalid_multi_topic_csv_fails_instead_of_silently_losing_tags(tmp_path):
    path = tmp_path / "topics.json"
    out = tmp_path / "dashboard.html"
    data = pd.DataFrame([dict(
        user="u", session_id="s", project="repo", model="gpt-5",
        date="2026-01-01", calls=1, total_tokens=10,
        topic_subjects='["Authentication",', topic_deliverables='["deck"]',
    )])
    with pytest.raises(ValueError, match="topic_subjects must be a JSON list"):
        dashboard.build_dashboard(data, str(out), "Bad tags", [], [], "bad", topic_file=path)
    assert not out.exists() and not path.exists()
