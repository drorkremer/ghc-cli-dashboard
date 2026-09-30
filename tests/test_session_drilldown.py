# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

import dashboard
from topic_classifier import empty_catalog, save_catalog


HARNESS = Path(__file__).with_name("dom_harness.js")


def render_dom(path, actions=()):
    return json.loads(subprocess.run(
        ["node", str(HARNESS), str(path), "{}", json.dumps(actions)],
        text=True, capture_output=True, check=True,
    ).stdout)


@pytest.mark.skipif(not shutil.which("node"), reason="Node.js not installed")
def test_session_explorer_reconciles_filter_and_group_costs(tmp_path):
    topics = empty_catalog()
    topics["topics"] = [
        {"id": "auth", "name": "Authentication", "kind": "user",
         "keywords": ["Sign-in"], "examples": []},
        {"id": "deploy", "name": "Deployment", "kind": "user",
         "keywords": ["Release"], "examples": []},
    ]
    catalog = tmp_path / "topics.json"
    save_catalog(catalog, topics)
    rows = [
        ("s1", "Sign-in flow", "alpha", "model-a", "2026-01-01", 100, 1e10),
        ("s1", "Sign-in flow", "alpha", "model-b", "2026-01-02", 200, 2e10),
        ("s2", "Release rollout", "alpha", "model-a", "2026-01-02", 50, 5e9),
        ("s3", "Sign-in tokens", "beta", "model-b", "2026-01-02", 70, 7e9),
    ]
    data = pd.DataFrame([
        dict(user="u", session_id=sid, task_summary=summary,
             project=project, model=model, date=date, calls=1,
             cost_data_calls=1, total_tokens=tokens, total_nano_aiu=aiu)
        for sid, summary, project, model, date, tokens, aiu in rows
    ])
    out = tmp_path / "dashboard.html"
    dashboard.build_dashboard(data, str(out), "Sessions", [], [], "session-scope", topic_file=catalog)
    initial = render_dom(out)
    session_table = initial["elements"]["session-table"]["innerHTML"]
    assert session_table.count('data-session-key=') == 3
    assert initial["elements"]["kpi-row"]["innerHTML"].count('class="kpi"') == 4
    assert '<div class="kpi-value">3</div>' in initial["elements"]["kpi-row"]["innerHTML"]
    assert "s1" in session_table and "s2" in session_table and "s3" in session_table
    assert "$0.30" in session_table and "300" in session_table
    session_key = json.dumps(["u", "s1"], separators=(",", ":"))
    detail = render_dom(out, [{"name": "selectSession", "args": [session_key]}])
    breakdown = detail["elements"]["session-detail"]["innerHTML"]
    assert "model-a" in breakdown and "model-b" in breakdown
    assert "$0.10" in breakdown and "$0.20" in breakdown and "$0.30" in breakdown

    auth_only = render_dom(out, [
        {"name": "setTopicFilter", "args": ["auth"]},
        {"name": "setSessionGroup", "args": ["model", "model-b"]},
        {"name": "selectSession", "args": [session_key]},
    ])
    table = auth_only["elements"]["session-table"]["innerHTML"]
    assert table.count('data-session-key=') == 2
    assert '<div class="kpi-value">2</div>' in auth_only["elements"]["kpi-row"]["innerHTML"]
    assert "s1" in table and "s3" in table and "s2" not in table
    assert "$0.20" in auth_only["elements"]["session-detail"]["innerHTML"]
    assert "$0.30" not in auth_only["elements"]["session-detail"]["innerHTML"]
    dated = render_dom(out, [
        {"name": "setDateFilter", "args": ["1"]},
        {"name": "selectSession", "args": [session_key]},
    ])
    assert "$0.20" in dated["elements"]["session-detail"]["innerHTML"]
    assert "$0.30" not in dated["elements"]["session-detail"]["innerHTML"]
    group = render_dom(out, [
        {"name": "drillToSessions", "args": ["project", "beta"]},
    ])
    assert group["elements"]["session-table"]["innerHTML"].count('data-session-key=') == 1
    assert "s3" in group["elements"]["session-table"]["innerHTML"]
    assert "$0.07" in group["elements"]["session-status"]["textContent"]
    chart = render_dom(out, [
        {"name": "clickChart", "args": ["fig_model", {"y": "model-b"}]},
    ])
    assert chart["elements"]["session-table"]["innerHTML"].count('data-session-key=') == 2
    assert "$0.27" in chart["elements"]["session-status"]["textContent"]
    topic_chart = render_dom(out, [
        {"name": "clickChart", "args": ["fig_topic", {"customdata": "deploy"}]},
    ])
    assert "s2" in topic_chart["elements"]["session-table"]["innerHTML"]
    assert "s1" not in topic_chart["elements"]["session-table"]["innerHTML"]
    trend = render_dom(out, [
        {"name": "clickChart", "args": ["fig_trend", {"x": "2026-01-01"}]},
    ])
    assert trend["elements"]["session-table"]["innerHTML"].count('data-session-key=') == 1
    assert "$0.10" in trend["elements"]["session-status"]["textContent"]


@pytest.mark.skipif(not shutil.which("node"), reason="Node.js not installed")
def test_legacy_rows_without_session_ids_are_not_falsely_merged(tmp_path):
    data = pd.DataFrame([
        dict(user="u", task_summary=f"Task {i}", project="repo",
             model="model-a", date="2026-01-01", calls=1,
             total_tokens=10, total_nano_aiu=1e9)
        for i in (1, 2)
    ])
    out = tmp_path / "dashboard.html"
    dashboard.build_dashboard(data, str(out), "Legacy", [], [], "legacy-sessions")
    rendered = render_dom(out)
    table = rendered["elements"]["session-table"]["innerHTML"]
    assert table.count("ID unavailable") == 2
    assert "Task 1" in table and "Task 2" in table
    assert "No session selected" in rendered["elements"]["session-detail"]["innerHTML"]
    kpis = rendered["elements"]["kpi-row"]["innerHTML"]
    assert "Identified sessions" in kpis
    assert "2 rows without session IDs excluded" in kpis
    assert '<div class="kpi-value">0</div>' in kpis


@pytest.mark.skipif(not shutil.which("node"), reason="Node.js not installed")
def test_session_kpi_distinguishes_users_with_same_session_id(tmp_path):
    data = pd.DataFrame([
        dict(user=user, session_id="same-id", task_summary="Some work", project="repo",
             model="gpt-5", date="2026-01-01", calls=1,
             cost_data_calls=1, total_tokens=10, total_nano_aiu=1e9)
        for user in ("alice", "bob")
    ])
    out = tmp_path / "dashboard.html"
    dashboard.build_dashboard(data, str(out), "Two users", [], [], "multi-user")
    rendered = render_dom(out)
    assert '<div class="kpi-value">2</div>' in rendered["elements"]["kpi-row"]["innerHTML"]
    assert rendered["elements"]["session-table"]["innerHTML"].count('data-session-key=') == 2


@pytest.mark.skipif(not shutil.which("node"), reason="Node.js not installed")
def test_session_explorer_escapes_summaries_and_bounds_rendered_rows(tmp_path):
    data = pd.DataFrame([
        dict(user="u", session_id=f"s{i:03}", task_summary="<img src=x onerror=alert(1)>",
             project="p", model="m", date="2026-01-01", calls=1, cost_data_calls=1,
             total_tokens=10, total_nano_aiu=1e9)
        for i in range(105)
    ])
    out = tmp_path / "dashboard.html"
    dashboard.build_dashboard(data, str(out), "Many", [], [], "many-sessions")
    rendered = render_dom(out, [{"name": "selectSession", "args": ['["u","s000"]']}])
    table = rendered["elements"]["session-table"]["innerHTML"]
    assert table.count('data-session-key=') == 100
    assert rendered["elements"]["session-more"]["hidden"] is False
    assert "<img " not in table
    assert "&lt;img " in table
    assert "<img " not in rendered["elements"]["session-detail"]["innerHTML"]
    expanded = render_dom(out, [{"name": "showMoreSessions"}])
    assert expanded["elements"]["session-table"]["innerHTML"].count('data-session-key=') == 105
