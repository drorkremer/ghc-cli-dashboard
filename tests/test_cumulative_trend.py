# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

import dashboard


pytestmark = pytest.mark.skipif(not shutil.which("node"), reason="Node.js not installed")
HARNESS = Path(__file__).with_name("dom_harness.js")


def render(path, actions=(), seed=None):
    result = subprocess.run(
        ["node", str(HARNESS), str(path), json.dumps(seed or {}), json.dumps(actions)],
        text=True, capture_output=True, check=True,
    )
    return json.loads(result.stdout)


def action(name, *args):
    return {"name": name, "args": list(args)}


def trend(result):
    return result["figures"]["fig_trend"]["data"][0]


def test_cumulative_day_respects_filters_date_window_metric_and_gap_days(tmp_path):
    data = pd.DataFrame([
        dict(user="u", project=project, model=model, date=date, calls=1,
             total_tokens=tokens, total_nano_aiu=tokens * 1e9)
        for date, project, model, tokens in [
            ("2026-01-01", "A", "gpt-5", 10),
            ("2026-01-03", "A", "gpt-5", 30),
            ("2026-01-05", "A", "gpt-5", 20),
            ("2026-01-03", "B", "claude-opus", 70),
        ]
    ])
    out = tmp_path / "trend.html"
    dashboard.build_dashboard(data, str(out), "Trend", [], [], "cumulative")
    baseline = render(out)
    assert trend(baseline)["x"] == ["2026-01-01", "2026-01-03", "2026-01-05"]
    assert trend(baseline)["y"] == [10, 100, 20]
    assert baseline["elements"]["trend-period"]["attributes"]["aria-pressed"] == "true"

    cumulative = render(out, [action("setTrendMode", "cumulative")])
    trace = trend(cumulative)
    assert trace["x"] == [
        "2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05",
    ]
    assert trace["y"] == [10, 10, 110, 110, 130]
    assert trace["customdata"] == [10, 0, 100, 0, 20]
    assert "%{customdata" in trace["hovertemplate"] and "%{y" in trace["hovertemplate"]
    assert cumulative["elements"]["trend-cumulative"]["attributes"]["aria-pressed"] == "true"
    widened = render(out, [
        action("setTrendMode", "cumulative"), action("setDateFilter", "7"),
    ])
    assert trend(widened)["x"][0] == "2025-12-30"
    assert trend(widened)["y"][:3] == [0, 0, 10]

    selected = render(out, [
        action("setTrendMode", "cumulative"),
        action("setDateFilter", "3"),
        action("setMetric", "cost"),
    ], seed={"copilot_usage_excluded_projects::cumulative": '["B"]'})
    trace = trend(selected)
    assert trace["x"] == ["2026-01-03", "2026-01-04", "2026-01-05"]
    assert trace["y"] == pytest.approx([0.30, 0.30, 0.50])
    assert trace["customdata"] == pytest.approx([0.30, 0, 0.20])
    assert selected["elements"]["trend-cumulative"]["attributes"]["aria-pressed"] == "true"
    assert selected["elements"]["kpi-row"]["innerHTML"].count("$0.50") >= 1
    zoomed = render(out, [
        action("setTrendMode", "cumulative"),
        action("zoomChart", "fig_trend", {
            "xaxis.range[0]": "2026-01-02 10:00:00",
            "xaxis.range[1]": "2026-01-04 18:00:00",
        }),
    ])
    assert zoomed["filteredCount"] == 2
    assert trend(zoomed)["x"] == ["2026-01-02", "2026-01-03", "2026-01-04"]
    assert trend(zoomed)["y"] == [0, 100, 100]
    assert zoomed["storage"]["copilot_usage_datefilter::cumulative"] == "custom"
    assert "2026-01-02" in zoomed["elements"]["date-range-hint"]["textContent"]
    quick_filter = render(out, [
        action("zoomChart", "fig_trend", {"xaxis.range": ["2026-01-02", "2026-01-04"]}),
        action("setDateFilter", "7"),
    ])
    assert "copilot_usage_date_range::cumulative" not in quick_filter["storage"]
    assert quick_filter["elements"]["date-7"]["attributes"]["aria-pressed"] == "true"


def test_cumulative_week_and_month_include_empty_buckets(tmp_path):
    data = pd.DataFrame([
        dict(user="u", project="A", model="gpt-5", date=date, calls=1,
             total_tokens=tokens, total_nano_aiu=tokens * 1e9)
        for date, tokens in [
            ("2026-01-01", 10), ("2026-01-03", 30), ("2026-01-05", 20),
        ]
    ])
    out = tmp_path / "trend.html"
    dashboard.build_dashboard(data, str(out), "Trend", [], [], "buckets")
    weekly = render(out, [action("setTrendMode", "cumulative"), action("setTrendGranularity", "week")])
    assert trend(weekly)["x"] == ["2025-12-29", "2026-01-05"]
    assert trend(weekly)["y"] == [40, 60]
    week_zoom = render(out, [
        action("setTrendGranularity", "week"),
        action("zoomChart", "fig_trend", {"xaxis.range": ["2026-01-02", "2026-01-05"]}),
    ])
    assert "2025-12-29" in week_zoom["elements"]["date-range-hint"]["textContent"]
    assert "2026-01-11" in week_zoom["elements"]["date-range-hint"]["textContent"]
    assert week_zoom["filteredCount"] == 3

    months = pd.DataFrame([
        dict(user="u", project="A", model="gpt-5", date=date, calls=1,
             total_tokens=tokens, total_nano_aiu=tokens * 1e9)
        for date, tokens in [("2026-01-01", 10), ("2026-03-01", 20)]
    ])
    dashboard.build_dashboard(months, str(out), "Trend", [], [], "buckets")
    monthly = render(out, [action("setTrendMode", "cumulative"), action("setTrendGranularity", "month")])
    assert trend(monthly)["x"] == ["2026-01", "2026-02", "2026-03"]
    assert trend(monthly)["y"] == [10, 10, 30]
    month_zoom = render(out, [
        action("setTrendMode", "cumulative"), action("setTrendGranularity", "month"),
        action("zoomChart", "fig_trend", {"xaxis.range": ["2026-01", "2026-02"]}),
    ])
    assert month_zoom["filteredCount"] == 1
    assert trend(month_zoom)["x"] == ["2026-01", "2026-02"]
    assert trend(month_zoom)["y"] == [10, 10]
