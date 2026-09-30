---
layout: default
title: Development
nav_order: 5
---

# Development

Install dependencies and run the test suite:

```powershell
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pytest tests/ -v
```

Optional topic discovery requires `python -m pip install -r requirements-topics.txt`;
the model weights download on first use. Topic tests inject a deterministic
encoder, so the normal test suite does not download a model or require the
optional package. For a real local smoke check, build a topic-enabled
dashboard from synthetic CSV data and inspect the localhost editor.

Optional local SLM summaries require `python -m pip install -r
requirements-summaries.txt` and a downloaded Qwen2.5 GGUF file. The
summarizer tests inject a deterministic generator, so CI does not download
weights or require GPT4All. The Windows x64 wheel is published, but a real
Windows inference smoke test is still needed before claiming full platform
validation.

The GitHub Actions workflow runs the test suite on Python 3.9 with Node.js.
Node executes the lightweight DOM harness used for generated-dashboard tests.
The harness also accepts a fourth argument containing a JSON array of named
UI actions (for example, `resetFilters` or `searchFilters`) for interaction
regressions.

`tests/test_dashboard_ux.py` includes browser layout and interaction checks at
390px, 768px and 1440px widths. These use the same optional Playwright/Chromium
installation as the screenshot tool below, and skip when it is unavailable.
Run them with `python -m pytest tests/test_dashboard_ux.py -q`.

Keep generated CSV and HTML files out of commits. They may contain personal
usage data, project names, and task summaries.

## Documentation screenshots

The screenshots in the README are built from synthetic data, so no real usage
data ever enters the repository. Regenerate them after changing the dashboard
layout:

```powershell
python -m pip install playwright
playwright install chromium

python tools/make_sample_data.py --out sample_usage.csv
python dashboard.py --in sample_usage.csv --out sample_dashboard.html
python tools/make_screenshots.py --html sample_dashboard.html --out-dir docs/images
```

`tools/make_sample_data.py` uses a fixed random seed and a fixed date range,
so repeated runs produce identical output. Both `sample_usage.csv` and
`sample_dashboard.html` are gitignored.

`dashboard.py` obtains the bundled Plotly.js source from
`plotly.offline.get_plotlyjs()`. When editing chart titles, use Plotly's
object form:

```javascript
title: { text: "Chart title" }
```
