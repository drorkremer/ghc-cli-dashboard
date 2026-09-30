---
layout: default
title: Advanced usage
nav_order: 4
---

# Advanced usage

## Privacy and sharing

Exports and generated dashboards can contain personal and confidential data:

- The default user label is the operating-system username.
- Project names, model names, dates, and session IDs are retained.
- Free-text task summaries are included by default (this tool is meant for
  personal viewing); pass `--exclude-task-summary` to `extract_usage.py` to
  leave them out, or `--omit-task-summaries` to `dashboard.py` to strip them
  at build time before sharing.
- A dashboard embeds its source rows in the HTML file.

Treat a CSV or generated dashboard as sensitive data. Review it before
sharing it.

The legacy `--include-task-summary` flag remains supported for existing
commands. It cannot be combined with `--exclude-task-summary`.

The sidebar filters and these options only change what the browser displays:

```powershell
python dashboard.py --exclude-default "Personal Project"
python dashboard.py --exclude-default-models "gemini-3.5-flash"
```

Use build-time options when content must be removed from a shared dashboard:

```powershell
python dashboard.py --in "copilot_usage_*.csv" --out shared_dashboard.html `
  --exclude-project "Personal Project" `
  --omit-task-summaries
```

`--exclude-project` can be repeated. It removes matching rows before the
dashboard is written. `--omit-task-summaries` removes all task-summary text.
These options do not anonymize the remaining fields.

The CSV export itself is intended for programmatic use. Import it as text
when opening it in spreadsheet software because project names and task
summaries can begin with formula characters.

## Combining exports

Point `dashboard.py` at a pattern that matches multiple CSV files:

```powershell
python dashboard.py --in "exports\copilot_usage_*.csv" --out combined_dashboard.html
```

Each export contains the available history at the time it was created.
`dashboard.py` removes overlapping rows by using the export timestamp and
session identity when available. Older exports without metadata still load,
but produce warnings because their ordering and identity are less certain.

Keep source CSV files in a location appropriate for their sensitivity. The
tool does not upload or transmit them.

## Topic classification

Topics are optional. Install `requirements-topics.txt`, then run
`python dashboard.py --in "copilot_usage_*.csv" --topics --serve`. The URL
printed by this command is an editable dashboard served **only on localhost**.
The ordinary generated HTML file is still available for read-only viewing.
The server saves topic changes atomically to
`~/.ghc-cli-dashboard/topics.json`; `--topics-file PATH` selects another
catalog. Keep this file private: it contains session IDs and potentially
sensitive topic names and example summaries. Do not commit it or share a
topic-enabled dashboard without reviewing its contents.

The local embedding model (`BAAI/bge-small-en-v1.5` through FastEmbed/ONNX)
is downloaded on first use (approximately 67 MB), with no paid service.
Only compact task summaries (or optional SLM-generated topic summaries) are
embedded. Assignment order is:

1. An explicit per-session correction in the catalog.
2. The saved automatic assignment from a previous build.
3. A user-defined topic's `keywords` (whole-word match).
4. Similarity to a user-defined topic's example summaries (cosine
   similarity at least 0.67), then to previously discovered topics.
5. Groups of at least two unmatched summaries at similarity 0.75 or higher
   become a discovered topic; unmatched singletons remain **Other**.

Discovered topics get stable IDs and names based on a shared summary word
or representative summary. They are approximate, not LLM-generated.
In editable mode, create a topic with a name and optional example, edit
example summaries and exact-phrase rules, rename or merge topics, or change
a session's selection. Changes to a session are visible immediately and
override future automatic builds; new matching rules are applied to
unassigned sessions on the *next* dashboard build. The topic
filter combines with existing project, model, provider, and date filters.
Session counts use distinct session IDs, while token/cost sums keep the
existing dashboard definitions. Rows from legacy CSVs without session IDs
remain in **Other** and do not contribute to the distinct-session count.
Session corrections are keyed by the export's user label and session ID;
changing `--user-label` when re-exporting will not carry over corrections.

Topic generation cannot be combined with `--omit-task-summaries`: a topic
name derived from a summary could reveal content that option was meant to
remove. Build-time `--exclude-project` runs before topic classification.
The browser cannot write the topic file from a standalone `file://` HTML
dashboard; run `--serve` for editing. The localhost editor checks a
session-specific token and same-origin requests; no external network
service receives your usage data. Only model weights are downloaded.

### Optional offline SLM summaries

If stored task summaries are generic, a separate, **opt-in** offline step
can summarize the first user turn. Install both optional requirements files:

```powershell
python -m pip install -r requirements-topics.txt -r requirements-summaries.txt
```

Download `qwen2.5-1.5b-instruct-q4_k_m.gguf` (about 1.1 GB) from the
[Qwen model repository](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF).
This step uses GPT4All's native GGUF runner; it does **not** require Ollama,
a background server, a cloud account, or a paid API. Pass the downloaded
file explicitly. The runner is never allowed to auto-download a model:

```powershell
python topic_summarizer.py --in usage.csv --out usage-with-topics.csv `
  --db "C:\first\.copilot\session-store.db" `
  --db "D:\other\.copilot\session-store.db" `
  --model "C:\Models\qwen2.5-1.5b-instruct-q4_k_m.gguf"
python dashboard.py --in usage-with-topics.csv --topics-file topics-slm.json --serve
```

Use the **same database order** as the extraction: the first store wins
when session IDs collide. `--limit 10` is useful for a trial; it leaves
the remaining sessions on their old task summaries, so do not interpret
the resulting topic totals as a fully classified dataset. Without a
limit, the first offline pass can take a while on a large store; the
versioned cache at `~/.ghc-cli-dashboard/topic-summary-cache.json`
avoids repeating unchanged work on subsequent runs. `--cache PATH` selects
a different cache. The limit counts **new** summaries per run; use
`--limit 0` to rebuild the export using only cached labels, or omit the
limit to finish the full dataset. The cache is keyed by user/session, the selected GGUF
file's metadata, and the input text; generated summaries are saved after
each session so an interrupted run can resume. No raw user turn is written
to the enriched CSV or cache. Both files may still contain sensitive
topic names and session IDs; keep them private. First turns are read from
read-only SQLite snapshots. Shared launcher lines are removed within
each store only when they occur in at least half of 20 or more first
messages; longer remaining text is summarized in bounded pieces and
combined. These are heuristics: inspect the results and use the
dashboard's manual corrections as needed.

Use a **new** `--topics-file` for the SLM-enriched export if the same
sessions were already classified from the old generic summaries: saved
automatic topic assignments intentionally retain their IDs until you
correct or merge them. The original CSV and existing cost calculations
are unchanged.

`extract_usage.py` accepts multiple `--db` arguments. Without one, it reads
`~/.copilot/session-store.db`; explicit arguments replace that default.
Pass each custom Copilot home's `session-store.db` to avoid partial usage.
Repeated references to the same physical path are silently deduplicated.
If the same session ID exists in different databases, a warning identifies
the overlap and the **first specified database wins** for that entire
session. Later rows for it are ignored, even if they contain newer calls.
The dashboard's separate deduplication of overlapping CSV exports still
applies after extraction.

## Filters and providers

Projects, models, providers, and date range all apply together. Disabling a
provider also excludes its models, even if the corresponding model checkboxes
remain selected. Those model controls are disabled and marked **Excluded by
provider** until the provider is included again. The saved model selection
does not change.

- Search projects and models without changing the active selection. The
  first eight matches are shown, with **Show more** revealing ten more.
- **Only** selects one project, model or provider within its group. Other
  groups and the date range still apply.
- **Select all** and **Select none** apply to the entire group, including
  items hidden by search or pagination.
- **Reset filters** includes every project, model, provider and date, and
  clears project/model searches. It does not restore build-time default
  exclusions, change the chart metric, or undo build-time redaction.

The sidebar's **Hide filters** button collapses the project/model/provider
list into a narrow reopen rail on desktop. The rail stays available while
scrolling, and the selection summary remains visible in the toolbar.
Reloading preserves the collapse choice, and resetting filters does not
reveal the panel. On small screens, the panel starts collapsed and expands
inline below the controls.

For demos, collapse the sidebar before presenting. This hides the filter list
only: project names remain in charts and task detail, and all source rows are
still embedded in the HTML. Use build-time redaction for a shareable artefact
that must not contain those projects.

Date presets are relative to the latest date in the loaded exports, not the
current date. The header shows **Data through** separately from the generation
time. The latest day, week or month can be incomplete.

Provider labels are inferred from model-name prefixes:

| Prefix | Provider |
| --- | --- |
| `claude*` | Anthropic |
| `gpt*`, `o1*`, `o3*`, `o4*` | OpenAI |
| `gemini*` | Google |
| `grok*` | xAI |
| Other values | `Other / Unknown` |

These labels help group usage. They are not billing metadata. To extend the
mapping, update `PROVIDER_PREFIX_RULES` in
[`provider_classifier.py`](../provider_classifier.py).

The provider palette is shared by the model ranking, model pricing-efficiency
chart and provider mix. Colours do not change with ranking, date range or
filter selection. Inline provider legends accompany those model charts.

**Model Mix per Top Project** instead uses a distinct colour for each model,
including models from the same provider. It uses Plotly's Alphabet categorical
palette, extended with non-repeating generated swatches beyond 26 models.
Assignments are made from the full exported model list in alphabetical order
and stay fixed through filtering and metric changes. Rebuilding with a
different model set can change the assignments. The model legend is ordered
alphabetically, with space reserved below the axes. Click a legend entry to
hide or show that model's segments in this chart.

Other charts choose palettes according to what colour encodes:

| Data type | Palette | Charts |
| --- | --- | --- |
| Nominal categories | Okabe-Ito, with fixed colours per category | Token categories, work themes and work modes |
| Continuous magnitude | Viridis, from dark purple to yellow | Task-theme-by-model heatmap |
| Ordered categories | Four Viridis samples, from purple to yellow | Reasoning effort: none, low, medium, high |
| One aggregate series | Okabe-Ito blue | Project totals, usage trends and user totals |

The heatmap scale starts at zero and extends to the selected maximum, using an
upper bound of one only for all-zero selections. Its colour bar shows the
current units and scale. Category colours stay fixed when filtered or ranked.
Unknown or unclassified categories retain grey. Warning colours keep their
separate meaning for data-quality messages.
