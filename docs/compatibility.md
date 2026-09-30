---
layout: default
title: Compatibility and data format
nav_order: 2
---

# Compatibility and data format

## Environment

The project requires Python 3.9 or later, `pandas`, and `plotly`.

The extractor reads the local Copilot CLI session store at
`~/.copilot/session-store.db` by default. Repeat `--db PATH` to read
additional/custom session stores; explicitly supplied paths replace the
default. GitHub documents the local session store, but
does not document the usage-event schema as a stable public API. The
integration has been validated against Copilot CLI 1.0.79 on Windows.
The store schema was also inspected on macOS with Copilot CLI 1.0.90-5;
topic inference and the local HTTP editor were exercised on macOS.

Linux support has not been verified against real Copilot CLI data.
The extractor validates the tables and columns it needs and stops with a
clear error if the local store is incompatible.

Optional topic discovery uses `fastembed>=0.7.4,<0.8` and a locally cached
ONNX embedding model. The topic editor runs through Python's standard-library
HTTP server on `127.0.0.1`; static HTML viewing needs neither the server nor
the embedding dependency.

Optional first-turn summarization uses `gpt4all>=2.8.2,<3` and an explicitly
downloaded local Qwen2.5 GGUF. GPT4All publishes a macOS universal2 wheel
and a Windows x64 wheel, and the summarizer uses only cross-platform Python
standard-library paths, SQLite, and CSV processing. Local inference was
exercised on macOS, including a CPU-only run; **Windows execution has not
yet been validated**. The optional SLM requires roughly a 1.1 GB model
download and enough memory for its 8K-token context. Windows ARM64 is not
covered by the published Windows x64 wheel. Without the SLM option, the
ordinary extractor and dashboard remain unchanged.

## Export format

`extract_usage.py` creates one CSV row per session, model, day, and reasoning
effort. The required dashboard columns are:

```text
user, date, project, model, calls, total_tokens
```

Current exports also include token categories, cost coverage, a session ID,
and export metadata. `export_format_version` identifies the CSV layout.
`exported_at` records when the file was written and lets the dashboard handle
overlapping exports deterministically.
`topic_summarizer.py` can add an optional `topic_summary` column containing
only the locally generated label, never the raw user message.

Older CSVs continue to load where possible. The dashboard warns when metadata
or optional fields are missing.

## Schema boundary

The extractor currently reads `assistant_usage_events` and `sessions` from
the local session store. The exact table layout can change with Copilot CLI
updates. This project uses a read-only SQLite snapshot and does not modify
the live database.

For implementation details, see `REQUIRED_SCHEMA` and `QUERY` in
[`extract_usage.py`](../extract_usage.py).
