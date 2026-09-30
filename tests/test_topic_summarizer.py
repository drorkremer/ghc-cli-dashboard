# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import csv
import json
import sqlite3

import pytest

from topic_summarizer import _one_line, _summarize, compact_messages, summarize_export


def make_store(path, sessions):
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY)")
        conn.execute(
            "CREATE TABLE turns (session_id TEXT, turn_index INTEGER, user_message TEXT)"
        )
        for session_id, message in sessions.items():
            conn.execute("INSERT INTO sessions VALUES (?)", (session_id,))
            conn.execute("INSERT INTO turns VALUES (?, 0, ?)", (session_id, message))


def make_csv(path, session_ids):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "user", "session_id", "date", "task_summary", "total_nano_aiu",
        ])
        writer.writeheader()
        for session_id in session_ids:
            writer.writerow(dict(
                user="u", session_id=session_id, date="2026-01-01",
                task_summary="Same boilerplate", total_nano_aiu=1000000000,
            ))


def test_compaction_removes_majority_boilerplate_but_preserves_topic():
    boilerplate = "Common launcher instructions repeated across all sessions"
    topic = "Refactor authentication and passkey sign-in flows for the next release"
    sessions = {str(i): f"{boilerplate}\n{topic} {i}" for i in range(20)}
    compact = compact_messages(sessions)
    assert all(boilerplate not in text for text in compact.values())
    assert compact["0"] == f"{topic} 0"
    assert compact_messages({"single": sessions["0"]})["single"] == sessions["0"]


def test_long_prompt_uses_all_parts_and_reduces_to_one_topic():
    prompts = []

    def generate(prompt):
        prompts.append(prompt)
        if "Combine these topic hints" in prompt:
            return "**Topic:** Authentication and deployment"
        return "Authentication" if "PASSKEY" in prompt else "Deployment"

    result = _summarize("PASSKEY " * 600 + "RELEASE " * 600, generate)
    assert result == "Authentication and deployment"
    assert len(prompts) == 3
    assert "PASSKEY" in prompts[0] and "RELEASE" in prompts[1]
    with pytest.raises(ValueError, match="empty topic"):
        _one_line("")


def test_summarizer_caches_per_session_and_preserves_cost_rows(tmp_path):
    db = tmp_path / "store.db"
    make_store(db, {
        "a": "Common launch setup\nRefactor passkey sign-in",
        "b": "Common launch setup\nInvestigate deployment failures",
    })
    source, dest, cache, model = (
        tmp_path / "usage.csv", tmp_path / "enriched.csv",
        tmp_path / "topic-cache.json", tmp_path / "qwen2.5.gguf",
    )
    model.write_bytes(b"model-placeholder")
    make_csv(source, ["a", "a", "b"])
    seen = []

    def generate(prompt):
        seen.append(prompt)
        if "passkey" in prompt:
            return "Authentication passkey sign-in"
        return "Deployment failures"

    summarize_export(source, dest, [db], cache, model, generate=generate)
    with dest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [r["topic_summary"] for r in rows] == [
        "Authentication passkey sign-in", "Authentication passkey sign-in", "Deployment failures",
    ]
    assert all(r["total_nano_aiu"] == "1000000000" for r in rows)
    assert len(seen) == 2
    assert cache.stat().st_mode & 0o077 == 0
    assert dest.stat().st_mode & 0o077 == 0
    summarize_export(
        source, dest, [db, db], cache, model,
        generate=lambda prompt: pytest.fail("cache hit should not call model"),
        limit=0,
    )
    assert json.loads(cache.read_text())["version"] == 1
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE turns SET user_message = ? WHERE session_id = 'b'",
            ("Investigate release pipeline",),
        )
    seen.clear()
    summarize_export(source, dest, [db], cache, model, generate=generate)
    assert len(seen) == 1


def test_session_collision_uses_first_store_and_warns(tmp_path, capsys):
    first, second = tmp_path / "first.db", tmp_path / "second.db"
    make_store(first, {"a": "First store authentication"})
    make_store(second, {"a": "Second store deployment"})
    source, dest, cache, model = (
        tmp_path / "usage.csv", tmp_path / "enriched.csv",
        tmp_path / "cache.json", tmp_path / "qwen2.5.gguf",
    )
    model.write_bytes(b"model")
    make_csv(source, ["a"])
    prompts = []
    summarize_export(
        source, dest, [first, second], cache, model,
        generate=lambda prompt: prompts.append(prompt) or "Authentication",
    )
    assert "First store" in prompts[0] and "Second store" not in prompts[0]
    assert "ignoring 1 duplicate" in capsys.readouterr().err


def test_missing_turns_table_is_explicit_error(tmp_path):
    db = tmp_path / "store.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE sessions (id TEXT)")
    source, dest, cache, model = (
        tmp_path / "usage.csv", tmp_path / "enriched.csv",
        tmp_path / "cache.json", tmp_path / "qwen2.5.gguf",
    )
    make_csv(source, ["a"])
    model.write_bytes(b"model")
    with pytest.raises(ValueError, match="turns"):
        summarize_export(source, dest, [db], cache, model, generate=lambda _: "label")
    assert not dest.exists()


def test_cache_path_cannot_overwrite_original_usage_export(tmp_path):
    db = tmp_path / "store.db"
    make_store(db, {"a": "Authentication request"})
    source, dest, model = tmp_path / "usage.csv", tmp_path / "out.csv", tmp_path / "qwen2.5.gguf"
    model.write_bytes(b"model")
    make_csv(source, ["a"])
    with pytest.raises(ValueError, match="cache"):
        summarize_export(
            source, dest, [db], source, model,
            generate=lambda _: "Authentication",
        )
    assert source.read_text(encoding="utf-8").startswith("user,session_id,")
