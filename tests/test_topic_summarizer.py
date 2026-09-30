# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import csv
import json
import sqlite3
from concurrent.futures import Future

import pytest

import topic_summarizer
from topic_summarizer import (
    _cache_lock, _one_line, _summarize, compact_messages, summarize_export,
)


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


def test_heading_only_first_line_uses_actual_model_topic():
    assert _one_line(
        "Substantive work topic:\n**Scenario descriptions and deck slide updates**"
    ) == "Scenario descriptions and deck slide updates"
    assert _one_line(
        "**Substantive Work Topic:**\nFixing design notes and deck slides"
    ) == "Fixing design notes and deck slides"
    assert _one_line("system\n\n### Content parity check") == "Content parity check"
    assert _one_line(
        "Consecutive parts of one session:\n"
        "The principal work topic is **Verification of stub files**."
    ) == "Verification of stub files."


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


def test_parallel_workers_persist_results_in_parent_and_use_spawn(tmp_path, monkeypatch):
    db = tmp_path / "store.db"
    make_store(db, {"a": "Authentication request", "b": "Deployment request"})
    source, dest, cache, model = (
        tmp_path / "usage.csv", tmp_path / "out.csv",
        tmp_path / "cache.json", tmp_path / "qwen2.5.gguf",
    )
    make_csv(source, ["a", "b"])
    model.write_bytes(b"model")
    created = []

    class FakeExecutor:
        def __init__(self, max_workers, mp_context, initializer, initargs):
            created.append((max_workers, mp_context.get_start_method(), initializer, initargs))

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def submit(self, function, message, multi_topic):
            future = Future()
            future.set_result("Authentication" if "Authentication" in message else "Deployment")
            return future

    monkeypatch.setattr(topic_summarizer, "ProcessPoolExecutor", FakeExecutor)
    assert summarize_export(source, dest, [db], cache, model, workers=2) == 2
    assert created[0][:2] == (2, "spawn")
    with dest.open(encoding="utf-8", newline="") as stream:
        assert [r["topic_summary"] for r in csv.DictReader(stream)] == [
            "Authentication", "Deployment",
        ]
    assert len(json.loads(cache.read_text())["entries"]) == 2
    with pytest.raises(ValueError, match="workers"):
        summarize_export(source, dest, [db], cache, model, workers=0)


def test_concurrent_cache_writers_are_rejected_and_lock_releases_after_failure(tmp_path):
    db = tmp_path / "store.db"
    make_store(db, {"a": "Authentication request"})
    source, dest, cache, model = (
        tmp_path / "usage.csv", tmp_path / "out.csv",
        tmp_path / "cache.json", tmp_path / "qwen2.5.gguf",
    )
    make_csv(source, ["a"])
    model.write_bytes(b"model")
    with _cache_lock(cache):
        with pytest.raises(RuntimeError, match="already in use"):
            summarize_export(source, dest, [db], cache, model, generate=lambda _: "Authentication")
        assert not cache.exists() and not dest.exists()
    with pytest.raises(ValueError, match="empty topic"):
        summarize_export(source, dest, [db], cache, model, generate=lambda _: "")
    assert not dest.exists()
    assert summarize_export(
        source, dest, [db], cache, model, generate=lambda _: "Authentication",
    ) == 1


def test_multi_topic_evidence_keeps_first_whole_and_samples_later_turns(tmp_path):
    db = tmp_path / "store.db"
    first = "FIRST " + "long context " * 1000
    make_store(db, {"s": first})
    with sqlite3.connect(db) as conn:
        for index in range(1, 20):
            conn.execute(
                "INSERT INTO turns VALUES (?, ?, ?)",
                ("s", index, f"LATER-{index} " + "details " * 400),
            )
        conn.commit()
        with topic_summarizer.ReadOnlySnapshot(db) as snapshot:
            evidence = topic_summarizer._sample_user_turns(snapshot, db)["s"]
            assert evidence == topic_summarizer._sample_user_turns(snapshot, db)["s"]
    assert first in evidence
    assert evidence.count("LATER-") <= 5
    assert len(evidence) <= len(first) + 8500
    assert "LATER-" in evidence


def test_multi_topic_export_validates_structured_labels_and_cache(tmp_path):
    db = tmp_path / "store.db"
    make_store(db, {"s": "Prepare a deck on passkey adoption"})
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO turns VALUES (?, ?, ?)",
            ("s", 1, "Compare rollout risk for authentication"),
        )
    source, dest, cache, model = (
        tmp_path / "usage.csv", tmp_path / "out.csv", tmp_path / "multi-cache.json",
        tmp_path / "qwen2.5.gguf",
    )
    make_csv(source, ["s", "s"])
    model.write_bytes(b"model")
    prompts = []

    def generate(prompt):
        prompts.append(prompt)
        return json.dumps({
            "primary_subject": "Passkey adoption",
            "other_subjects": ["Authentication rollout"],
            "deliverables": ["deck", "doc"],
        })

    assert summarize_export(
        source, dest, [db], cache, model, generate=generate, multi_topic=True,
    ) == 1
    with dest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [json.loads(row["topic_subjects"]) for row in rows] == [
        ["Passkey adoption", "Authentication rollout"],
        ["Passkey adoption", "Authentication rollout"],
    ]
    assert all(json.loads(row["topic_deliverables"]) == ["deck", "doc"] for row in rows)
    assert "Compare rollout risk" in prompts[0]
    assert all(row["total_nano_aiu"] == "1000000000" for row in rows)
    assert summarize_export(
        source, dest, [db], cache, model, generate=lambda _: pytest.fail("cache miss"),
        multi_topic=True, limit=0,
    ) == 0
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE turns SET user_message = ? WHERE session_id = 's' AND turn_index = 1",
            ("Review documentation for authentication rollout",),
        )
    prompts.clear()
    assert summarize_export(
        source, dest, [db], cache, model, generate=generate, multi_topic=True,
    ) == 1
    assert "Review documentation" in prompts[0]
    with pytest.warns(RuntimeWarning, match="generic primary subject"):
        assert topic_summarizer._parse_multi_topic(
            '{"primary_subject": "Deck", "other_subjects": [], "deliverables": ["deck"]}'
        )["primary_subject"] == "Unknown"
    with pytest.raises(ValueError, match="deliverables"):
        topic_summarizer._parse_multi_topic('{"primary_subject": "Passkeys", "other_subjects": [], "deliverables": ["deck", "deck"]}')
    assert topic_summarizer._parse_multi_topic(
        '{"primary_subject":"Code review","other_subjects":[],"deliverables":["rubber-duck-pre-execution.md"]}'
    )["deliverables"] == ["doc"]
    with pytest.warns(RuntimeWarning, match="unrecognized deliverable"):
        assert topic_summarizer._parse_multi_topic(
            '{"primary_subject":"Worktree status","other_subjects":[],"deliverables":["worktree_path"]}'
        )["deliverables"] == ["other"]
    with pytest.warns(RuntimeWarning, match="unrecognized deliverable"):
        parsed = topic_summarizer._parse_multi_topic(
            '{"primary_subject":"Documentation audit","other_subjects":[],"deliverables":'
            '["line_count","link_count","mermaid_diagram_count",'
            '"components_documented","integration_surfaces_documented"]}'
        )
    assert parsed["deliverables"] == ["other"]
    assert "unrecognized_deliverable" in parsed["review_reasons"]
    with pytest.warns(RuntimeWarning, match="additional deliverable"):
        parsed = topic_summarizer._parse_multi_topic(
            '{"primary_subject":"Passkeys","other_subjects":[],'
            '"deliverables":["code","deck","doc","info"]}'
        )
    assert parsed["deliverables"] == ["code", "deck", "doc"]
    assert "extra_deliverables" in parsed["review_reasons"]
    with pytest.warns(RuntimeWarning, match="additional subject"):
        assert topic_summarizer._parse_multi_topic(
            '{"primary_subject":"Passkeys","other_subjects":["Rollout","Audit","Identity"],'
            '"deliverables":[]}'
        )["other_subjects"] == ["Rollout", "Audit"]


def test_multi_topic_export_continues_after_unrepairable_session(tmp_path, capsys):
    db = tmp_path / "store.db"
    make_store(db, {"bad": "broken request", "good": "Build passkey rollout slides"})
    source, dest, cache, model = (
        tmp_path / "usage.csv", tmp_path / "out.csv", tmp_path / "cache.json",
        tmp_path / "local.gguf",
    )
    make_csv(source, ["bad", "good"])
    model.write_bytes(b"model")

    def generate(prompt):
        if "broken request" in prompt:
            return "not JSON"
        return '{"primary_subject":"Passkey rollout","other_subjects":[],"deliverables":["deck"]}'

    assert summarize_export(
        source, dest, [db], cache, model, generate=generate, multi_topic=True,
    ) == 1
    with dest.open(encoding="utf-8", newline="") as stream:
        rows = {row["session_id"]: row for row in csv.DictReader(stream)}
    assert rows["bad"]["topic_status"] == "error"
    assert rows["bad"]["topic_subjects"] == "[]"
    assert rows["good"]["topic_status"] == "classified"
    assert rows["good"]["topic_deliverables"] == '["deck"]'
    assert "bad" in capsys.readouterr().err
    assert summarize_export(
        source, dest, [db], cache, model, generate=generate, multi_topic=True,
    ) == 0
    with dest.open(encoding="utf-8", newline="") as stream:
        rows = {row["session_id"]: row for row in csv.DictReader(stream)}
    assert rows["bad"]["topic_status"] == "error"


def test_multi_topic_response_repairs_invalid_primary_once():
    responses = iter([
        "not JSON",
        '{"primary_subject":"Passkey rollout","other_subjects":[],"deliverables":["deck"]}',
    ])
    prompts = []
    result = topic_summarizer._summarize_multi(
        "Prepare passkey rollout slides",
        lambda prompt: prompts.append(prompt) or next(responses),
    )
    assert result["primary_subject"] == "Passkey rollout"
    assert len(prompts) == 2
    assert "not JSON" in prompts[1]
    with pytest.raises(ValueError, match="JSON"):
        topic_summarizer._parse_multi_topic("not JSON")


def test_combined_subject_does_not_inherit_unknown_chunk_review_flag():
    responses = iter([
        '{"primary_subject":"Unknown","other_subjects":[],"deliverables":[]}',
        '{"primary_subject":"Identity audit","other_subjects":[],"deliverables":["doc"]}',
        '{"primary_subject":"Identity audit","other_subjects":[],"deliverables":["doc"]}',
    ])
    result = topic_summarizer._summarize_multi(
        "a" * 12, lambda _: next(responses), chunk_chars=6,
    )
    assert result["primary_subject"] == "Identity audit"
    assert "unknown_subject" not in result["review_reasons"]


def test_cached_resolved_subject_drops_stale_unknown_review_flag(tmp_path):
    db = tmp_path / "store.db"
    make_store(db, {"s": "Investigate identity records"})
    source, dest, cache, model = (
        tmp_path / "usage.csv", tmp_path / "out.csv",
        tmp_path / "cache.json", tmp_path / "model.gguf",
    )
    make_csv(source, ["s"])
    model.write_bytes(b"model")
    result = '{"primary_subject":"Identity audit","other_subjects":[],"deliverables":["doc"]}'
    summarize_export(source, dest, [db], cache, model, multi_topic=True, generate=lambda _: result)
    persisted = json.loads(cache.read_text())
    next(iter(persisted["entries"].values()))["review_reasons"] = ["unknown_subject"]
    cache.write_text(json.dumps(persisted))
    summarize_export(source, dest, [db], cache, model, multi_topic=True,
                     generate=lambda _: pytest.fail("cache miss"), limit=0)
    with dest.open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["topic_status"] == "classified"
    assert row["topic_review"] == "[]"


def test_large_context_model_preserves_entire_first_turn_in_one_request():
    prompts = []
    text = "First user turn:\n" + "Review authentication rollout. " * 420
    result = topic_summarizer._summarize_multi(
        text,
        lambda prompt: prompts.append(prompt) or '{"primary_subject":"Authentication rollout",'
                                                    '"other_subjects":[],"deliverables":["doc"]}',
        chunk_chars=24000,
    )
    assert len(text) > topic_summarizer.CHUNK_CHARS
    assert len(prompts) == 1
    assert text in prompts[0]
    assert result["primary_subject"] == "Authentication rollout"


def test_ollama_generator_uses_loopback_and_constrained_types(monkeypatch):
    from io import BytesIO

    calls = []

    class Response(BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.close()

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        return Response(json.dumps({
            "message": {"content": '{"primary_subject":"Passkey rollout",'
                       '"other_subjects":[],"deliverables":["doc"]}'},
        }).encode())

    monkeypatch.setattr(topic_summarizer.urllib.request, "urlopen", fake_urlopen)
    prompt = topic_summarizer._topic_prompt("Draft passkey rollout notes", multi_topic=True)
    response = topic_summarizer._ollama_generate(prompt, "gpt-oss:20b")
    assert json.loads(response)["deliverables"] == ["doc"]
    request, timeout = calls[0]
    assert request.full_url == "http://127.0.0.1:11434/api/chat"
    body = json.loads(request.data)
    assert body["messages"][1]["content"] == "Draft passkey rollout notes"
    assert body["format"]["properties"]["deliverables"]["items"]["enum"] == sorted(
        topic_summarizer.DELIVERABLE_TYPES
    )
    assert body["think"] == "low"
    assert timeout > 30


def test_ollama_retries_empty_length_limited_response(monkeypatch):
    from io import BytesIO

    token_budgets = []

    def fake_urlopen(request, timeout):
        token_budgets.append(json.loads(request.data)["options"]["num_predict"])
        response = (
            {"done_reason": "length", "message": {"content": ""}}
            if len(token_budgets) == 1 else
            {"done_reason": "stop", "message": {
                "content": '{"primary_subject":"Rule migration",'
                           '"other_subjects":[],"deliverables":["doc"]}',
            }}
        )
        return BytesIO(json.dumps(response).encode())

    monkeypatch.setattr(topic_summarizer.urllib.request, "urlopen", fake_urlopen)
    prompt = topic_summarizer._topic_prompt("Review rule migration", multi_topic=True)
    assert "Rule migration" in topic_summarizer._ollama_generate(prompt, "gpt-oss:20b")
    assert token_budgets == [256, 512]


def test_ollama_preflight_reports_missing_service_and_model(monkeypatch):
    from io import BytesIO
    from urllib.error import URLError

    def disconnected(*_, **__):
        raise URLError("connection refused")

    monkeypatch.setattr(topic_summarizer.urllib.request, "urlopen", disconnected)
    with pytest.raises(RuntimeError, match="Start Ollama"):
        topic_summarizer._ollama_identity("gpt-oss:20b")
    monkeypatch.setattr(
        topic_summarizer.urllib.request, "urlopen",
        lambda *_, **__: BytesIO(b'{"models": []}'),
    )
    with pytest.raises(ValueError, match=r"ollama pull gpt-oss:20b"):
        topic_summarizer._ollama_identity("gpt-oss:20b")


@pytest.mark.parametrize("workers", [1, 2])
def test_ollama_transport_failure_is_recorded_per_session(tmp_path, monkeypatch, workers):
    from urllib.error import URLError

    db = tmp_path / "store.db"
    make_store(db, {"a": "Draft passkey rollout notes", "b": "Code passkey API"})
    source, dest, cache = tmp_path / "usage.csv", tmp_path / "out.csv", tmp_path / "cache.json"
    make_csv(source, ["a", "b"])
    monkeypatch.setattr(topic_summarizer, "_ollama_identity", lambda model: f"ollama:{model}:digest")
    calls = []

    def generate(prompt, model):
        calls.append(prompt)
        if "Draft passkey rollout notes" in prompt:
            raise URLError("temporary local model failure")
        return '{"primary_subject":"Passkey API","other_subjects":[],"deliverables":["code"]}'

    monkeypatch.setattr(topic_summarizer, "_ollama_generate", generate)
    summarize_export(source, dest, [db], cache, None, multi_topic=True,
                     ollama_model="gpt-oss:20b", workers=workers)
    with dest.open(encoding="utf-8", newline="") as stream:
        rows = {row["session_id"]: row for row in csv.DictReader(stream)}
    assert rows["a"]["topic_status"] == "error"
    assert rows["b"]["topic_status"] == "classified"
    assert rows["b"]["topic_subjects"] == '["Passkey API"]'

    calls.clear()
    summarize_export(source, dest, [db], cache, None, multi_topic=True,
                     ollama_model="gpt-oss:20b", workers=workers)
    assert len(calls) == 1


def test_ollama_export_uses_shared_model_threads_and_distinct_cache(tmp_path, monkeypatch):
    db = tmp_path / "store.db"
    make_store(db, {"a": "Draft passkey rollout notes", "b": "Code passkey API"})
    source, dest, cache = tmp_path / "usage.csv", tmp_path / "out.csv", tmp_path / "cache.json"
    make_csv(source, ["a", "b"])
    monkeypatch.setattr(topic_summarizer, "_ollama_identity", lambda model: f"ollama:{model}:digest")
    monkeypatch.setattr(
        topic_summarizer, "_ollama_generate",
        lambda prompt, model: '{"primary_subject":"Passkey rollout",'
                              '"other_subjects":[],"deliverables":["doc"]}',
    )
    assert summarize_export(
        source, dest, [db], cache, None, multi_topic=True,
        ollama_model="gpt-oss:20b", workers=2,
    ) == 2
    with dest.open(encoding="utf-8", newline="") as stream:
        assert {row["topic_status"] for row in csv.DictReader(stream)} == {"classified"}
    assert summarize_export(
        source, dest, [db], cache, None, multi_topic=True,
        ollama_model="gpt-oss:20b", workers=2, limit=0,
    ) == 0
