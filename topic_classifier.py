# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Optional, entirely local topic assignments for Copilot CLI sessions."""
import json
import math
import os
import re
import tempfile
import uuid
from collections import Counter
from pathlib import Path

import numpy as np


OTHER = "other"
MODEL = "BAAI/bge-small-en-v1.5"
STOPWORDS = set(
    "a an and are as at be by for from in into is it of on or the this to with "
    "about after before can how what why your my please session sessions copilot "
    "cli task tasks project repo repository file files using use implement help "
    "bootstrap watcher".split()
)


def session_key(user, session_id):
    return json.dumps([user, session_id], ensure_ascii=False, separators=(",", ":"))


def empty_catalog():
    return {"version": 1, "topics": [], "assignments": {}, "overrides": {}}


def validate_catalog(catalog):
    if not isinstance(catalog, dict) or catalog.get("version") != 1:
        raise ValueError("invalid topic catalog version")
    topics = catalog.get("topics")
    if not isinstance(topics, list):
        raise ValueError("invalid topics list")
    ids = {OTHER}
    for topic in topics:
        if not isinstance(topic, dict) or not isinstance(topic.get("id"), str) or not topic["id"]:
            raise ValueError("invalid topic ID")
        if topic["id"] in ids:
            raise ValueError("invalid topic: duplicate or reserved ID")
        ids.add(topic["id"])
        if not isinstance(topic.get("name"), str) or not topic["name"].strip() or len(topic["name"]) > 120:
            raise ValueError("invalid topic name")
        if topic.get("kind") not in ("user", "auto"):
            raise ValueError("invalid topic kind")
        for field in ("examples", "keywords"):
            values = topic.get(field)
            if not isinstance(values, list) or any(
                not isinstance(value, str) or not value.strip() or len(value) > 500
                for value in values
            ):
                raise ValueError(f"invalid topic {field}")
    for field in ("assignments", "overrides"):
        values = catalog.get(field)
        if not isinstance(values, dict) or any(
            not isinstance(key, str) or not isinstance(value, str) or value not in ids
            for key, value in values.items()
        ):
            raise ValueError(f"invalid topic {field}")


def load_catalog(path):
    path = Path(path)
    if not path.exists():
        return empty_catalog()
    try:
        with path.open(encoding="utf-8") as stream:
            catalog = json.load(stream)
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not read topic catalog {path}: {exc}") from exc
    validate_catalog(catalog)
    return catalog


def save_catalog(path, catalog):
    validate_catalog(catalog)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".topics-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(catalog, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def apply_edit(catalog, edit):
    validate_catalog(catalog)
    action = edit.get("action")
    topics = catalog["topics"]
    by_id = {topic["id"]: topic for topic in topics}
    if action == "create":
        name = edit.get("name")
        examples = edit.get("examples", [])
        if not isinstance(name, str) or not name.strip() or len(name) > 120:
            raise ValueError("invalid topic name")
        if not isinstance(examples, list):
            raise ValueError("invalid topic examples")
        if not examples:
            examples = [name.strip()]
        topics.append({
            "id": uuid.uuid4().hex, "name": name.strip(), "kind": "user",
            "examples": examples, "keywords": [],
        })
    elif action == "rename":
        topic_id = edit.get("topic_id")
        topic = by_id.get(topic_id) if isinstance(topic_id, str) else None
        name = edit.get("name")
        if topic is None or not isinstance(name, str) or not name.strip() or len(name) > 120:
            raise ValueError("invalid topic rename")
        topic["name"] = name.strip()
    elif action == "update":
        topic_id = edit.get("topic_id")
        topic = by_id.get(topic_id) if isinstance(topic_id, str) else None
        if topic is None:
            raise ValueError("unknown topic")
        for field in ("examples", "keywords"):
            values = edit.get(field)
            if not isinstance(values, list) or any(
                not isinstance(value, str) or not value.strip() or len(value) > 500
                for value in values
            ):
                raise ValueError(f"invalid topic {field}")
        topic["examples"] = edit["examples"]
        topic["keywords"] = edit["keywords"]
    elif action == "merge":
        source, target = edit.get("source_id"), edit.get("target_id")
        if (not isinstance(source, str) or not isinstance(target, str)
                or source not in by_id or target not in by_id or source == target):
            raise ValueError("invalid topic merge")
        by_id[target]["examples"].extend(
            example for example in by_id[source]["examples"]
            if example not in by_id[target]["examples"]
        )
        for field in ("assignments", "overrides"):
            for key, value in catalog[field].items():
                if value == source:
                    catalog[field][key] = target
        topics.remove(by_id[source])
    elif action == "override":
        user, session_id, topic_id = (edit.get(key) for key in ("user", "session_id", "topic_id"))
        if not isinstance(user, str) or not user or not isinstance(session_id, str) or not session_id:
            raise ValueError("invalid session identity")
        if not isinstance(topic_id, str) or (topic_id not in by_id and topic_id != OTHER):
            raise ValueError("unknown topic")
        catalog["overrides"][session_key(user, session_id)] = topic_id
    else:
        raise ValueError("unknown topic edit action")
    validate_catalog(catalog)
    return catalog


def _local_encoder(texts):
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise RuntimeError(
            "Local topic discovery requires fastembed. Install requirements-topics.txt "
            "(the model is downloaded once and used locally)."
        ) from exc
    return list(TextEmbedding(model_name=MODEL, threads=1).embed(texts))


def _cosine(left, right):
    if len(left) != len(right):
        raise ValueError("embedding model returned vectors of different dimensions")
    numerator = sum(float(a) * float(b) for a, b in zip(left, right))
    magnitude = math.sqrt(sum(float(a) ** 2 for a in left)) * math.sqrt(
        sum(float(b) ** 2 for b in right)
    )
    return numerator / magnitude if magnitude else 0.0


def _label(texts):
    terms = [
        {word.casefold() for word in re.findall(r"[^\W_]+", text, re.UNICODE)
         if len(word) > 2 and not word.isdigit() and word.casefold() not in STOPWORDS}
        for text in texts
    ]
    counts = Counter(word for term_set in terms for word in term_set)
    common = [word for word, count in counts.items() if count > 1]
    if common:
        best = sorted(common, key=lambda word: (-counts[word], word))[0]
        return next(word for word in re.findall(r"[^\W_]+", texts[0]) if word.casefold() == best)[:120]
    return (texts[0].splitlines()[0][:60].strip() or "Unclassified topic")


def classify(sessions, path, encoder=None):
    """Return session-key -> topic ID, preserving earlier decisions across builds.

    sessions contains (user, session_id, summary) tuples. Classification is
    performed only for as-yet-unassigned sessions; no Copilot process is used.
    """
    catalog = load_catalog(path)
    assignments = {}
    pending = {}
    for user, session_id, summary in sessions:
        if not isinstance(session_id, str) or not session_id:
            continue
        key = session_key(str(user), session_id)
        if key in assignments:
            continue
        if key in catalog["overrides"]:
            assignments[key] = catalog["overrides"][key]
        elif key in catalog["assignments"]:
            assignments[key] = catalog["assignments"][key]
        elif isinstance(summary, str) and summary.strip():
            pending[key] = summary.strip()[:500].strip()
        else:
            assignments[key] = OTHER

    for key, summary in list(pending.items()):
        for topic in catalog["topics"]:
            if any(
                re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", summary, re.IGNORECASE)
                for word in topic["keywords"]
            ):
                assignments[key] = topic["id"]
                del pending[key]
                break

    examples = [(topic["id"], example) for topic in catalog["topics"] for example in topic["examples"]]
    keys = list(pending)
    if keys and (examples or len(keys) >= 2):
        encode = encoder or _local_encoder
        vectors = list(encode([pending[key] for key in keys] + [text for _, text in examples]))
        if len(vectors) != len(keys) + len(examples):
            raise ValueError("embedding model returned an unexpected number of vectors")
        topic_kinds = {topic["id"]: topic["kind"] for topic in catalog["topics"]}
        prior = [
            (topic_id, vector, topic_kinds[topic_id])
            for (topic_id, _), vector in zip(examples, vectors[len(keys):])
        ]
        unassigned = []
        for key, vector in zip(keys, vectors[:len(keys)]):
            for kind in ("user", "auto"):
                match = max(
                    (item for item in prior if item[2] == kind),
                    key=lambda item: _cosine(vector, item[1]), default=None,
                )
                if match and _cosine(vector, match[1]) >= 0.67:
                    assignments[key] = match[0]
                    break
            if key not in assignments:
                unassigned.append((key, vector))

        if unassigned:
            matrix = np.asarray([vector for _, vector in unassigned], dtype=np.float32)
            if matrix.ndim != 2:
                raise ValueError("embedding model returned invalid vectors")
            norms = np.linalg.norm(matrix, axis=1)
            matrix = matrix / np.where(norms == 0, 1, norms)[:, None]
            similarity = matrix @ matrix.T
            remaining = list(range(len(unassigned)))
            while remaining:
                seed = remaining.pop(0)
                group = [seed]
                rest = []
                for index in remaining:
                    if all(similarity[index, member] >= 0.75 for member in group):
                        group.append(index)
                    else:
                        rest.append(index)
                remaining = rest
                if len(group) < 2:
                    assignments[unassigned[seed][0]] = OTHER
                    continue
                texts = [pending[unassigned[index][0]] for index in group]
                topic_id = uuid.uuid4().hex
                catalog["topics"].append({
                    "id": topic_id, "name": _label(texts),
                    "examples": [texts[0]], "keywords": [], "kind": "auto",
                })
                for index in group:
                    assignments[unassigned[index][0]] = topic_id

    for key in pending:
        assignments.setdefault(key, OTHER)
    for key, topic_id in assignments.items():
        if topic_id != OTHER and key not in catalog["overrides"]:
            catalog["assignments"][key] = topic_id
    save_catalog(path, catalog)
    return assignments, catalog
