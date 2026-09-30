# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Optional, entirely local topic assignments for Copilot CLI sessions."""
import json
import math
import os
import sqlite3
from contextlib import contextmanager
import re
import tempfile
import uuid
from collections import Counter
from pathlib import Path

import numpy as np


OTHER = "other"
UNCATEGORIZED = "uncategorized"
MODEL = "BAAI/bge-small-en-v1.5"
DELIVERABLE_TYPES = {"code", "deck", "doc", "info", "data", "config", "other"}
STOPWORDS = set(
    "a an and are as at be by for from in into is it of on or the this to with "
    "about after before can how what why your my please session sessions copilot "
    "cli task tasks project repo repository file files using use implement help "
    "bootstrap watcher audit auditing verify verification verified final current "
    "expanded specific review reviewing optimization substantive work topic "
    "architecture".split()
)


def session_key(user, session_id):
    return json.dumps([user, session_id], ensure_ascii=False, separators=(",", ":"))


def empty_catalog():
    return {
        "version": 1, "topics": [], "assignments": {}, "overrides": {},
        "tag_overrides": {}, "deliverable_overrides": {},
        "families": [], "family_assignments": {}, "family_overrides": {},
    }


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
    families = catalog.get("families", [])
    if not isinstance(families, list):
        raise ValueError("invalid topic families")
    family_ids = {UNCATEGORIZED}
    family_names = set()
    for family in families:
        if (not isinstance(family, dict) or not isinstance(family.get("id"), str)
                or not family["id"] or family["id"] in family_ids
                or not isinstance(family.get("name"), str)
                or not family["name"].strip() or len(family["name"]) > 120
                or family["name"].strip().casefold() in family_names
                or family["name"].strip().casefold() == "uncategorized"):
            raise ValueError("invalid topic family")
        family_ids.add(family["id"])
        family_names.add(family["name"].casefold())
    for field in ("family_assignments", "family_overrides"):
        values = catalog.get(field, {})
        if not isinstance(values, dict) or any(
            not isinstance(topic_id, str) or topic_id not in ids
            or topic_id == OTHER or not isinstance(family_id, str)
            or family_id not in family_ids
            for topic_id, family_id in values.items()
        ):
            raise ValueError(f"invalid topic {field}")
    for field in ("assignments", "overrides"):
        values = catalog.get(field)
        if not isinstance(values, dict) or any(
            not isinstance(key, str) or not isinstance(value, str) or value not in ids
            for key, value in values.items()
        ):
            raise ValueError(f"invalid topic {field}")
    tag_overrides = catalog.get("tag_overrides", {})
    if not isinstance(tag_overrides, dict) or any(
        not isinstance(key, str) or not isinstance(values, list) or len(values) > 2
        or any(not isinstance(value, str) or value == OTHER or value not in ids for value in values)
        or len(set(values)) != len(values)
        for key, values in tag_overrides.items()
    ):
        raise ValueError("invalid topic tag_overrides")
    deliverable_overrides = catalog.get("deliverable_overrides", {})
    if not isinstance(deliverable_overrides, dict) or any(
        not isinstance(key, str) or not isinstance(values, list) or len(values) > len(DELIVERABLE_TYPES)
        or any(not isinstance(value, str) or value not in DELIVERABLE_TYPES for value in values)
        or len(set(values)) != len(values)
        for key, values in deliverable_overrides.items()
    ):
        raise ValueError("invalid topic deliverable_overrides")


@contextmanager
def catalog_lock(path):
    lock_path = Path(f"{path}.lock.sqlite3")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(lock_path, timeout=300) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        finally:
            connection.rollback()


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
        for key, values in catalog.get("tag_overrides", {}).items():
            catalog["tag_overrides"][key] = list(dict.fromkeys(
                target if value == source else value for value in values
            ))
        for field in ("family_assignments", "family_overrides"):
            mapping = catalog.setdefault(field, {})
            if target not in mapping and source in mapping:
                mapping[target] = mapping[source]
            mapping.pop(source, None)
        topics.remove(by_id[source])
    elif action == "create_family":
        name = edit.get("name")
        if (not isinstance(name, str) or not name.strip() or len(name) > 120
                or name.strip().casefold() == "uncategorized"
                or any(f["name"].casefold() == name.strip().casefold()
                       for f in catalog.get("families", []))):
            raise ValueError("invalid family name")
        catalog.setdefault("families", []).append({"id": uuid.uuid4().hex, "name": name.strip()})
    elif action == "rename_family":
        family = next((f for f in catalog.get("families", [])
                       if f["id"] == edit.get("family_id")), None)
        name = edit.get("name")
        if (family is None or not isinstance(name, str) or not name.strip()
                or len(name) > 120
                or name.strip().casefold() == "uncategorized"
                or any(f["id"] != family["id"] and f["name"].casefold() == name.strip().casefold()
                       for f in catalog["families"])):
            raise ValueError("invalid family rename")
        family["name"] = name.strip()
    elif action == "assign_family":
        topic_id, family_id = edit.get("topic_id"), edit.get("family_id")
        if not isinstance(topic_id, str) or topic_id not in by_id:
            raise ValueError("unknown topic")
        if not isinstance(family_id, str) or family_id not in {
            UNCATEGORIZED, *(family["id"] for family in catalog.get("families", []))
        }:
            raise ValueError("unknown family")
        catalog.setdefault("family_overrides", {})[topic_id] = family_id
    elif action == "override":
        user, session_id, topic_id = (edit.get(key) for key in ("user", "session_id", "topic_id"))
        if not isinstance(user, str) or not user or not isinstance(session_id, str) or not session_id:
            raise ValueError("invalid session identity")
        if not isinstance(topic_id, str) or (topic_id not in by_id and topic_id != OTHER):
            raise ValueError("unknown topic")
        catalog["overrides"][session_key(user, session_id)] = topic_id
    elif action == "override_tags":
        user, session_id, topic_ids = (edit.get(key) for key in ("user", "session_id", "topic_ids"))
        if not isinstance(user, str) or not user or not isinstance(session_id, str) or not session_id:
            raise ValueError("invalid session identity")
        if (not isinstance(topic_ids, list) or len(topic_ids) > 2
                or any(not isinstance(topic_id, str) or topic_id not in by_id for topic_id in topic_ids)
                or len(set(topic_ids)) != len(topic_ids)):
            raise ValueError("invalid additional subject topics")
        catalog.setdefault("tag_overrides", {})[session_key(user, session_id)] = topic_ids
    elif action == "override_deliverables":
        user, session_id, deliverables = (edit.get(key) for key in ("user", "session_id", "deliverables"))
        if not isinstance(user, str) or not user or not isinstance(session_id, str) or not session_id:
            raise ValueError("invalid session identity")
        if (not isinstance(deliverables, list) or len(deliverables) > len(DELIVERABLE_TYPES)
                or any(not isinstance(value, str) or value not in DELIVERABLE_TYPES
                       for value in deliverables)
                or len(set(deliverables)) != len(deliverables)):
            raise ValueError("invalid deliverable types")
        catalog.setdefault("deliverable_overrides", {})[session_key(user, session_id)] = deliverables
    else:
        raise ValueError("unknown topic edit action")
    validate_catalog(catalog)
    return catalog


def assign_family_batch(catalog, topic_ids, family_ids):
    """Record validated model suggestions without replacing explicit corrections."""
    validate_catalog(catalog)
    known_topics = {topic["id"] for topic in catalog["topics"]}
    known_families = {UNCATEGORIZED, *(family["id"] for family in catalog.get("families", []))}
    if (not isinstance(topic_ids, list) or not isinstance(family_ids, list)
            or len(topic_ids) != len(family_ids) or len(set(topic_ids)) != len(topic_ids)
            or any(not isinstance(topic_id, str) or topic_id not in known_topics
                   for topic_id in topic_ids)):
        raise ValueError("invalid family model response: unexpected topics")
    if any(not isinstance(family_id, str) or family_id not in known_families
           for family_id in family_ids):
        raise ValueError("unknown family in model response")
    assignments = catalog.setdefault("family_assignments", {})
    overrides = catalog.get("family_overrides", {})
    for topic_id, family_id in zip(topic_ids, family_ids):
        if topic_id not in overrides:
            assignments[topic_id] = family_id
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


def _anchor_tokens(text):
    return {
        word.casefold() for word in re.findall(r"[^\W_]+", text, re.UNICODE)
        if len(word) > 2 and not word.isdigit() and word.casefold() not in STOPWORDS
    }


def _uninformative_summary(text):
    text = text.strip()
    return (
        text.casefold().rstrip(".!?") == "unknown"
        or bool(re.match(r"first read the cairn protocol context at\s", text, re.IGNORECASE))
        or (text.startswith("`") and text.endswith("`")
            and not any(char.isspace() for char in text[1:-1]))
    )


def _label(texts):
    terms = [_anchor_tokens(text) for text in texts]
    counts = Counter(word for term_set in terms for word in term_set)
    common = [word for word, count in counts.items() if count > 1]
    if common:
        if len(common) > 1:
            shared = set(common)
            label = []
            seen = set()
            for word in re.findall(r"[^\W_]+", texts[0]):
                normalized = word.casefold()
                if normalized in shared and normalized not in seen:
                    label.append(word)
                    seen.add(normalized)
                    if len(label) == 4:
                        break
            if len(label) > 1:
                return " ".join(label)[:120]
        best = sorted(common, key=lambda word: (-counts[word], word))[0]
        return next(word for word in re.findall(r"[^\W_]+", texts[0]) if word.casefold() == best)[:120]
    return (texts[0].splitlines()[0][:60].strip() or "Unclassified topic")


def _unique_auto_name(texts, used_names):
    def key(name):
        return re.sub(r"[^\w]+", " ", name.casefold()).strip()

    label = _label(texts)
    candidates = [label, *(text.splitlines()[0][:60].strip() for text in texts)]
    for candidate in candidates:
        if candidate and key(candidate) not in used_names:
            used_names.add(key(candidate))
            return candidate
    number = 2
    while key(f"{label[:110]} ({number})") in used_names:
        number += 1
    name = f"{label[:110]} ({number})"
    used_names.add(key(name))
    return name


def _classify_locked(sessions, path, encoder=None):
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
            if _uninformative_summary(summary):
                assignments[key] = OTHER
            else:
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
            (topic_id, vector, topic_kinds[topic_id], _anchor_tokens(example))
            for (topic_id, example), vector in zip(examples, vectors[len(keys):])
        ]
        unassigned = []
        for key, vector in zip(keys, vectors[:len(keys)]):
            anchors = _anchor_tokens(pending[key])
            for kind in ("user", "auto"):
                match = max(
                    (item for item in prior if item[2] == kind
                     and (kind == "user" or anchors & item[3])),
                    key=lambda item: _cosine(vector, item[1]), default=None,
                )
                if match and _cosine(vector, match[1]) >= 0.67:
                    assignments[key] = match[0]
                    break
            if key not in assignments:
                unassigned.append((key, vector))

        if unassigned:
            used_names = {
                re.sub(r"[^\w]+", " ", topic["name"].casefold()).strip()
                for topic in catalog["topics"]
            }
            matrix = np.asarray([vector for _, vector in unassigned], dtype=np.float32)
            if matrix.ndim != 2:
                raise ValueError("embedding model returned invalid vectors")
            norms = np.linalg.norm(matrix, axis=1)
            matrix = matrix / np.where(norms == 0, 1, norms)[:, None]
            similarity = matrix @ matrix.T
            anchors = [_anchor_tokens(pending[key]) for key, _ in unassigned]
            remaining = list(range(len(unassigned)))
            while remaining:
                seed = remaining.pop(0)
                group = [seed]
                rest = []
                for index in remaining:
                    if all(similarity[index, member] >= 0.75
                           and anchors[index] & anchors[member] for member in group):
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
                    "id": topic_id, "name": _unique_auto_name(texts, used_names),
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


def classify(sessions, path, encoder=None):
    with catalog_lock(path):
        return _classify_locked(sessions, path, encoder=encoder)


def _additional_subject_session_id(session_id, index, label):
    return f"{session_id}\0subject\0{index}\0{label.strip().casefold()}"


def classify_tagged(sessions, path, encoder=None):
    """Assign one primary subject for cost totals and up to two additional subjects."""
    sessions = list(sessions)
    expanded = []
    for user, session_id, primary, additional in sessions:
        expanded.append((user, session_id, primary))
        for index, label in enumerate(additional):
            expanded.append((user, _additional_subject_session_id(session_id, index, label), label))
    assignments, catalog = classify(expanded, path, encoder=encoder)
    primary_assignments = {}
    subject_tags = {}
    for user, session_id, _, additional in sessions:
        key = session_key(str(user), session_id)
        primary = assignments.get(key, OTHER)
        primary_assignments[key] = primary
        extra = catalog.get("tag_overrides", {}).get(key)
        if extra is None:
            extra = [
                assignments.get(
                    session_key(str(user), _additional_subject_session_id(session_id, index, label)),
                    OTHER,
                )
                for index, label in enumerate(additional)
            ]
        subject_tags[key] = list(dict.fromkeys(
            topic_id for topic_id in [primary, *extra] if topic_id != OTHER
        ))
    return primary_assignments, subject_tags, catalog
