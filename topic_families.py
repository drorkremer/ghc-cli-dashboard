# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Opt-in, local Ollama pass grouping existing topics into editable families."""
import argparse
import json
import urllib.request

from topic_classifier import assign_family_batch, catalog_lock, load_catalog, save_catalog, UNCATEGORIZED
from topic_summarizer import OLLAMA_URL, _ollama_identity


FAMILY_NAMES = {
    "developer-tooling": "Developer tooling",
    "software-engineering": "Software engineering",
    "systems-architecture": "Systems architecture",
    "data-integration": "Data & integration",
    "identity-people": "Identity & people",
    "security-compliance": "Security & compliance",
    "documents-content": "Documents & content",
    "presentations-visuals": "Presentations & visuals",
    "research-analysis": "Research & analysis",
    "operations-process": "Operations & process",
    "finance-expenses": "Finance & expenses",
    "education-onboarding": "Education & onboarding",
    "communications": "Communication & collaboration",
    "product-strategy": "Product & strategy",
}
BATCH_SIZE = 12


def _infer(topics, model):
    choices = [*FAMILY_NAMES, UNCATEGORIZED]
    descriptions = "\n".join(f"{key}: {name}" for key, name in FAMILY_NAMES.items())
    examples = [
        {"name": topic["name"], "example": topic["examples"][0][:240] if topic["examples"] else ""}
        for topic in topics
    ]
    payload = {
        "model": model, "stream": False,
        "think": "low" if model.startswith("gpt-oss:") else False,
        "messages": [
            {"role": "system", "content": (
                "Group distinct subject topics into broad families. Choose the most specific "
                "subject, using the example to resolve vague names. Classify the subject, not "
                "the document/deck/code deliverable unless the work is about creating or "
                "improving that medium. 'developer-tooling' includes Copilot/Cairn CLI "
                "tools, Cairn inbox relay, session watchers, handbacks and worktrees. "
                "'identity-people' includes roster correctness, directory identity records, "
                "personhood and people verification; 'data-integration' is for data pipelines, "
                "storage, mappings and integration of systems. 'operations-process' is for "
                "administrative/business workflows, not software automation or every task. "
                "Use 'uncategorized' if the subject is genuinely "
                "ambiguous. Return one family ID per input topic, in exactly the same order. "
                "Never combine topics or invent new IDs.\nFamilies:\n" + descriptions
            )},
            {"role": "user", "content": json.dumps(examples, ensure_ascii=False)},
        ],
        "format": {
            "type": "object",
            "properties": {"families": {"type": "array", "items": {
                "type": "string", "enum": choices,
            }, "minItems": len(topics), "maxItems": len(topics)}},
            "required": ["families"], "additionalProperties": False,
        },
        "options": {"temperature": 0, "num_ctx": 16384, "num_predict": 2048},
    }
    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=240) as response:
        output = json.load(response)
    content = output["message"]["content"]
    if not content:
        raise ValueError("family model response is empty")
    result = json.loads(content)
    families = result.get("families")
    if not isinstance(families, list) or len(families) != len(topics):
        raise ValueError("family model response has an unexpected number of assignments")
    if any(not isinstance(family, str) or family not in choices for family in families):
        raise ValueError("unknown family in model response")
    return families


def classify_catalog(path, model, *, refresh=False, limit=None, batch_size=BATCH_SIZE, infer=None):
    if batch_size < 1 or limit is not None and limit < 0:
        raise ValueError("batch size must be positive and limit must be nonnegative")
    catalog = load_catalog(path)
    pending = [
        topic for topic in catalog["topics"]
        if topic["id"] not in catalog.get("family_overrides", {})
        and (refresh or topic["id"] not in catalog.get("family_assignments", {}))
    ]
    if limit is not None:
        pending = pending[:limit]
    if not pending:
        return 0
    if infer is None:
        _ollama_identity(model)
        infer = lambda batch: _infer(batch, model)
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        suggestions = infer(batch)
        if len(suggestions) != len(batch):
            raise ValueError("invalid family model response: unexpected topics")
        with catalog_lock(path):
            catalog = load_catalog(path)
            current_topics = {topic["id"] for topic in catalog["topics"]}
            applicable = [
                (topic["id"], family_id) for topic, family_id in zip(batch, suggestions)
                if topic["id"] in current_topics
                and topic["id"] not in catalog.get("family_overrides", {})
                and (refresh or topic["id"] not in catalog.get("family_assignments", {}))
            ]
            if applicable:
                families = catalog.setdefault("families", [])
                by_name = {family["name"].casefold(): family["id"] for family in families}
                resolved_ids = {}
                for family_id, name in FAMILY_NAMES.items():
                    existing = next((family for family in families if family["id"] == family_id), None)
                    if existing:
                        resolved_ids[family_id] = existing["id"]
                    elif name.casefold() in by_name:
                        resolved_ids[family_id] = by_name[name.casefold()]
                    else:
                        families.append({"id": family_id, "name": name})
                        resolved_ids[family_id] = family_id
                assign_family_batch(
                    catalog, [topic_id for topic_id, _ in applicable],
                    [resolved_ids.get(family_id, family_id) for _, family_id in applicable],
                )
                save_catalog(path, catalog)
        print(f"Classified {min(start + len(batch), len(pending))}/{len(pending)} topics", flush=True)
    return len(pending)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, help="Private topic catalog used by dashboard.py")
    parser.add_argument("--ollama-model", required=True, help="Installed local Ollama model")
    parser.add_argument("--refresh", action="store_true",
                        help="Reclassify automatic family suggestions; keep manual corrections")
    parser.add_argument("--limit", type=int, default=None, help="Maximum new topics this run")
    args = parser.parse_args()
    count = classify_catalog(args.catalog, args.ollama_model,
                             refresh=args.refresh, limit=args.limit)
    print(f"Family classification complete: {count} topic(s)")


if __name__ == "__main__":
    main()
