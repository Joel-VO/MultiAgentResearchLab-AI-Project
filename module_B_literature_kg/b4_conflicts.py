"""Detect potential contradictory findings between related papers."""
from __future__ import annotations

import json
from itertools import combinations
from typing import Callable

from .interface import Conflict, KnowledgeBase


def detect_finding_conflicts(
    kb: KnowledgeBase,
    reviews=None,
    compare_fn: Callable[[dict, dict], dict] | None = None,
    max_pairs: int = 50,
) -> list[Conflict]:
    """Compare mutual-similarity neighbours that share a dataset or metric.

    ``compare_fn`` receives two compact review dictionaries and should return
    ``{"relation": "contradict|agree|unrelated", "summary": ..., 
    "evidence_text": ..., "claim_a": ..., "claim_b": ...}``.
    Without a callback, no LLM is called and this function returns no findings;
    this makes the module safe to test offline.
    """
    if compare_fn is None:
        return []
    review_map = {r.paper_id: r for r in (reviews or [])}
    for pid in kb.ids:
        if pid not in review_map:
            review_map[pid] = kb.get_review(pid)
    pairs = set()
    for pid in kb.ids:
        for neighbor in kb.neighbors(pid, kinds=("similar",)):
            other = neighbor.paper_id
            if pid >= other or other not in review_map or review_map[other] is None or review_map[pid] is None:
                continue
            a, b = review_map[pid], review_map[other]
            shared_datasets = {x.casefold() for x in a.datasets} & {x.casefold() for x in b.datasets}
            shared_metrics = {x.casefold() for x in a.metrics} & {x.casefold() for x in b.metrics}
            if shared_datasets or shared_metrics:
                pairs.add((pid, other))
    conflicts = []
    for index, (pid_a, pid_b) in enumerate(sorted(pairs)):
        if index >= max(0, max_pairs):
            break
        a, b = review_map[pid_a], review_map[pid_b]
        result = compare_fn(
            {"paper_id": pid_a, "idea": a.idea, "method": a.method, "pros": a.pros,
             "cons": a.cons, "datasets": a.datasets, "metrics": a.metrics,
             "evidence_quotes": a.evidence_quotes},
            {"paper_id": pid_b, "idea": b.idea, "method": b.method, "pros": b.pros,
             "cons": b.cons, "datasets": b.datasets, "metrics": b.metrics,
             "evidence_quotes": b.evidence_quotes},
        )
        if not isinstance(result, dict) or result.get("relation") != "contradict":
            continue
        conflicts.append(Conflict(
            id=f"CF-F{len(conflicts) + 1}", type="finding", paper_ids=[pid_a, pid_b],
            summary=str(result.get("summary", "Contradictory findings detected")),
            evidence_text=str(result.get("evidence_text", "")),
            shared_dataset=next(iter(sorted({x.casefold() for x in a.datasets} & {x.casefold() for x in b.datasets})), None),
            shared_metric=next(iter(sorted({x.casefold() for x in a.metrics} & {x.casefold() for x in b.metrics})), None),
            claim_a=result.get("claim_a"), claim_b=result.get("claim_b"),
        ))
    return conflicts


def ollama_compare_fn(client):
    """Create a comparison callback using the project's OllamaClient."""
    system = (
        "Compare scientific findings carefully. Do not infer contradiction from different methods alone. "
        "Return only JSON with relation (contradict, agree, or unrelated), summary, evidence_text, claim_a, claim_b. "
        "Use contradict only when the reported findings genuinely conflict."
    )
    def compare(a, b):
        prompt = json.dumps({"paper_a": a, "paper_b": b}, ensure_ascii=False)
        raw = client.generate(user_prompt=prompt, system_prompt=system, format="json", temperature=0)
        try:
            result = json.loads(raw)
        except (TypeError, ValueError):
            return {"relation": "unrelated", "summary": "", "evidence_text": ""}
        if result.get("relation") not in {"contradict", "agree", "unrelated"}:
            result["relation"] = "unrelated"
        return result
    return compare
