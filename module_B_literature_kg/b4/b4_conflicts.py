"""module_B_literature_kg/b4/b4_conflicts.py: find papers whose findings CONTRADICT each other.

Idea (simple version)
    1. take pairs of papers that are close in the graph ("similar" edges) AND share a dataset or a metric
    2. sort the pairs, most similar first, and keep the first B4_MAX_PAIRS (every pair costs one LLM call)
    3. ask the LLM: do the two papers contradict each other? (answer form: schemas.ConflictCheck)
    4. keep only the pairs the LLM calls "contradict" and for which it named both claims
"""

import json
from typing import Literal, Optional

from pydantic import BaseModel

from module_A_tooling.config import B4_MAX_PAIRS, B4_MODEL, B4_PROMPT_VERSION
from module_A_tooling.llm_client import LLMError
from module_B_literature_kg.b3.b3_normalize import normalize_name
from ..interface import Conflict, KnowledgeBase


class ConflictCheck(BaseModel):
    """The answer form the LLM must fill for one pair of papers."""
    relation: Literal["contradict", "agree", "unrelated"]
    summary: str = ""
    evidence_text: str = ""
    claim_a: Optional[str] = None
    claim_b: Optional[str] = None


SYSTEM_PROMPT = ("You compare the findings of two scientific papers. Use ONLY the information you are given. "
                 "Different methods alone are NOT a contradiction. Use 'contradict' only when the reported "
                 "results genuinely conflict. Reply with one JSON object and nothing else.")

PROMPT = """PAPER A: {a}
PAPER B: {b}

Both papers use the same dataset or metric. Do their findings contradict each other?
Reply with this JSON form (for "agree" or "unrelated" the other fields may be empty):
{{"relation": "contradict" or "agree" or "unrelated", "summary": "...", "evidence_text": "...",
 "claim_a": "what paper A reports", "claim_b": "what paper B reports"}}"""


def make_compare_fn(client, model=B4_MODEL):
    """Returns compare(a, b) -> dict. a and b are short review dicts. Uses the same LLMClient as B2
    (so answers are cached and checked). A question that keeps failing counts as "unrelated"."""
    def compare(a, b):
        prompt = PROMPT.format(a=json.dumps(a, ensure_ascii=False), b=json.dumps(b, ensure_ascii=False))
        try:
            return client.call("b4_compare", SYSTEM_PROMPT, prompt, ConflictCheck,
                               model=model, version=B4_PROMPT_VERSION).data
        except LLMError:
            return {"relation": "unrelated", "summary": "", "evidence_text": ""}
    return compare


def short_review(r):
    """Only what the LLM needs to compare two papers (keeps the question short)."""
    return {"paper_id": r.paper_id, "idea": r.idea, "method": r.method, "pros": r.pros,
            "datasets": r.datasets, "metrics": r.metrics}


def shared_names(names_a, names_b):
    """Names both papers have, after normalising ("Truthful QA" == "TruthfulQA")."""
    a = {normalize_name(x) for x in names_a}
    b = {normalize_name(x) for x in names_b}
    return sorted((a & b) - {""})


def detect_finding_conflicts(kb: KnowledgeBase, compare_fn, max_pairs: int = B4_MAX_PAIRS) -> list[Conflict]:
    # 1. candidate pairs: similar in the graph, share a dataset or metric, and at least one states a result
    candidates = []
    for pid in kb.ids:
        for nb in kb.neighbors(pid, kinds=("similar",)):
            other = nb.paper_id
            if pid >= other:                      # "similar" edges exist in both directions: keep one
                continue
            a, b = kb.get_review(pid), kb.get_review(other)
            if a is None or b is None or not (a.pros or b.pros):
                continue
            datasets = shared_names(a.datasets, b.datasets)
            metrics = shared_names(a.metrics, b.metrics)
            if datasets or metrics:
                candidates.append((nb.weight or 0.0, pid, other, datasets, metrics))

    # 2. most similar pairs first, then cut
    candidates.sort(key=lambda c: -c[0])
    candidates = candidates[:max(0, max_pairs)]

    # 3. ask the LLM about every pair
    conflicts = []
    for n, (weight, pid_a, pid_b, datasets, metrics) in enumerate(candidates, start=1):
        result = compare_fn(short_review(kb.get_review(pid_a)), short_review(kb.get_review(pid_b)))
        print(f"B4 pair {n}/{len(candidates)} {pid_a} vs {pid_b}: {result.get('relation')}")
        if result.get("relation") != "contradict" or not result.get("claim_a") or not result.get("claim_b"):
            continue
        conflicts.append(Conflict(
            id=f"CF-F{len(conflicts) + 1}", type="finding", paper_ids=[pid_a, pid_b],
            summary=result.get("summary") or "Contradictory findings detected",
            evidence_text=result.get("evidence_text") or "",
            shared_dataset=datasets[0] if datasets else None,
            shared_metric=metrics[0] if metrics else None,
            claim_a=result["claim_a"], claim_b=result["claim_b"]))
    return conflicts