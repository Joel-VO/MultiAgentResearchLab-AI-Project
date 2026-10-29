"""Detect disagreement in OpenReview reviewer scores."""
from __future__ import annotations

from statistics import variance
from typing import Callable, Iterable

from .interface import Conflict


def detect_reviewer_conflicts(
    reviews: Iterable[dict],
    threshold: float = 2.0,
    summarize_fn: Callable[[dict, float], str] | None = None,
) -> list[Conflict]:
    """Flag submissions whose reviewer-score sample variance exceeds threshold.

    Each input is a dict with ``paper_id`` (or ``submission_id``), optional
    ``reviewer_scores`` (list of numbers) and optional ``reviewer_comments``.
    The caller is responsible for fetching/normalizing venue data from OpenReview.
    """
    output = []
    for item in reviews:
        if not isinstance(item, dict):
            continue
        paper_id = str(item.get("paper_id") or item.get("submission_id") or "")
        scores = item.get("reviewer_scores", item.get("scores", []))
        if not paper_id or not isinstance(scores, (list, tuple)):
            continue
        valid_scores = [float(s) for s in scores if isinstance(s, (int, float)) and not isinstance(s, bool)]
        if len(valid_scores) < 2:
            continue
        score_var = variance(valid_scores)
        if score_var <= threshold:
            continue
        summary = summarize_fn(item, score_var) if summarize_fn else (
            "Reviewers gave substantially different scores for this submission."
        )
        comments = item.get("reviewer_comments", [])
        evidence = " | ".join(str(c) for c in comments if c) if isinstance(comments, list) else str(comments or "")
        output.append(Conflict(
            id=f"CF-R{len(output) + 1}", type="reviewer", paper_ids=[paper_id],
            summary=summary, evidence_text=evidence or f"Reviewer scores: {valid_scores}",
            score_variance=score_var,
        ))
    return output
