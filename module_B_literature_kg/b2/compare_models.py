"""module_B_literature_kg/b2/compare_models.py: run the SAME abstracts through several LOCAL models and compare them.

    python -m module_B_literature_kg.b2.compare_models --n 20 --models qwen14 llama8 mistral7
    python -m module_B_literature_kg.b2.compare_models --n 1 --models local            # first live check of the server

The names after --models are the nicknames in config.LLM_MODELS. To compare several models, add one nickname per
model there (same base_url, different "model" name), and pull/load those models in your local server.

What it does
    1. picks N papers (fixed random seed, so every model sees the same papers) from runs/<id>/papers.json,
       only papers with a real abstract
    2. for each model, runs the B2a review (b2a_review.review_paper) on every paper (answers are cached,
       so a second run costs nothing)
    3. prints one table and writes, into runs/<id>/model_compare/:
         summary.json            the numbers
         <model>.json            every review + its record, for that model
         side_by_side.md         the first papers with every model's idea / method / lists, to READ and judge

How to read the table (no model is "right" by itself, you need the numbers AND a look at side_by_side.md)
    quote pass %   share of claims whose quote was found in the abstract. Low = the model invents or paraphrases.
    claims kept    claims per paper that survived. Very low with a high pass % = a lazy model that says little.
    entity pass %  share of datasets/metrics/methods that really appear in the text.
    failed/empty   papers with no usable answer / with nothing supported.
    retried        papers that needed the second, corrected question.
    agreement      how often two models name the same datasets/metrics/methods (1.0 = identical lists).
    min/150        estimated minutes for 150 papers (average seconds per paper, answers read from the cache
                   are not counted). Local models cost time, not money.
A model that cannot be used (server not running, model not found, name still "SET_ME") is skipped with a message.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from itertools import combinations
from pathlib import Path
from typing import Optional

from module_A_tooling.config import LLM_MODELS, run_dir, run_file
from module_A_tooling.llm_client import LLMClient, LLMConfigError
from .b2a_review import ReviewOutcome, clean_abstract, review_paper

MIN_ABSTRACT_CHARS = 150          # same rule as the B1 "no abstract" check (placeholder until it lives in config.py)
SIDE_BY_SIDE_PAPERS = 5


# --------------------------------------------------------------------------- choosing the papers

def sample_papers(papers: dict[str, dict], n: int, seed: int = 0, include_seeds: bool = False) -> list[dict]:
    """n papers with a real abstract, chosen with a fixed seed. Seeds are left out by default
    (B2a is for non-seed papers)."""
    usable = [p for _, p in sorted(papers.items(), key=lambda kv: int(kv[0][1:]))
              if len(clean_abstract(p.get("abstract"))) >= MIN_ABSTRACT_CHARS and (include_seeds or not p.get("is_seed"))]
    return random.Random(seed).sample(usable, min(n, len(usable)))


# --------------------------------------------------------------------------- numbers

def _entity_set(o: ReviewOutcome) -> set[str]:
    r = o.review
    return {n.lower() for n in (*r.datasets, *r.metrics, *r.methods)}


def summarise(model: str, outcomes: list[ReviewOutcome]) -> dict:
    n = len(outcomes)
    made = sum(o.claims_made for o in outcomes)
    kept = sum(o.claims_kept for o in outcomes)
    e_made = sum(o.entities_made for o in outcomes)
    e_kept = sum(o.entities_kept for o in outcomes)
    usable = [o for o in outcomes if o.status != "failed"]
    fresh = [o for o in outcomes if not o.cached]
    avg_in = sum(o.input_tokens for o in outcomes) / n if n else 0.0
    avg_out = sum(o.output_tokens for o in outcomes) / n if n else 0.0
    return {"model": model, "papers": n,
            "failed": sum(o.status == "failed" for o in outcomes),
            "empty": sum(o.status == "empty" for o in outcomes),
            "retried": sum(o.retried for o in outcomes),
            "claims_made": made, "claims_kept": kept,
            "quote_pass_rate": round(kept / made, 3) if made else None,
            "claims_kept_per_paper": round(kept / len(usable), 2) if usable else None,
            "entity_pass_rate": round(e_kept / e_made, 3) if e_made else None,
            "avg_input_tokens": round(avg_in), "avg_output_tokens": round(avg_out),
            "avg_seconds": round(sum(o.seconds for o in fresh) / len(fresh), 2) if fresh else None,
            "est_minutes_per_150_papers": round(sum(o.seconds for o in fresh) / len(fresh) * 150 / 60, 1) if fresh else None}


def agreement(results: dict[str, list[ReviewOutcome]]) -> dict[str, float]:
    """For every pair of models: mean Jaccard overlap of the dataset/metric/method names, over the papers
    where at least one of the two named something."""
    out = {}
    for a, b in combinations(results, 2):
        scores = []
        for oa, ob in zip(results[a], results[b]):
            sa, sb = _entity_set(oa), _entity_set(ob)
            if sa | sb:
                scores.append(len(sa & sb) / len(sa | sb))
        if scores:
            out[f"{a} vs {b}"] = round(sum(scores) / len(scores), 3)
    return out


def _fmt(x, pct: bool = False) -> str:
    if x is None:
        return "-"
    return f"{x * 100:.0f}%" if pct else str(x)


def format_table(summaries: list[dict], agree: dict[str, float], skipped: dict[str, str]) -> str:
    cols = [("model", "model"), ("papers", "papers"), ("quote pass", "quote_pass_rate"),
            ("claims kept", "claims_kept_per_paper"), ("entity pass", "entity_pass_rate"),
            ("failed", "failed"), ("empty", "empty"), ("retried", "retried"),
            ("in tok", "avg_input_tokens"), ("out tok", "avg_output_tokens"), ("sec", "avg_seconds"),
            ("min/150", "est_minutes_per_150_papers")]
    rows = [[h for h, _ in cols]]
    for s in summaries:
        rows.append([_fmt(s[k], pct=k in ("quote_pass_rate", "entity_pass_rate")) for _, k in cols])
    widths = [max(len(r[i]) for r in rows) for i in range(len(cols))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in rows]
    lines.insert(1, "  ".join("-" * w for w in widths))
    if agree:
        lines += ["", "agreement on dataset/metric/method names: " + "; ".join(f"{k} = {v}" for k, v in agree.items())]
    for model, why in skipped.items():
        lines.append(f"SKIPPED {model}: {why}")
    return "\n".join(lines)


def side_by_side(papers: list[dict], results: dict[str, list[ReviewOutcome]], k: int = SIDE_BY_SIDE_PAPERS) -> str:
    parts = ["# Side by side (read these and judge the wording yourself)\n"]
    for i, paper in enumerate(papers[:k]):
        parts += [f"## {paper['paper_id']}: {paper['title']}\n", clean_abstract(paper.get("abstract")) + "\n"]
        for model, outcomes in results.items():
            o, r = outcomes[i], outcomes[i].review
            parts += [f"### {model} (status: {o.status}, claims kept {o.claims_kept}/{o.claims_made})",
                      f"- idea: {r.idea}", f"- method: {r.method}",
                      f"- pros: {r.pros}",
                      f"- concepts: {r.concepts}", f"- datasets: {r.datasets}; metrics: {r.metrics}; methods: {r.methods}",
                      ""]
    return "\n".join(parts)


# --------------------------------------------------------------------------- the run

def run_comparison(papers: list[dict], models: list[str], client: LLMClient,
                   out_dir: Optional[Path] = None) -> dict:
    results: dict[str, list[ReviewOutcome]] = {}
    skipped: dict[str, str] = {}
    for model in models:
        try:
            outcomes = [review_paper(p["paper_id"], p["title"], p.get("abstract", ""), client, model)
                        for p in papers]
        except LLMConfigError as exc:
            skipped[model] = str(exc)
            continue
        results[model] = outcomes
    summaries = [summarise(m, o) for m, o in results.items()]
    agree = agreement(results)
    report = {"n_papers": len(papers), "paper_ids": [p["paper_id"] for p in papers],
              "summaries": summaries, "agreement": agree, "skipped": skipped}
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        for model, outcomes in results.items():
            (out_dir / f"{model}.json").write_text(json.dumps(
                {p["paper_id"]: o.to_record() for p, o in zip(papers, outcomes)}, indent=2, ensure_ascii=False),
                encoding="utf-8")
        (out_dir / "side_by_side.md").write_text(side_by_side(papers, results), encoding="utf-8")
    report["table"] = format_table(summaries, agree, skipped)
    return report


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m module_B_literature_kg.b2.compare_models",
                                 description="Compare local LLMs on the B2a abstract review.")
    ap.add_argument("--papers", help="papers.json of a finished B0 run (default: runs/<run-id>/papers.json)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--n", type=int, default=20, help="how many papers to review (default 20)")
    ap.add_argument("--models", nargs="+", default=list(LLM_MODELS), help="nicknames from config.LLM_MODELS")
    ap.add_argument("--seed", type=int, default=0, help="random seed for the paper sample")
    ap.add_argument("--include-seeds", action="store_true")
    ap.add_argument("--out", help="output folder (default: runs/<run-id>/model_compare)")
    args = ap.parse_args(argv)

    papers_path = Path(args.papers) if args.papers else run_file("papers", *([args.run_id] if args.run_id else []))
    if not papers_path.exists():
        print(f"ERROR: {papers_path} not found: run B0 first or pass --papers", file=sys.stderr)
        return 1
    papers = sample_papers(json.loads(papers_path.read_text(encoding="utf-8")), args.n, args.seed, args.include_seeds)
    if not papers:
        print("ERROR: no paper with a usable abstract in that file", file=sys.stderr)
        return 1
    out_dir = Path(args.out) if args.out else (run_dir(args.run_id) if args.run_id else papers_path.parent) / "model_compare"
    report = run_comparison(papers, args.models, LLMClient(), out_dir)
    print(f"{len(papers)} papers, models: {', '.join(args.models)}\n")
    print(report["table"])
    print(f"\nfiles written to {out_dir}")
    return 0 if report["summaries"] else 1


if __name__ == "__main__":                       # pragma: no cover
    sys.exit(main())
