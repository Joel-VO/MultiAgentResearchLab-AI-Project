"""tests/run_anchor_weight_experiments.py

Runs B1 five times, changing ONLY ANCHOR_TOPIC_WEIGHT (anchor = w*topic_sim + (1-w)*seed_sim):
    A 0.00 | B 0.25 | C 0.50 | D 0.75 | E 1.00
Everything else is identical: same seeds.json, same references.json, same embedding model,
same budget, same other settings from config.py.

It needs the files of an EARLIER finished run (seeds.json, references.json, request.json), so
run the pipeline once first:   python -m module_B_literature_kg.b0.pipeline --topic ... --seed ...

Usage:
    python tests/run_anchor_weight_experiments.py --run-id run_001
    python tests/run_anchor_weight_experiments.py --run-id run_001 --labels tests/relevance_labels.json

What "relevant" means (there is no ground truth in the project yet):
    * With --labels: a JSON file {"s2:abc...": true, "s2:def...": false, ...} that YOU fill in by hand
      (paper key from discovery.json). Papers not in the file are counted as "unlabelled".
    * Without --labels: a PROXY is used: topic_sim >= --topic-cut (default 0.45). topic_sim does not
      depend on the weight, so the proxy is the same for all 5 experiments. But it is the same number
      experiment E uses to decide, so E looks artificially good under the proxy. Use real labels for
      conclusions.

Cost: Semantic Scholar answers and embeddings come from the cache (cache.db) when they were fetched
before. A paper that a new setting expands for the first time is fetched live (a few seconds each).
Output: a table on screen + runs/<run-id>/experiments/anchor_weight_results.json
"""
from __future__ import annotations

import argparse
import datetime
import json
from dataclasses import asdict
from pathlib import Path

from module_A_tooling.config import RUN_FILES, run_dir
from module_B_literature_kg.b0.b1_discovery import B1Settings, discover, normalise_budget
from module_B_literature_kg.b0.clients import S2Client
from module_B_literature_kg.b0.embed import Embedder

EXPERIMENTS = {"A": 0.00, "B": 0.25, "C": 0.50, "D": 0.75, "E": 1.00}   # name -> topic weight
DOORS = ("seed", "guaranteed", "anchor", "seed_sim", "explore")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_labels(path):
    return read(Path(path)) if path else None


def relevance(key: str, topic_sim: float, labels, topic_cut: float):
    """True / False / None (None = unlabelled)."""
    if labels is not None:
        return labels.get(key)
    return topic_sim >= topic_cut


def run_one(seeds_doc, refs_doc, topic, budget, weight, s2, embedder, as_of_year) -> dict:
    settings = B1Settings(anchor_topic_weight=weight)        # the ONLY change
    return discover(seeds_doc, refs_doc, topic, embedder, s2, budget, settings, as_of_year)


def measure(doc: dict, universe_relevant: set, labels, topic_cut: float) -> dict:
    scores, meta = doc["scores"], doc["meta"]
    accepted = {k for k, s in scores.items() if s["accepted"] and s["accepted_by"] != "seed"}
    rel_acc = irr_acc = unl_acc = 0
    for k in accepted:
        r = relevance(k, scores[k]["topic_sim"], labels, topic_cut)
        if r is None:
            unl_acc += 1
        elif r:
            rel_acc += 1
        else:
            irr_acc += 1
    cov = meta["seed_coverage"]
    reached = [v["reached"] for v in cov.values()]
    return {
        "accepted_total": meta["n_accepted"],            # includes the seeds
        "accepted_non_seed": len(accepted),
        "rejected": meta["n_rejected"],
        "relevant_accepted": rel_acc,
        "irrelevant_accepted": irr_acc,
        "unlabelled_accepted": unl_acc,
        "relevant_missed": len(universe_relevant - accepted),
        "seed_reach_min": min(reached), "seed_reach_max": max(reached),
        "seed_reach_mean": round(sum(reached) / len(reached), 1),
        "seed_coverage": cov,
        "doors": {d: meta["accepted_by"].get(d, 0) for d in DOORS},
        "stop_reason": meta["stop_reason"], "expansions": meta["expansions"],
    }


def print_table(results: dict) -> None:
    names = list(results)
    rows = [
        ("topic weight / seed weight", lambda n, r: f"{EXPERIMENTS[n]:.2f}/{1 - EXPERIMENTS[n]:.2f}"),
        ("accepted (incl. seeds)", lambda n, r: r["accepted_total"]),
        ("rejected", lambda n, r: r["rejected"]),
        ("relevant accepted", lambda n, r: r["relevant_accepted"]),
        ("irrelevant accepted", lambda n, r: r["irrelevant_accepted"]),
        ("unlabelled accepted", lambda n, r: r["unlabelled_accepted"]),
        ("relevant missed", lambda n, r: r["relevant_missed"]),
        ("seed reach min/mean/max", lambda n, r: f"{r['seed_reach_min']}/{r['seed_reach_mean']}/{r['seed_reach_max']}"),
        *[(f"door: {d}", (lambda d: lambda n, r: r["doors"][d])(d)) for d in DOORS],
        ("stop reason", lambda n, r: r["stop_reason"]),
        ("expansions", lambda n, r: r["expansions"]),
    ]
    width = 28
    print("\n" + "".ljust(width) + "".join(f"{n:>14}" for n in names))
    for label, fn in rows:
        print(label.ljust(width) + "".join(f"{str(fn(n, results[n])):>14}" for n in names))
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare 5 anchor weights in B1.")
    ap.add_argument("--run-id", default="run_001", help="an earlier run that has seeds.json, references.json, request.json")
    ap.add_argument("--labels", help="JSON {paper_key: true/false} with hand-made relevance labels")
    ap.add_argument("--topic-cut", type=float, default=0.45, help="proxy: topic_sim at or above this = relevant")
    ap.add_argument("--as-of-year", type=int, default=datetime.date.today().year)
    args = ap.parse_args()

    root = run_dir(args.run_id)
    seeds_doc = read(root / RUN_FILES["seeds"])
    refs_doc = read(root / RUN_FILES["references"])
    request = read(root / RUN_FILES["request"])
    topic, budget = request["topic"], normalise_budget(request.get("budget"))
    labels = load_labels(args.labels)

    s2, embedder = S2Client(), Embedder()            # use the shared cache: repeated questions cost nothing
    docs = {}
    for name, w in EXPERIMENTS.items():
        print(f"running experiment {name} (topic weight {w:.2f}) ...")
        docs[name] = run_one(seeds_doc, refs_doc, topic, budget, w, s2, embedder, args.as_of_year)

    # One shared list of "relevant papers": every non-seed paper judged in ANY experiment.
    # (A paper that no experiment ever looked at cannot be counted as missed.)
    universe_relevant: set = set()
    for doc in docs.values():
        for k, s in doc["scores"].items():
            if s["accepted_by"] != "seed" and relevance(k, s["topic_sim"], labels, args.topic_cut):
                universe_relevant.add(k)

    results = {n: measure(d, universe_relevant, labels, args.topic_cut) for n, d in docs.items()}
    print_table(results)
    print(f"relevance: {'hand labels from ' + args.labels if labels else f'PROXY topic_sim >= {args.topic_cut}'}; "
          f"{len(universe_relevant)} relevant papers in the shared list")

    out_dir = root / "experiments"
    out_dir.mkdir(exist_ok=True)
    out = {"topic": topic, "budget": budget, "weights": EXPERIMENTS, "relevance_source": args.labels or "proxy",
           "topic_cut": args.topic_cut, "n_relevant_in_universe": len(universe_relevant),
           "base_settings": asdict(B1Settings()), "results": results}
    (out_dir / "anchor_weight_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved: {out_dir / 'anchor_weight_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
