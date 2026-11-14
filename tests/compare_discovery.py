"""compare_discovery.py: how different are the accepted papers of two B1 discovery runs?

Typical use (one run folder per mode, e.g. made with --run-id):
    python compare_discovery.py runs/run_classic runs/run_groups
    python compare_discovery.py runs/run_classic runs/run_groups --top 50 --csv diff.csv

You can also pass the discovery.json files directly. Only the standard library is needed.

What it prints
    1. each run: mode, budget, how many papers were accepted, which door let them in
    2. the overlap: papers in both runs, papers only in A, only in B, Jaccard score
    3. for the papers in both runs: did they come in through the same door?
    4. for the papers in only one run: WHY the other run did not take them
         "below the door"  -> the other run judged the paper and its score was too low
         "lost the slot"   -> the other run judged it, it passed a door, but the budget/fanout was used up
         "never seen"      -> the other run never reached this paper (it was never scored there)
    5. the top papers that only one run took (title, year, door, scores)
Seeds are always accepted by both runs, so they are left out of the overlap numbers.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path


# --------------------------------------------------------------------------- loading

def load_run(path: str) -> dict:
    p = Path(path)
    file = p / "discovery.json" if p.is_dir() else p
    if not file.exists():
        sys.exit(f"ERROR: {file} not found")
    doc = json.loads(file.read_text(encoding="utf-8"))
    titles = {k: v.get("title", "") for k, v in doc.get("stubs", {}).items()}
    years = {k: v.get("year") for k, v in doc.get("stubs", {}).items()}
    seeds_file = file.parent / "seeds.json"                      # seeds are not stubs: take their titles here
    if seeds_file.exists():
        for rec in json.loads(seeds_file.read_text(encoding="utf-8")).get("seeds", []):
            paper = rec["paper"]
            titles[paper["key"]] = paper.get("title", "")
            years[paper["key"]] = paper.get("year")
    scores = doc["scores"]
    return {"label": p.name if p.is_dir() else p.parent.name, "doc": doc, "meta": doc["meta"],
            "scores": scores, "titles": titles, "years": years,
            "accepted": {k for k, s in scores.items() if s["accepted"]},
            "seeds": {k for k, s in scores.items() if s["accepted_by"] == "seed"}}


def mode_of(run: dict) -> str:
    return run["meta"].get("mode", "classic")        # old files (before the mode switch) are classic


def passes_a_door(run: dict, score: dict) -> bool:
    """Would this judged-but-rejected paper have passed the run's own relevance door?"""
    st = run["meta"].get("settings", {})
    if mode_of(run) == "groups":
        return score["anchor"] >= st.get("group_accept_threshold", 0.5) - 1e-9
    return (score["anchor"] >= st.get("sim_threshold", 0.35)
            or score["seed_sim"] >= st.get("seed_accept_threshold", 0.60))


def why_missing(other: dict, key: str) -> str:
    """Why is `key` (accepted in the first run) not accepted in `other`?"""
    sc = other["scores"].get(key)
    if sc is None:
        return "never seen"
    if sc["accepted"]:
        return "accepted"
    return "lost the slot" if passes_a_door(other, sc) else "below the door"


def short(text: str, n: int = 70) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def fmt(x) -> str:
    return "  -  " if x is None else f"{x:.2f}"


# --------------------------------------------------------------------------- report

def describe(run: dict) -> list[str]:
    m, plan = run["meta"], run["meta"].get("plan", {})
    lines = [f"  {run['label']}: mode={mode_of(run)}, accepted={m['n_accepted']} (of max {plan.get('max_papers')}), "
             f"judged={m['n_judged']}, expansions={m['expansions']}, stop={m['stop_reason']}",
             f"      accepted by: " + ", ".join(f"{k}={v}" for k, v in sorted(m["accepted_by"].items()))]
    for g in m.get("groups") or []:
        lines.append(f"      group {g['id']}: {g['size']} seed(s), threshold {fmt(g['threshold'])}, "
                     f"guaranteed {g['guaranteed']}")
    return lines


def row(run: dict, other: dict, key: str) -> dict:
    sc = run["scores"][key]
    sc_other = other["scores"].get(key)
    return {"key": key, "title": run["titles"].get(key, ""), "year": run["years"].get(key),
            "door": sc["accepted_by"], "depth": sc["depth"], "anchor": sc["anchor"],
            "topic_sim": sc["topic_sim"], "seed_sim": sc["seed_sim"],
            "other_status": why_missing(other, key),
            "other_anchor": None if sc_other is None else sc_other["anchor"],
            "reached_from": len(sc.get("reached_from") or [])}


def print_table(rows: list[dict], run_label: str, other_label: str, top: int) -> None:
    rows = sorted(rows, key=lambda r: (-r["anchor"], r["key"]))[:top]
    print(f"    {'title':<70} {'year':>4}  {'door':<10} {'dep':>3} {'anchor':>6} {'topic':>5} {'seed':>5}  "
          f"{other_label[:12]+':':<13} status")
    for r in rows:
        print(f"    {short(r['title'] or r['key']):<70} {r['year'] or '':>4}  {r['door']:<10} {r['depth']!s:>3} "
              f"{fmt(r['anchor']):>6} {fmt(r['topic_sim']):>5} {fmt(r['seed_sim']):>5}  "
              f"{fmt(r['other_anchor']):<13} {r['other_status']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compare the accepted papers of two B1 discovery runs.")
    ap.add_argument("run_a", help="run folder (or discovery.json) of run A, e.g. runs/run_classic")
    ap.add_argument("run_b", help="run folder (or discovery.json) of run B, e.g. runs/run_groups")
    ap.add_argument("--top", type=int, default=25, help="how many papers to list per side (default 25)")
    ap.add_argument("--csv", help="write every paper of the union to this CSV file")
    args = ap.parse_args(argv)

    a, b = load_run(args.run_a), load_run(args.run_b)
    if a["label"] == b["label"]:
        a["label"], b["label"] = a["label"] + " (A)", b["label"] + " (B)"

    print("=" * 100)
    print("RUNS")
    for line in describe(a) + describe(b):
        print(line)
    if a["seeds"] != b["seeds"]:
        print("  WARNING: the two runs have different seeds, so the comparison is not like for like")

    acc_a, acc_b = a["accepted"] - a["seeds"], b["accepted"] - b["seeds"]
    both, only_a, only_b = acc_a & acc_b, acc_a - acc_b, acc_b - acc_a
    union = acc_a | acc_b
    jaccard = len(both) / len(union) if union else 1.0
    print("\n" + "=" * 100)
    print("OVERLAP (seeds left out)")
    print(f"  accepted in A: {len(acc_a)}   accepted in B: {len(acc_b)}")
    print(f"  in both: {len(both)}   only A: {len(only_a)}   only B: {len(only_b)}")
    print(f"  Jaccard (both / either): {jaccard:.2f}")
    if acc_a:
        print(f"  share of A also in B: {len(both) / len(acc_a):.0%}")
    if acc_b:
        print(f"  share of B also in A: {len(both) / len(acc_b):.0%}")

    if both:
        print("\n" + "=" * 100)
        print("PAPERS IN BOTH RUNS: door in A -> door in B")
        cross = Counter((a["scores"][k]["accepted_by"], b["scores"][k]["accepted_by"]) for k in both)
        for (da, db), n in sorted(cross.items(), key=lambda t: -t[1]):
            print(f"  {da:<11} -> {db:<11} {n}")

    for first, second, only in ((a, b, only_a), (b, a, only_b)):
        print("\n" + "=" * 100)
        print(f"ONLY IN {first['label']} ({len(only)} papers): why did {second['label']} not take them?")
        if not only:
            print("  none")
            continue
        reasons = Counter(why_missing(second, k) for k in only)
        for reason, n in reasons.most_common():
            print(f"  {reason:<15} {n}")
        doors = Counter(first["scores"][k]["accepted_by"] for k in only)
        print("  door they used here: " + ", ".join(f"{k}={v}" for k, v in doors.most_common()))
        print(f"\n  top {min(args.top, len(only))} by anchor in {first['label']}:")
        print_table([row(first, second, k) for k in only], first["label"], second["label"], args.top)

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["key", "title", "year", "in_A", "in_B", "door_A", "door_B", "anchor_A", "anchor_B",
                        "topic_sim_A", "seed_sim_A", "status_in_other"])
            for key in sorted(union):
                sa, sb = a["scores"].get(key), b["scores"].get(key)
                where = "both" if key in both else ("only_A" if key in only_a else "only_B")
                other_status = "" if where == "both" else (why_missing(b, key) if where == "only_A" else why_missing(a, key))
                w.writerow([key, a["titles"].get(key) or b["titles"].get(key, ""),
                            a["years"].get(key) or b["years"].get(key), key in acc_a, key in acc_b,
                            sa["accepted_by"] if sa and sa["accepted"] else "", sb["accepted_by"] if sb and sb["accepted"] else "",
                            sa["anchor"] if sa else "", sb["anchor"] if sb else "",
                            sa["topic_sim"] if sa else "", sa["seed_sim"] if sa else "", f"{where}: {other_status}".strip(": ")])
        print(f"\nCSV written: {args.csv} ({len(union)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
