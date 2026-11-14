"""tests/test_b1_door_overlap.py: do B1's acceptance doors accept DIFFERENT papers?

Plain-language idea
    A paper can be let in by door 1 (anchor >= SIM_THRESHOLD), by door 2 (seed_sim >= SEED_ACCEPT_THRESHOLD),
    by the guarantee (top references of a seed) or by exploration. discovery.json stores the label of the FIRST
    door that matched ("accepted_by"), which hides overlap. This file recomputes, from the stored numbers only,
    which doors each accepted paper would pass, and answers: "if I removed door X, how many papers would I lose?"
    Overlap itself is NOT a bug (the doors are "or" conditions). It only tells you which doors are redundant.

Use as a script on a real run (prints the table, exits with 1 if the labels are inconsistent):
    python tests/test_b1_door_overlap.py runs/run_001/discovery.json

Use as tests: pytest tests/test_b1_door_overlap.py
    The synthetic checks always run. If runs/run_001/discovery.json exists (or the environment variable
    B1_DISCOVERY_JSON points at one), the labels of that real run are checked too.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

try:                                    # pytest is only needed for the tests, not for script use
    import pytest
except ImportError:                     # pragma: no cover
    pytest = None

TOL = 1e-6        # scores are rounded to 6 decimals in discovery.json


def door_overlap(doc: dict, sim_threshold: Optional[float] = None,
                 seed_accept_threshold: Optional[float] = None) -> dict:
    """Which doors would each paper pass? Thresholds default to the ones the run used (doc["meta"]["settings"])."""
    settings = doc.get("meta", {}).get("settings", {})
    t1 = settings.get("sim_threshold") if sim_threshold is None else sim_threshold
    t2 = settings.get("seed_accept_threshold") if seed_accept_threshold is None else seed_accept_threshold
    if t1 is None or t2 is None:
        raise ValueError("thresholds are not stored in the document: pass sim_threshold and seed_accept_threshold")

    accepted, rejected_passing = [], []
    for key, sc in doc["scores"].items():
        if sc["accepted_by"] == "seed":                      # seeds are always in: no door involved
            continue
        door1, door2 = sc["anchor"] >= t1 - TOL, sc["seed_sim"] >= t2 - TOL
        row = {"key": key, "label": sc["accepted_by"], "door1": door1, "door2": door2,
               "guaranteed": sc["guaranteed_for"] is not None}
        if sc["accepted"]:
            accepted.append(row)
        elif door1 or door2:                                 # passed a door but was not taken: the budget stopped it
            rejected_passing.append(key)

    def count(pred):
        return sum(1 for r in accepted if pred(r))

    breakdown = {"both_doors": count(lambda r: r["door1"] and r["door2"]),
                 "only_door1": count(lambda r: r["door1"] and not r["door2"]),
                 "only_door2": count(lambda r: r["door2"] and not r["door1"]),
                 "neither_door": count(lambda r: not r["door1"] and not r["door2"])}
    guaranteed = [r for r in accepted if r["guaranteed"]]
    return {
        "thresholds": {"sim_threshold": t1, "seed_accept_threshold": t2},
        "n_accepted_non_seed": len(accepted),
        "labels": dict(Counter(r["label"] for r in accepted)),
        "breakdown": breakdown,
        # what would be lost if a door were removed (guaranteed papers are kept by the guarantee anyway)
        "lost_without_door1": sorted(r["key"] for r in accepted
                                     if r["door1"] and not r["door2"] and not r["guaranteed"]),
        "lost_without_door2": sorted(r["key"] for r in accepted
                                     if r["door2"] and not r["door1"] and not r["guaranteed"]),
        "guaranteed": {"total": len(guaranteed),
                       "also_pass_a_door": sum(1 for r in guaranteed if r["door1"] or r["door2"]),
                       "only_because_guaranteed": sum(1 for r in guaranteed if not r["door1"] and not r["door2"])},
        "explore_total": sum(1 for r in accepted if r["label"] == "explore"),
        "rejected_but_passing_a_door": sorted(rejected_passing),
    }


def label_problems(doc: dict, sim_threshold: Optional[float] = None,
                   seed_accept_threshold: Optional[float] = None) -> list[str]:
    """Does every label agree with the numbers? (a bug check: "anchor" needs door 1, "seed_sim" needs door 2
    and must NOT pass door 1, because the code tries door 1 first.)"""
    settings = doc.get("meta", {}).get("settings", {})
    t1 = settings.get("sim_threshold") if sim_threshold is None else sim_threshold
    t2 = settings.get("seed_accept_threshold") if seed_accept_threshold is None else seed_accept_threshold
    problems = []
    for key, sc in doc["scores"].items():
        label = sc["accepted_by"]
        if label == "anchor" and not sc["anchor"] >= t1 - TOL:
            problems.append(f"{key}: labelled 'anchor' but anchor {sc['anchor']} < {t1}")
        if label == "seed_sim" and (sc["anchor"] >= t1 - TOL or not sc["seed_sim"] >= t2 - TOL):
            problems.append(f"{key}: labelled 'seed_sim' but it does not pass exactly door 2")
        if label == "guaranteed" and sc["guaranteed_for"] is None:
            problems.append(f"{key}: labelled 'guaranteed' without a seed")
        if sc["accepted"] != (label is not None):
            problems.append(f"{key}: 'accepted' and the label disagree")
    return problems


def format_overlap(result: dict) -> str:
    b, g, t = result["breakdown"], result["guaranteed"], result["thresholds"]
    lines = [f"Accepted papers that are not seeds: {result['n_accepted_non_seed']}   "
             f"(door 1 at anchor >= {t['sim_threshold']}, door 2 at seed_sim >= {t['seed_accept_threshold']})",
             f"  labels in discovery.json : {result['labels']}",
             f"  pass both doors          : {b['both_doors']}",
             f"  pass only door 1         : {b['only_door1']}",
             f"  pass only door 2         : {b['only_door2']}   <- what door 2 adds beyond door 1",
             f"  pass neither door        : {b['neither_door']}   (guaranteed or exploration papers)",
             f"  guaranteed               : {g['total']} ({g['only_because_guaranteed']} only because of the guarantee)",
             f"  exploration              : {result['explore_total']}",
             f"  lost if door 1 is removed: {len(result['lost_without_door1'])}",
             f"  lost if door 2 is removed: {len(result['lost_without_door2'])}",
             f"  passed a door but NOT accepted (stopped by the budget): {len(result['rejected_but_passing_a_door'])}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- tests

def _score(accepted_by, anchor, seed_sim, guaranteed_for=None):
    return {"accepted": accepted_by is not None, "accepted_by": accepted_by, "anchor": anchor,
            "seed_sim": seed_sim, "guaranteed_for": guaranteed_for}


def synthetic_doc():
    return {"meta": {"settings": {"sim_threshold": 0.35, "seed_accept_threshold": 0.60}},
            "scores": {"S": _score("seed", 0.9, 1.0),
                       "both": _score("anchor", 0.50, 0.70),
                       "only1": _score("anchor", 0.50, 0.30),
                       "only2": _score("seed_sim", 0.30, 0.65),
                       "g_neither": _score("guaranteed", 0.10, 0.20, guaranteed_for="S"),
                       "g_door1": _score("guaranteed", 0.50, 0.30, guaranteed_for="S"),
                       "explore": _score("explore", 0.20, 0.20),
                       "rej_none": _score(None, 0.10, 0.10),
                       "rej_passing": _score(None, 0.60, 0.10)}}


def test_door_overlap_counts_on_a_synthetic_run():
    r = door_overlap(synthetic_doc())
    assert r["n_accepted_non_seed"] == 6                                     # the seed and the 2 rejected are not counted
    assert r["breakdown"] == {"both_doors": 1, "only_door1": 2, "only_door2": 1, "neither_door": 2}
    assert r["lost_without_door2"] == ["only2"]                              # the one paper only door 2 brings in
    assert r["lost_without_door1"] == ["only1"]                              # g_door1 is kept by its guarantee
    assert r["guaranteed"] == {"total": 2, "also_pass_a_door": 1, "only_because_guaranteed": 1}
    assert r["explore_total"] == 1
    assert r["rejected_but_passing_a_door"] == ["rej_passing"]
    assert "pass only door 2" in format_overlap(r)


def test_door_overlap_can_try_other_thresholds_and_needs_them_when_missing():
    doc = synthetic_doc()
    strict = door_overlap(doc, sim_threshold=0.99, seed_accept_threshold=0.99)
    assert strict["breakdown"]["both_doors"] == 0 and strict["breakdown"]["neither_door"] == 6
    del doc["meta"]["settings"]
    with pytest.raises(ValueError):
        door_overlap(doc)


def test_door_two_is_redundant_when_its_threshold_is_at_least_sim_threshold_over_one_minus_w():
    """The algebra from the design chat: with anchor = w*T + (1-w)*S, door 1 means T+S >= 0.7 (w = 0.5 and
    threshold 0.35). Door 2 can only add a paper with S >= t2 and anchor < t1. With T in 0..1 the best case
    is T = 0, so door 2 adds nothing as soon as t2 >= t1 / (1 - w) = 0.7."""
    w, t1 = 0.5, 0.35
    for t2, may_add in ((0.60, True), (0.70, False), (0.80, False)):
        best_anchor = w * 0.0 + (1 - w) * t2                                 # a paper with S = t2 and no topic similarity
        adds_something = best_anchor < t1 - 1e-12
        assert adds_something == may_add


def test_label_problems_finds_a_wrong_label():
    assert label_problems(synthetic_doc()) == []
    bad = synthetic_doc()
    bad["scores"]["only1"]["accepted_by"] = "seed_sim"                       # passes door 1, so it can never carry this label
    bad["scores"]["only2"]["anchor"] = 0.10
    bad["scores"]["only2"]["accepted_by"] = "anchor"                         # fails door 1
    problems = label_problems(bad)
    assert any("only1" in p for p in problems) and any("only2" in p for p in problems)


def test_labels_of_a_real_run_agree_with_their_numbers():
    path = Path(os.environ.get("B1_DISCOVERY_JSON", "runs/run_001/discovery.json"))
    if not path.exists():
        pytest.skip(f"no real run found at {path}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert label_problems(doc) == []
    print("\n" + format_overlap(door_overlap(doc)))


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/run_001/discovery.json")
    document = json.loads(target.read_text(encoding="utf-8"))
    print(format_overlap(door_overlap(document)))
    issues = label_problems(document)
    print("\nlabel check:", "OK" if not issues else f"{len(issues)} problem(s)")
    for issue in issues[:10]:
        print("  -", issue)
    sys.exit(1 if issues else 0)
