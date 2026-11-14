"""tests/test_b1_discovery.py: B1 (b1_discovery.py) with a fake Semantic Scholar and a fake encoder.

No internet and no BGE-M3 needed. The fake encoder turns words into numbers, so papers that share words are
"similar". Topic used everywhere: "alpha beta". Relevant papers contain those words, irrelevant ones do not.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import replace

import numpy as np
import pytest

from module_A_tooling.cache import Cache
from module_A_tooling.config import RUN_FILES
from module_B_literature_kg.b0 import pipeline
from module_B_literature_kg.b0.assemble import run_assemble_step
from module_B_literature_kg.b0.b1_discovery import (B1ConfigError, B1Settings, DISCOVERY_FILE,
                                                    cite_score, discover, normalise_budget, replay_doors,
                                                    run_b1_step, seed_similarity)
from module_B_literature_kg.b0.clients import HttpError, new_paper
from module_B_literature_kg.b0.dedupe import run_dedupe_step
from module_B_literature_kg.b0.embed import Embedder, run_embed_step
from module_B_literature_kg.b0.fulltext import FulltextFetcher, run_fulltext_step
from module_B_literature_kg.b0.manifest import Manifest, atomic_write_json, read_json

TOPIC = "alpha beta"
DIM = 64


# --------------------------------------------------------------------------- fakes

def hash_encoder(texts):
    """Bag of words -> 64 numbers. Same words = similar vectors. Deterministic (no Python hash())."""
    out = np.zeros((len(texts), DIM))
    for i, text in enumerate(texts):
        for word in text.lower().split():
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            out[i, h % DIM] += 1.0
            out[i, (h // DIM) % DIM] += 0.5
    return out


def make_embedder():
    return Embedder(encoder=hash_encoder, cache=Cache(":memory:"))


class FakeS2:
    """get_references(s2_id, cap) from a dict {s2_id: [paper dicts]}; counts the calls."""

    def __init__(self, graph, fail_ids=()):
        self.graph, self.fail_ids, self.calls = graph, set(fail_ids), []

    def get_references(self, s2_id, cap):
        self.calls.append(s2_id)
        if s2_id in self.fail_ids:
            raise HttpError(500, "fake")
        return {"references": [copy.deepcopy(p) for p in self.graph.get(s2_id, [])][:cap],
                "partial": False, "skipped_no_id": 0}


def paper(n, words, *, abstract="", year=2020, cites=40, title_extra=""):
    """A paper dict in the shape clients.new_paper makes. Title = words, abstract = the same words by default."""
    return new_paper(s2_id=f"s{n:03d}", title=f"{words} {title_extra} p{n}".replace("  ", " "),
                     abstract=abstract if abstract != "" else f"{words} study of {words}",
                     year=year, citation_count=cites, authors=[f"Author{n}"])


def docs(seeds, graph):
    """seeds.json and references.json documents for these seed papers and this citation graph."""
    seed_keys = {p["key"] for p in seeds}
    stubs, edges, by_seed = {}, [], {}
    for s in seeds:
        refs = graph.get(s["s2_id"], [])
        by_seed[s["key"]] = {"status": "ok", "n_references": len(refs), "partial": False, "skipped_no_id": 0}
        for r in refs:
            edges.append({"source": s["key"], "target": r["key"], "kind": "cites"})
            if r["key"] not in seed_keys:
                stubs[r["key"]] = {**r, "is_seed": False, "cited_by_seeds": [s["key"]]}
    seeds_doc = {"version": 1, "seeds": [{"input": p["title"], "input_kind": "title", "status": "resolved",
                                          "paper": {**p, "is_seed": True}, "match": {}, "abstract_source": "s2",
                                          "flags": [], "notes": []} for p in seeds],
                 "duplicates": [], "unresolved": [], "needs_confirmation": [], "warnings": []}
    refs_doc = {"version": 1, "cap": 200, "by_seed": by_seed, "stubs": stubs,
                "edges": sorted(edges, key=lambda e: (e["source"], e["target"])), "warnings": []}
    return seeds_doc, refs_doc


def run(seeds, graph, *, budget=None, topic=TOPIC, as_of_year=2026, fail_ids=(), **overrides):
    seeds_doc, refs_doc = docs(seeds, graph)
    s2 = FakeS2(graph, fail_ids)
    st = replace(B1Settings(), **overrides)
    doc = discover(seeds_doc, refs_doc, topic, make_embedder(), s2, budget, st, as_of_year)
    return doc, s2


def accepted(doc, how=None):
    return sorted(k for k, s in doc["scores"].items() if s["accepted"] and (how is None or s["accepted_by"] == how))


def relevant_graph(n_children=6, n_grand=3):
    """seed -> n_children relevant refs -> each cites n_grand relevant refs. Returns (seed, graph, ids)."""
    seed = paper(1, "alpha beta seed")
    graph, nxt, children = {}, 100, []
    for _ in range(n_children):
        c = paper(nxt, f"alpha beta topic{nxt}")
        nxt += 1
        children.append(c)
        graph[c["s2_id"]] = []
        for _ in range(n_grand):
            g = paper(nxt, f"alpha beta deeper{nxt}")
            nxt += 1
            graph[c["s2_id"]].append(g)
    graph[seed["s2_id"]] = children
    return seed, graph


# --------------------------------------------------------------------------- small rules

def test_cite_score_fixed_rule_and_missing_data():
    st = B1Settings()
    p = {"citation_count": 100, "year": 2016}                       # 100 / 10 years = 10 per year
    score, missing = cite_score(p, 2026, st)
    assert not missing and score == pytest.approx(np.log1p(10) / np.log1p(100))
    assert cite_score({"citation_count": 10**6, "year": 2000}, 2026, st)[0] == 1.0      # clipped
    assert cite_score({"citation_count": 0, "year": 2020}, 2026, st)[0] == 0.0
    # brand-new paper: age is at least one year, so no division by zero
    assert cite_score({"citation_count": 5, "year": 2026}, 2026, st)[0] == pytest.approx(np.log1p(5) / np.log1p(100))
    for missing_one in ({"citation_count": None, "year": 2020}, {"citation_count": 5, "year": None}):
        assert cite_score(missing_one, 2026, st) == (st.cite_missing_score, True)


def test_seed_similarity_modes():
    sims = np.array([0.9, 0.5, 0.1])
    assert seed_similarity(sims, 0.3, "max", 3) == pytest.approx(0.9)
    assert seed_similarity(sims, 0.3, "top_k_mean", 2) == pytest.approx(0.7)
    assert seed_similarity(sims, 0.3, "top_k_mean", 9) == pytest.approx(0.5)     # K larger than the seeds
    assert seed_similarity(sims, 0.3, "centroid", 3) == pytest.approx(0.3)
    assert seed_similarity(np.array([-0.4]), 0.0, "max", 3) == 0.0               # clamped to 0..1
    with pytest.raises(ValueError):
        seed_similarity(sims, 0.3, "nope", 3)


def test_normalise_budget():
    assert normalise_budget(None) == {"max_papers": None, "max_depth": 2, "fanout_per_node": 20}   # None = grow with the seeds
    assert normalise_budget({"max_papers": 30})["max_papers"] == 30
    with pytest.raises(B1ConfigError):
        normalise_budget({"max_papers": 0})
    with pytest.raises(B1ConfigError):
        normalise_budget({"max_depth": 1.5})

    class Obj:                                                       # like schemas.Budget
        max_papers, max_depth, fanout_per_node = 10, 3, 5
    assert normalise_budget(Obj()) == {"max_papers": 10, "max_depth": 3, "fanout_per_node": 5}
    assert pipeline.read_budget({"topic": "x", "budget": {"max_papers": 7}})["max_papers"] == 7
    assert pipeline.read_budget({"topic": "x"})["max_papers"] is None


def test_pipeline_order_has_b1_between_references_and_fulltext():
    order = pipeline.STEP_ORDER
    assert order.index("b0_references") + 1 == order.index("b1_discovery") == order.index("b0_fulltext") - 1


# --------------------------------------------------------------------------- budget and planning

def test_budget_is_a_hard_limit_and_quota_is_derived():
    seeds = [paper(i, f"alpha beta seed{i}") for i in range(1, 11)]          # 10 seeds
    graph = {s["s2_id"]: [paper(1000 + i * 10 + j, f"alpha beta ref{i}x{j}") for j in range(5)]
             for i, s in enumerate(seeds)}
    doc, _ = run(seeds, graph, budget={"max_papers": 40, "max_depth": 2, "fanout_per_node": 20})
    plan = doc["meta"]["plan"]
    assert plan["guarantee_budget"] == 12 and plan["quota"] == 1             # 12 // 10 seeds = 1 each
    assert len(accepted(doc)) <= 40
    assert len(accepted(doc, "guaranteed")) <= plan["guarantee_budget"]
    assert len(accepted(doc, "seed")) == 10                                  # seeds are never removed


def test_not_enough_room_for_the_seeds_is_a_clear_error():
    seeds = [paper(i, f"alpha beta seed{i}") for i in range(1, 6)]
    with pytest.raises(B1ConfigError, match="do not fit"):
        run(seeds, {}, budget={"max_papers": 4})
    with pytest.raises(B1ConfigError, match="MAX_EXPANSIONS"):
        run(seeds, {}, budget={"max_papers": 100}, max_expansions=3)


def test_no_room_for_guarantees_warns_instead_of_breaking():
    seeds = [paper(i, f"alpha beta seed{i}") for i in range(1, 5)]
    doc, _ = run(seeds, {}, budget={"max_papers": 4})
    assert doc["meta"]["plan"]["quota"] == 0
    assert any("no guaranteed children" in w for w in doc["warnings"])


# --------------------------------------------------------------------------- guaranteed children

def test_guaranteed_child_counts_once_even_if_two_seeds_want_it():
    a, b = paper(1, "alpha beta seed"), paper(2, "alpha beta seed")
    shared = [paper(10 + i, f"alpha beta shared{i}") for i in range(4)]
    doc, _ = run([a, b], {a["s2_id"]: shared, b["s2_id"]: shared},
                 budget={"max_papers": 100, "max_depth": 2, "fanout_per_node": 20}, phase0_quota=3)
    g = doc["scores"]
    by_seed = {k: sorted(x for x, s in g.items() if s["guaranteed_for"] == k) for k in (a["key"], b["key"])}
    assert len(by_seed[a["key"]]) == 3                                      # first seed takes its 3
    assert len(by_seed[b["key"]]) == 1                                      # only 1 distinct paper is left for the second
    assert not set(by_seed[a["key"]]) & set(by_seed[b["key"]])
    assert doc["meta"]["seed_coverage"][a["key"]]["guaranteed"] == 3


def test_off_topic_seed_still_gets_its_guaranteed_children():
    topic_seed = paper(1, "alpha beta seed")
    off_seed = paper(2, "zeta omega theta seed")
    off_refs = [paper(20 + i, f"zeta omega theta bg{i}") for i in range(5)]
    graph = {topic_seed["s2_id"]: [paper(10 + i, f"alpha beta near{i}") for i in range(5)],
             off_seed["s2_id"]: off_refs}
    # both relevance doors closed: only the guarantee can let papers in
    doc, _ = run([topic_seed, off_seed], graph, budget={"max_papers": 60, "max_depth": 2, "fanout_per_node": 20},
                 phase0_quota=3, sim_threshold=0.99, seed_accept_threshold=0.999)
    off_guaranteed = [k for k, s in doc["scores"].items() if s["guaranteed_for"] == off_seed["key"]]
    assert len(off_guaranteed) == 3
    assert all(k in {r["key"] for r in off_refs} for k in off_guaranteed)
    assert accepted(doc, "anchor") == [] and accepted(doc, "seed_sim") == []


def test_guaranteed_children_are_expanded_and_their_references_judged_normally():
    seed, graph = relevant_graph(n_children=3, n_grand=2)
    doc, s2 = run([seed], graph, budget={"max_papers": 50, "max_depth": 2, "fanout_per_node": 20}, phase0_quota=3)
    kids = [r["key"] for r in graph[seed["s2_id"]]]
    assert all(doc["scores"][k]["expanded"] for k in kids)
    assert seed["s2_id"] not in s2.calls                                    # the seed's list came from references.json
    assert len(s2.calls) >= 3


# --------------------------------------------------------------------------- the same paper found twice

def test_paper_cited_by_two_seeds_is_one_paper_with_both_seeds_listed():
    a, b = paper(1, "alpha beta seed"), paper(2, "alpha beta seed")
    x = paper(10, "alpha beta shared")
    doc, _ = run([a, b], {a["s2_id"]: [x], b["s2_id"]: [x]}, budget={"max_papers": 50, "max_depth": 2, "fanout_per_node": 20})
    assert list(doc["stubs"]) == [x["key"]]                                 # stored once
    sc = doc["scores"][x["key"]]
    assert sc["cited_by"] == sorted([a["key"], b["key"]])
    assert sc["reached_from"] == sorted([a["key"], b["key"]])
    links = {(e["source"], e["target"]) for e in doc["edges"]}
    assert (a["key"], x["key"]) in links and (b["key"], x["key"]) in links and len(doc["edges"]) == 2
    cov = doc["meta"]["seed_coverage"]
    assert cov[a["key"]]["shared"] == 1 and cov[a["key"]]["unique"] == 1   # x is shared; the seed itself is unique to it


def test_shortest_depth_wins_and_priority_only_goes_up():
    seed = paper(1, "alpha beta seed")
    child = paper(10, "alpha beta child")
    x = paper(11, "alpha beta target")
    # x is cited by the seed (depth 1) AND by the child (depth 2)
    graph = {seed["s2_id"]: [child, x], child["s2_id"]: [x], x["s2_id"]: []}
    doc, _ = run([seed], graph, budget={"max_papers": 50, "max_depth": 3, "fanout_per_node": 20})
    sc = doc["scores"][x["key"]]
    assert sc["depth"] == 1
    assert sc["cited_by"] == sorted([seed["key"], child["key"]])
    assert sc["parent_best"] >= doc["scores"][child["key"]]["priority"] or sc["parent_best"] == 1.0
    assert len([e for e in doc["edges"] if e["target"] == x["key"]]) == 2


# --------------------------------------------------------------------------- scoring details

def test_missing_citations_get_the_neutral_score_and_a_flag():
    seed = paper(1, "alpha beta seed")
    known, unknown = paper(10, "alpha beta known"), paper(11, "alpha beta unknown")
    unknown["citation_count"] = None
    doc, _ = run([seed], {seed["s2_id"]: [known, unknown]}, budget={"max_papers": 20, "max_depth": 2, "fanout_per_node": 20})
    assert doc["scores"][unknown["key"]]["cite_score"] == 0.3
    assert "cite_missing" in doc["scores"][unknown["key"]]["flags"]
    assert "cite_missing" not in doc["scores"][known["key"]]["flags"]


def test_paper_without_abstract_is_scored_on_its_title_and_flagged_and_empty_ones_are_skipped():
    seed = paper(1, "alpha beta seed")
    title_only = new_paper(s2_id="s010", title="alpha beta title only", abstract="", year=2020, citation_count=3)
    empty = new_paper(s2_id="s011", title="", abstract="", year=2020, citation_count=3)
    doc, _ = run([seed], {seed["s2_id"]: [title_only, empty]}, budget={"max_papers": 20, "max_depth": 2, "fanout_per_node": 20})
    assert "title_only" in doc["scores"][title_only["key"]]["flags"]
    assert empty["key"] not in doc["scores"]
    assert doc["meta"]["n_unscorable"] == 1


def test_every_judged_paper_has_all_scores_including_rejected_ones_and_the_similarity_to_each_seed():
    seed = paper(1, "alpha beta seed")
    good, bad = paper(10, "alpha beta good"), paper(11, "zeta omega unrelated")
    doc, _ = run([seed], {seed["s2_id"]: [good, bad]}, budget={"max_papers": 20, "max_depth": 2, "fanout_per_node": 20},
                 phase0_quota=1)
    assert doc["scores"][good["key"]]["accepted"] and not doc["scores"][bad["key"]]["accepted"]
    for key in (good["key"], bad["key"]):
        sc = doc["scores"][key]
        for field in ("topic_sim", "seed_sim", "anchor", "cite_score", "priority", "sim_to_seeds", "depth"):
            assert sc[field] is not None
        assert list(sc["sim_to_seeds"]) == [seed["key"]]
    assert bad["key"] not in doc["stubs"] and good["key"] in doc["stubs"]


def test_priority_is_anchor_plus_citations_plus_parent_best_and_there_is_no_base_score():
    seed = paper(1, "alpha beta seed")
    child = paper(10, "alpha beta child")
    doc, _ = run([seed], {seed["s2_id"]: [child]}, budget={"max_papers": 20, "max_depth": 2, "fanout_per_node": 20})
    sc = doc["scores"][child["key"]]
    pw = B1Settings().priority_weights
    assert "base" not in sc
    assert sc["parent_best"] == 1.0                                          # its parent is a seed
    assert sc["priority"] == pytest.approx(pw["anchor"] * sc["anchor"] + pw["cite"] * sc["cite_score"]
                                           + pw["parent"] * 1.0, abs=1e-5)


def test_citations_order_the_expansion_but_never_decide_who_is_accepted():
    seed = paper(1, "alpha beta seed")
    guaranteed = paper(10, "alpha beta seed guaranteed")                    # takes the guarantee
    plain = paper(11, "alpha beta seed topic", cites=0, year=2020)           # more relevant, not cited
    famous = paper(12, "alpha beta gamma", cites=10**6, year=2000)           # less relevant, very cited
    doc, _ = run([seed], {seed["s2_id"]: [guaranteed, plain, famous]},
                 budget={"max_papers": 50, "max_depth": 2, "fanout_per_node": 1},       # ONE slot for the two others
                 phase0_quota=1, sim_threshold=0.0)
    p, f = doc["scores"][plain["key"]], doc["scores"][famous["key"]]
    assert p["anchor"] > f["anchor"] and f["priority"] > p["priority"]        # the case where the two rules disagree
    assert p["accepted"] and not f["accepted"]                                # the anchor decided, not the citations


# --------------------------------------------------------------------------- the cap grows with the seeds

def test_max_papers_grows_linearly_with_the_number_of_seeds_when_the_user_gives_none():
    for n_seeds, expected in ((1, 65), (2, 90), (5, 165), (10, 290), (20, 540)):     # 40 + 25 * n, no ceiling
        seeds = [paper(i, f"alpha beta seed{i}") for i in range(1, n_seeds + 1)]
        doc, _ = run(seeds, {})
        plan = doc["meta"]["plan"]
        assert plan["max_papers"] == expected and plan["max_papers_source"] == "formula"
        assert plan["quota"] >= 5                                           # many seeds no longer starve the guarantees


def test_a_number_given_in_the_request_wins_over_the_formula():
    seeds = [paper(i, f"alpha beta seed{i}") for i in range(1, 4)]
    doc, _ = run(seeds, {}, budget={"max_papers": 77})
    assert doc["meta"]["plan"]["max_papers"] == 77 and doc["meta"]["plan"]["max_papers_source"] == "request"
    doc, _ = run(seeds, {}, papers_base=10, papers_per_seed=5)               # the formula numbers are settings
    assert doc["meta"]["plan"]["max_papers"] == 25


def test_the_expansion_limit_scales_with_the_paper_budget():
    seeds = [paper(i, f"alpha beta seed{i}") for i in range(1, 3)]
    doc, _ = run(seeds, {})                                                  # 90 papers -> ceil(1.5 * 90)
    assert doc["meta"]["plan"]["max_expansions"] == 135
    doc, _ = run(seeds, {}, max_expansions_factor=2.0)
    assert doc["meta"]["plan"]["max_expansions"] == 180
    doc, _ = run(seeds, {}, max_expansions=7, phase0_quota=1)                # an explicit number wins (2 x (1+1) fits)
    assert doc["meta"]["plan"]["max_expansions"] == 7


# --------------------------------------------------------------------------- stop rules

def test_stop_when_max_papers_is_reached_and_never_go_over():
    seed, graph = relevant_graph(n_children=8, n_grand=6)
    doc, _ = run([seed], graph, budget={"max_papers": 10, "max_depth": 3, "fanout_per_node": 20})
    assert doc["meta"]["stop_reason"] == "max_papers"
    assert len(accepted(doc)) == 10


def test_stop_when_max_expansions_is_reached():
    seed, graph = relevant_graph(n_children=6, n_grand=3)
    doc, s2 = run([seed], graph, budget={"max_papers": 100, "max_depth": 3, "fanout_per_node": 20},
                  phase0_quota=1, max_expansions=3)
    assert doc["meta"]["stop_reason"] == "max_expansions"
    assert doc["meta"]["expansions"] == 3                                    # seed + 1 guaranteed child + 1 best-first


def test_stop_when_nothing_is_left():
    seed, graph = relevant_graph(n_children=2, n_grand=0)
    doc, _ = run([seed], graph, budget={"max_papers": 100, "max_depth": 3, "fanout_per_node": 20})
    assert doc["meta"]["stop_reason"] == "frontier_empty"


def test_depth_limit_soft_and_the_high_sim_exemption():
    seed = paper(1, "alpha beta seed")
    c1, c2, c3 = paper(10, "alpha beta one"), paper(11, "alpha beta two"), paper(12, "alpha beta three")
    graph = {seed["s2_id"]: [c1], c1["s2_id"]: [c2], c2["s2_id"]: [c3], c3["s2_id"]: []}
    budget = {"max_papers": 50, "max_depth": 1, "fanout_per_node": 20}
    # soft depth 1: only the seed's children (guaranteed ones) are read; c2 is never reached by the best-first loop
    doc, _ = run([seed], graph, budget=budget, high_sim=1.01, phase0_quota=1)
    assert c1["key"] in doc["scores"] and c2["key"] in doc["scores"] and c3["key"] not in doc["scores"]
    # relevant papers (anchor >= HIGH_SIM) may go deeper than the soft depth, up to ABS_MAX_DEPTH
    doc, _ = run([seed], graph, budget=budget, high_sim=0.0, abs_max_depth=3, phase0_quota=1)
    assert c3["key"] in doc["scores"] and doc["scores"][c3["key"]]["depth"] == 3
    doc, _ = run([seed], graph, budget=budget, high_sim=0.0, abs_max_depth=2, phase0_quota=1)
    assert c3["key"] not in doc["scores"]


def test_every_seed_is_expanded_and_a_failed_fetch_is_recorded_not_fatal():
    seed, graph = relevant_graph(n_children=3, n_grand=2)
    bad_child = graph[seed["s2_id"]][0]
    doc, _ = run([seed], graph, budget={"max_papers": 50, "max_depth": 2, "fanout_per_node": 20},
                 phase0_quota=3, fail_ids={bad_child["s2_id"]})
    assert doc["scores"][seed["key"]]["expanded"]
    assert "expand_error" in doc["scores"][bad_child["key"]]
    assert any("reference fetch failed" in w for w in doc["warnings"])


# --------------------------------------------------------------------------- exploration

def explore_case(**extra):
    seed = paper(1, "alpha beta seed")
    refs = [paper(10, "alpha gamma delta epsilon"), paper(11, "beta gamma delta zeta"),
            paper(12, "alpha beta seed", abstract="alpha beta seed study of alpha beta seed", title_extra=""),
            paper(13, "omega theta iota kappa")]
    dup = copy.deepcopy(seed)                                                # a near-copy of the seed
    dup.update(s2_id="s012", title=seed["title"], abstract=seed["abstract"])
    refs[2] = new_paper(**{k: dup[k] for k in ("s2_id", "title", "abstract", "year", "citation_count", "authors")})
    graph = {seed["s2_id"]: refs}
    base = dict(sim_threshold=0.97, seed_accept_threshold=0.999, high_sim=0.999, explore_min_anchor=0.05,
                novelty_min=0.2, phase0_quota=1)
    base.update(extra)
    return run([seed], graph, budget={"max_papers": 20, "max_depth": 2, "fanout_per_node": 20}, **base), refs


def test_exploration_is_off_by_default():
    (doc, _), _ = explore_case()
    assert accepted(doc, "explore") == []
    assert all(s["novelty"] is None for s in doc["scores"].values())


def test_exploration_respects_floor_novelty_slots_and_budget():
    (doc, _), refs = explore_case(explore_fraction=0.2)
    explore = accepted(doc, "explore")
    assert 1 <= len(explore) <= 4                                           # round(0.2 * 20) slots
    assert len(accepted(doc)) <= 20
    for k in explore:
        sc = doc["scores"][k]
        assert sc["anchor"] >= 0.05 and sc["novelty"] >= 0.2
        assert not sc["expanded"]                                           # exploration papers are leaves
    assert refs[2]["key"] not in explore                                    # the near-copy of the seed is not novel
    assert doc["meta"]["plan"]["normal_cap"] == 16


def test_bridge_paper_beats_a_more_novel_paper_for_the_exploration_slot():
    a, b = paper(1, "alpha beta seed"), paper(2, "zeta omega seed")
    a1, b1 = paper(10, "alpha beta guaranteed"), paper(11, "zeta omega guaranteed")
    x = paper(20, "alpha zeta bridge")                                      # cited by BOTH territories
    y = paper(21, "alpha beta theta iota only from a")                      # cited from one territory only
    graph = {a["s2_id"]: [a1], b["s2_id"]: [b1], a1["s2_id"]: [x, y], b1["s2_id"]: [x]}
    doc, _ = run([a, b], graph, budget={"max_papers": 10, "max_depth": 2, "fanout_per_node": 20},
                 sim_threshold=0.999, seed_accept_threshold=0.999, high_sim=0.999, phase0_quota=1,
                 explore_fraction=0.1, explore_min_anchor=0.01, novelty_min=0.0)
    assert doc["meta"]["plan"]["explore_slots"] == 1
    assert accepted(doc, "explore") == [x["key"]]
    assert doc["scores"][x["key"]]["bridge"] == 1 and doc["scores"][y["key"]]["bridge"] == 0


# --------------------------------------------------------------------------- reproducibility

def test_same_input_twice_gives_identical_output_even_if_the_input_order_changes():
    seed, graph = relevant_graph(n_children=6, n_grand=4)
    budget = {"max_papers": 14, "max_depth": 3, "fanout_per_node": 3}
    first, _ = run([seed], graph, budget=budget, phase0_quota=2)
    second, _ = run([seed], graph, budget=budget, phase0_quota=2)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)

    shuffled = {k: random.Random(7).sample(v, len(v)) for k, v in graph.items()}
    third, _ = run([seed], shuffled, budget=budget, phase0_quota=2)
    assert accepted(first) == accepted(third)


# --------------------------------------------------------------------------- replay

def test_replay_doors_reproduces_the_run_and_lets_you_try_other_settings():
    seed, graph = relevant_graph(n_children=4, n_grand=2)
    doc, _ = run([seed], graph, budget={"max_papers": 50, "max_depth": 2, "fanout_per_node": 20}, phase0_quota=1)
    same = replay_doors(doc)
    door_accepted = sorted(k for k, s in doc["scores"].items() if s["accepted_by"] in ("anchor", "seed_sim"))
    assert set(door_accepted) <= set(same["would_pass"])
    strict = replay_doors(doc, sim_threshold=0.999, seed_accept_threshold=0.999)
    assert strict["n_would_pass"] == 0
    assert replay_doors(doc, seed_sim_mode="top_k_mean")["n_would_pass"] >= 0
    with pytest.raises(ValueError):
        replay_doors(doc, seed_sim_mode="centroid")


# --------------------------------------------------------------------------- the step, with B0's manifest

def write_run(root, seeds, graph):
    seeds_doc, refs_doc = docs(seeds, graph)
    atomic_write_json(root / RUN_FILES["seeds"], seeds_doc)
    atomic_write_json(root / RUN_FILES["references"], refs_doc)
    return seeds_doc, refs_doc


def write_run_as_steps(manifest, seeds, graph):
    """Same files as write_run, but written THROUGH the manifest, so b0_seeds and b0_references count as done."""
    from module_B_literature_kg.b0.manifest import run_step
    seeds_doc, refs_doc = docs(seeds, graph)
    run_step(manifest, "b0_seeds", lambda st: atomic_write_json(st / RUN_FILES["seeds"], seeds_doc),
             outputs=[RUN_FILES["seeds"]])
    run_step(manifest, "b0_references", lambda st: atomic_write_json(st / RUN_FILES["references"], refs_doc),
             outputs=[RUN_FILES["references"]], inputs=[RUN_FILES["seeds"]])


def test_step_writes_discovery_json_skips_when_nothing_changed_and_reruns_when_the_budget_changes(tmp_path):
    seed, graph = relevant_graph(n_children=4, n_grand=2)
    write_run(tmp_path, [seed], graph)
    manifest = Manifest("t", root=tmp_path)
    s2, emb = FakeS2(graph), make_embedder()
    budget = {"max_papers": 30, "max_depth": 2, "fanout_per_node": 20}
    first = run_b1_step(manifest, TOPIC, s2, emb, budget, as_of_year=2026)
    assert first.status == "done" and (tmp_path / DISCOVERY_FILE).exists()
    assert first.notes["stop_reason"] and first.notes["n_accepted"] >= 1
    calls = len(s2.calls)
    again = run_b1_step(manifest, TOPIC, s2, emb, budget, as_of_year=2026)
    assert again.status == "skipped" and len(s2.calls) == calls              # nothing re-fetched
    changed = run_b1_step(manifest, TOPIC, s2, emb, {**budget, "max_papers": 12}, as_of_year=2026)
    assert changed.status == "done"                                          # the budget is part of the fingerprint
    also = run_b1_step(manifest, TOPIC, s2, emb, {**budget, "max_papers": 12}, as_of_year=2027)
    assert also.status == "done"                                             # and so is the year of the citation score


def test_step_with_no_max_papers_uses_the_formula_and_checks_the_output_against_it(tmp_path):
    seed, graph = relevant_graph(n_children=4, n_grand=2)
    write_run(tmp_path, [seed], graph)
    manifest = Manifest("t", root=tmp_path)
    res = run_b1_step(manifest, TOPIC, FakeS2(graph), make_embedder(), None, as_of_year=2026)
    assert res.status == "done"
    meta = read_json(tmp_path / DISCOVERY_FILE)["meta"]
    assert meta["plan"]["max_papers"] == 65 and meta["plan"]["max_papers_source"] == "formula"


def test_step_needs_the_earlier_steps(tmp_path):
    from module_B_literature_kg.b0.manifest import MissingInputError
    with pytest.raises(MissingInputError):
        run_b1_step(Manifest("t", root=tmp_path), TOPIC, FakeS2({}), make_embedder())


def test_dedupe_reads_discovery_json_and_rejected_papers_never_get_an_id(tmp_path):
    seed = paper(1, "alpha beta seed")
    good = [paper(10 + i, f"alpha beta good{i}") for i in range(3)]
    bad = [paper(50 + i, f"zeta omega junk{i}") for i in range(4)]
    graph = {seed["s2_id"]: good + bad}
    write_run(tmp_path, [seed], graph)
    manifest = Manifest("t", root=tmp_path)
    run_b1_step(manifest, TOPIC, FakeS2(graph), make_embedder(),
                {"max_papers": 30, "max_depth": 2, "fanout_per_node": 20}, settings=B1Settings(phase0_quota=1),
                as_of_year=2026)
    disc = read_json(tmp_path / DISCOVERY_FILE)
    assert len(read_json(tmp_path / RUN_FILES["references"])["stubs"]) == 7
    res = run_dedupe_step(manifest)
    assert res.status == "done"
    canonical = read_json(tmp_path / RUN_FILES["canonical"])["papers"]
    assert len(canonical) == 1 + len(disc["stubs"]) < 8                      # ids P1..Pn only for accepted papers
    assert set(canonical) == {f"P{i}" for i in range(1, len(canonical) + 1)}  # no gaps
    junk_keys = {p["key"] for p in bad}
    assert not junk_keys & {k for p in canonical.values() for k in p["aliases"]}


def test_whole_chain_b1_then_fulltext_dedupe_embed_assemble(tmp_path):
    seed, graph = relevant_graph(n_children=4, n_grand=2)
    manifest = Manifest("t", root=tmp_path)
    write_run_as_steps(manifest, [seed], graph)
    emb = make_embedder()
    run_b1_step(manifest, TOPIC, FakeS2(graph), emb, {"max_papers": 30, "max_depth": 2, "fanout_per_node": 20},
                as_of_year=2026)
    fetcher = FulltextFetcher(cache=Cache(":memory:"), pdf_dir=tmp_path / "pdfs")   # seeds have no arXiv id: no download
    run_fulltext_step(manifest, fetcher)
    run_dedupe_step(manifest)
    run_embed_step(manifest, TOPIC, emb)
    res = run_assemble_step(manifest)
    assert res.status == "done"
    report = read_json(tmp_path / RUN_FILES["b0_report"])
    assert report["all_gates_passed"] and report["discovery"]["n_accepted"] == report["counts"]["n_papers"]
    records = read_json(tmp_path / RUN_FILES["paper_records"])
    assert all("discovery" in r for r in records.values())                    # every paper keeps its B1 numbers
    assert any(r["discovery"]["accepted_by"] == "seed" for r in records.values())


class FakeS2Full(FakeS2):
    """FakeS2 that can also look seeds up by DOI, so the whole run_b0 entry point can run offline."""

    def __init__(self, graph, by_doi):
        super().__init__(graph)
        self.by_doi = by_doi

    def get_paper(self, identifier):
        p = self.by_doi.get(identifier.split(":", 1)[1].lower()) if identifier.upper().startswith("DOI:") else None
        return copy.deepcopy(p)


class NoArxiv:
    def get_metadata(self, arxiv_id):
        return None


def test_run_b0_whole_pipeline_uses_the_request_budget_and_runs_b1_in_order(tmp_path):
    s1 = {**paper(1, "alpha beta seed one"), "doi": "10.1000/seed1"}
    s2_ = {**paper(2, "alpha beta seed two"), "doi": "10.1000/seed2"}
    graph = {s1["s2_id"]: [paper(10 + i, f"alpha beta ref{i}") for i in range(6)],
             s2_["s2_id"]: [paper(30 + i, f"alpha beta other{i}") for i in range(6)]}
    fake = FakeS2Full(graph, {"10.1000/seed1": s1, "10.1000/seed2": s2_})
    request = {"topic": TOPIC, "seed_papers": ["10.1000/seed1", "10.1000/seed2"],
               "budget": {"max_papers": 9, "max_depth": 2, "fanout_per_node": 5}}
    result = pipeline.run_b0(request, "t", root=tmp_path, s2=fake, arxiv=NoArxiv(),
                             fetcher=FulltextFetcher(cache=Cache(":memory:"), pdf_dir=tmp_path / "pdfs"),
                             embedder=make_embedder(), as_of_year=2026)
    assert result.completed
    assert [s.step for s in result.steps] == pipeline.STEP_ORDER
    assert result.report["all_gates_passed"]
    assert result.report["counts"]["n_papers"] <= 9                          # the request budget reached B1
    assert result.report["discovery"]["plan"]["max_papers"] == 9
    assert "b1_discovery" in result.report["steps"]
    assert "b1_discovery" in pipeline.format_summary(result)
    # stopping right after B1 works too, and a second full run only skips
    again = pipeline.run_b0(request, "t", root=tmp_path, s2=fake, arxiv=NoArxiv(), embedder=make_embedder(),
                            fetcher=FulltextFetcher(cache=Cache(":memory:"), pdf_dir=tmp_path / "pdfs"),
                            as_of_year=2026)
    assert all(s.status == "skipped" for s in again.steps)
