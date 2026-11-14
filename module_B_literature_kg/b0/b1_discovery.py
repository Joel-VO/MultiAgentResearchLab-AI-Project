"""module_B_literature_kg/b0/b1_discovery.py: step b1_discovery. Decide WHICH papers go into the knowledge base.

Plain-language idea (the B1 v2.2 spec)
    Start from the seed papers. Follow their citations. Score every paper that shows up ("is it relevant
    to the topic and to the seeds?"), keep the relevant ones, and always read the most promising paper
    next (best-first search). Stop when the paper budget, the expansion budget or the frontier runs out.

Where it sits
    b0_seeds -> b0_references -> [b1_discovery] -> b0_fulltext -> b0_dedupe -> b0_embed -> b0_assemble
    Reads   seeds.json and references.json (the seeds' reference lists: no new API call for them).
    Writes  discovery.json, in the SAME SHAPE as references.json ("stubs" + "edges") plus the scores:
        {"version", "stubs": {key: paper dict}, "edges": [{"source","target","kind":"cites"}],
         "scores": {key: every number for EVERY judged paper, accepted or rejected},
         "meta": {...}, "warnings": [...]}
    b0_dedupe reads discovery.json instead of references.json; final ids P1..Pn are given there, so
    rejected papers never leave gaps in the numbering.

The numbers (all 0..1)
    topic_sim  = w*sim(topic, title) + (1-w)*sim(topic, title+abstract)          w = TOPIC_TITLE_WEIGHT
    seed_sim   = similarity to the closest single seed (SEED_SIM_MODE "max"; the similarity to EVERY seed
                 is stored too, so other modes can be replayed offline)
    anchor     = ANCHOR_TOPIC_WEIGHT*topic_sim + (1-ANCHOR_TOPIC_WEIGHT)*seed_sim        "is it relevant?"
    cite_score = min(1, log1p(citations / max(as_of_year - year, 1)) / log1p(CITE_SATURATION)),
                 CITE_MISSING_SCORE (+ flag "cite_missing") when citations or year are unknown
    priority   = w.anchor*anchor + w.cite*cite_score + w.parent*parent_best     "which paper do I open next?"
                 (PRIORITY_WEIGHTS). It only orders the expansion queue: WHO IS ACCEPTED is decided by the anchor
                 alone, so a famous paper can never take the slot of a more relevant one.
                 parent_best = highest priority of an expanded paper that cites it (seed = 1.0); only goes up

Doors (a paper is accepted by one of them)
    anchor     anchor >= SIM_THRESHOLD
    seed_sim   seed_sim >= SEED_ACCEPT_THRESHOLD   (protects the neighbours of a seed that is off-topic)
    guaranteed the top-Q references of every seed, by similarity to THAT seed, no threshold (Phase 0)
    explore    Phase 2, only when EXPLORE_FRACTION > 0

Two modes (config.DISCOVERY_MODE, or B1Settings(mode=...), or --discovery-mode on the command line)
    "classic"  the rule described above (anchor door + seed_sim door, guaranteed children per seed).
    "groups"   seeds that look alike (average cosine >= GROUP_LINK_THRESHOLD) are clustered into groups; each
               group is ONE centroid (mean of its seeds, length 1). A paper is accepted through ONE door:
               its similarity to some group centroid reaches that group's threshold. The topic sentence is
               kept as one extra centroid (GROUP_INCLUDE_TOPIC). Thresholds can be per group (loose groups
               get a lower bar). Every seed must pass its own group's bar (self-test). The guaranteed
               children are given per GROUP, not per seed, so a big group cannot eat the budget.
               In groups mode "anchor" means the group score: max over groups of
               (similarity - group threshold + GROUP_ACCEPT_THRESHOLD), so one bar fits every group and
               priority / depth exemption / exploration keep working unchanged.
    Phase 2 (exploration) is the same in both modes. Nothing else (priority, budget, checks) changes.

The budget is a HARD limit. It is split BEFORE anything is accepted, so no repair (trim) step is needed:
    max_papers = the request's number if the user gave one, otherwise PAPERS_BASE + PAPERS_PER_SEED * n_seeds
    max_expansions = ceil(MAX_EXPANSIONS_FACTOR * max_papers)    (a safety valve: a paper is read at most once)
    explore_slots = round(EXPLORE_FRACTION * max_papers)        normal_cap = max_papers - explore_slots
    guarantee_budget = min(floor(GUARANTEE_FRACTION * max_papers), normal_cap - n_seeds)
    Q = min(PHASE0_QUOTA, fanout, guarantee_budget // n_seeds)

Phases
    0  accept the seeds; guaranteed children for every seed; other references through the doors;
       expand the seeds and the guaranteed children
    1  best-first until max_expansions, normal_cap or an empty queue
    2  exploration pass over papers already scored (no API call), only if explore_slots > 0
    finish: exact coverage report (which seeds reach which paper), drop links to rejected papers, checks.

Rules this file follows (so that running it twice gives the same result: "idempotent")
    * it writes only into the staging folder it is given, by key (never appends);
    * every choice is deterministic: ties are broken by depth, then by paper key;
    * the same paper found twice is ONE record: links are a set, ALL citers are kept, the shortest depth wins,
      the priority only goes up.
"""
from __future__ import annotations

import datetime
import heapq
import math
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import requests

from module_A_tooling.config import (ABS_MAX_DEPTH, ANCHOR_TOPIC_WEIGHT, B1_DEFAULT_BUDGET,
                                     DISCOVERY_MODE, GROUP_ACCEPT_THRESHOLD, GROUP_EXPLORE_MIN_ANCHOR,
                                     GROUP_HIGH_SIM, GROUP_INCLUDE_TOPIC, GROUP_LINK_THRESHOLD,
                                     GROUP_MIN_THRESHOLD, GROUP_SCORE_MODE, GROUP_SELF_TEST,
                                     GROUP_THRESHOLD_MODE, GROUP_TOPIC_THRESHOLD,
                                     CITE_MISSING_SCORE, CITE_SATURATION, EXPLORE_FRACTION,
                                     EXPLORE_MIN_ANCHOR, GUARANTEE_FRACTION, HIGH_SIM,
                                     MAX_EXPANSIONS_FACTOR, MAX_REFERENCES_PER_PAPER, NOVELTY_MIN,
                                     PAPERS_BASE, PAPERS_PER_SEED, PHASE0_QUOTA, PRIORITY_WEIGHTS, RUN_FILES, SEED_ACCEPT_THRESHOLD,
                                     SEED_SIM_K, SEED_SIM_MODE, SIM_THRESHOLD, TOPIC_TITLE_WEIGHT)
from .clients import HttpError, S2Client
from .embed import Embedder, build_embedding_text
from .manifest import (Manifest, MissingInputError, StepResult, atomic_write_json, read_json,
                       run_step)
from .references import REFERENCES_FILE
from .seeds import SEEDS_FILE

DISCOVERY_FILE = RUN_FILES["discovery"]
DISCOVERY_VERSION = 1
SCORE_DECIMALS = 6                      # scores are rounded like this in discovery.json (stable, readable)


# --------------------------------------------------------------------------- errors

class B1Error(RuntimeError):
    """Base class for B1 problems."""


class B1ConfigError(B1Error):
    """The request/budget and the settings cannot work together (stop before any API call)."""


class B1InvariantError(B1Error):
    """A result broke one of the final checks (a bug or corrupt input). Nothing is committed."""


# --------------------------------------------------------------------------- settings and budget

@dataclass(frozen=True)
class B1Settings:
    """Every number B1 uses. Defaults come from config.py; tests can pass their own."""
    seed_sim_mode: str = SEED_SIM_MODE
    seed_sim_k: int = SEED_SIM_K
    topic_title_weight: float = TOPIC_TITLE_WEIGHT
    anchor_topic_weight: float = ANCHOR_TOPIC_WEIGHT
    cite_saturation: float = CITE_SATURATION
    cite_missing_score: float = CITE_MISSING_SCORE
    priority_weights: dict = field(default_factory=lambda: dict(PRIORITY_WEIGHTS))
    sim_threshold: float = SIM_THRESHOLD
    seed_accept_threshold: float = SEED_ACCEPT_THRESHOLD
    high_sim: float = HIGH_SIM
    abs_max_depth: int = ABS_MAX_DEPTH
    max_expansions_factor: float = MAX_EXPANSIONS_FACTOR
    max_expansions: Optional[int] = None      # explicit override (tests, experiments); None = factor * max_papers
    papers_base: int = PAPERS_BASE
    papers_per_seed: int = PAPERS_PER_SEED
    phase0_quota: int = PHASE0_QUOTA
    guarantee_fraction: float = GUARANTEE_FRACTION
    explore_fraction: float = EXPLORE_FRACTION
    explore_min_anchor: float = EXPLORE_MIN_ANCHOR
    novelty_min: float = NOVELTY_MIN
    max_references: int = MAX_REFERENCES_PER_PAPER
    # --- discovery mode ("classic" or "groups") and the settings that only "groups" uses
    mode: str = DISCOVERY_MODE
    group_link_threshold: float = GROUP_LINK_THRESHOLD
    group_accept_threshold: float = GROUP_ACCEPT_THRESHOLD
    group_threshold_mode: str = GROUP_THRESHOLD_MODE
    group_min_threshold: float = GROUP_MIN_THRESHOLD
    group_score_mode: str = GROUP_SCORE_MODE
    group_include_topic: bool = GROUP_INCLUDE_TOPIC
    group_topic_threshold: float = GROUP_TOPIC_THRESHOLD
    group_high_sim: float = GROUP_HIGH_SIM
    group_explore_min_anchor: float = GROUP_EXPLORE_MIN_ANCHOR
    group_self_test: str = GROUP_SELF_TEST

    def __post_init__(self) -> None:
        for name, allowed in (("mode", ("classic", "groups")), ("group_threshold_mode", ("per_group", "global")),
                              ("group_score_mode", ("centroid", "centroid_or_member")),
                              ("group_self_test", ("error", "warn"))):
            if getattr(self, name) not in allowed:
                raise B1ConfigError(f"{name} must be one of {allowed}, got {getattr(self, name)!r}")
        for name in ("group_link_threshold", "group_accept_threshold", "group_min_threshold",
                     "group_topic_threshold", "group_high_sim", "group_explore_min_anchor"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise B1ConfigError(f"{name} must be between 0 and 1, got {getattr(self, name)}")
        if self.group_min_threshold > self.group_accept_threshold:
            raise B1ConfigError("group_min_threshold must not be higher than group_accept_threshold")
        if self.group_high_sim < self.group_accept_threshold:
            raise B1ConfigError("group_high_sim must not be lower than group_accept_threshold")
        if self.group_explore_min_anchor > self.group_accept_threshold:
            raise B1ConfigError("group_explore_min_anchor must not be higher than group_accept_threshold")


def normalise_budget(budget: Any) -> dict:
    """{max_papers, max_depth, fanout_per_node} from None, a dict, or a schemas.Budget-like object.
    Anything missing falls back to config.B1_DEFAULT_BUDGET. Each must be a whole number >= 1, except that
    max_papers may be None: "the user did not choose it", see resolve_max_papers()."""
    if budget is None:
        raw: dict = {}
    elif isinstance(budget, dict):
        raw = budget
    elif hasattr(budget, "model_dump"):
        raw = budget.model_dump()
    else:
        raw = {k: getattr(budget, k) for k in B1_DEFAULT_BUDGET if hasattr(budget, k)}
    out = {}
    for name, default in B1_DEFAULT_BUDGET.items():
        value = raw.get(name, default)
        if name == "max_papers" and value is None:
            out[name] = None
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value or value < 1:
            raise B1ConfigError(f"budget.{name} must be a whole number >= 1, got {value!r}")
        out[name] = int(value)
    return out


def resolve_max_papers(budget: dict, n_seeds: int, st: B1Settings) -> tuple[int, str]:
    """(max_papers, where it came from). The user's number wins; without one it grows with the seeds:
    PAPERS_BASE + PAPERS_PER_SEED * n_seeds (linear on purpose: the later steps cost per paper)."""
    if budget.get("max_papers") is not None:
        return int(budget["max_papers"]), "request"
    return st.papers_base + st.papers_per_seed * n_seeds, "formula"


def resolve_max_expansions(max_papers: int, st: B1Settings) -> int:
    return st.max_expansions if st.max_expansions is not None else int(math.ceil(st.max_expansions_factor * max_papers))


# --------------------------------------------------------------------------- small maths helpers

def clamp01(x: float) -> float:
    return float(min(1.0, max(0.0, float(x))))


def _squash(text: Optional[str]) -> str:
    return " ".join((text or "").split())


def cite_score(paper: dict, as_of_year: int, st: B1Settings) -> tuple[float, bool]:
    """(score 0..1, was_missing). Fixed rule: log of citations per year of age, age at least 1 year."""
    citations, year = paper.get("citation_count"), paper.get("year")
    if citations is None or year is None:
        return float(st.cite_missing_score), True
    age = max(as_of_year - int(year), 1)
    rate = max(float(citations), 0.0) / age
    return clamp01(math.log1p(rate) / math.log1p(st.cite_saturation)), False


def seed_similarity(sims: np.ndarray, centroid_sim: float, mode: str, k: int) -> float:
    """How close a paper is to the seeds, from its similarity to each seed (sims) and to their centroid."""
    if mode == "max":
        return clamp01(np.max(sims))
    if mode == "top_k_mean":
        top = np.sort(sims)[::-1][:max(1, min(k, len(sims)))]
        return clamp01(np.mean(top))
    if mode == "centroid":
        return clamp01(centroid_sim)
    raise ValueError(f"unknown seed similarity mode {mode!r}")


# --------------------------------------------------------------------------- seed groups ("groups" mode)

def cluster_seeds(vectors: np.ndarray, link_threshold: float) -> list[list[int]]:
    """Average-linkage clustering of the seeds. Start with one group per seed, then keep merging the two
    groups whose AVERAGE pairwise cosine is highest, as long as that average is >= link_threshold.
    Plain numpy (no scipy needed): there are only a handful of seeds. Deterministic: on a tie the pair
    with the lowest group positions wins. Returns lists of seed indexes, ordered by first seed."""
    sims = np.asarray(vectors) @ np.asarray(vectors).T
    clusters = [[i] for i in range(len(sims))]
    while len(clusters) > 1:
        best, pair = -2.0, None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                avg = float(np.mean(sims[np.ix_(clusters[a], clusters[b])]))
                if avg > best + 1e-12:
                    best, pair = avg, (a, b)
        if pair is None or best < link_threshold:
            break
        a, b = pair
        clusters[a] = sorted(clusters[a] + clusters[b])
        del clusters[b]
    return sorted(clusters, key=lambda c: c[0])


def build_groups(vectors: np.ndarray, seed_keys: list[str], st: "B1Settings") -> list[dict]:
    """One dict per group: id G1.., member seed keys, centroid (length 1), tightness, threshold.
    tightness = average similarity of the members to their own centroid (a single seed: 1.0)."""
    groups = []
    for n, members in enumerate(cluster_seeds(vectors, st.group_link_threshold), start=1):
        centroid = vectors[members].mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        centroid = centroid / norm if norm > 0 else vectors[members[0]]
        tightness = float(np.mean(vectors[members] @ centroid))
        if st.group_threshold_mode == "per_group":
            threshold = max(st.group_min_threshold, st.group_accept_threshold * tightness)
        else:
            threshold = st.group_accept_threshold
        groups.append({"id": f"G{n}", "members": [seed_keys[i] for i in members], "member_index": members,
                       "centroid": centroid, "tightness": tightness, "threshold": threshold})
    return groups


# --------------------------------------------------------------------------- the algorithm

class Discovery:
    """One run of B1. Use discover(...) unless you need the object (tests do)."""

    def __init__(self, seeds_doc: dict, refs_doc: dict, topic: str, embedder: Embedder, s2: S2Client,
                 budget: Any = None, settings: Optional[B1Settings] = None,
                 as_of_year: Optional[int] = None):
        self.seeds_doc, self.refs_doc, self.topic = seeds_doc, refs_doc, topic
        self.embedder, self.s2 = embedder, s2
        self.budget = normalise_budget(budget)
        self.st = settings or B1Settings()
        self.as_of_year = int(as_of_year) if as_of_year is not None else datetime.date.today().year

        self.records: dict[str, dict] = {}       # every judged paper (accepted or not), key -> record
        self.papers: dict[str, dict] = {}        # key -> the paper dict (S2 shape)
        self.full_vecs: dict[str, np.ndarray] = {}
        self.unscorable: set[str] = set()        # no title and no abstract: cannot be judged
        self.selected: dict[str, dict] = {}      # accepted papers in acceptance order
        self.edges: set[tuple[str, str]] = set() # (citing, cited)
        self.heap: list[tuple[float, int, str]] = []
        self.expansions = 0
        self.warnings: list[str] = []
        self.stop_reason: Optional[str] = None
        self.n_partial_expansions = 0
        # mode-dependent numbers: bar = minimum anchor to be accepted, high_bar = depth exemption,
        # explore_floor = relevance floor of an exploration paper
        self.groups_mode = self.st.mode == "groups"
        self.groups: list[dict] = []
        self.group_of: dict[str, str] = {}       # seed key -> group id (groups mode)
        self.unit_ids: list[str] = []            # who gets guaranteed children: seeds (classic) or groups
        self.accept_bar = self.st.group_accept_threshold if self.groups_mode else self.st.sim_threshold
        self.high_bar = self.st.group_high_sim if self.groups_mode else self.st.high_sim
        self.explore_floor = (self.st.group_explore_min_anchor if self.groups_mode
                              else self.st.explore_min_anchor)

    # ------------------------------------------------------------------ planning (before any API call)
    def _make_plan(self) -> dict:
        st, b = self.st, self.budget
        n = len(self.seed_keys)
        if n < 1:
            raise B1ConfigError("B1 needs at least one resolved seed")
        u = len(self.unit_ids)                       # classic: one unit per seed. groups: one unit per group
        max_papers, max_papers_source = resolve_max_papers(b, n, st)
        max_expansions = resolve_max_expansions(max_papers, st)
        explore_slots = int(round(st.explore_fraction * max_papers))
        normal_cap = max_papers - explore_slots
        if n > normal_cap:
            raise B1ConfigError(f"{n} seeds do not fit into the normal budget of {normal_cap} papers "
                                f"(max_papers={max_papers}, explore slots={explore_slots})")
        requested = int(math.floor(st.guarantee_fraction * max_papers))
        guarantee_budget = max(0, min(requested, normal_cap - n))
        quota = min(st.phase0_quota, b["fanout_per_node"], guarantee_budget // u)
        needed = n + u * quota                       # classic: n*(1+Q). groups: every seed + Q per group
        if needed > max_expansions:
            raise B1ConfigError(f"Phase 0 needs up to {needed} expansions "
                                f"({n} seeds + {u} x {quota} guaranteed) but MAX_EXPANSIONS is {max_expansions}")
        if quota == 0:
            self.warnings.append("no guaranteed children: the budget leaves no room for them "
                                 f"(max_papers={max_papers}, {n} seeds)")
        return {"mode": st.mode, "n_units": u, "n_seeds": n, "max_papers": max_papers, "max_papers_source": max_papers_source,
                "max_expansions": max_expansions, "explore_slots": explore_slots,
                "normal_cap": normal_cap, "guarantee_budget_requested": requested,
                "guarantee_budget": guarantee_budget, "quota": quota,
                "fanout": b["fanout_per_node"], "max_depth": b["max_depth"],
                "soft_depth": min(b["max_depth"], st.abs_max_depth)}

    # ------------------------------------------------------------------ embedding and scoring
    def _embed_papers(self, papers: list[dict]) -> list[Optional[tuple]]:
        """For each paper: (title_vec or None, full_vec, flags), or None if it has no usable text."""
        texts = [build_embedding_text(p) for p in papers]
        usable = [i for i, (text, _) in enumerate(texts) if text is not None]
        out: list[Optional[tuple]] = [None] * len(papers)
        if not usable:
            return out
        fulls = self.embedder.embed_texts([texts[i][0] for i in usable])
        titled = [i for i in usable if _squash(papers[i].get("title"))]
        tvecs = self.embedder.embed_texts([_squash(papers[i]["title"]) for i in titled]) if titled else []
        title_of = {i: v for i, v in zip(titled, tvecs)}
        for row, i in enumerate(usable):
            out[i] = (title_of.get(i), fulls[row], list(texts[i][1]))
        return out

    def _make_record(self, paper: dict, title_vec: Optional[np.ndarray], full_vec: np.ndarray,
                     flags: list[str], seed_index: Optional[int] = None) -> dict:
        st = self.st
        sim_title = clamp01(title_vec @ self.topic_vec) if title_vec is not None else None
        sim_full = clamp01(full_vec @ self.topic_vec)
        topic_sim = (st.topic_title_weight * sim_title + (1 - st.topic_title_weight) * sim_full
                     if sim_title is not None else sim_full)
        sims = self.seed_full @ full_vec
        if seed_index is None:                       # a normal paper: all seeds count
            seed_sim = seed_similarity(sims, float(full_vec @ self.seed_centroid), st.seed_sim_mode, st.seed_sim_k)
        elif len(sims) == 1:                         # the only seed: nothing to compare with
            seed_sim = 1.0
        else:                                        # a seed: compare with the OTHER seeds (report only)
            others = np.delete(sims, seed_index)
            seed_sim = seed_similarity(others, float(full_vec @ self.seed_centroid), st.seed_sim_mode, st.seed_sim_k)
        group_info = None
        if self.groups_mode:                         # the group score replaces the topic/seed blend
            group_info = self._group_scores(full_vec, sims, topic_sim)
            anchor = group_info["anchor"]
        else:
            anchor = st.anchor_topic_weight * topic_sim + (1 - st.anchor_topic_weight) * seed_sim
        cite, cite_missing = cite_score(paper, self.as_of_year, st)
        flags = list(flags) + (["cite_missing"] if cite_missing else [])
        self.papers[paper["key"]] = paper
        self.full_vecs[paper["key"]] = full_vec
        rec = {"key": paper["key"], "flags": flags, "sim_topic_title": sim_title, "sim_topic_full": sim_full,
                "topic_sim": topic_sim,
                "sim_to_seeds": {k: clamp01(s) for k, s in zip(self.seed_keys, sims)},
                "seed_sim": seed_sim, "anchor": anchor, "cite_score": cite,
                "parent_best": 0.0, "priority": 0.0, "depth": None, "cited_by": set(),
                "accepted_by": None, "guaranteed_for": None, "expanded": False,
                "novelty": None, "bridge": None, "reached_from": [], "primary_origin": None}
        if group_info is not None:
            rec.update(best_group=group_info["best_group"], sim_to_groups=group_info["sim_to_groups"],
                       group_scores=group_info["group_scores"])
        return rec

    def _group_scores(self, full_vec: np.ndarray, seed_sims: np.ndarray, topic_sim: float) -> dict:
        """groups mode. One score per group (and one for the topic): similarity - bar of that group + the common bar.
        A score >= the common bar means "passes that group's own bar", so max over groups is the paper's anchor."""
        st, bar = self.st, self.accept_bar
        sim_to_groups: dict[str, float] = {}
        scores: dict[str, float] = {}
        for g, raw in zip(self.groups, self.group_matrix @ full_vec):
            sim = clamp01(raw)
            sim_to_groups[g["id"]] = sim
            effective = sim
            if st.group_score_mode == "centroid_or_member":
                effective = max(sim, clamp01(np.max(seed_sims[g["member_index"]])))
            scores[g["id"]] = effective - g["threshold"] + bar
        if st.group_include_topic:
            scores["topic"] = topic_sim - st.group_topic_threshold + bar
        best = max(scores, key=scores.get)           # first maximum in insertion order = deterministic
        return {"anchor": clamp01(scores[best]), "best_group": best, "sim_to_groups": sim_to_groups,
                "group_scores": scores}

    def _prepare(self, papers: list[dict]) -> None:
        """Score every paper we have not seen yet (one batch, so the encoder is used efficiently)."""
        new, seen = [], set()
        for p in papers:
            k = p["key"]
            if k in self.records or k in self.unscorable or k in seen:
                continue
            seen.add(k)
            new.append(p)
        if not new:
            return
        for p, emb in zip(new, self._embed_papers(new)):
            if emb is None:
                self.unscorable.add(p["key"])
                continue
            self.records[p["key"]] = self._make_record(p, *emb)

    # ------------------------------------------------------------------ the rules
    def _can_expand(self, rec: dict) -> bool:
        return (rec["depth"] < self.plan["soft_depth"]
                or (rec["anchor"] >= self.high_bar and rec["depth"] < self.st.abs_max_depth))

    def _door(self, rec: dict) -> Optional[str]:
        if self.groups_mode:                         # one door: close enough to some group (or to the topic)
            return "group" if rec["anchor"] >= self.accept_bar - 1e-9 else None
        if rec["anchor"] >= self.st.sim_threshold:
            return "anchor"
        if rec["seed_sim"] >= self.st.seed_accept_threshold:
            return "seed_sim"
        return None

    def _push(self, rec: dict) -> None:
        heapq.heappush(self.heap, (-rec["priority"], rec["depth"], rec["key"]))

    def _pop(self) -> Optional[dict]:
        while self.heap:
            _, _, key = heapq.heappop(self.heap)
            rec = self.records[key]
            if not rec["expanded"]:                 # a stale duplicate entry is simply skipped
                return rec
        return None

    def _register(self, parent: dict, paper: dict) -> Optional[dict]:
        """One reference of an expanded paper. Handles "found for the first time" AND "found again":
        the link is a set, every citer is kept, the shortest depth wins, the priority only goes up."""
        key = paper["key"]
        if key == parent["key"]:
            return None
        self.edges.add((parent["key"], key))
        rec = self.records.get(key)
        if rec is None:                              # unscorable
            return None
        rec["cited_by"].add(parent["key"])
        if rec["accepted_by"] == "seed":             # a seed cited by another paper: link only
            return None
        old_depth, old_priority = rec["depth"], rec["priority"]
        rec["depth"] = parent["depth"] + 1 if old_depth is None else min(old_depth, parent["depth"] + 1)
        rec["parent_best"] = max(rec["parent_best"], parent["priority"])
        pw = self.st.priority_weights
        rec["priority"] = max(old_priority, pw["anchor"] * rec["anchor"] + pw["cite"] * rec["cite_score"]
                              + pw["parent"] * rec["parent_best"])
        if (rec["accepted_by"] is not None and not rec["expanded"]
                and (rec["priority"] > old_priority or rec["depth"] != old_depth) and self._can_expand(rec)):
            self._push(rec)                          # found again by a better route: queue it again
        return rec

    def _register_batch(self, parent: dict, papers: list[dict]) -> list[dict]:
        self._prepare(papers)
        out: dict[str, dict] = {}
        for p in papers:
            rec = self._register(parent, p)
            if rec is not None:
                out[rec["key"]] = rec
        return list(out.values())

    def _accept(self, rec: dict, how: str, cap: int, guaranteed_for: Optional[str] = None) -> bool:
        if rec["accepted_by"] is not None or len(self.selected) >= cap:
            return False
        rec["accepted_by"], rec["guaranteed_for"] = how, guaranteed_for
        self.selected[rec["key"]] = rec
        if how != "explore" and not rec["expanded"] and self._can_expand(rec):
            self._push(rec)
        return True

    def _accept_best(self, batch: list[dict], cap: int) -> int:
        """From one expansion: accept up to `fanout` papers that pass a door, most relevant (anchor) first.
        Citations are NOT used here: they only order the expansion queue (priority)."""
        ranked = sorted((r for r in batch if r["accepted_by"] is None and self._door(r)),
                        key=lambda r: (-r["anchor"], r["key"]))
        taken = 0
        for rec in ranked:
            if taken >= self.plan["fanout"] or len(self.selected) >= cap:
                break
            if self._accept(rec, self._door(rec), cap):
                taken += 1
        return taken

    def _expand(self, rec: dict) -> Optional[list[dict]]:
        """Fetch the references of one accepted paper. None if it cannot be (or the call failed)."""
        if rec["expanded"]:
            return None
        paper = self.papers[rec["key"]]
        rec["expanded"] = True                       # whatever happens, never try this paper twice
        if not paper.get("s2_id"):
            self.warnings.append(f"{rec['key']}: no Semantic Scholar id, references cannot be fetched")
            return None
        try:
            got = self.s2.get_references(paper["s2_id"], self.st.max_references)
        except (HttpError, requests.RequestException) as exc:
            rec["expand_error"] = f"{type(exc).__name__}: {exc}"
            self.warnings.append(f"{rec['key']}: reference fetch failed ({exc})")
            return None
        self.expansions += 1
        if got.get("partial"):
            self.n_partial_expansions += 1
        return list(got["references"])

    # ------------------------------------------------------------------ setup
    def _setup(self) -> None:
        seeds = [r["paper"] for r in self.seeds_doc.get("seeds", [])]
        self.seed_papers = seeds
        self.seed_keys = [p["key"] for p in seeds]
        if len(set(self.seed_keys)) != len(self.seed_keys):
            raise B1ConfigError("seeds.json holds two seeds with the same key")
        if not self.seed_keys:
            raise B1ConfigError("B1 needs at least one resolved seed")

        self.topic_vec = self.embedder.embed_topic(self.topic)
        embedded = self._embed_papers(seeds)
        for p, emb in zip(seeds, embedded):
            if emb is None:
                raise B1ConfigError(f"seed {p.get('key')!r} has no title and no abstract")
        self.seed_full = np.array([emb[1] for emb in embedded])
        if self.seed_full.shape[1] != self.topic_vec.shape[0]:
            raise B1ConfigError("topic vector and paper vectors have different sizes")
        centroid = self.seed_full.mean(axis=0)
        norm = np.linalg.norm(centroid)
        self.seed_centroid = centroid / norm if norm > 0 else self.seed_full[0]
        self._setup_units()                          # groups (groups mode), then the budget plan
        self.plan = self._make_plan()

        for i, (p, emb) in enumerate(zip(seeds, embedded)):
            rec = self._make_record(p, *emb, seed_index=i)
            rec.update(accepted_by="seed", depth=0, priority=1.0, parent_best=1.0)
            self.records[p["key"]] = rec
            self.selected[p["key"]] = rec            # seeds are always accepted and never removed

    def _setup_units(self) -> None:
        """classic: units are the seeds. groups: cluster the seeds, make the centroids, run the self-test."""
        if not self.groups_mode:
            self.unit_ids = list(self.seed_keys)
            return
        self.groups = build_groups(self.seed_full, self.seed_keys, self.st)
        self.group_matrix = np.array([g["centroid"] for g in self.groups])
        self.group_of = {k: g["id"] for g in self.groups for k in g["members"]}
        self.group_by_id = {g["id"]: g for g in self.groups}
        self.unit_ids = [g["id"] for g in self.groups]
        self._self_test()

    def _self_test(self) -> None:
        """Every seed must pass the bar of its OWN group (centroid similarity only). If not, the group is too loose."""
        bad = []
        for g in self.groups:
            for i, key in zip(g["member_index"], g["members"]):
                sim = float(self.seed_full[i] @ g["centroid"])
                if sim < g["threshold"] - 1e-9:
                    bad.append(f"seed {key!r} scores {sim:.2f} against group {g['id']} (needs {g['threshold']:.2f})")
        if not bad:
            return
        msg = ("grouping self-test failed: " + "; ".join(bad[:5])
               + ". The groups are too loose: raise GROUP_LINK_THRESHOLD (or lower GROUP_ACCEPT_THRESHOLD).")
        if self.st.group_self_test == "error":
            raise B1ConfigError(msg)
        self.warnings.append(msg)

    def _seed_reference_papers(self) -> dict[str, list[dict]]:
        """Each seed's references, taken from references.json (fetched by B0: no new API call)."""
        by_seed = self.refs_doc.get("by_seed", {})
        stubs = self.refs_doc.get("stubs", {})
        targets: dict[str, list[str]] = {}
        for e in self.refs_doc.get("edges", []):
            targets.setdefault(e["source"], []).append(e["target"])
        out: dict[str, list[dict]] = {}
        for key in self.seed_keys:
            if by_seed.get(key, {}).get("status") != "ok":
                self.warnings.append(f"seed {key!r}: no reference list in references.json "
                                     f"({by_seed.get(key, {}).get('status', 'missing')}), it cannot be expanded")
                continue
            papers = []
            for t in targets.get(key, []):
                if t in self.records and self.records[t]["accepted_by"] == "seed":
                    self.edges.add((key, t))         # one seed cites another seed
                    self.records[t]["cited_by"].add(key)
                elif t in stubs:
                    papers.append(stubs[t])
            out[key] = papers
        return out

    # ------------------------------------------------------------------ phase 0
    def _phase0(self) -> None:
        plan, cap = self.plan, self.plan["normal_cap"]
        seed_refs = self._seed_reference_papers()
        self._prepare([p for papers in seed_refs.values() for p in papers])   # one big batch
        child_recs: dict[str, list[dict]] = {}
        for key in self.seed_keys:                   # fixed order = order of the seed list
            if key in seed_refs:
                child_recs[key] = self._register_batch(self.records[key], seed_refs[key])

        # (a) guaranteed children per UNIT (classic: every seed; groups: every group of seeds), FIRST for all
        #     units, so "others" can never starve them. A group ranks the references of ALL its seeds by
        #     similarity to the group centroid.
        guaranteed: list[dict] = []
        for key in self.unit_ids:
            if self.groups_mode:
                pool: dict[str, dict] = {}
                for member in self.group_by_id[key]["members"]:
                    for r in child_recs.get(member, []):
                        pool[r["key"]] = r
                candidates = list(pool.values())
                ranked = sorted(candidates, key=lambda r: (-r["sim_to_groups"][key], r["key"]))
            else:
                ranked = sorted(child_recs.get(key, []), key=lambda r: (-r["sim_to_seeds"][key], r["key"]))
            taken = 0
            for rec in ranked:
                if taken >= plan["quota"]:
                    break
                if rec["accepted_by"] is not None:   # already a seed, or guaranteed for an earlier seed
                    continue
                if self._accept(rec, "guaranteed", cap, guaranteed_for=key):
                    guaranteed.append(rec)
                    taken += 1
        # (b) the seeds' other references go through the doors
        for key in self.seed_keys:
            self._accept_best(child_recs.get(key, []), cap)
        # (c) expand the seeds (their lists are already known) and then every guaranteed child
        for key in self.seed_keys:
            if key in seed_refs:
                self.records[key]["expanded"] = True
                self.expansions += 1
        for rec in guaranteed:
            refs = self._expand(rec)
            if refs is not None:
                self._accept_best(self._register_batch(rec, refs), cap)

    # ------------------------------------------------------------------ phase 1
    def _phase1(self) -> None:
        cap = self.plan["normal_cap"]
        while True:
            if self.expansions >= self.plan["max_expansions"]:
                self.stop_reason = "max_expansions"
                return
            if len(self.selected) >= cap:
                self.stop_reason = "max_papers"
                return
            rec = self._pop()
            if rec is None:
                self.stop_reason = "frontier_empty"
                return
            refs = self._expand(rec)
            if refs is None:
                continue
            self._accept_best(self._register_batch(rec, refs), cap)

    # ------------------------------------------------------------------ reach (which seeds lead to which paper)
    def _reach(self, keys: set[str]) -> dict[str, dict[str, int]]:
        """{seed: {paper: hops}} following citation links (citing -> cited) between accepted papers."""
        adj: dict[str, list[str]] = {}
        for s, t in sorted(self.edges):
            if s in keys and t in keys:
                adj.setdefault(s, []).append(t)
        out: dict[str, dict[str, int]] = {}
        for seed in self.seed_keys:
            dist = {seed: 0}
            queue = deque([seed])
            while queue:
                cur = queue.popleft()
                for nxt in adj.get(cur, []):
                    if nxt not in dist:
                        dist[nxt] = dist[cur] + 1
                        queue.append(nxt)
            out[seed] = dist
        return out

    # ------------------------------------------------------------------ phase 2
    def _phase2(self) -> None:
        slots = self.plan["explore_slots"]
        if slots <= 0:
            return
        reach = self._reach(set(self.selected))
        reached_by = {k: {s for s in self.seed_keys if k in reach[s]} for k in self.selected}
        pool = [r for r in self.records.values()
                if r["accepted_by"] is None and r["anchor"] >= self.explore_floor]
        if not pool:
            return
        sel_matrix = np.array([self.full_vecs[k] for k in self.selected])
        kept = []
        for rec in pool:
            rec["novelty"] = clamp01(1.0 - float(np.max(sel_matrix @ self.full_vecs[rec["key"]])))
            if rec["novelty"] < self.st.novelty_min:
                continue
            seeds_behind = set()                      # seeds that lead to the papers citing this one
            for citer in rec["cited_by"]:
                seeds_behind |= reached_by.get(citer, set())
            rec["bridge"] = 1 if len(seeds_behind) >= 2 else 0
            kept.append(rec)
        kept.sort(key=lambda r: (-r["bridge"], -r["novelty"], -r["anchor"], r["key"]))
        for rec in kept[:slots]:
            self._accept(rec, "explore", self.plan["max_papers"])

    # ------------------------------------------------------------------ finish
    def _finish(self) -> dict:
        plan = self.plan
        keys = set(self.selected)
        final_edges = sorted((s, t) for s, t in self.edges if s in keys and t in keys)
        reach = self._reach(keys)
        for key, rec in self.selected.items():
            seeds_here = [s for s in self.seed_keys if key in reach[s]]
            rec["reached_from"] = seeds_here
            if seeds_here:
                rec["primary_origin"] = min(
                    seeds_here, key=lambda s: (reach[s][key], -rec["sim_to_seeds"][s], s))
        self._check(final_edges, reach)

        seed_set = set(self.seed_keys)
        stubs = {}
        for key, rec in self.selected.items():
            if rec["accepted_by"] == "seed":
                continue
            paper = dict(self.papers[key])
            paper["is_seed"] = False
            paper["cited_by_seeds"] = sorted(c for c in rec["cited_by"] if c in seed_set)
            stubs[key] = paper

        accepted_by: dict[str, int] = {}
        for rec in self.selected.values():
            accepted_by[rec["accepted_by"]] = accepted_by.get(rec["accepted_by"], 0) + 1
        coverage = {}
        for s in self.seed_keys:
            mine = [r for r in self.selected.values() if s in r["reached_from"]]
            unit = self.group_of[s] if self.groups_mode else s     # groups mode: the seed's group holds the guarantee
            coverage[s] = {"reached": len(mine),
                           "unique": sum(1 for r in mine if r["reached_from"] == [s]),
                           "shared": sum(1 for r in mine if len(r["reached_from"]) > 1),
                           "primary": sum(1 for r in self.selected.values() if r["primary_origin"] == s),
                           "guaranteed": sum(1 for r in self.selected.values() if r["guaranteed_for"] == unit)}
        group_report = [{"id": g["id"], "members": g["members"], "size": len(g["members"]),
                         "tightness": round(g["tightness"], SCORE_DECIMALS),
                         "threshold": round(g["threshold"], SCORE_DECIMALS),
                         "guaranteed": sum(1 for r in self.selected.values() if r["guaranteed_for"] == g["id"]),
                         "best_match_for": sum(1 for r in self.selected.values()
                                               if r["accepted_by"] != "seed" and r.get("best_group") == g["id"])}
                        for g in self.groups]
        if self.groups_mode and self.st.group_include_topic:
            group_report.append({"id": "topic", "members": [], "size": 0, "tightness": None,
                                 "threshold": self.st.group_topic_threshold, "guaranteed": 0,
                                 "best_match_for": sum(1 for r in self.selected.values()
                                                       if r["accepted_by"] != "seed" and r.get("best_group") == "topic")})
        pairwise = (self.seed_full @ self.seed_full.T).tolist()
        meta = {"mode": self.st.mode, "groups": group_report, "stop_reason": self.stop_reason,
                "as_of_year": self.as_of_year, "plan": plan,
                "expansions": self.expansions, "n_partial_expansions": self.n_partial_expansions,
                "n_judged": len(self.records), "n_accepted": len(self.selected),
                "n_rejected": len(self.records) - len(self.selected),
                "n_unscorable": len(self.unscorable), "accepted_by": accepted_by,
                "seed_coverage": coverage,
                "seed_pairwise_similarity": {a: {b: round(pairwise[i][j], SCORE_DECIMALS)
                                                 for j, b in enumerate(self.seed_keys)}
                                             for i, a in enumerate(self.seed_keys)},
                "settings": asdict(self.st)}
        return {"version": DISCOVERY_VERSION, "stubs": stubs,
                "edges": [{"source": s, "target": t, "kind": "cites"} for s, t in final_edges],
                "scores": {k: self._public(self.records[k]) for k in sorted(self.records)},
                "meta": meta, "warnings": list(self.warnings)}

    def _check(self, final_edges: list[tuple[str, str]], reach: dict) -> None:
        problems = []
        if len(self.selected) > self.plan["max_papers"]:
            problems.append(f"{len(self.selected)} papers accepted, more than max_papers={self.plan['max_papers']}")
        by_seed = self.refs_doc.get("by_seed", {})
        for key in self.seed_keys:
            if by_seed.get(key, {}).get("status") == "ok" and not self.records[key]["expanded"]:
                problems.append(f"seed {key!r} was not expanded")
        guaranteed = [r for r in self.selected.values() if r["guaranteed_for"] is not None]
        if len({r["key"] for r in guaranteed}) != len(guaranteed):
            problems.append("a guaranteed paper is counted twice")
        for key in self.unit_ids:
            n = sum(1 for r in guaranteed if r["guaranteed_for"] == key)
            if n > self.plan["quota"]:
                problems.append(f"{key!r} has {n} guaranteed papers, more than Q={self.plan['quota']}")
        if len(guaranteed) > self.plan["guarantee_budget"]:
            problems.append("guaranteed papers exceed the guarantee budget")
        for key, rec in self.selected.items():
            if not any(key in reach[s] for s in self.seed_keys):
                problems.append(f"accepted paper {key!r} is not reachable from any seed")
        if problems:
            raise B1InvariantError("B1 final checks failed: " + "; ".join(problems[:5]))

    @staticmethod
    def _public(rec: dict) -> dict:
        """The record as it is written into discovery.json."""
        def r(x):
            return None if x is None else round(float(x), SCORE_DECIMALS)
        return {"accepted": rec["accepted_by"] is not None, "accepted_by": rec["accepted_by"],
                "guaranteed_for": rec["guaranteed_for"], "depth": rec["depth"], "expanded": rec["expanded"],
                "flags": rec["flags"], "topic_sim": r(rec["topic_sim"]),
                "sim_topic_title": r(rec["sim_topic_title"]), "sim_topic_full": r(rec["sim_topic_full"]),
                "seed_sim": r(rec["seed_sim"]), "sim_to_seeds": {k: r(v) for k, v in rec["sim_to_seeds"].items()},
                "anchor": r(rec["anchor"]), "cite_score": r(rec["cite_score"]),
                "parent_best": r(rec["parent_best"]), "priority": r(rec["priority"]),
                "cited_by": sorted(rec["cited_by"]), "reached_from": rec["reached_from"],
                "primary_origin": rec["primary_origin"], "novelty": r(rec["novelty"]), "bridge": rec["bridge"],
                **({"expand_error": rec["expand_error"]} if "expand_error" in rec else {}),
                **({"best_group": rec["best_group"],
                    "sim_to_groups": {k: r(v) for k, v in rec["sim_to_groups"].items()},
                    "group_scores": {k: r(v) for k, v in rec["group_scores"].items()}}
                   if "best_group" in rec else {})}

    # ------------------------------------------------------------------ run
    def run(self) -> dict:
        self._setup()
        self._phase0()
        self._phase1()
        if self.stop_reason is None:                 # only possible if Phase 0 left nothing to do
            self.stop_reason = "frontier_empty"
        self._phase2()
        return self._finish()


def discover(seeds_doc: dict, refs_doc: dict, topic: str, embedder: Embedder, s2: S2Client,
             budget: Any = None, settings: Optional[B1Settings] = None,
             as_of_year: Optional[int] = None) -> dict:
    """Run B1 on in-memory documents and return the discovery.json document."""
    return Discovery(seeds_doc, refs_doc, topic, embedder, s2, budget, settings, as_of_year).run()


# --------------------------------------------------------------------------- offline replay

def replay_doors(doc: dict, *, seed_sim_mode: str = "max", seed_sim_k: int = SEED_SIM_K,
                 anchor_topic_weight: float = ANCHOR_TOPIC_WEIGHT, sim_threshold: float = SIM_THRESHOLD,
                 seed_accept_threshold: float = SEED_ACCEPT_THRESHOLD) -> dict:
    """What would the two relevance doors let in under other settings? Uses only the numbers stored in
    discovery.json (no API call, no encoder). Modes: "max" and "top_k_mean" (the "centroid" mode needs the
    vectors, so it is not available here). Guaranteed and exploration papers are not door decisions and
    are listed separately."""
    if seed_sim_mode not in ("max", "top_k_mean"):
        raise ValueError("replay_doors supports seed_sim_mode 'max' and 'top_k_mean' only")
    would_pass, by_door = [], {"anchor": 0, "seed_sim": 0}
    for key, sc in doc["scores"].items():
        if sc["accepted_by"] == "seed" or not sc["sim_to_seeds"]:
            continue
        sims = np.array(list(sc["sim_to_seeds"].values()), dtype=float)
        seed_sim = seed_similarity(sims, 0.0, seed_sim_mode, seed_sim_k)
        anchor = anchor_topic_weight * sc["topic_sim"] + (1 - anchor_topic_weight) * seed_sim
        if anchor >= sim_threshold:
            would_pass.append(key)
            by_door["anchor"] += 1
        elif seed_sim >= seed_accept_threshold:
            would_pass.append(key)
            by_door["seed_sim"] += 1
    special = sorted(k for k, sc in doc["scores"].items() if sc["accepted_by"] in ("guaranteed", "explore"))
    return {"would_pass": sorted(would_pass), "n_would_pass": len(would_pass), "by_door": by_door,
            "guaranteed_or_explore": special}


def replay_group_doors(doc: dict, *, accept_threshold: Optional[float] = None,
                       topic_threshold: Optional[float] = None, threshold_mode: Optional[str] = None,
                       min_threshold: Optional[float] = None, include_topic: Optional[bool] = None) -> dict:
    """groups mode: what would the group door let in under other settings? Uses only discovery.json (no API call,
    no encoder). Not replayable here: GROUP_LINK_THRESHOLD (the groups themselves) and GROUP_SCORE_MODE
    "centroid_or_member" (this replay uses the centroid similarity). Guaranteed and exploration papers are listed
    separately because they are not door decisions."""
    meta = doc["meta"]
    if meta.get("mode") != "groups":
        raise ValueError("this discovery.json was made in 'classic' mode; use replay_doors instead")
    s = meta["settings"]
    accept = s["group_accept_threshold"] if accept_threshold is None else accept_threshold
    topic_thr = s["group_topic_threshold"] if topic_threshold is None else topic_threshold
    mode = s["group_threshold_mode"] if threshold_mode is None else threshold_mode
    floor = s["group_min_threshold"] if min_threshold is None else min_threshold
    use_topic = s["group_include_topic"] if include_topic is None else include_topic
    if mode not in ("per_group", "global"):
        raise ValueError("threshold_mode must be 'per_group' or 'global'")
    thresholds = {g["id"]: (max(floor, accept * g["tightness"]) if mode == "per_group" else accept)
                  for g in meta["groups"] if g["id"] != "topic"}
    would_pass, by_group = [], {}
    for key, sc in doc["scores"].items():
        if sc["accepted_by"] == "seed" or "sim_to_groups" not in sc:
            continue
        hits = [gid for gid, sim in sc["sim_to_groups"].items() if sim >= thresholds[gid]]
        if use_topic and sc["topic_sim"] >= topic_thr:
            hits.append("topic")
        if hits:
            would_pass.append(key)
            by_group[hits[0]] = by_group.get(hits[0], 0) + 1
    special = sorted(k for k, sc in doc["scores"].items() if sc["accepted_by"] in ("guaranteed", "explore"))
    return {"would_pass": sorted(would_pass), "n_would_pass": len(would_pass), "by_first_matching_group": by_group,
            "thresholds": {k: round(v, 4) for k, v in thresholds.items()}, "guaranteed_or_explore": special}


# --------------------------------------------------------------------------- the step

def validate_discovery_output(staging: Path, seed_keys: set[str], max_papers: int) -> list[str]:
    doc = read_json(staging / DISCOVERY_FILE)
    if not isinstance(doc, dict):
        return [f"{DISCOVERY_FILE} is missing or not valid JSON"]
    stubs = doc.get("stubs", {})
    known = seed_keys | set(stubs)
    problems = []
    if len(known) > max_papers:
        problems.append(f"{len(known)} accepted papers, more than max_papers={max_papers}")
    if seed_keys & set(stubs):
        problems.append("a seed also appears as a stub")
    seen = set()
    for e in doc.get("edges", []):
        pair = (e["source"], e["target"])
        if pair in seen:
            problems.append(f"duplicate edge {pair}")
        seen.add(pair)
        if e["source"] not in known or e["target"] not in known:
            problems.append(f"edge {pair} joins a paper that was not accepted")
        if e["source"] == e["target"]:
            problems.append(f"self edge on {e['source']!r}")
    scores = doc.get("scores", {})
    for key in known:
        if key not in scores or not scores[key].get("accepted"):
            problems.append(f"accepted paper {key!r} has no accepted score entry")
    return problems


def run_b1_step(manifest: Manifest, topic: str, s2: S2Client, embedder: Embedder, budget: Any = None, *,
                settings: Optional[B1Settings] = None, as_of_year: Optional[int] = None,
                cap: int = MAX_REFERENCES_PER_PAPER) -> StepResult:
    """Run b1_discovery under the manifest. Reads seeds.json and references.json."""
    root = manifest.root
    seeds_doc, refs_doc = read_json(root / SEEDS_FILE), read_json(root / REFERENCES_FILE)
    if seeds_doc is None or refs_doc is None:
        raise MissingInputError(f"{SEEDS_FILE} and {REFERENCES_FILE} are needed: run steps b0_seeds and b0_references first")
    budget = normalise_budget(budget)
    year = int(as_of_year) if as_of_year is not None else datetime.date.today().year
    st = settings or B1Settings()
    st = B1Settings(**{**asdict(st), "max_references": cap})
    seed_keys = {rec["paper"]["key"] for rec in seeds_doc.get("seeds", [])}
    max_papers, _ = resolve_max_papers(budget, len(seed_keys), st)

    def work(staging: Path) -> dict:
        doc = discover(seeds_doc, refs_doc, topic, embedder, s2, budget, st, year)
        atomic_write_json(staging / DISCOVERY_FILE, doc)
        m = doc["meta"]
        return {"n_accepted": m["n_accepted"], "n_judged": m["n_judged"], "n_rejected": m["n_rejected"],
                "expansions": m["expansions"], "stop_reason": m["stop_reason"],
                "accepted_by": m["accepted_by"], "warnings": doc["warnings"]}

    return run_step(manifest, "b1_discovery", work, outputs=[DISCOVERY_FILE],
                    inputs=[SEEDS_FILE, REFERENCES_FILE],
                    extra_inputs={"topic": _squash(topic), "budget": budget, "as_of_year": year,
                                  "settings": asdict(st)},
                    validate=lambda staging: validate_discovery_output(staging, seed_keys, max_papers))
