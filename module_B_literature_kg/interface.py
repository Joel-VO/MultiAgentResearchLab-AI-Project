"""
Module B -> Module C interface (PROPOSED CONTRACT, agreed choices from the design discussion).

What lives where
----------------
* KBHandle      : tiny, paths-only object. Safe to keep inside LabState (it gets saved to SQLite).
* open_kb(h)    : turns a KBHandle into a KnowledgeBase object with query methods (loads files).
* KnowledgeBase : query methods + raw access (.graph, .embeddings, .ids, .topic_vector)
                  so Module C can write any extra query it needs.

Conventions (Module C can rely on these)
----------------------------------------
IDs        : every paper has a short internal ID "P1", "P2", ... (stable once assigned).
             PaperMeta holds the real IDs (s2_id, arxiv_id, doi). kb.resolve_id() maps back.
Missing    : list fields use [] when the paper does not say anything ("not stated").
             idea / method are plain strings and use "not stated" (a list can't be a string).
Embeddings : row i of kb.embeddings belongs to paper kb.ids[i]. Rows are L2-normalised,
             so dot product == cosine similarity.
Graph      : kb.graph is a networkx.MultiDiGraph.
             Paper nodes   : node id = "P1"..., attrs: type="paper"
             Concept nodes : node id = "<type>:<normalised name>", e.g. "dataset:truthfulqa",
                             attrs: type in {"concept","method","dataset","metric"}, name="TruthfulQA"
             Edges carry attr kind:
                 "cites"        paper -> paper   (citing -> cited)
                 "similar"      paper <-> paper  (stored in BOTH directions), attr weight = cosine
                 "has_concept" / "has_method" / "has_dataset" / "has_metric"
                                paper -> concept-layer node
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Literal, Optional

import networkx as nx
import numpy as np
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- types


class KBHandle(BaseModel):
    """Lives in LabState.kb. Paths only."""
    graph_path: str        # node-link JSON of the MultiDiGraph
    emb_path: str          # .npy, shape (n_papers, dim), L2-normalised
    ids_path: str          # ids.json: ["P1", "P2", ...] in embedding-row order
    meta_path: str         # papers.json: {paper_id: PaperMeta}
    reviews_path: str      # reviews.json: {paper_id: PaperReview}
    topic_emb_path: str    # .npy, shape (dim,), the topic query vector


class PaperMeta(BaseModel):
    paper_id: str                      # internal "P1"
    title: str
    abstract: str = ""
    year: Optional[int] = None
    venue: Optional[str] = None
    citation_count: Optional[int] = None
    s2_id: Optional[str] = None        # Semantic Scholar paperId
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    is_seed: bool = False
    has_full_text: bool = False


class PaperReview(BaseModel):
    paper_id: str
    title: str = ""
    idea: str = "not stated"
    method: str = "not stated"
    pros: list[str] = Field(default_factory=list)          # [] == not stated
    cons: list[str] = Field(default_factory=list)
    future_work: list[str] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    # field name -> exact quote from the paper that supports it (from the quote check)
    evidence_quotes: dict[str, str] = Field(default_factory=dict)


class Conflict(BaseModel):
    id: str
    type: Literal["finding", "reviewer"]
    paper_ids: list[str]               # 2 papers for "finding", 1 for "reviewer"
    summary: str
    evidence_text: str
    # extras, filled for type == "finding" (None otherwise)
    shared_dataset: Optional[str] = None
    shared_metric: Optional[str] = None
    claim_a: Optional[str] = None
    claim_b: Optional[str] = None
    score_variance: Optional[float] = None  # reviewer conflicts only


class ScoredPaper(BaseModel):
    paper_id: str
    score: float                       # cosine similarity


class Neighbor(BaseModel):
    paper_id: str
    kind: Literal["cites", "cited_by", "similar"]
    weight: Optional[float] = None     # cosine for "similar", None otherwise


# --------------------------------------------------------------------------- loading


def _load_graph(path: str) -> nx.MultiDiGraph:
    data = json.loads(Path(path).read_text())
    return nx.node_link_graph(data, directed=True, multigraph=True, edges="edges")


def save_graph(g: nx.MultiDiGraph, path: str) -> None:
    Path(path).write_text(json.dumps(nx.node_link_data(g, edges="edges")))


def open_kb(handle: KBHandle,
            embed_fn: Optional[Callable[[str], np.ndarray]] = None) -> "KnowledgeBase":
    """embed_fn: text -> vector, using the SAME embedding model as the KB.
    Only needed for closest_to_text(). Module A provides the real one."""
    return KnowledgeBase(handle, embed_fn)


# --------------------------------------------------------------------------- the KB


class KnowledgeBase:
    def __init__(self, handle: KBHandle, embed_fn=None):
        self.handle = handle
        self._embed_fn = embed_fn
        self.graph: nx.MultiDiGraph = _load_graph(handle.graph_path)
        self.embeddings: np.ndarray = np.load(handle.emb_path)
        self.ids: list[str] = json.loads(Path(handle.ids_path).read_text())
        self.topic_vector: np.ndarray = np.load(handle.topic_emb_path)
        self._row = {pid: i for i, pid in enumerate(self.ids)}
        meta = json.loads(Path(handle.meta_path).read_text())
        self._meta = {k: PaperMeta(**v) for k, v in meta.items()}
        rev = json.loads(Path(handle.reviews_path).read_text())
        self._reviews = {k: PaperReview(**v) for k, v in rev.items()}
        self._external = {}
        for m in self._meta.values():
            for ext in (m.s2_id, m.arxiv_id, m.doi):
                if ext:
                    self._external[ext.lower()] = m.paper_id

    # ---- lookups
    def get_paper(self, paper_id: str) -> PaperMeta:
        return self._meta[paper_id]

    def get_review(self, paper_id: str) -> Optional[PaperReview]:
        return self._reviews.get(paper_id)

    def resolve_id(self, external_id: str) -> Optional[str]:
        """S2 id / arXiv id / DOI -> internal 'P#' (None if not in the KB)."""
        return self._external.get(external_id.lower())

    def get_embeddings(self, paper_ids: list[str]) -> np.ndarray:
        return self.embeddings[[self._row[p] for p in paper_ids]]

    # ---- similarity search
    def closest_to_paper(self, paper_id: str, k: int = 10) -> list[ScoredPaper]:
        sims = self.embeddings @ self.embeddings[self._row[paper_id]]
        return self._top_k(sims, k, skip=paper_id)

    def closest_to_text(self, text: str, k: int = 10) -> list[ScoredPaper]:
        if self._embed_fn is None:
            raise RuntimeError("open_kb() was called without embed_fn; cannot embed text")
        v = np.asarray(self._embed_fn(text), dtype=float)
        v = v / (np.linalg.norm(v) or 1.0)
        return self._top_k(self.embeddings @ v, k)

    def closest_to(self, text: str, k: int = 10) -> list[ScoredPaper]:
        """Compatibility alias for semantic search by query text."""
        return self.closest_to_text(text, k)

    def _top_k(self, sims, k, skip=None) -> list[ScoredPaper]:
        order = np.argsort(-sims)
        out = []
        for i in order:
            if self.ids[i] == skip:
                continue
            out.append(ScoredPaper(paper_id=self.ids[i], score=float(sims[i])))
            if len(out) == k:
                break
        return out

    # ---- graph queries (papers only)
    def neighbors(self, paper_id: str,
                  kinds: tuple[str, ...] = ("cites", "cited_by", "similar")) -> list[Neighbor]:
        out = []
        for _, v, d in self.graph.out_edges(paper_id, data=True):
            if d["kind"] == "cites" and "cites" in kinds:
                out.append(Neighbor(paper_id=v, kind="cites"))
            elif d["kind"] == "similar" and "similar" in kinds:
                out.append(Neighbor(paper_id=v, kind="similar", weight=d.get("weight")))
        if "cited_by" in kinds:
            for u, _, d in self.graph.in_edges(paper_id, data=True):
                if d["kind"] == "cites":
                    out.append(Neighbor(paper_id=u, kind="cited_by"))
        return out

    def citation_path(self, a: str, b: str, undirected: bool = False) -> Optional[list[str]]:
        view = nx.subgraph_view(
            self.graph, filter_edge=lambda u, v, key: self.graph[u][v][key]["kind"] == "cites")
        if undirected:
            view = view.to_undirected(as_view=True)
        try:
            return nx.shortest_path(view, a, b)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None


# --------------------------------------------------------------------------- B's output


class ModuleBResult(BaseModel):
    kb: KBHandle
    reviews: list[PaperReview]          # same list type as LabState.reviews
    conflicts: list[Conflict]
    ledger_entries: list = Field(default_factory=list)   # list[LedgerEntry] (defined by Module A)


def module_b_node(state: dict, run_module_b: Callable[..., ModuleBResult]) -> dict:
    """Thin LangGraph wrapper: returns a partial state update.
    Ledger handling must be agreed with Module A (Module A appends the entries)."""
    res = run_module_b(state["request"])
    return {"kb": res.kb, "reviews": res.reviews, "conflicts": res.conflicts}
