"""Build Module B's citation, mutual-kNN, and concept graph."""
from __future__ import annotations

from itertools import combinations
from typing import Iterable

import networkx as nx
import numpy as np

from .b3_normalize import concept_node_id, normalize_name

CONCEPT_FIELDS = {
    "concepts": "concept",
    "methods": "method",
    "datasets": "dataset",
    "metrics": "metric",
}


def _value(item, key, default=None):
    return item.get(key, default) if isinstance(item, dict) else getattr(item, key, default)


def build_graph(papers, embeddings, reviews=(), citation_edges=(), k=5) -> nx.MultiDiGraph:
    """Build a graph using paper metadata, normalized embeddings, reviews, and citations.

    ``papers`` and ``reviews`` may contain dictionaries or objects with matching
    fields. ``citation_edges`` contains directed ``(citing_id, cited_id)`` pairs.
    Embedding row i must correspond to papers[i].paper_id (or papers[i].id).
    """
    papers = list(papers)
    reviews = list(reviews)
    ids = [str(_value(p, "paper_id", _value(p, "id", ""))) for p in papers]
    if any(not pid for pid in ids):
        raise ValueError("Every paper needs a paper_id or id")
    if len(set(ids)) != len(ids):
        raise ValueError("Paper IDs must be unique")
    vectors = np.asarray(embeddings, dtype=float)
    if vectors.ndim != 2 or vectors.shape[0] != len(ids):
        raise ValueError("embeddings must be a 2-D array with one row per paper")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.where(norms == 0, 1.0, norms)

    graph = nx.MultiDiGraph()
    for paper, pid in zip(papers, ids):
        graph.add_node(pid, type="paper", title=_value(paper, "title", ""))

    valid = set(ids)
    for source, target in citation_edges:
        if source in valid and target in valid and source != target:
            graph.add_edge(source, target, kind="cites")

    # Keep an edge only when the two papers occur in each other's top-k lists.
    similarities = vectors @ vectors.T
    neighbor_sets = {}
    for i, pid in enumerate(ids):
        order = [j for j in np.argsort(-similarities[i]) if j != i]
        neighbor_sets[pid] = set(ids[j] for j in order[:max(0, int(k))])
    for i, j in combinations(range(len(ids)), 2):
        a, b = ids[i], ids[j]
        if b in neighbor_sets[a] and a in neighbor_sets[b]:
            score = float(similarities[i, j])
            graph.add_edge(a, b, kind="similar", weight=score)
            graph.add_edge(b, a, kind="similar", weight=score)

    # Keep isolated papers connected to their best available neighbour.
    for i, pid in enumerate(ids):
        if len(ids) < 2 or any(d.get("kind") == "similar" for _, _, d in graph.out_edges(pid, data=True)):
            continue
        j = max((j for j in range(len(ids)) if j != i), key=lambda j: similarities[i, j])
        other = ids[j]
        score = float(similarities[i, j])
        graph.add_edge(pid, other, kind="similar", weight=score)
        graph.add_edge(other, pid, kind="similar", weight=score)

    review_by_id = {str(_value(r, "paper_id", "")): r for r in reviews}
    for pid, review in review_by_id.items():
        if pid not in valid:
            continue
        for field, kind in CONCEPT_FIELDS.items():
            values = _value(review, field, []) or []
            for raw_name in values:
                if not isinstance(raw_name, str) or not raw_name.strip():
                    continue
                node_id = concept_node_id(kind, raw_name)
                graph.add_node(node_id, type=kind, name=normalize_name(raw_name))
                graph.add_edge(pid, node_id, kind=f"has_{kind}")
    return graph
