"""module_B_literature_kg/b3/b3_step.py: build the knowledge graph (graph.json) from the B0 and B2 files.

Reads:  papers.json, ids.json, embeddings.npy, citation_edges.json (B0)  and  reviews.json (B2)
Writes: graph.json   (node-link JSON of the networkx graph, the format interface.open_kb() reads)

Also here: make_kb_handle(), which fills the KBHandle (the list of file paths Module C uses).
"""
from pathlib import Path

import numpy as np

from module_A_tooling.config import RUN_FILES
from module_B_literature_kg.b0.assemble import PAPERS_FILE
from module_B_literature_kg.b0.dedupe import EDGES_FILE
from module_B_literature_kg.b0.embed import EMBEDDINGS_FILE, IDS_FILE, TOPIC_FILE
from module_B_literature_kg.b0.manifest import read_json, run_step
from ..interface import KBHandle, save_graph
from ..b2.b2_merge import REVIEWS_FILE
from .b3_graph import build_graph

GRAPH_FILE = RUN_FILES["graph"]          # "graph.json"
K_NEIGHBOURS = 5                         # each paper is linked to its 5 closest papers (if they link back)


def run_b3_step(manifest, k=K_NEIGHBOURS):
    root = manifest.root

    def work(staging):
        papers = read_json(root / PAPERS_FILE)
        ids = read_json(root / IDS_FILE)                  # only papers that have a vector
        embeddings = np.load(root / EMBEDDINGS_FILE)
        reviews = read_json(root / REVIEWS_FILE)
        edges = read_json(root / EDGES_FILE)

        graph = build_graph(
            papers=[papers[i] for i in ids],              # same order as the embedding rows
            embeddings=embeddings,
            reviews=list(reviews.values()),               # build_graph wants a list, not a dict
            citation_edges=[(e["source"], e["target"]) for e in edges],   # and (source, target) pairs
            k=k)

        n_papers = sum(1 for _, d in graph.nodes(data=True) if d.get("type") == "paper")
        if n_papers != len(ids):
            raise ValueError(f"graph has {n_papers} paper nodes but ids.json has {len(ids)}")
        save_graph(graph, str(staging / GRAPH_FILE))
        return {"n_papers": n_papers, "n_concept_nodes": graph.number_of_nodes() - n_papers,
                "n_edges": graph.number_of_edges()}

    return run_step(manifest, "b3_graph", work, outputs=[GRAPH_FILE],
                    inputs=[PAPERS_FILE, IDS_FILE, EMBEDDINGS_FILE, REVIEWS_FILE, EDGES_FILE],
                    extra_inputs={"k": k})


def make_kb_handle(root):
    """The paths of all Module B result files, in the shape interface.KBHandle expects."""
    root = Path(root)
    return KBHandle(graph_path=str(root / GRAPH_FILE), emb_path=str(root / EMBEDDINGS_FILE),
                    ids_path=str(root / IDS_FILE), meta_path=str(root / PAPERS_FILE),
                    reviews_path=str(root / REVIEWS_FILE), topic_emb_path=str(root / TOPIC_FILE))
