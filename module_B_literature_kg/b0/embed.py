"""module_B_literature_kg/b0/embed.py: step 5 of B0. Turn paper text into numbers (vectors) so that "how related are
two papers" becomes one calculation.

Plain-language idea
    The embedding model (BAAI/bge-m3) reads "title + abstract" and returns a list of numbers. Papers on
    similar topics get similar lists. We scale every list to length 1, so the dot product of two lists
    IS their cosine similarity (1.0 = same direction, 0 = unrelated). The research topic is embedded the
    same way, so "how relevant is this paper to the topic" is also one dot product.

What this file gives
    Embedder        text -> unit vectors, with cache + batches (B1 reuses it for papers it discovers)
    EmbeddingStore  the matrix + the ids, row i of the matrix belongs to ids[i] (the interface.py contract)
    run_embed_step  step 5 under the manifest

Outputs: embeddings.npy, ids.json, topic.npy, embedding_status.json
    ids.json lists ONLY papers that were embedded. Skipped papers are listed in embedding_status.json.

Rules
    * Input text is title + abstract. Title only is allowed (flag "title_only"); abstract only too
      (flag "abstract_only"); neither means skipped with status "missing_input".
    * Each raw vector is cached under (text hash, model, revision, settings), so a re-run or a second
      paper with the same text costs nothing. The model is loaded only if something is NOT cached.
    * Texts are encoded on CPU in batches of EMBEDDING_BATCH_SIZE (8).
"""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

try:
    from module_A_tooling.cache import Cache, get_default_cache, hash_text, make_key
except ImportError:                            # pragma: no cover
    from module_A_tooling.cache import Cache, get_default_cache, hash_text, make_key

from module_A_tooling.config import (EMBEDDING_BATCH_SIZE, EMBEDDING_DEVICE, EMBEDDING_INPUT, EMBEDDING_MAX_SEQ_LENGTH,
                    EMBEDDING_MODEL, EMBEDDING_REVISION, RUN_FILES, VECTOR_NORM_TOLERANCE)
from .manifest import (Manifest, MissingInputError, StepResult, atomic_write_bytes, atomic_write_json,
                       read_json, run_step)
from .dedupe import CANONICAL_FILE

EMBEDDINGS_FILE = RUN_FILES["embeddings"]
IDS_FILE = RUN_FILES["ids"]
TOPIC_FILE = RUN_FILES["topic_embedding"]
STATUS_FILE = RUN_FILES["embedding_status"]
UNIT_NORM_TOLERANCE = VECTOR_NORM_TOLERANCE
_MISSING = object()

Encoder = Callable[[list[str]], np.ndarray]      # list of texts -> array (len(texts), dim)


class EmbeddingLibraryMissing(RuntimeError):
    """sentence-transformers is not installed."""


# --------------------------------------------------------------------------- text and maths

def _squash(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def build_embedding_text(paper: dict) -> tuple[Optional[str], list[str]]:
    """(text to embed, flags). text is None when there is nothing usable."""
    title, abstract = _squash(paper.get("title")), _squash(paper.get("abstract"))
    if not title and not abstract:
        return None, ["missing_input"]
    if not abstract:
        return title, ["title_only"]
    if not title:
        return abstract, ["abstract_only"]
    return f"{title}\n\n{abstract}", []


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale every row to length 1. A zero or non-finite row is an error, never silently kept."""
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim != 2:
        raise ValueError(f"expected a 2-D array of vectors, got shape {m.shape}")
    if not np.isfinite(m).all():
        raise ValueError("embedding contains NaN or infinite values")
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("embedding contains an all-zero vector")
    return m / norms


def load_default_encoder(model: str = EMBEDDING_MODEL, revision: str = EMBEDDING_REVISION,
                         device: str = EMBEDDING_DEVICE, max_seq_length: int = EMBEDDING_MAX_SEQ_LENGTH,
                         batch_size: int = EMBEDDING_BATCH_SIZE) -> Encoder:
    """Load BGE-M3 through sentence-transformers (first call downloads the model, about 2 GB)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbeddingLibraryMissing("install it with: pip install sentence-transformers") from exc
    st = SentenceTransformer(model, device=device, revision=revision)
    st.max_seq_length = max_seq_length

    def encode(texts: list[str]) -> np.ndarray:
        return st.encode(texts, batch_size=batch_size, normalize_embeddings=False,
                         convert_to_numpy=True, show_progress_bar=False)

    return encode


# --------------------------------------------------------------------------- the embedder

class Embedder:
    def __init__(self, encoder: Optional[Encoder] = None, cache: Optional[Cache] = None,
                 batch_size: int = EMBEDDING_BATCH_SIZE, model: str = EMBEDDING_MODEL,
                 revision: str = EMBEDDING_REVISION):
        self._encoder = encoder              # None = load BGE-M3 the first time something needs encoding
        self.cache = cache if cache is not None else get_default_cache()
        self.batch_size = batch_size
        self.model, self.revision = model, revision
        self.encoder_calls = 0

    def _key(self, text: str) -> str:
        settings = hash_text(f"{EMBEDDING_INPUT}|{EMBEDDING_MAX_SEQ_LENGTH}")
        return make_key("embedding", {"text": hash_text(text)}, model=self.model,
                        revision=self.revision, prompt_hash=settings)

    def _encode(self, batch: list[str]) -> np.ndarray:
        if self._encoder is None:
            self._encoder = load_default_encoder(self.model, self.revision)
        self.encoder_calls += 1
        out = np.asarray(self._encoder(batch), dtype=np.float64)
        if out.ndim != 2 or out.shape[0] != len(batch):
            raise ValueError(f"encoder returned shape {out.shape} for {len(batch)} texts")
        return out

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Unit vectors, one row per text, in the same order. Identical texts are encoded once."""
        unique = list(dict.fromkeys(texts))
        vectors: dict[str, list[float]] = {}
        missing: list[str] = []
        for t in unique:
            cached = self.cache.get(self._key(t), _MISSING)
            if cached is _MISSING:
                missing.append(t)
            else:
                vectors[t] = cached
        for i in range(0, len(missing), self.batch_size):
            batch = missing[i:i + self.batch_size]
            out = self._encode(batch)
            for t, v in zip(batch, out):
                vectors[t] = v.tolist()
                self.cache.set(self._key(t), vectors[t], kind="embedding")    # saved at once: a crash loses nothing
        return l2_normalize(np.array([vectors[t] for t in texts]))

    def embed_topic(self, topic: str) -> np.ndarray:
        if not _squash(topic):
            raise ValueError("the research topic is empty")
        return self.embed_texts([_squash(topic)])[0]


# --------------------------------------------------------------------------- the store

class EmbeddingStore:
    """ids[i] <-> matrix[i]. Rows are unit vectors (float32 on disk)."""

    def __init__(self, ids: Optional[list[str]] = None, matrix: Optional[np.ndarray] = None):
        self.ids: list[str] = list(ids or [])
        self.matrix = np.zeros((0, 0)) if matrix is None else np.asarray(matrix)
        if len(self.ids) != len(self.matrix) and not (not self.ids and self.matrix.size == 0):
            raise ValueError("ids and matrix have different lengths")
        self._row = {pid: i for i, pid in enumerate(self.ids)}
        if len(self._row) != len(self.ids):
            raise ValueError("duplicate paper ids in the store")

    @property
    def dim(self) -> int:
        return int(self.matrix.shape[1]) if self.matrix.ndim == 2 and self.matrix.size else 0

    def __contains__(self, pid: str) -> bool:
        return pid in self._row

    def get(self, pid: str) -> np.ndarray:
        return self.matrix[self._row[pid]]

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        """Append rows (B1 uses this for newly discovered papers). Existing ids are refused."""
        vectors = np.asarray(vectors)
        clash = [p for p in ids if p in self._row]
        if clash:
            raise ValueError(f"already in the store: {clash[:3]}")
        if self.ids and vectors.shape[1] != self.dim:
            raise ValueError("vector size differs from the store")
        self.matrix = vectors if not self.ids else np.vstack([self.matrix, vectors])
        for pid in ids:
            self._row[pid] = len(self.ids)
            self.ids.append(pid)

    def save(self, emb_path: Path, ids_path: Path) -> None:
        buf = io.BytesIO()
        np.save(buf, self.matrix.astype(np.float32))
        atomic_write_bytes(emb_path, buf.getvalue())
        atomic_write_json(ids_path, self.ids)

    @classmethod
    def load(cls, emb_path: Path, ids_path: Path) -> "EmbeddingStore":
        return cls(read_json(ids_path, []), np.load(emb_path))


def _numeric(pid: str) -> int:
    return int(pid[1:])


def embed_papers(embedder: Embedder, papers: dict[str, dict]) -> tuple[EmbeddingStore, dict]:
    """Embed every paper that has usable text. Returns (store, status per paper id)."""
    status: dict[str, dict] = {}
    to_embed: list[tuple[str, str]] = []
    for pid in sorted(papers, key=_numeric):
        text, flags = build_embedding_text(papers[pid])
        if text is None:
            status[pid] = {"status": "skipped", "flags": flags}
        else:
            status[pid] = {"status": "embedded", "flags": flags}
            to_embed.append((pid, text))
    if not to_embed:
        return EmbeddingStore(), status
    matrix = embedder.embed_texts([t for _, t in to_embed])
    return EmbeddingStore([p for p, _ in to_embed], matrix), status


# --------------------------------------------------------------------------- the step

def validate_embedding_output(staging: Path, paper_ids: set[str]) -> list[str]:
    try:
        matrix = np.load(staging / EMBEDDINGS_FILE)
        topic = np.load(staging / TOPIC_FILE)
    except (FileNotFoundError, ValueError) as exc:
        return [f"embedding files unreadable: {exc}"]
    ids = read_json(staging / IDS_FILE, None)
    status = read_json(staging / STATUS_FILE, None)
    if ids is None or status is None:
        return ["ids.json or embedding_status.json is missing"]
    problems = []
    if matrix.ndim != 2 or len(ids) != matrix.shape[0] or not len(ids):
        problems.append("embeddings.npy and ids.json do not line up (or are empty)")
        return problems
    if len(set(ids)) != len(ids) or not set(ids) <= paper_ids:
        problems.append("ids.json has duplicates or unknown paper ids")
    if not np.isfinite(matrix).all():
        problems.append("embeddings contain NaN or infinite values")
    if np.abs(np.linalg.norm(matrix, axis=1) - 1.0).max() > UNIT_NORM_TOLERANCE:
        problems.append("some vectors are not unit length")
    if topic.shape != (matrix.shape[1],) or abs(np.linalg.norm(topic) - 1.0) > UNIT_NORM_TOLERANCE:
        problems.append("topic vector has the wrong size or is not unit length")
    if set(status) != paper_ids:
        problems.append("embedding_status.json does not cover every paper")
    if {p for p, s in status.items() if s["status"] == "embedded"} != set(ids):
        problems.append("embedding_status.json and ids.json disagree")
    return problems


def run_embed_step(manifest: Manifest, topic: str, embedder: Embedder) -> StepResult:
    """Step 5 under the manifest. Reads canonical_papers.json; embeds the papers and the topic."""
    doc = read_json(manifest.root / CANONICAL_FILE)
    if doc is None:
        raise MissingInputError(f"{CANONICAL_FILE} not found: run step 4 (b0_dedupe) first")
    papers = doc["papers"]

    def work(staging: Path) -> dict:
        store, status = embed_papers(embedder, papers)
        if not store.ids:
            raise ValueError("no paper has usable text (title or abstract): nothing to embed")
        topic_vec = embedder.embed_topic(topic)
        if topic_vec.shape[0] != store.dim:
            raise ValueError("topic vector and paper vectors have different sizes")
        store.save(staging / EMBEDDINGS_FILE, staging / IDS_FILE)
        buf = io.BytesIO()
        np.save(buf, topic_vec.astype(np.float32))
        atomic_write_bytes(staging / TOPIC_FILE, buf.getvalue())
        atomic_write_json(staging / STATUS_FILE, status)
        flagged = sum(1 for s in status.values() if s["flags"])
        return {"n_embedded": len(store.ids), "n_skipped": len(status) - len(store.ids),
                "n_flagged": flagged, "dim": store.dim}

    return run_step(manifest, "b0_embed", work,
                    outputs=[EMBEDDINGS_FILE, IDS_FILE, TOPIC_FILE, STATUS_FILE], inputs=[CANONICAL_FILE],
                    extra_inputs={"topic": _squash(topic)},
                    validate=lambda staging: validate_embedding_output(staging, set(papers)))
