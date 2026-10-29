import shutil
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock

import numpy as np

import module_A_tooling.config
from module_A_tooling.cache import Cache
from module_B_literature_kg.b0 import embed
from module_B_literature_kg.b0.dedupe import run_dedupe_step
from module_B_literature_kg.b0.embed import (Embedder, EmbeddingLibraryMissing, EmbeddingStore, build_embedding_text,
                               embed_papers, l2_normalize, load_default_encoder, run_embed_step)
from module_B_literature_kg.b0.manifest import Manifest, MissingInputError, atomic_write_json, hash_path, read_json

DIM = 64


def fake_encoder(calls):
    """Bag-of-words hashing: texts that share words get similar vectors. Records every call."""
    def encode(texts):
        calls.append(list(texts))
        out = np.zeros((len(texts), DIM))
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, zlib.crc32(w.encode()) % DIM] += 1.0
        return out
    return encode


def make_embedder(calls, batch_size=8):
    return Embedder(encoder=fake_encoder(calls), cache=Cache(":memory:"), batch_size=batch_size)


def canonical(*papers):
    return {"version": 1, "papers": {f"P{i}": {"paper_id": f"P{i}", **p} for i, p in enumerate(papers, 1)}}


class TextTests(unittest.TestCase):
    def test_input_text_and_flags(self):
        self.assertEqual(build_embedding_text({"title": " A  Title ", "abstract": "Some\nabstract."}),
                         ("A Title\n\nSome abstract.", []))
        self.assertEqual(build_embedding_text({"title": "Only title", "abstract": ""}), ("Only title", ["title_only"]))
        self.assertEqual(build_embedding_text({"title": "", "abstract": "Only abstract"}), ("Only abstract", ["abstract_only"]))
        self.assertEqual(build_embedding_text({"title": " ", "abstract": ""}), (None, ["missing_input"]))

    def test_normalize(self):
        m = l2_normalize(np.array([[3.0, 4.0], [0.0, 2.0]]))
        np.testing.assert_allclose(np.linalg.norm(m, axis=1), 1.0)
        with self.assertRaises(ValueError):
            l2_normalize(np.array([[0.0, 0.0]]))
        with self.assertRaises(ValueError):
            l2_normalize(np.array([[1.0, float("nan")]]))


class EmbedderTests(unittest.TestCase):
    def test_unit_vectors_order_and_similarity(self):
        calls = []
        e = make_embedder(calls)
        m = e.embed_texts(["language models for translation", "translation with language models", "protein folding in cells"])
        np.testing.assert_allclose(np.linalg.norm(m, axis=1), 1.0)
        self.assertGreater(m[0] @ m[1], m[0] @ m[2])                 # related texts are closer

    def test_batches_of_eight_and_duplicates_encoded_once(self):
        calls = []
        e = make_embedder(calls, batch_size=8)
        texts = [f"text number {i}" for i in range(20)] + ["text number 3"]
        m = e.embed_texts(texts)
        self.assertEqual([len(c) for c in calls], [8, 8, 4])         # 20 unique texts, not 21
        np.testing.assert_allclose(m[3], m[20])

    def test_cache_means_no_second_encoding(self):
        calls = []
        e = make_embedder(calls)
        first = e.embed_texts(["alpha beta", "gamma delta"])
        n = len(calls)
        second = e.embed_texts(["gamma delta", "alpha beta"])
        self.assertEqual(len(calls), n)
        np.testing.assert_allclose(first[0], second[1])

    def test_model_revision_is_part_of_the_cache_key(self):
        cache, calls = Cache(":memory:"), []
        Embedder(encoder=fake_encoder(calls), cache=cache, revision="v1").embed_texts(["alpha"])
        Embedder(encoder=fake_encoder(calls), cache=cache, revision="v2").embed_texts(["alpha"])
        self.assertEqual(len(calls), 2)

    def test_wrong_encoder_output_is_rejected(self):
        e = Embedder(encoder=lambda texts: np.zeros((1, 4)) + 1, cache=Cache(":memory:"))
        with self.assertRaises(ValueError):
            e.embed_texts(["a", "b"])

    def test_empty_topic_is_rejected(self):
        with self.assertRaises(ValueError):
            make_embedder([]).embed_topic("   ")

    def test_default_encoder_needs_the_library(self):
        try:
            import sentence_transformers      # noqa: F401
            self.skipTest("sentence-transformers is installed")
        except ImportError:
            with self.assertRaises(EmbeddingLibraryMissing):
                load_default_encoder()

    def test_model_is_not_loaded_when_everything_is_cached(self):
        cache = Cache(":memory:")
        Embedder(encoder=fake_encoder([]), cache=cache).embed_texts(["alpha beta"])
        with mock.patch.object(embed, "load_default_encoder", side_effect=AssertionError("model loaded")):
            Embedder(encoder=None, cache=cache).embed_texts(["alpha beta"])


class StoreTests(unittest.TestCase):
    def test_papers_skipped_and_flagged(self):
        e = make_embedder([])
        papers = {"P1": {"title": "Full", "abstract": "Has both"}, "P2": {"title": "", "abstract": ""},
                  "P3": {"title": "Title only", "abstract": ""}}
        store, status = embed_papers(e, papers)
        self.assertEqual(store.ids, ["P1", "P3"])
        self.assertEqual((status["P2"]["status"], status["P2"]["flags"]), ("skipped", ["missing_input"]))
        self.assertEqual(status["P3"]["flags"], ["title_only"])
        self.assertEqual(store.matrix.shape, (2, DIM))

    def test_add_get_save_load(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            store = EmbeddingStore(["P1"], l2_normalize(np.array([[1.0, 0.0, 0.0]])))
            store.add(["P2"], l2_normalize(np.array([[0.0, 1.0, 0.0]])))
            with self.assertRaises(ValueError):
                store.add(["P1"], np.array([[1.0, 0.0, 0.0]]))
            store.save(tmp / "e.npy", tmp / "ids.json")
            back = EmbeddingStore.load(tmp / "e.npy", tmp / "ids.json")
            self.assertEqual(back.ids, ["P1", "P2"])
            np.testing.assert_allclose(back.get("P2"), [0.0, 1.0, 0.0])
            self.assertEqual(back.matrix.dtype, np.float32)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class StepTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.m = Manifest(root=self.tmp)
        atomic_write_json(self.tmp / "canonical_papers.json", canonical(
            {"title": "Language models for translation", "abstract": "We study translation."},
            {"title": "Protein folding", "abstract": "Cells and proteins."},
            {"title": "", "abstract": ""}))
        self.calls = []
        self.embedder = make_embedder(self.calls)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_step_outputs_line_up(self):
        res = run_embed_step(self.m, "machine translation with language models", self.embedder)
        self.assertEqual((res.status, res.notes["n_embedded"], res.notes["n_skipped"], res.notes["dim"]), ("done", 2, 1, DIM))
        ids = read_json(self.tmp / "ids.json")
        matrix, topic = np.load(self.tmp / "embeddings.npy"), np.load(self.tmp / "topic.npy")
        self.assertEqual((ids, matrix.shape, topic.shape), (["P1", "P2"], (2, DIM), (DIM,)))     # row i <-> ids[i]
        np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5)
        self.assertGreater(matrix[0] @ topic, matrix[1] @ topic)                                  # P1 is closer to the topic
        self.assertEqual(read_json(self.tmp / "embedding_status.json")["P3"]["status"], "skipped")

    def test_skip_then_rerun_on_topic_or_model_change(self):
        run_embed_step(self.m, "topic one", self.embedder)
        n = len(self.calls)
        self.assertEqual(run_embed_step(self.m, "topic one", self.embedder).status, "skipped")
        self.assertEqual(len(self.calls), n)
        self.assertEqual(run_embed_step(self.m, "a different topic", self.embedder).status, "done")
        old = module_A_tooling.config.EMBEDDING_REVISION
        try:
            module_A_tooling.config.EMBEDDING_REVISION = "abc123"
            self.assertEqual(run_embed_step(self.m, "a different topic", self.embedder).status, "done")
        finally:
            module_A_tooling.config.EMBEDDING_REVISION = old

    def test_new_input_paper_reruns(self):
        run_embed_step(self.m, "topic", self.embedder)
        atomic_write_json(self.tmp / "canonical_papers.json", canonical(
            {"title": "Language models for translation", "abstract": "We study translation."},
            {"title": "Protein folding", "abstract": "Cells and proteins."},
            {"title": "Graph networks", "abstract": "Nodes and edges."}))
        self.assertEqual(run_embed_step(self.m, "topic", self.embedder).status, "done")
        self.assertEqual(len(read_json(self.tmp / "ids.json")), 3)

    def test_crash_rerun_gives_identical_files(self):
        run_embed_step(self.m, "topic", self.embedder)
        names = ["embeddings.npy", "ids.json", "topic.npy", "embedding_status.json"]
        before = {n: hash_path(self.tmp / n) for n in names}
        self.m.set("b0_embed", {**self.m.get("b0_embed"), "status": "running"})
        self.assertEqual(run_embed_step(Manifest(root=self.tmp), "topic", make_embedder([])).status, "done")
        self.assertEqual(before, {n: hash_path(self.tmp / n) for n in names})

    def test_nothing_embeddable_fails_the_step(self):
        atomic_write_json(self.tmp / "canonical_papers.json", canonical({"title": "", "abstract": ""}))
        with self.assertRaises(ValueError):
            run_embed_step(self.m, "topic", self.embedder)
        self.assertEqual(self.m.status("b0_embed"), "failed")
        self.assertFalse((self.tmp / "ids.json").exists())

    def test_needs_dedupe_first(self):
        (self.tmp / "canonical_papers.json").unlink()
        with self.assertRaises(MissingInputError):
            run_embed_step(self.m, "topic", self.embedder)

    def test_dedupe_then_embed_chain(self):
        from module_B_literature_kg.b0.clients import new_paper
        seed = new_paper(title="Language models for translation", abstract="We study translation.", s2_id="S1",
                         authors=["Ada Lovelace"], year=2020)
        seed["is_seed"] = True
        stub = new_paper(title="Protein folding in cells", abstract="Cells.", s2_id="R1", authors=["Bob Roe"], year=2019)
        atomic_write_json(self.tmp / "seeds.json", {"seeds": [{"input": "x", "status": "resolved", "paper": seed, "flags": []}]})
        atomic_write_json(self.tmp / "references.json", {"stubs": {stub["key"]: stub},
                          "edges": [{"source": seed["key"], "target": stub["key"], "kind": "cites"}]})
        (self.tmp / "canonical_papers.json").unlink()
        self.assertEqual(run_dedupe_step(self.m).status, "done")
        self.assertEqual(run_embed_step(self.m, "translation", self.embedder).status, "done")
        self.assertEqual(read_json(self.tmp / "ids.json"), ["P1", "P2"])
        # change upstream: embedding must become stale automatically
        atomic_write_json(self.tmp / "references.json", {"stubs": {}, "edges": []})
        self.assertEqual(run_dedupe_step(self.m).status, "done")
        self.assertEqual(run_embed_step(self.m, "translation", self.embedder).status, "done")
        self.assertEqual(read_json(self.tmp / "ids.json"), ["P1"])


if __name__ == "__main__":
    unittest.main()
