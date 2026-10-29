import shutil
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock

import numpy as np

from module_A_tooling.cache import Cache
from module_B_literature_kg.b0 import assemble, pipeline
from module_B_literature_kg.b0 import fulltext as ft
from module_B_literature_kg.b0.assemble import run_assemble_step
from module_B_literature_kg.b0.clients import ArxivClient, S2Client
from module_B_literature_kg.b0.embed import Embedder
from module_B_literature_kg.b0.fulltext import FulltextFetcher, Thresholds
from module_B_literature_kg.b0.manifest import Manifest, StepValidationError, hash_path, read_json
from module_B_literature_kg.b0.pipeline import (B0Error, B0Result, STEP_ORDER, build_parser, format_summary, read_request,
                                  run_b0)
from tests.Module_B.B0.fakes  import ARXIV_FEED, FakeResponse, FakeSession, s2_raw
from tests.Module_B.B0.test_fulltext import TITLE, ar5iv_html

TH = Thresholds(min_chars=300, min_chars_per_page=50)
REQUEST = {"topic": "attention models for translation", "seed_papers": ["2305.12345"]}
EMPTY_FEED = "<feed xmlns='http://www.w3.org/2005/Atom'/>"


def encoder(texts):
    out = np.zeros((len(texts), 64))
    for i, t in enumerate(texts):
        for w in t.lower().split():
            out[i, zlib.crc32(w.encode()) % 64] += 1.0
    return out


def services(tmp, *, found=True, ar5iv=True):
    def s2_router(path, params):
        if not found:
            return FakeResponse(404)
        if path.endswith("/references"):
            return FakeResponse(200, {"data": [
                {"citedPaper": s2_raw("r1", "Reference One About Vision Models", arxiv="2001.00001")},
                {"citedPaper": s2_raw("r2", "Reference Two About Speech Systems")}]})
        if "ARXIV:2305.12345" in path:
            return FakeResponse(200, s2_raw("p1", TITLE, arxiv="2305.12345"))
        return FakeResponse(404)

    def html_router(url, params):
        if ar5iv and url == ft.AR5IV_BASE_URL + "2305.12345":
            return FakeResponse(200, text=ar5iv_html())
        return FakeResponse(404)

    arxiv_text = ARXIV_FEED.format(aid="2305.12345") if found else EMPTY_FEED
    return dict(
        s2=S2Client(cache=Cache(":memory:"), session=FakeSession(s2_router), api_key=None, min_interval=0.0,
                    sleep=lambda s: None),
        arxiv=ArxivClient(cache=Cache(":memory:"), session=FakeSession(lambda p, q: FakeResponse(200, text=arxiv_text)),
                          min_interval=0.0, sleep=lambda s: None),
        fetcher=FulltextFetcher(cache=Cache(":memory:"), session=FakeSession(html_router), pdf_dir=tmp / "pdfs",
                                min_interval=0.0, sleep=lambda s: None, now=lambda: 1000.0),
        embedder=Embedder(encoder=encoder, cache=Cache(":memory:")), thresholds=TH)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "run"
        self.svc = services(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_b0(self, request=REQUEST, **kw):
        return run_b0(request, root=self.root, **{**self.svc, **kw})

    def test_full_run_then_everything_skipped(self):
        res = self.run_b0()
        self.assertTrue(res.completed)
        self.assertEqual([s.step for s in res.steps], STEP_ORDER)
        self.assertEqual({s.status for s in res.steps}, {"done"})
        again = self.run_b0()
        self.assertEqual({s.status for s in again.steps}, {"skipped"})
        self.assertIn("COMPLETE", format_summary(again))

    def test_outputs_follow_the_interface_contract(self):
        self.run_b0()
        papers = read_json(self.root / "papers.json")
        self.assertEqual(sorted(papers), ["P1", "P2", "P3"])
        self.assertEqual(set(papers["P1"]), set(assemble.PAPER_META_TYPES))
        self.assertEqual((papers["P1"]["is_seed"], papers["P1"]["has_full_text"], papers["P2"]["has_full_text"]),
                         (True, True, False))
        ids, m, topic = read_json(self.root / "ids.json"), np.load(self.root / "embeddings.npy"), np.load(self.root / "topic.npy")
        self.assertEqual((ids, m.shape, topic.shape), (["P1", "P2", "P3"], (3, 64), (64,)))
        np.testing.assert_allclose(np.linalg.norm(m, axis=1), 1.0, atol=1e-5)
        self.assertEqual([(e["source"], e["target"]) for e in read_json(self.root / "citation_edges.json")],
                         [("P1", "P2"), ("P1", "P3")])
        self.assertEqual(read_json(self.root / "request.json")["seed_papers"], ["2305.12345"])

    def test_sidecar_records_where_facts_came_from(self):
        self.run_b0()
        rec = read_json(self.root / "paper_records.json")
        seed = rec["P1"]
        self.assertEqual(seed["seed_resolution"]["match"]["method"], "s2_arxiv_id")
        self.assertEqual((seed["references"]["status"], seed["references"]["n_references"]), ("ok", 2))
        self.assertEqual((seed["fulltext"]["status"], seed["fulltext"]["source"]), ("ok", "ar5iv"))
        self.assertEqual(seed["fulltext"]["attempts"][0]["source"], "ar5iv")
        self.assertEqual(seed["embedding"], {"status": "embedded", "flags": []})
        self.assertEqual(rec["P2"]["cited_by_seeds"], ["P1"])
        self.assertNotIn("seed_resolution", rec["P2"])

    def test_report_and_gates(self):
        res = self.run_b0()
        c = res.report["counts"]
        self.assertEqual((c["n_papers"], c["n_seeds"], c["n_edges"], c["n_embedded"], c["n_with_full_text"]), (3, 1, 2, 3, 1))
        self.assertTrue(res.report["all_gates_passed"])
        self.assertEqual(len(res.report["gates"]), 9)
        self.assertEqual(set(res.paths), {"emb_path", "ids_path", "meta_path", "topic_emb_path", "citation_edges_path",
                                          "paper_records_path", "report_path", "graph_path", "reviews_path"})

    def test_abstract_only_seed_is_a_warning_not_a_failure(self):
        self.svc = services(self.tmp, ar5iv=False)
        res = self.run_b0()
        self.assertTrue(res.completed)
        self.assertFalse(read_json(self.root / "papers.json")["P1"]["has_full_text"])
        self.assertTrue(any("abstract-only" in w for w in res.report["warnings"]))

    def test_no_seed_resolved_stops_at_step_one(self):
        self.svc = services(self.tmp, found=False)
        with self.assertRaises(B0Error) as ctx:
            self.run_b0()
        self.assertEqual(ctx.exception.step, "b0_seeds")
        self.assertFalse((self.root / "papers.json").exists())
        self.assertNotIn("b0_references", Manifest(root=self.root).steps())

    def test_tampered_vectors_fail_the_gates_then_heal(self):
        self.run_b0()
        good_papers = hash_path(self.root / "papers.json")
        np.save(self.root / "embeddings.npy", np.load(self.root / "embeddings.npy") * 2.0)   # no longer unit length
        with self.assertRaises(StepValidationError) as ctx:
            run_assemble_step(Manifest(root=self.root))
        self.assertIn("vectors_valid", str(ctx.exception))
        self.assertIn("steps_intact", str(ctx.exception))
        self.assertEqual(hash_path(self.root / "papers.json"), good_papers)                   # old outputs untouched
        self.assertTrue((self.root / ".staging" / "b0_assemble" / "b0_report.json").exists()) # diagnostics kept
        healed = self.run_b0()                                                                # embed re-runs, assemble passes
        self.assertTrue(healed.completed)
        self.assertEqual(next(s.status for s in healed.steps if s.step == "b0_embed"), "done")
        self.assertTrue(healed.report["all_gates_passed"])

    def test_missing_text_file_fails_the_gate(self):
        self.run_b0()
        (self.root / "fulltext" / "s2_p1.txt").unlink()
        with self.assertRaises(StepValidationError) as ctx:
            run_assemble_step(Manifest(root=self.root))
        self.assertIn("fulltext_files", str(ctx.exception))

    def test_force_redoes_everything_and_stop_after_stops(self):
        self.run_b0()
        forced = self.run_b0(force=True)
        self.assertEqual({s.status for s in forced.steps}, {"done"})
        early = self.run_b0(stop_after="b0_references")
        self.assertEqual(([s.step for s in early.steps], early.completed), (["b0_seeds", "b0_references"], False))
        with self.assertRaises(ValueError):
            self.run_b0(stop_after="nonsense")

    def test_changed_seed_list_reruns_from_step_one(self):
        self.run_b0()
        self.svc = services(self.tmp)
        res = self.run_b0({"topic": REQUEST["topic"], "seed_papers": ["2305.12345", "2305.12345"]})
        self.assertEqual(res.steps[0].status, "skipped")           # a repeated line is removed: same seed list
        res = self.run_b0({"topic": REQUEST["topic"], "seed_papers": ["https://arxiv.org/abs/2305.12345v2"]})
        self.assertEqual(res.steps[0].status, "done")              # another spelling = another input text: redo
        res = self.run_b0({"topic": "a different topic", "seed_papers": ["https://arxiv.org/abs/2305.12345v2"]})
        self.assertEqual({s.step: s.status for s in res.steps}["b0_embed"], "done")
        self.assertEqual({s.step: s.status for s in res.steps}["b0_dedupe"], "skipped")


class UnitTests(unittest.TestCase):
    def test_read_request(self):
        class Obj:
            topic, seed_papers = " T ", ["a", " a ", "b", ""]
        self.assertEqual(read_request(Obj()), ("T", ["a", "b"]))
        self.assertEqual(read_request({"topic": "t", "seed_papers": ["x"]}), ("t", ["x"]))
        with self.assertRaises(ValueError):
            read_request({"topic": " ", "seed_papers": ["x"]})
        with self.assertRaises(ValueError):
            read_request({"topic": "t", "seed_papers": []})

    def test_paper_meta_gate_catches_wrong_types(self):
        good = {"paper_id": "P1", "title": "T", "abstract": "", "year": 2020, "venue": None, "citation_count": None,
                "s2_id": None, "arxiv_id": None, "doi": None, "is_seed": True, "has_full_text": False}
        self.assertEqual(assemble._gate_paper_meta({"P1": good}), [])
        self.assertTrue(assemble._gate_paper_meta({"P1": {**good, "year": "2020"}}))
        self.assertTrue(assemble._gate_paper_meta({"P1": {**good, "year": True}}))
        self.assertTrue(assemble._gate_paper_meta({"P1": {k: v for k, v in good.items() if k != "doi"}}))

    def test_command_line(self):
        args = build_parser().parse_args(["--topic", "t", "--seed", "a", "--seed", "b", "--force", "--stop-after", "b0_dedupe"])
        self.assertEqual((args.seed, args.force, args.stop_after), (["a", "b"], True, "b0_dedupe"))
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["--seed", "a"])                                   # topic is required
        done = B0Result(run_id="r", root=Path("."), completed=True)
        with mock.patch.object(pipeline, "run_b0", return_value=done) as m:
            self.assertEqual(pipeline.main(["--topic", "t", "--seed", "a"]), 0)
            self.assertEqual(m.call_args[0][0], {"topic": "t", "seed_papers": ["a"]})
        with mock.patch.object(pipeline, "run_b0", side_effect=B0Error("b0_seeds", ValueError("x"))):
            self.assertEqual(pipeline.main(["--topic", "t", "--seed", "a"]), 1)


if __name__ == "__main__":
    unittest.main()
