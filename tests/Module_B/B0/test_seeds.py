import shutil
import tempfile
import unittest
from pathlib import Path

from module_A_tooling.cache import Cache
from module_B_literature_kg.b0.clients import ArxivClient, S2Client
from module_B_literature_kg.b0.manifest import Manifest, read_json
from module_B_literature_kg.b0.seeds import (NoUsableSeedsError, classify_seed, resolve_seeds, run_seeds_step,
                               title_similarity)
from tests.Module_B.B0.fakes import ARXIV_FEED, FakeResponse, FakeSession, s2_raw


def build(s2_router, arxiv_router=None):
    s2 = S2Client(
        cache=Cache(":memory:"),
        session=FakeSession(s2_router),
        api_key=None,
        min_interval=0.0, sleep=lambda s: None)
    ax_router = arxiv_router or (lambda p, q: FakeResponse(200, text=ARXIV_FEED.format(aid="2305.12345")))
    arxiv = ArxivClient(cache=Cache(":memory:"), session=FakeSession(ax_router), min_interval=0.0,
                        sleep=lambda s: None)
    return s2, arxiv


class ClassifyTests(unittest.TestCase):
    def test_kinds(self):
        self.assertEqual(classify_seed("arXiv:2305.12345v2"), ("arxiv", "2305.12345"))
        self.assertEqual(classify_seed("https://doi.org/10.1145/ABC"), ("doi", "10.1145/abc"))
        self.assertEqual(classify_seed("a" * 40), ("s2", "a" * 40))
        self.assertEqual(classify_seed("Attention Is All You Need"), ("title", "Attention Is All You Need"))

    def test_title_similarity(self):
        self.assertEqual(title_similarity("Attention is ALL you need!", "attention is all you need"), 1.0)
        self.assertLess(title_similarity("Attention is all you need", "Deep residual learning"), 0.5)


class ResolveTests(unittest.TestCase):
    def test_arxiv_seed_found_in_s2(self):
        s2, arxiv = build(lambda path, params: FakeResponse(200, s2_raw("p1", "Paper One", arxiv="2305.12345")))
        out = resolve_seeds(["2305.12345"], s2, arxiv)
        rec = out["seeds"][0]
        self.assertEqual((rec["status"], rec["abstract_source"], rec["flags"]), ("resolved", "s2", []))
        self.assertEqual(rec["match"]["confidence"], 1.0)
        self.assertTrue(rec["paper"]["is_seed"])

    def test_missing_abstract_comes_from_arxiv_api(self):
        s2, arxiv = build(lambda path, params: FakeResponse(200, s2_raw("p1", "Paper One", abstract="", arxiv="2305.12345")))
        rec = resolve_seeds(["2305.12345"], s2, arxiv)["seeds"][0]
        self.assertEqual(rec["abstract_source"], "arxiv_api")
        self.assertEqual(rec["paper"]["abstract"], "The abstract text.")
        self.assertNotIn("no_abstract", rec["flags"])

    def test_no_abstract_anywhere_is_flagged(self):
        s2, arxiv = build(lambda path, params: FakeResponse(200, s2_raw("p1", "Paper One", abstract="", arxiv="2305.12345")),
                          lambda p, q: FakeResponse(200, text="<feed xmlns='http://www.w3.org/2005/Atom'/>"))
        rec = resolve_seeds(["2305.12345"], s2, arxiv)["seeds"][0]
        self.assertIn("no_abstract", rec["flags"])

    def test_unknown_to_s2_falls_back_to_arxiv_api(self):
        s2, arxiv = build(lambda path, params: FakeResponse(404))
        rec = resolve_seeds(["2305.12345"], s2, arxiv)["seeds"][0]
        self.assertEqual(rec["match"]["method"], "arxiv_api")
        self.assertIn("not_in_semantic_scholar", rec["flags"])
        self.assertIn("no_s2_id", rec["flags"])
        self.assertEqual(rec["paper"]["key"], "arxiv:2305.12345")

    def test_doi_seed_without_arxiv_id_is_flagged_abstract_only(self):
        s2, arxiv = build(lambda path, params: FakeResponse(200, s2_raw("p9", "Journal Paper", doi="10.1000/abc")))
        rec = resolve_seeds(["10.1000/abc"], s2, arxiv)["seeds"][0]
        self.assertIn("no_arxiv_id", rec["flags"])

    def test_title_thresholds(self):
        def router(path, params):
            return FakeResponse(200, {"data": [s2_raw("p1", "Attention Is All You Need")]})
        s2, arxiv = build(router)
        ok = resolve_seeds(["attention is all you need"], s2, arxiv)
        self.assertEqual(ok["seeds"][0]["flags"].count("confirm_title_match"), 0)

        s2, arxiv = build(router)
        mid = resolve_seeds(["Attention Is All You Need: transformers"], s2, arxiv, title_min=0.5, title_confirm=0.99)
        self.assertIn("confirm_title_match", mid["seeds"][0]["flags"])
        self.assertEqual(mid["needs_confirmation"], ["Attention Is All You Need: transformers"])

        s2, arxiv = build(router)
        bad = resolve_seeds(["Completely different thing about biology"], s2, arxiv)
        self.assertEqual(len(bad["seeds"]), 0)
        self.assertEqual(len(bad["unresolved"]), 1)
        self.assertTrue(bad["unresolved"][0]["match"]["candidates"])     # the near-miss is logged

    def test_duplicate_seeds_are_merged(self):
        s2, arxiv = build(lambda path, params: FakeResponse(200, s2_raw("p1", "Paper One", arxiv="2305.12345", doi="10.1000/abc")))
        out = resolve_seeds(["2305.12345", "10.1000/abc"], s2, arxiv)
        self.assertEqual(len(out["seeds"]), 1)
        self.assertEqual(out["duplicates"][0]["duplicate_of"], "s2:p1")

    def test_server_error_for_one_seed_does_not_stop_others(self):
        def router(path, params):
            if "2305.99999" in path:
                return FakeResponse(500)
            return FakeResponse(200, s2_raw("p1", "Paper One", arxiv="2305.12345"))
        s2, arxiv = build(router)
        out = resolve_seeds(["2305.99999", "2305.12345"], s2, arxiv)
        self.assertEqual(len(out["seeds"]), 1)
        self.assertIn("lookup_error", out["unresolved"][0]["flags"])


class StepTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.m = Manifest(root=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_step_writes_seeds_and_skips_on_rerun(self):
        s2, arxiv = build(lambda path, params: FakeResponse(200, s2_raw("p1", "Paper One", arxiv="2305.12345")))
        self.assertEqual(run_seeds_step(self.m, ["2305.12345"], s2, arxiv).status, "done")
        self.assertEqual(read_json(self.tmp / "seeds.json")["seeds"][0]["paper"]["key"], "s2:p1")
        n = len(s2.session.calls)
        self.assertEqual(run_seeds_step(self.m, ["2305.12345"], s2, arxiv).status, "skipped")
        self.assertEqual(len(s2.session.calls), n)
        self.assertEqual(run_seeds_step(self.m, ["2305.12345", "2306.00001"], s2, arxiv).status, "done")  # new seed list

    def test_all_seeds_unresolved_stops_the_run(self):
        s2, arxiv = build(lambda path, params: FakeResponse(404),
                          lambda p, q: FakeResponse(200, text="<feed xmlns='http://www.w3.org/2005/Atom'/>"))
        with self.assertRaises(NoUsableSeedsError):
            run_seeds_step(self.m, ["2305.12345", "2306.00001"], s2, arxiv)
        self.assertEqual(self.m.status("b0_seeds"), "failed")
        self.assertFalse((self.tmp / "seeds.json").exists())

    def test_some_unresolved_continues_with_warning(self):
        def router(path, params):
            return FakeResponse(200, s2_raw("p1", "Paper One", arxiv="2305.12345")) if "2305.12345" in path else FakeResponse(404)
        s2, arxiv = build(router, lambda p, q: FakeResponse(200, text="<feed xmlns='http://www.w3.org/2005/Atom'/>"))
        res = run_seeds_step(self.m, ["2305.12345", "2306.00001"], s2, arxiv)
        self.assertEqual(res.status, "done")
        self.assertTrue(any("could not be resolved" in w for w in res.notes["warnings"]))


if __name__ == "__main__":
    unittest.main()
