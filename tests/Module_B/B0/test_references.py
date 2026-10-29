import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from module_A_tooling.cache import Cache
from module_B_literature_kg.b0 import clients
from module_B_literature_kg.b0.clients import S2Client
from module_B_literature_kg.b0.manifest import Manifest, MissingInputError, atomic_write_json, hash_path, read_json
from module_B_literature_kg.b0.references import ReferencesUnavailableError, fetch_references, run_references_step
from tests.Module_B.B0.fakes import FakeResponse, FakeSession, s2_raw


def seeds_doc(*pairs):
    """pairs of (s2_id or None, title)"""
    seeds = []
    for pid, title in pairs:
        paper = clients.new_paper(s2_id=pid, title=title, arxiv_id=None if pid else "2301.00001")
        paper["is_seed"] = True
        seeds.append({"input": title, "status": "resolved", "paper": paper, "flags": []})
    return {"seeds": seeds}


def refs_router(table):
    """table: {seed s2 id: [raw cited papers]}; one page per seed."""
    def router(path, params):
        sid = path.split("/")[2]
        data = [{"citedPaper": c} for c in table[sid]]
        return FakeResponse(200, {"offset": 0, "data": data})
    return router


def make_s2(router):
    return S2Client(cache=Cache(":memory:"), session=FakeSession(router), api_key=None,
                    min_interval=0.0, sleep=lambda s: None)


class FetchTests(unittest.TestCase):
    def test_edges_stubs_and_overlap(self):
        table = {"A": [s2_raw("x", "X"), s2_raw("y", "Y", abstract="")],
                 "B": [s2_raw("y", "Y", abstract="has abstract now"), s2_raw("A", "Seed A"), s2_raw("z", "Z")]}
        doc = fetch_references(seeds_doc(("A", "Seed A"), ("B", "Seed B")), make_s2(refs_router(table)), cap=10)
        self.assertEqual(set(doc["stubs"]), {"s2:x", "s2:y", "s2:z"})            # seed A is not a stub
        self.assertEqual(doc["stubs"]["s2:y"]["abstract"], "has abstract now")    # richer copy wins
        self.assertEqual(doc["stubs"]["s2:y"]["cited_by_seeds"], ["s2:A", "s2:B"])
        pairs = [(e["source"], e["target"]) for e in doc["edges"]]
        self.assertEqual(pairs, sorted(set(pairs)))                              # sorted, no repeats
        self.assertIn(("s2:B", "s2:A"), pairs)                                   # seed cites seed: edge kept
        self.assertTrue(all(e["kind"] == "cites" for e in doc["edges"]))

    def test_self_citation_is_ignored(self):
        table = {"A": [s2_raw("A", "Seed A"), s2_raw("x", "X")]}
        doc = fetch_references(seeds_doc(("A", "Seed A")), make_s2(refs_router(table)), cap=10)
        self.assertEqual([(e["source"], e["target"]) for e in doc["edges"]], [("s2:A", "s2:x")])

    def test_partial_flag_when_cut_by_cap(self):
        with mock.patch.object(clients, "S2_REFERENCES_PAGE_SIZE", 2):
            def router(path, params):
                off, lim, total = params["offset"], params["limit"], 6
                data = [{"citedPaper": s2_raw(f"r{i}", f"R{i}")} for i in range(off, min(off + lim, total))]
                body = {"data": data}
                if off + lim < total:
                    body["next"] = off + lim
                return FakeResponse(200, body)
            doc = fetch_references(seeds_doc(("A", "Seed A")), make_s2(router), cap=3)
        self.assertTrue(doc["by_seed"]["s2:A"]["partial"])
        self.assertEqual(len(doc["stubs"]), 3)
        self.assertTrue(any("partial" in w for w in doc["warnings"]))

    def test_seed_without_s2_id_is_unavailable(self):
        table = {"A": [s2_raw("x", "X")]}
        doc = fetch_references(seeds_doc(("A", "Seed A"), (None, "arXiv-only seed")),
                               make_s2(refs_router(table)), cap=10)
        self.assertEqual(doc["by_seed"]["arxiv:2301.00001"]["status"], "unavailable")
        self.assertEqual(doc["by_seed"]["s2:A"]["status"], "ok")

    def test_error_for_one_seed_is_recorded_and_others_continue(self):
        def router(path, params):
            if path.split("/")[2] == "BAD":
                return FakeResponse(500)
            return FakeResponse(200, {"data": [{"citedPaper": s2_raw("x", "X")}]})
        doc = fetch_references(seeds_doc(("BAD", "Bad"), ("A", "Seed A")), make_s2(router), cap=10)
        self.assertEqual(doc["by_seed"]["s2:BAD"]["status"], "error")
        self.assertEqual(doc["by_seed"]["s2:A"]["status"], "ok")


class StepTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.m = Manifest(root=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_needs_seeds_first(self):
        with self.assertRaises(MissingInputError):
            run_references_step(self.m, make_s2(refs_router({})))

    def test_step_runs_skips_and_reruns_when_seeds_change(self):
        atomic_write_json(self.tmp / "seeds.json", seeds_doc(("A", "Seed A")))
        s2 = make_s2(refs_router({"A": [s2_raw("x", "X")], "B": [s2_raw("y", "Y")]}))
        self.assertEqual(run_references_step(self.m, s2, cap=10).status, "done")
        self.assertEqual(run_references_step(self.m, s2, cap=10).status, "skipped")
        atomic_write_json(self.tmp / "seeds.json", seeds_doc(("A", "Seed A"), ("B", "Seed B")))
        self.assertEqual(run_references_step(self.m, s2, cap=10).status, "done")
        self.assertEqual(set(read_json(self.tmp / "references.json")["stubs"]), {"s2:x", "s2:y"})

    def test_cap_change_reruns(self):
        atomic_write_json(self.tmp / "seeds.json", seeds_doc(("A", "Seed A")))
        s2 = make_s2(refs_router({"A": [s2_raw("x", "X")]}))
        run_references_step(self.m, s2, cap=10)
        self.assertEqual(run_references_step(self.m, s2, cap=5).status, "done")

    def test_rerun_after_simulated_crash_gives_identical_file(self):
        atomic_write_json(self.tmp / "seeds.json", seeds_doc(("A", "Seed A")))
        s2 = make_s2(refs_router({"A": [s2_raw("x", "X"), s2_raw("y", "Y")]}))
        run_references_step(self.m, s2, cap=10)
        before = hash_path(self.tmp / "references.json")
        self.m.set("b0_references", {**self.m.get("b0_references"), "status": "running"})
        self.assertEqual(run_references_step(Manifest(root=self.tmp), s2, cap=10).status, "done")
        self.assertEqual(hash_path(self.tmp / "references.json"), before)       # no duplicates, same bytes

    def test_no_reference_list_at_all_fails_the_step(self):
        atomic_write_json(self.tmp / "seeds.json", seeds_doc((None, "arXiv-only seed")))
        with self.assertRaises(ReferencesUnavailableError):
            run_references_step(self.m, make_s2(refs_router({})), cap=10)
        self.assertEqual(self.m.status("b0_references"), "failed")


if __name__ == "__main__":
    unittest.main()
