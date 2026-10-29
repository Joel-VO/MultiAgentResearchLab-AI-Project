import shutil
import tempfile
import unittest
from pathlib import Path

import module_A_tooling.config
from module_B_literature_kg.b0.clients import new_paper
from module_B_literature_kg.b0.dedupe import (build_canonical, dedupe, find_decisions, first_author_surname,
                                collect_records, run_dedupe_step)
from module_B_literature_kg.b0.manifest import Manifest, MissingInputError, atomic_write_json, hash_path, read_json

LONG = "A Long Enough Title About Language Models"


def paper(title=LONG, authors=("Ada Lovelace",), year=2020, is_seed=False, **ids):
    p = new_paper(title=title, authors=list(authors), year=year, **ids)
    p["is_seed"] = is_seed
    return p


def seeds_of(*papers):
    return {"seeds": [{"input": p["title"], "status": "resolved", "paper": p, "flags": []} for p in papers]}


def refs_of(stubs, edges=()):
    return {"stubs": {s["key"]: s for s in stubs},
            "edges": [{"source": a, "target": b, "kind": "cites"} for a, b in edges]}


class NameTests(unittest.TestCase):
    def test_surname(self):
        self.assertEqual(first_author_surname(["Ada Lovelace"]), "lovelace")
        self.assertEqual(first_author_surname(["Lovelace, Ada"]), "lovelace")
        self.assertEqual(first_author_surname(["Zoë Müller"]), "muller")
        self.assertEqual(first_author_surname([]), "")


class DecisionTests(unittest.TestCase):
    def decide(self, *records, **kw):
        return find_decisions(list(records), **kw)

    def test_exact_ids_merge_with_full_confidence(self):
        a = paper(arxiv_id="2305.12345", is_seed=True)                     # key arxiv:2305.12345
        b = paper(title="Different words entirely here", s2_id="s2b", arxiv_id="2305.12345", authors=("Bob Roe",))
        (d,) = self.decide(a, b)
        self.assertEqual((d["rule"], d["status"], d["confidence"]), ("exact_id", "merged", 1.0))
        self.assertIn("same arxiv_id", d["evidence"])

    def test_preprint_and_published_version_merge_on_title_author_year(self):
        pre = paper(arxiv_id="2001.00001", year=2020)
        pub = paper(doi="10.1000/pub", year=2021)
        (d,) = self.decide(pre, pub)
        self.assertEqual((d["rule"], d["status"], d["confidence"]), ("title_author_year", "merged", 0.9))

    def test_years_too_far_apart_is_only_a_candidate(self):
        (d,) = self.decide(paper(arxiv_id="2001.00001", year=2018), paper(doi="10.1000/p", year=2021))
        self.assertEqual(d["status"], "candidate")
        self.assertIn("years differ", d["evidence"])

    def test_same_title_different_authors_is_never_merged(self):
        (d,) = self.decide(paper(arxiv_id="2001.00001"), paper(doi="10.1/x", authors=("Bob Roe",)))
        self.assertEqual(d["status"], "candidate")

    def test_unknown_authors_or_year_cannot_confirm(self):
        (d,) = self.decide(paper(arxiv_id="2001.00001", authors=()), paper(doi="10.1/x"))
        self.assertEqual(d["status"], "candidate")
        (d,) = self.decide(paper(arxiv_id="2001.00001", year=None), paper(doi="10.1/x"))
        self.assertEqual(d["status"], "candidate")

    def test_generic_short_title_never_matches(self):
        self.assertEqual(self.decide(paper(title="Introduction", arxiv_id="2001.00001"),
                                     paper(title="Introduction", doi="10.1/x")), [])

    def test_near_identical_titles_are_candidates_only(self):
        a = paper(title="Attention Is All You Need For Language Modelling", arxiv_id="2001.00001")
        b = paper(title="Attention Is All You Need For Language Modeling", doi="10.1/x", authors=("Bob Roe",))
        (d,) = self.decide(a, b)
        self.assertEqual((d["status"], d["rule"]), ("candidate", "similar_title"))
        self.assertGreaterEqual(d["confidence"], module_A_tooling.config.DEDUPE_TITLE_SIM)

    def test_different_titles_give_no_decision(self):
        self.assertEqual(self.decide(paper(arxiv_id="2001.00001"),
                                     paper(title="Something Else Altogether About Vision", doi="10.1/x")), [])

    def test_undo_override_cancels_a_merge(self):
        a, b = paper(arxiv_id="2001.00001"), paper(doi="10.1000/pub")
        (d,) = self.decide(a, b, undo=[(b["key"], a["key"])])                  # order does not matter
        self.assertEqual(d["status"], "undone")
        canonical, alias = build_canonical([a, b], [d])
        self.assertEqual(len(canonical), 2)

    def test_manual_merge_promotes_a_candidate(self):
        a, b = paper(arxiv_id="2001.00001", year=2010), paper(doi="10.1/x", year=2021)
        (d,) = self.decide(a, b, force_merge=[(a["key"], b["key"])])
        self.assertEqual((d["status"], d["rule"]), ("merged", "manual"))


class MergeTests(unittest.TestCase):
    def test_ids_seeds_first_then_references_and_alias_map_covers_everything(self):
        seed = paper(title="Seed One Paper About Planning Agents", s2_id="S1", is_seed=True, authors=("A Seed",))
        r1 = paper(title="Reference Alpha About Vision Transformers", s2_id="R1", authors=("B Ref",))
        r2 = paper(title="Reference Beta About Speech Recognition", s2_id="R2", authors=("C Ref",))
        out = dedupe(seeds_of(seed), refs_of([r1, r2], [(seed["key"], r1["key"]), (seed["key"], r2["key"])]))
        alias = out["alias_map"]
        self.assertEqual([alias[seed["key"]], alias[r1["key"]], alias[r2["key"]]], ["P1", "P2", "P3"])
        self.assertTrue(out["canonical"]["papers"]["P1"]["is_seed"])
        self.assertEqual(out["canonical"]["papers"]["P2"]["cited_by_seeds"], ["P1"])

    def test_merged_record_fills_blanks_and_seed_wins(self):
        seed = paper(arxiv_id="2001.00001", is_seed=True, abstract="", citation_count=3)
        stub = paper(s2_id="X", arxiv_id="2001.00001", abstract="Abstract from the stub.", citation_count=9,
                     venue="NeurIPS", year=2020)
        out = dedupe(seeds_of(seed), refs_of([stub]))
        (p,) = out["canonical"]["papers"].values()
        self.assertEqual((p["is_seed"], p["abstract"], p["citation_count"], p["venue"], p["s2_id"]),
                         (True, "Abstract from the stub.", 9, "NeurIPS", "X"))
        self.assertEqual(p["key"], seed["key"])                                 # the seed is the primary record
        self.assertEqual(sorted(p["aliases"]), sorted([seed["key"], stub["key"]]))

    def test_identifier_conflict_is_reported_not_hidden(self):
        a = paper(s2_id="A", doi="10.1/same", arxiv_id="2001.00001", is_seed=True)
        b = paper(title="Other Title Words Entirely Different", s2_id="B", doi="10.1/same", arxiv_id="2002.00002")
        out = dedupe(seeds_of(a), refs_of([b]))
        self.assertEqual(len(out["canonical"]["papers"]), 1)
        self.assertTrue(any("disagree on identifiers" in w for w in out["warnings"]))

    def test_transitive_merge(self):
        a = paper(arxiv_id="2001.00001", is_seed=True)
        b = paper(arxiv_id="2001.00001", doi="10.1/x", title="Another Different Title For The Same Work", s2_id="B")
        c = paper(doi="10.1/x", title="Third Title For The Same Work Again", s2_id="C")
        out = dedupe(seeds_of(a), refs_of([b, c]))
        self.assertEqual(len(out["canonical"]["papers"]), 1)

    def test_edges_are_remapped_and_cleaned(self):
        seed = paper(s2_id="S1", is_seed=True)
        dup = paper(title="Different Title Words For Reference", s2_id="D", arxiv_id="2001.00001")
        same = paper(title="Another Different Title For Reference", s2_id="E", arxiv_id="2001.00001")
        other = paper(title="Unrelated Paper About Graph Networks", s2_id="O", authors=("Z Zed",))
        edges = [(seed["key"], dup["key"]), (seed["key"], same["key"]),       # both become the same target
                 (seed["key"], seed["key"]),                                  # self edge
                 (seed["key"], other["key"])]
        out = dedupe(seeds_of(seed), refs_of([dup, same, other], edges))
        self.assertEqual([(e["source"], e["target"]) for e in out["edges"]], [("P1", "P2"), ("P1", "P3")])

    def test_numbering_is_deterministic(self):
        def build():
            seed = paper(s2_id="S1", is_seed=True)
            stubs = [paper(title=f"Reference number {i} about topic {i}", s2_id=f"R{i}", authors=(f"X{i} Y{i}",)) for i in range(5)]
            return dedupe(seeds_of(seed), refs_of(stubs))
        self.assertEqual(build()["alias_map"], build()["alias_map"])

    def test_duplicate_keys_are_rejected(self):
        a = paper(s2_id="S1", is_seed=True)
        with self.assertRaises(ValueError):
            collect_records(seeds_of(a), refs_of([a]))


class StepTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.m = Manifest(root=self.tmp)
        self.seed = paper(s2_id="S1", arxiv_id="2001.00001", is_seed=True)
        self.pub = paper(doi="10.1000/pub", year=2021, s2_id="PUB")
        self.other = paper(title="Unrelated Paper About Graph Networks", s2_id="O", authors=("Z Zed",))
        atomic_write_json(self.tmp / "seeds.json", seeds_of(self.seed))
        atomic_write_json(self.tmp / "references.json", refs_of(
            [self.pub, self.other], [(self.seed["key"], self.pub["key"]), (self.seed["key"], self.other["key"])]))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_step_runs_skips_and_writes_all_outputs(self):
        res = run_dedupe_step(self.m)
        self.assertEqual((res.status, res.notes["n_papers"], res.notes["merged"]), ("done", 2, 1))
        self.assertEqual(run_dedupe_step(self.m).status, "skipped")
        log = read_json(self.tmp / "dedupe_log.json")
        self.assertEqual(log["decisions"][0]["rule"], "title_author_year")
        alias = read_json(self.tmp / "alias_map.json")
        self.assertEqual(alias[self.pub["key"]], alias[self.seed["key"]])      # originals still findable
        self.assertEqual([(e["source"], e["target"]) for e in read_json(self.tmp / "citation_edges.json")], [("P1", "P2")])

    def test_override_file_undoes_a_merge_and_reruns(self):
        run_dedupe_step(self.m)
        atomic_write_json(self.tmp / "dedupe_overrides.json", {"undo": [[self.seed["key"], self.pub["key"]]]})
        self.assertEqual(run_dedupe_step(self.m).status, "done")
        self.assertEqual(len(read_json(self.tmp / "canonical_papers.json")["papers"]), 3)
        self.assertEqual(read_json(self.tmp / "dedupe_log.json")["decisions"][0]["status"], "undone")

    def test_changed_reference_input_reruns(self):
        run_dedupe_step(self.m)
        atomic_write_json(self.tmp / "references.json", refs_of([self.other]))
        self.assertEqual(run_dedupe_step(self.m).status, "done")

    def test_rerun_after_simulated_crash_is_identical(self):
        run_dedupe_step(self.m)
        before = {n: hash_path(self.tmp / n) for n in ("canonical_papers.json", "alias_map.json", "dedupe_log.json", "citation_edges.json")}
        self.m.set("b0_dedupe", {**self.m.get("b0_dedupe"), "status": "running"})
        self.assertEqual(run_dedupe_step(Manifest(root=self.tmp)).status, "done")
        self.assertEqual(before, {n: hash_path(self.tmp / n) for n in before})

    def test_needs_earlier_steps(self):
        (self.tmp / "references.json").unlink()
        with self.assertRaises(MissingInputError):
            run_dedupe_step(self.m)


if __name__ == "__main__":
    unittest.main()
