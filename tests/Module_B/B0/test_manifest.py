import shutil
import tempfile
import unittest
from pathlib import Path

import module_A_tooling.config
from module_B_literature_kg.b0 import manifest as mf
from module_B_literature_kg.b0.manifest import (Manifest, MissingInputError, StepValidationError, atomic_write_json,
                                  hash_path, read_json, run_step)

STEP_A, STEP_B = "b0_seeds", "b0_references"     # any two names that exist in config.STEP_SETTINGS


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.m = Manifest(root=self.tmp)
        self.calls = 0

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def step_a(self, value=1, **kw):
        def work(staging):
            self.calls += 1
            atomic_write_json(staging / "a.json", {"value": value})
            return {"n": 1}
        return run_step(self.m, STEP_A, work, outputs=["a.json"], extra_inputs={"v": value}, **kw)

    def test_second_run_is_skipped(self):
        self.assertEqual(self.step_a().status, "done")
        self.assertEqual(self.step_a().status, "skipped")
        self.assertEqual(self.calls, 1)
        self.assertEqual(read_json(self.tmp / "a.json"), {"value": 1})

    def test_extra_input_change_reruns(self):
        self.step_a(1)
        self.assertEqual(self.step_a(2).status, "done")
        self.assertEqual(read_json(self.tmp / "a.json"), {"value": 2})

    def test_config_change_reruns(self):
        def work(staging):
            self.calls += 1
            atomic_write_json(staging / "b.json", {})
        run_step(self.m, STEP_B, work, outputs=["b.json"])
        old = module_A_tooling.config.MAX_REFERENCES_PER_PAPER
        try:
            module_A_tooling.config.MAX_REFERENCES_PER_PAPER = old + 1
            self.assertEqual(run_step(self.m, STEP_B, work, outputs=["b.json"]).status, "done")
        finally:
            module_A_tooling.config.MAX_REFERENCES_PER_PAPER = old
        self.assertEqual(self.calls, 2)

    def test_input_file_change_reruns_even_if_settings_same(self):
        self.step_a(1)
        def work_b(staging):
            self.calls += 1
            atomic_write_json(staging / "b.json", {"from": read_json(self.tmp / "a.json")})
        kw = dict(outputs=["b.json"], inputs=["a.json"])
        run_step(self.m, STEP_B, work_b, **kw)
        self.assertEqual(run_step(self.m, STEP_B, work_b, **kw).status, "skipped")
        self.step_a(2)                                       # upstream output changes
        self.assertEqual(run_step(self.m, STEP_B, work_b, **kw).status, "done")
        self.assertEqual(read_json(self.tmp / "b.json"), {"from": {"value": 2}})

    def test_tampered_output_reruns(self):
        self.step_a()
        (self.tmp / "a.json").write_text('{"value": 999}')
        self.assertEqual(self.step_a().status, "done")
        self.assertEqual(read_json(self.tmp / "a.json"), {"value": 1})

    def test_crash_after_commit_before_done_is_harmless(self):
        self.step_a()
        before = hash_path(self.tmp / "a.json")
        entry = self.m.get(STEP_A)
        self.m.set(STEP_A, {**entry, "status": "running"})   # what a crash at that moment leaves
        fresh = Manifest(root=self.tmp)                      # a new process reads the file
        res = run_step(fresh, STEP_A, lambda st: atomic_write_json(st / "a.json", {"value": 1}),
                       outputs=["a.json"], extra_inputs={"v": 1})
        self.assertEqual(res.status, "done")
        self.assertEqual(hash_path(self.tmp / "a.json"), before)

    def test_validation_failure_keeps_old_output_and_marks_failed(self):
        self.step_a(1)
        with self.assertRaises(StepValidationError):
            self.step_a(2, validate=lambda staging: ["bad data"])
        self.assertEqual(read_json(self.tmp / "a.json"), {"value": 1})       # old file untouched
        self.assertEqual(self.m.status(STEP_A), "failed")
        self.assertTrue((self.tmp / ".staging" / STEP_A).exists())           # diagnostics kept

    def test_exception_marks_failed(self):
        def boom(staging):
            raise ValueError("boom")
        with self.assertRaises(ValueError):
            run_step(self.m, STEP_A, boom, outputs=["a.json"])
        self.assertEqual(self.m.status(STEP_A), "failed")
        self.assertIn("boom", self.m.get(STEP_A)["error"])

    def test_missing_output_is_a_failure(self):
        with self.assertRaises(StepValidationError):
            run_step(self.m, STEP_A, lambda st: None, outputs=["a.json"])

    def test_missing_input_raises(self):
        with self.assertRaises(MissingInputError):
            run_step(self.m, STEP_B, lambda st: None, outputs=["b.json"], inputs=["nope.json"])

    def test_validate_prior_state_marks_stale(self):
        self.step_a()
        (self.tmp / "a.json").unlink()
        problems = Manifest(root=self.tmp).validate_prior_state()
        self.assertIn(STEP_A, problems)
        self.assertEqual(Manifest(root=self.tmp).status(STEP_A), "stale")

    def test_folder_outputs(self):
        def work(staging):
            (staging / "fulltext").mkdir()
            (staging / "fulltext" / "P1.txt").write_text("hello")
        run_step(self.m, STEP_A, work, outputs=["fulltext"])
        self.assertEqual(run_step(self.m, STEP_A, work, outputs=["fulltext"]).status, "skipped")
        (self.tmp / "fulltext" / "P1.txt").write_text("changed")
        self.assertEqual(run_step(self.m, STEP_A, work, outputs=["fulltext"]).status, "done")
        self.assertEqual((self.tmp / "fulltext" / "P1.txt").read_text(), "hello")

    def test_no_temp_files_left_behind(self):
        self.step_a()
        leftovers = [p.name for p in self.tmp.rglob("*") if ".tmp" in p.name]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
