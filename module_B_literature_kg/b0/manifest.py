"""module_B_literature_kg/b0/manifest.py: the step wrapper every B0 step runs inside.

Plain-language idea
    A run is a series of steps. Each step reads some files and writes some files. The manifest is a
    notebook (runs/<run_id>/manifest.json) that remembers, per step: did it finish, with which
    settings, and on which input files. That lets us:
      * skip a step when nothing changed (same settings, same inputs, outputs still intact),
      * re-run a step when a setting OR an input file changed (stale-skip protection),
      * survive a crash: a step writes into a staging folder first, is checked, then moved into
        place, and only then marked "done" (LAST). A crash at any moment leaves either the old
        files or the new files, never a half-written mix.

Typical use
    manifest = Manifest(run_id="run_001")
    manifest.validate_prior_state()
    def work(staging):                           # write the output files into `staging`
        atomic_write_json(staging / "seeds.json", {...})
        return {"n_seeds": 3}                    # optional notes, saved in the manifest
    run_step(manifest, "b0_seeds", work, outputs=["seeds.json"], extra_inputs={"seeds": [...]})

Rules a step function must follow (so that running it twice gives the same result: "idempotent")
    * write only into the staging folder it is given,
    * store records by key (upsert), never append to a file that already holds earlier results.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from module_A_tooling.config import DEFAULT_RUN_ID, RUN_FILES, config_hash, run_dir

MANIFEST_VERSION = 1
STAGING_DIRNAME = ".staging"


class StepError(RuntimeError):
    """Base class for step problems."""


class MissingInputError(StepError):
    """A file the step needs (written by an earlier step) is not there."""


class StepValidationError(StepError):
    """The step produced output that failed its checks. Nothing was committed."""


# --------------------------------------------------------------------------- hashing and safe writes

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_json(obj: Any) -> str:
    """Fingerprint of any JSON-friendly value (key order does not matter)."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return sha256_bytes(blob.encode("utf-8"))


def hash_path(path: Path) -> Optional[str]:
    """Fingerprint of a file, or of a whole folder (names + contents). None if it does not exist."""
    path = Path(path)
    if not path.exists():
        return None
    if path.is_file():
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    h = hashlib.sha256()
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(str(child.relative_to(path)).encode("utf-8"))
        h.update(b"\0")
        h.update((hash_path(child) or "").encode("ascii"))
        h.update(b"\0")
    return h.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    """Write to a temporary file next to the target, then rename. A crash never leaves a half file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Same idea as atomic_write_text, for binary files (downloaded PDFs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2))


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


# --------------------------------------------------------------------------- the manifest file

class Manifest:
    """runs/<run_id>/manifest.json. Per step: status, fingerprint, output hashes, notes.

    status is one of: "running" (started, not finished: also what a crash leaves behind),
    "done", "failed", "stale" (was done, but its outputs changed or vanished afterwards).
    `root` lets tests point at a temporary folder instead of runs/<run_id>/.
    """

    def __init__(self, run_id: str = DEFAULT_RUN_ID, root: Optional[Path] = None):
        self.run_id = run_id
        self.root = Path(root) if root is not None else run_dir(run_id)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / RUN_FILES["manifest"]
        data = read_json(self.path, None)
        if not isinstance(data, dict) or data.get("version") != MANIFEST_VERSION:
            data = {"version": MANIFEST_VERSION, "steps": {}}
        self._data = data

    def get(self, step: str) -> dict:
        return dict(self._data["steps"].get(step, {}))

    def set(self, step: str, entry: dict) -> None:
        self._data["steps"][step] = entry
        atomic_write_json(self.path, self._data)

    def status(self, step: str) -> Optional[str]:
        return self._data["steps"].get(step, {}).get("status")

    def steps(self) -> dict:
        return {k: dict(v) for k, v in self._data["steps"].items()}

    def output_problem(self, step: str) -> Optional[str]:
        """Read-only check of one step: None if it is done and its output files still match the
        recorded hashes, otherwise a short reason. (validate_prior_state also MARKS bad steps stale.)"""
        entry = self._data["steps"].get(step)
        if entry is None:
            return "step has not run"
        if entry.get("status") != "done":
            return f"step status is {entry.get('status')!r}, not 'done'"
        return _output_problem(self.root, entry.get("output_hashes", {}))

    def validate_prior_state(self) -> dict[str, str]:
        """Check every step marked done: are its output files still there and unchanged?
        Steps that fail the check are marked "stale" so they re-run. Returns {step: reason}."""
        problems: dict[str, str] = {}
        for step, entry in list(self._data["steps"].items()):
            if entry.get("status") != "done":
                continue
            reason = _output_problem(self.root, entry.get("output_hashes", {}))
            if reason:
                problems[step] = reason
                self.set(step, {**entry, "status": "stale", "stale_reason": reason})
        return problems


# --------------------------------------------------------------------------- running a step

@dataclass
class StepResult:
    step: str
    status: str                       # "done" (ran now) or "skipped" (nothing changed)
    notes: dict = field(default_factory=dict)
    output_hashes: dict = field(default_factory=dict)


def make_fingerprint(step: str, root: Path, inputs: Iterable[str],
                     extra_inputs: Optional[Any] = None) -> dict:
    """What a step's result depends on: its settings + its input files + any extra input values."""
    hashes: dict[str, Optional[str]] = {}
    for name in sorted(inputs):
        h = hash_path(root / name)
        if h is None:
            raise MissingInputError(f"step {step!r} needs {name!r}, which does not exist in {root}")
        hashes[name] = h
    return {"config": config_hash(step), "inputs": hashes,
            "extra": None if extra_inputs is None else hash_json(extra_inputs)}


def _output_problem(root: Path, expected: dict) -> Optional[str]:
    for name, want in expected.items():
        have = hash_path(root / name)
        if have is None:
            return f"output {name!r} is missing"
        if have != want:
            return f"output {name!r} changed after the step finished"
    return None


def _can_skip(entry: dict, fingerprint: dict, root: Path, outputs: list[str]) -> bool:
    if entry.get("status") != "done" or entry.get("fingerprint") != fingerprint:
        return False
    expected = entry.get("output_hashes", {})
    if set(expected) != set(outputs):
        return False
    return _output_problem(root, expected) is None


def _commit(staging: Path, root: Path, outputs: list[str]) -> None:
    """Move each output from staging into the run folder. Each move is atomic for files."""
    for name in outputs:
        src, dst = staging / name, root / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            backup = None
            if dst.exists():
                backup = dst.with_name(dst.name + ".old")
                if backup.exists():
                    shutil.rmtree(backup)
                os.replace(dst, backup)
            os.replace(src, dst)
            if backup is not None:
                shutil.rmtree(backup, ignore_errors=True)
        else:
            os.replace(src, dst)


def run_step(manifest: Manifest, step: str, fn: Callable[[Path], Optional[dict]], *,
             outputs: list[str], inputs: Iterable[str] = (), extra_inputs: Optional[Any] = None,
             validate: Optional[Callable[[Path], list[str]]] = None,
             force: bool = False) -> StepResult:
    """Run one step safely.

    fn(staging)   writes every file listed in `outputs` into the staging folder and may return a dict
                  of notes (warnings, counts). Raising an exception means the step failed.
    inputs        names of files in the run folder (written by earlier steps) this step reads.
    extra_inputs  any other JSON-friendly value the result depends on (e.g. the seed list).
    validate      optional check on the staging folder; returns a list of problems (empty = fine).
    """
    root = manifest.root
    fingerprint = make_fingerprint(step, root, inputs, extra_inputs)
    entry = manifest.get(step)
    if not force and _can_skip(entry, fingerprint, root, outputs):
        return StepResult(step, "skipped", entry.get("notes", {}), entry.get("output_hashes", {}))

    staging = root / STAGING_DIRNAME / step
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    started = time.time()
    manifest.set(step, {"status": "running", "fingerprint": fingerprint, "started_at": started})

    try:
        notes = fn(staging) or {}
        problems = list(validate(staging)) if validate else []
        problems += [f"step did not write {name!r}" for name in outputs if not (staging / name).exists()]
        if problems:
            raise StepValidationError(f"step {step!r} failed its checks: " + "; ".join(problems))
        _commit(staging, root, outputs)
        output_hashes = {name: hash_path(root / name) for name in outputs}
        # "done" is written LAST, after the outputs are validated and committed.
        manifest.set(step, {"status": "done", "fingerprint": fingerprint,
                            "output_hashes": output_hashes, "notes": notes,
                            "started_at": started, "finished_at": time.time()})
        shutil.rmtree(staging, ignore_errors=True)
        return StepResult(step, "done", notes, output_hashes)
    except Exception as exc:
        # keep the staging folder: it holds diagnostics. The next run of this step clears it.
        manifest.set(step, {"status": "failed", "fingerprint": fingerprint,
                            "error": f"{type(exc).__name__}: {exc}", "started_at": started,
                            "finished_at": time.time()})
        raise
