"""module_a/cache.py (A1): a notebook that remembers answers, so we never ask the same question twice.

Before calling the internet or an LLM, look in the notebook. If the answer is there, reuse it.
If not, ask, then write the answer down. It is one SQLite file (cache.db), safe to delete at any time.

Typical use
    cache = get_default_cache()
    key = make_key("s2.paper", {"arxiv_id": "2305.12345"})
    paper = cache.get_or_compute(key, lambda: fetch_paper("2305.12345"), kind="s2")

Values must be JSON-friendly: str, numbers, bool, None, lists, dicts. pydantic models, numpy arrays,
sets and Paths are converted automatically (a model comes back as a plain dict).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional, Union

from .config import CACHE_DB_PATH, CACHE_TTL_SECONDS

_UNSET: Any = object()   # "caller did not say": use the default expiry of the kind


def _json_fallback(obj: Any) -> Any:
    """Turn the common non-JSON objects into JSON. Anything else raises, so a bad key is never silent."""
    if hasattr(obj, "model_dump"):                    # pydantic model
        return obj.model_dump(mode="json")
    if hasattr(obj, "tolist"):                        # numpy array or scalar
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=repr)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"cannot cache an object of type {type(obj).__name__}")


def hash_text(text: str) -> str:
    """Fingerprint of a prompt or any long text, to put into a key."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_key(function: str, inputs: Any, *, model: Optional[str] = None,
             revision: Optional[str] = None, prompt_hash: Optional[str] = None) -> str:
    """Cache key = function + inputs + model + revision + prompt hash (the agreed B0 rule).
    Same ingredients always give the same key; changing any ingredient gives a new key."""
    payload = {"fn": function, "inputs": inputs, "model": model,
               "revision": revision, "prompt": prompt_hash}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=_json_fallback)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class Cache:
    def __init__(self, path: Union[str, Path] = CACHE_DB_PATH,
                 default_ttls: Optional[dict[str, Optional[float]]] = None,
                 clock: Callable[[], float] = time.time):
        """path ':memory:' gives a throw-away cache (handy in tests). clock can be faked in tests."""
        self._ttls = dict(CACHE_TTL_SECONDS if default_ttls is None else default_ttls)
        self._clock = clock
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        with self._lock, self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS cache ("
                " key TEXT PRIMARY KEY, kind TEXT NOT NULL, value TEXT NOT NULL,"
                " created_at REAL NOT NULL, expires_at REAL)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_kind ON cache(kind)")

    # ---------------------------------------------------------------- reading
    def _lookup(self, key: str) -> tuple[bool, Any]:
        """(found, value). Expired or unreadable rows count as not found and are removed."""
        with self._lock:
            row = self._conn.execute(
                "SELECT value, expires_at FROM cache WHERE key = ?", (key,)).fetchone()
            if row is None:
                self.misses += 1
                return False, None
            value_text, expires_at = row
            if expires_at is not None and expires_at <= self._clock():
                self.delete(key)
                self.misses += 1
                return False, None
            try:
                value = json.loads(value_text)
            except json.JSONDecodeError:
                self.delete(key)
                self.misses += 1
                return False, None
            self.hits += 1
            return True, value

    def get(self, key: str, default: Any = None) -> Any:
        """The stored value, or `default` if missing/expired. Use contains() to tell a stored None apart."""
        found, value = self._lookup(key)
        return value if found else default

    def contains(self, key: str) -> bool:
        return self._lookup(key)[0]

    # ---------------------------------------------------------------- writing
    def set(self, key: str, value: Any, *, kind: str = "", ttl_seconds: Any = _UNSET) -> None:
        """Store a value. ttl_seconds: omit = default for this kind, None = never expires, number = seconds."""
        text = json.dumps(value, ensure_ascii=False, default=_json_fallback)   # fails before touching the DB
        ttl = self._ttls.get(kind) if ttl_seconds is _UNSET else ttl_seconds
        now = self._clock()
        expires_at = None if ttl is None else now + ttl
        with self._lock, self._conn:   # one transaction: all or nothing
            self._conn.execute(
                "INSERT OR REPLACE INTO cache (key, kind, value, created_at, expires_at)"
                " VALUES (?, ?, ?, ?, ?)", (key, kind, text, now, expires_at))

    def get_or_compute(self, key: str, compute: Callable[[], Any], *, kind: str = "",
                       ttl_seconds: Any = _UNSET) -> Any:
        """Return the stored value, or run compute(), store the result and return it.
        If compute() raises, nothing is stored and the error passes through."""
        found, value = self._lookup(key)
        if found:
            return value
        value = compute()
        self.set(key, value, kind=kind, ttl_seconds=ttl_seconds)
        return json.loads(json.dumps(value, ensure_ascii=False, default=_json_fallback))

    # ---------------------------------------------------------------- housekeeping
    def delete(self, key: str) -> bool:
        with self._lock, self._conn:
            return self._conn.execute("DELETE FROM cache WHERE key = ?", (key,)).rowcount > 0

    def purge_expired(self) -> int:
        """Remove all expired rows. Returns how many."""
        with self._lock, self._conn:
            return self._conn.execute(
                "DELETE FROM cache WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (self._clock(),)).rowcount

    def clear(self, kind: Optional[str] = None) -> int:
        """Remove everything, or only one kind. Returns how many rows were removed."""
        with self._lock, self._conn:
            if kind is None:
                return self._conn.execute("DELETE FROM cache").rowcount
            return self._conn.execute("DELETE FROM cache WHERE kind = ?", (kind,)).rowcount

    def stats(self) -> dict[str, Any]:
        with self._lock:
            by_kind = dict(self._conn.execute(
                "SELECT kind, COUNT(*) FROM cache GROUP BY kind").fetchall())
        return {"entries": sum(by_kind.values()), "by_kind": by_kind,
                "hits": self.hits, "misses": self.misses}

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "Cache":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


_default: Optional[Cache] = None


def get_default_cache() -> Cache:
    """The shared cache at config.CACHE_DB_PATH, opened on first use."""
    global _default
    if _default is None:
        _default = Cache()
    return _default
