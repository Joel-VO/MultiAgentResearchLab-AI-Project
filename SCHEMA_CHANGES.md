# schemas.py: what changed and why (A0)

Rule used: schemas.py is the project overview; interface.py is the Module B contract. Where they
disagreed, interface.py won and schemas.py now mirrors it. `tests/test_schema_sync.py` keeps them identical.

## Changes
| What | Before (schemas.py) | After | Reason |
|---|---|---|---|
| `Paper` | S2 paperId as id, `peer_reviewed`, `depth`, `relevance`, ... | replaced by `PaperMeta` (internal id `P1`, plus `s2_id`, `arxiv_id`, `doi`, `is_seed`, `has_full_text`) | interface.py wins |
| `KBHandle` | `papers_path`, `embedding_model`, `n_papers` | `meta_path`, `reviews_path`, `topic_emb_path` | interface.py wins |
| `PaperReview` | `paper_id`, `source` + draft fields | `paper_id`, `title`, draft fields, `evidence_quotes`; no `source`; defaults `"not stated"` / `[]` | interface.py wins |
| `PaperReviewDraft` | no defaults | same fields, with the interface defaults | stays (LLM output shape), aligned |
| `Conflict` | `id` auto-generated, `score_variance` | `id` required; adds `shared_dataset`, `shared_metric`, `claim_a`, `claim_b`; no `score_variance` | interface.py wins |
| `ConflictCheck` | 3 fields | + 4 optional extras (same names as `Conflict`) | lets B4a fill the extras |
| Paper ID prefix | Semantic Scholar paperId | `P1, P2, ...` internal | interface.py wins |
| Letters | tooling = "Module B", literature = "Module A" | tooling = A, literature = B (headers, `LabState` comments, `"B.conflicts"` example) | matches the letter swap |
| `LedgerEntry` and everything else | | unchanged | used as is |

## Three things I did NOT change in interface.py (your call)
1. `module_B_literature_kg_node` returns the key `"reviews"`. In `LabState`, `reviews` is the peer-review list (Module E),
   and paper summaries live under `paper_reviews` (a dict keyed by paper id). Suggested fix, one line:
   `return {"kb": res.kb, "paper_reviews": {r.paper_id: r for r in res.reviews}, "conflicts": {c.id: c for c in res.conflicts}}`
   (`conflicts` is also a dict keyed by id in `LabState`).
2. `Conflict` has no place for the reviewer score variance that B4b computes. Suggest `score_variance: Optional[float] = None`
   in interface.py, then copy it to schemas.py (the sync test will remind you).
3. `PaperReview` has no `source` ("abstract" or "fulltext"). Optional: add `source: Literal["abstract","fulltext"] = "abstract"` to both files.

## Push it to GitHub
```
git checkout main && git pull
git checkout -b a0-a1-config-cache-schemas
git apply --check schemas.patch && git apply schemas.patch    # or just overwrite schemas.py with the new file
# copy in: config.py  module_a/__init__.py  module_a/cache.py  tests/  .gitignore (lines below)
python -m unittest discover tests -v
git add schemas.py config.py module_a tests .gitignore
git commit -m "A0: config + schemas aligned with interface.py; A1: SQLite cache"
git push -u origin a0-a1-config-cache-schemas
```
Then open a pull request on GitHub. `test_schema_sync` is skipped until `interface.py` is in the same checkout
(it activates after the Module B branch is merged). `.gitignore` lines: `cache.db*`, `runs/`, `__pycache__/`.
