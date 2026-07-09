# Design: dense-corroborated subject filter

**Date:** 2026-07-10
**Status:** approved (design), pending implementation
**Supersedes:** the `"nhat"` stopword band-aid (PR #5, to be closed unmerged)

## Problem

The pipeline's **subject hard-filter** (`_constraint_filter` in `pipeline.py`) can pin the
entire result set to a single, unrelated POI when a common query word coincidentally appears
in that POI's proper name.

Reported query: **`quan an ngon nhat`** ("best/most-delicious restaurant") returned **Công viên
Thống Nhất — a park** as the only result, despite the category parsing correctly as `Nhà hàng`.

### Root cause (evidence-backed)

The parser marks any residual token that is alphabetic, length ≥ 3, and has corpus
document-frequency `df ∈ [1,2]` as a **distinctive subject** (`content_terms`), intended to catch
dish/brand names ("bún chả", "phở"). It then hard-filters results to POIs whose name/text contains
all such terms.

The heuristic uses **corpus rarity as a proxy for query intent**, and that proxy breaks:

- `"nhất"` (the superlative marker "most/best/-est") is rare in POI *names* (df = 1 — only "Thống
  **Nhất**"), so it qualifies as "distinctive" — even though in a query it is a quality *modifier*,
  never a subject.
- A denylist of such words (the band-aid) can never be complete; it enumerates a language.

**The decisive measurement** — for the same term, the two retrievers disagree exactly where the
heuristic fails:

| Query | Subject token | POI matched | BM25 rank | **Dense rank** |
|---|---|---|---|---|
| `quan an ngon nhat` | `nhat` | Công viên Thống Nhất (park) | **1** | **45 / 111** |
| `cay xang gan nhat` | `nhat` | Công viên Thống Nhất (park) | **1** | **49 / 111** |
| `quan bun cha …` | `bun` / `cha` | Bún Chả Hương Liên | 1 | **1 / 111** |

BM25 ranks the coincidental high-IDF token at #1; the **dense retriever understands the query** and
ranks the spurious match far down, while a **genuine** subject is dense-rank 1. That gap (1 vs 45+)
is the signal.

## Approach

**Semantic corroboration.** A distinctive term is allowed to trigger the subject hard-filter **only
when the dense retriever also ranks a POI matching it near the top** — i.e. keyword *and* meaning
agree the term is the subject. No vocabulary lists; the fix is driven by the retriever the system
already runs, and self-corrects as data changes.

One-line story for judges: *"we enforce a subject constraint only when keyword and semantic search
agree on it."*

### Why not the alternatives (recorded for posterity)

- **A — Vietnamese common-word list.** Principled denylist (vendored frequency/stopword resource),
  but still a word-filter dependent on the list's coverage; needs a vendored asset. Rejected in
  favor of a mechanism fix.
- **C — soft signal only.** Drop the hard filter, make subject match a ranking boost. Most robust to
  collisions and aligns with the pipeline's "no destructive filtering" philosophy, but gives up the
  hard "only bún chả" guarantee and needs broader eval re-validation. Rejected as larger scope;
  B keeps the guarantee where it is genuinely warranted.

## Design

Boundary: the **parser stays retrieval-free** (pure structural parse). The retrieval-dependent
judgment — "is this term corroborated?" — lives in the **pipeline**, where dense ranks already exist.

### 1. `parse.py` — expose full residual

Add one additive field to `QueryIntent`:

```python
residual_terms: list[str]   # every unexplained non-stopword token (⊇ content_terms)
```

`content_terms` (distinctive subset) is unchanged. `residual_terms` lets the pipeline distinguish a
*discredited distinctive term* from *genuine unexplained content* when deciding category eligibility.

### 2. `pipeline.py` — corroboration

`rank_scored` computes `dense_ids` **once** and reuses it for both `_relevance` and corroboration
(no second dense call → no latency regression, G4 safe).

```python
DENSE_SUBJECT_TOPK = 10   # a distinctive subject must have a name-match within the dense top-K

def _corroborated_subjects(self, intent, dense_ids) -> set[str]:
    """Keep only the parser's distinctive content_terms that the DENSE retriever
    corroborates as central — some POI whose content contains the term is in the
    dense top-K. Filters out coincidental high-IDF proper-name collisions (e.g.
    'nhat' in 'Thống Nhất'), which BM25 ranks #1 but dense ranks far down."""
    dense_top = set(dense_ids[:DENSE_SUBJECT_TOPK])
    return {t for t in intent.content_terms
            if any(t in self._content[pid] for pid in dense_top)}
```

### 3. `pipeline.py` — `_constraint_filter` uses corroborated subjects

```python
subject_terms = self._corroborated_subjects(intent, dense_ids)
discredited   = set(intent.content_terms) - subject_terms          # spurious distinctive terms
meaningful_residual = [t for t in intent.residual_terms if t not in discredited]

if intent.district or intent.city:
    filters.append(location_filter)
if subject_terms:                                                  # corroborated subject → hard-filter
    filters.append(lambda pid: subject_terms <= self._content[pid])
elif intent.category and not meaningful_residual:                  # else category, if nothing real is left unexplained
    filters.append(category_filter)
```

Relax-most-specific-first (existing loop) is unchanged.

### Behavior on the key cases

| Query | `content_terms` | corroborated? | filter applied | result |
|---|---|---|---|---|
| `quan an ngon nhat` | `{nhat}` | no (dense 45) | category `Nhà hàng` | restaurants ✓ |
| `cay xang gan nhat` | `{nhat}` | no (dense 49) | category `Trạm xăng` | gas stations ✓ |
| `quán bún chả …` | `{bun,cha}` | yes (dense 1) | subject | only bún chả (R003) ✓ |
| `công viên thống nhất` | (park query) | yes — park *is* the dense-top match for its own name | subject (`thong`, corroborated) or category `Công viên` | park still #1 ✓ |
| `nơi mua sắm có nhiều nhà hàng` (P055) | `{}` | — | none (real residual `mua/sam`) | guard preserved ✓ |

### Threshold justification

`DENSE_SUBJECT_TOPK = 10` ≈ top 9% of 111 POIs. Genuine subjects sit at dense rank 1; spurious ones
at 45+. The 1-vs-45 margin means any K in [5, 30] behaves identically, so this is a **structural
constant**, documented as such, and **not tuned on the eval set** (NFR-6 preserved).

## Scope / non-goals

- **No change** to: the keyword compare-lane (`engine=keyword`, naive BM25 by design), the
  `/v1/search` contract DTOs, ranker weights, or any script-generated report.
- Revert the `"nhat"` stopword addition; leave pre-existing quality-adjective stopwords
  (`ngon`/`tốt`/`rẻ`/…) untouched — they are original design, out of scope.
- No LLM parse, no vendored linguistic resources.

## Testing

Failing-first, at the level the fix operates (pipeline):

- `test_superlative_does_not_hijack_to_proper_name` — `quan an ngon nhat` → all `Nhà hàng`.
- Superlative on another category — `cay xang gan nhat` → all `Trạm xăng`.
- Genuine subject preserved — `quán bún chả …` → top result `R003` (existing
  `test_subject_term_isolates_bun_cha` must stay green).
- Genuine proper-name query preserved — `công viên thống nhất` → the park ranks #1.
- P055-type guard preserved — mis-parse query still not category-filtered.

Replace the parser-level `test_superlative_marker_is_stopword` (obsolete under B — the parser still
emits `content_terms=['nhat']`; corroboration happens in the pipeline).

## Verification gates

1. `uv run python -m pytest -q` green.
2. `uv run python scripts/run_eval.py --engine full --split tune` — NDCG@5 ≥ 0.959 (no regression).
3. Test split evaluated **once** for the milestone (NFR-6): confirm G3 (NDCG@5 ≥ 0.80, Recall@3 ≥ 0.75) holds.
4. Latency spot-check — warm p95 still < 200 ms (G4).
5. End-to-end manual: the five rows in the behavior table above.

## Delivery

Fresh branch off `develop` (`fix/subject-corroboration`); PR into `develop`. Close PR #5 unmerged.
Commit style follows the repo: `fix(rank): …` with the numbers.
