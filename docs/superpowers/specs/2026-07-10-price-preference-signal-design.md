# Design: price-preference ranking signal

**Date:** 2026-07-10
**Status:** approved (design), pending implementation
**Related:** builds on the 8-signal ranker; adds a 9th signal.

## Problem

The query `cafe rẻ nhất` ("cheapest café") returns cafés (category correct, after the
subject-corroboration fix) but **completely ignores the "rẻ nhất / cheapest" intent** — the #1
result is `price_level=3` (expensive) and the cheapest café sits at #4. Opposite intents (`rẻ nhất`
vs `đắt nhất`) produce near-identical, price-blind orderings.

**Root cause:** `"rẻ"` is a generic-adjective stopword (dropped), and the ranker's 8 signals have
**no price dimension** — even though every POI carries a `price_level` field. So all
affordability intents (`giá rẻ`, `bình dân`, `rẻ nhất`, `sang`, `cao cấp`) are silently unhandled.

## Data ground truth

- `POI.price_level: int` ∈ {1,2,3,4}, **all 111 POIs populated**, evenly distributed
  (23/39/23/26), higher = pricier.
- `QueryIntent` already reserves an unused `price_max` field — price was anticipated, never wired.
- **Only 2 of 60 eval queries** contain a price word (both `giá rẻ`, i.e. cheap):
  `quán ăn giá rẻ mở khuya quận 1`, `cafe giá rẻ có sân vườn`. Price is *neutral* on the other 58.

## Approach — a soft, weighted 9th signal (not a hard filter)

Consistent with the pipeline's "no destructive filtering; intents flow through weighted signals"
philosophy. Price is **bidirectional** (cheap vs expensive), unlike rating/popularity — so the
signal is computed **relative to the parsed intent direction**, and is **neutral when no price
intent is present** (like `distance` without an anchor).

### 1. `parse.py` — price direction

Add `price_pref: str | None` to `QueryIntent` (`"cheap"` | `"expensive"` | `None`). Scan the folded
+ expanded query haystack (independent of the stopword/residual logic, so `rẻ`/`sang` inform price
without ever becoming spurious subjects). Token-boundary matched, multiword-first:

- **cheap:** `re`, `gia re`, `binh dan`, `tiet kiem`, `cheap`, `budget`
- **expensive:** `sang`, `sang trong`, `cao cap`, `sang chanh`, `luxury`

Deliberately **exclude bare `dat`** — folded "dat" collides with `đặt` ("to book", e.g. "đặt bàn").
Multiword cheap/expensive keys are checked before unigrams; if both directions somehow match, cheap
wins (the far more common intent) — documented, not expected in practice.

### 2. `rank.py` — `price_signal`

```python
PRICE_MIN, PRICE_MAX = 1, 4

def price_signal(intent: QueryIntent, poi: POI) -> float:
    if intent.price_pref is None or poi.price_level is None:
        return NEUTRAL                                   # constant across POIs -> no ranking effect
    norm = (poi.price_level - PRICE_MIN) / (PRICE_MAX - PRICE_MIN)   # 0=cheapest .. 1=priciest
    return _clamp01(1.0 - norm) if intent.price_pref == "cheap" else _clamp01(norm)
```

Add `"price"` to `SIGNALS` and to `signals()` — it then gets a breakdown entry, an explanation
reason, and a UI bar for free.

### 3. Weighting — fixed, untuned

`DEFAULT_WEIGHTS["price"] = 0.20` (on par with `distance`/`popularity`). **`data/weights.json` is
left untouched** — it has no `price` key, so `load_weights()` supplies price from
`DEFAULT_WEIGHTS` via its existing per-key fallback, and the proven 8 tuned weights are unchanged.

**Why not tuned:** only 2 eval queries exercise price, and we must not fit to them (NFR-6);
coordinate ascent would leave price at the 0.05 floor — too weak to be a real preference. `0.20` is
a deliberate design weight. Documented honestly as *not eval-tuned*.

**Why ranking still holds on the 58 non-price queries:** with `price_pref=None`, `price_signal`
returns `0.5` for every POI. `score = (Σ w·b)/Σw`; a constant `b_price=0.5` adds the same
`w_price·0.5` to every POI's numerator and the same `w_price` to every denominator → **order
preserved**. Only the 2 price queries can change; measured on tune, checked once on test.

### Behavior after the change

| Query | `price_pref` | effect |
|---|---|---|
| `cafe rẻ nhất` | cheap | cheaper cafés float up (level 1 → signal 1.0) |
| `nhà hàng sang trọng` | expensive | pricier restaurants float up |
| `cafe yên tĩnh` (no price word) | None | signal 0.5 everywhere → ranking unchanged |

## Scope / non-goals

- No hard price filter; no `price_max` numeric parsing (direction only — `rẻ`/`sang`, not "< 200k").
- No re-tuning; `weights.json` untouched.
- Keyword compare-lane, `/v1/search` contract DTOs unchanged (price rides in `breakdown` on the
  enriched `/v1/semantic-search` only).

## Docs reconciliation (8 → 9 signals)

Same pattern as the earlier 7→8. Update: `README.md` (signal count + table), `docs/methodology.md`
(signal table + count), the UI signal palette/glossary (`ui/index.html` — add a 9th Okabe-Ito hue
and a `price` gloss), `scripts/sample_queries.py` `SIG_LABEL` (add `price`: "giá"), and the deck
outline. `SPEC.md`'s stale "7 signals" references remain out of scope (pre-existing debt).

## Testing (failing-first)

- `test_price_cheap_prefers_cheaper` (parser): `cafe rẻ nhất` → `price_pref == "cheap"`.
- `test_price_expensive_parsed`: `nhà hàng sang trọng` → `price_pref == "expensive"`.
- `test_dat_not_confused_with_booking`: `đặt bàn nhà hàng` → `price_pref is None`.
- `test_price_signal_directions` (rank): cheap intent scores level 1 > level 4; expensive inverts;
  no intent → 0.5.
- `test_cheapest_cafe_ranks_above_priciest` (pipeline): in `cafe rẻ nhất`, the cheapest café
  outranks the priciest among the returned cafés.
- `test_no_price_word_ranking_unchanged` (pipeline): the `cafe yên tĩnh` ordering is identical with
  the `price` weight at 0.20 vs 0.0 (the constant-neutral property, checked by mutating
  `ranker.weights["price"]`).

## Verification gates

1. `uv run python -m pytest -q` green.
2. `run_eval.py --engine full --split tune` — NDCG@5 ≥ 0.959 (no regression; adjust weight down if
   the 2 price queries regress tune, never using test to choose it).
3. Test split evaluated **once** — G3 (NDCG@5 ≥ 0.80, Recall@3 ≥ 0.75) holds.
4. G4 warm p95 < 200 ms; G5 138/138.
5. End-to-end: `cafe rẻ nhất` (cheap on top), `nhà hàng sang trọng` (pricey on top),
   `cafe yên tĩnh` (unchanged).

## Delivery

New branch off `develop` (`feat/price-signal`); PR into `develop`. Commit style `feat(rank): …`.
Regenerate `reports/` only if metrics actually move (script-generated).
