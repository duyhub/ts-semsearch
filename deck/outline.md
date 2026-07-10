# Deck outline + live demo script — Tasco Semantic Search & Ranking

**AABW 2026 · Mobility track · P7 "AI Semantic Search & Ranking" · sponsored by Tasco Maps**

Deck arc (SPEC/PLAN): **problem → live demo → architecture → metrics/ablation → integration-readiness → roadmap.**
All numbers below are the fresh, post-hard-constraints figures from `reports/*.json` (test split held out; never tuned on).

**Ranking is 9 interpretable signals — 6 of Tasco's 7 published `Ranking_Signals` implemented, plus `category`-fit and `price`-preference signals we added.** (`business_attributes` splits into `attributes` + `open_now`; the 7th sponsor signal, `freshness`, is documented-but-not-implementable — no recency field in the dataset.)

**The five official judging dimensions** (`docs/Q&A.md`) — every slide is tagged with the ones it serves:

- **D1** Search relevance & semantic understanding
- **D2** Retrieval & ranking quality
- **D3** Explainable ranking
- **D4** User experience
- **D5** Technical design & production readiness

Judges said IR metrics are **optional** — which is exactly why we bring them: self-reported metrics on a held-out split are the credibility edge almost no team will have (serves D1 + D2 as *evidence*, not assertion).

---

## Part A — Slide-by-slide deck outline (11 slides)

### Slide 1 — Title / hook  · `D1`
- **Tasco.tìm — semantic search & ranking for Vietnamese POIs.** Search by *need*, not by name.
- One real query: **"quán cà phê yên tĩnh để làm việc"** (a quiet café to work from). No place name — a need.
- Keyword search can't map that to a café whose attributes are `wifi; yên tĩnh; phù hợp làm việc; ổ cắm`. We can.
- Team, track (Mobility P7), sponsor (Tasco Maps). One line: *"We built the retrieval + ranking engine that understands the query and explains every result — and we measured it."*
- **Visual:** the query typed into the app, semantic column already showing 3 quiet work-cafés with reason chips. Freeze frame of the money shot.

### Slide 2 — The problem  · `D1`
- Vietnamese users search by need, preference, attribute, natural language — **they rarely know the exact place name** (sponsor's own framing, `docs/problem-statement.md`).
- Keyword search breaks on Vietnamese behavior the sponsor explicitly expects (briefing recap §9): **missing accents, typos, abbreviations, slang, mixed VI/EN, incomplete + ambiguous queries, coordinates.**
- Three concrete failures: "gần hồ gươm" is a distance constraint keyword search can't resolve; "ca phe yen tinh" (no diacritics) misses the accented index; "nơi hẹn hò" has **no category word at all** to match.
- The bar the sponsor set: understand meaning, rank on multiple signals, and **return results with relevance explanations**.
- **Visual:** split screen — a keyword engine returning literal/empty junk for "nơi hẹn hò lãng mạn có view đẹp" vs. what the user actually wanted.

### Slide 3 — The solution in one diagram  · `D1 D2 D5`
- **query → normalize** (fold diacritics, expand abbreviations: q1→Quận 1, cf→cà phê, ks→khách sạn) **→ parse intent** (rules: category, anchor, required/soft attributes, time — closed to the sponsor's vocabularies).
- **→ hybrid retrieval:** BM25 (folded lexical) + dense `bge-m3` multilingual embeddings over the full 111-doc corpus, fused with **RRF** — no vector DB (111 docs fit a numpy matrix; a vector DB is live-demo risk + résumé noise, and we say so).
- **→ 8-signal interpretable re-ranker** (linear, tuned — 7 sponsor-aligned + a `category`-fit signal) → **hard-constraint filter** (category/location/subject → matches only) → **faithful explanations** derived only from signal values.
- **→ Tasco `/v1/search` contract** (+ extended `/v1/semantic-search`) → **live UI** with keyword-vs-semantic side-by-side.
- **Visual:** the README pipeline diagram, one box per stage, `bge-m3` and `RRF` labeled; a small "no vector DB / runs fully local" badge.

### Slide 4 — Live demo (switch to the app)  · `D1 D2 D3 D4`
- **"Everything from here is the running system — localhost, local embeddings, no cloud in the loop."**
- We'll do six things in ~3 minutes: keyword-vs-semantic side-by-side; audit a result's signals + reasons; an intent query with no category word; a hard-constraint correctness check; a non-accented/slang query; and the live latency badge.
- One-tap query chips = the exact scenarios; no live Vietnamese typing risk on stage.
- **→ Full script in Part B.**
- **Visual:** switch to `http://127.0.0.1:8000/` — the two-column UI with the chip row.

### Slide 5 — Why our ranking is interpretable  · `D2 D3`
- **9 interpretable signals — 6 of the sponsor's 7 published `Ranking_Signals` implemented, plus `category`-fit and `price`-preference signals we added.** The sponsor-aligned ones, nothing invented: `semantic`←relevance, `attributes`←business_attributes, `distance`←distance, `rating`←rating (Bayesian-smoothed), `popularity`←popularity, `open_now`←business_attributes (time), `review`←review_signal (tags/description) — `business_attributes` powers two of ours (`attributes` + `open_now`).
- **Our additions — `category` and `price`:** `category` is a category-consistency prior (1.0 on category match, 0.0 on mismatch, 0.5 when none parsed) — a *soft* signal, not a filter, that fixed malls/gas stations outranking cafés on "cà phê" without banishing a true answer on a mis-parse. `price` is an affordability preference from `price_level` (cheaper floats up on `rẻ`/`bình dân`, pricier on `sang`/`cao cấp`; neutral when no price is named) — it carries a fixed 0.20 weight, not a tuned one, since only 2/60 eval queries mention price. Both are presented transparently as ours, not sponsor signals.
- The sponsor's 7th listed signal, **`freshness`, is honestly disposed of**: the dataset has no recency field, so it's documented as a production roadmap item. Of the sponsor's seven, **six are implemented + freshness documented** — every sponsor signal is implemented *or* explicitly accounted for.
- **Every result carries a per-signal score breakdown** (top-3 bars on the card, expand to all 9) + **1–4 Vietnamese reasons** — and each reason is traceable to a real value ("✓ yên tĩnh", "cách Hồ Gươm 820 m", "4.7★ · 940 đánh giá", "mở đến 22:30"). The LLM only phrases; it never invents facts.
- Kills the hallucination question before it's asked: explanations are auditable, not generated prose.
- **Visual:** one result card, top-3 colored signal bars, the "▾ Xem đủ 9 tín hiệu" expander open; a small table mapping our sponsor-aligned signals to the sponsor's list, with `category` and `price` shown as our labeled additions.

### Slide 6 — The measurement edge (THE WIN)  · `D1 D2`
- **Almost no hackathon team reports real IR metrics.** We do — on a **held-out test split** (stratified 40 tune / 20 test, committed seed) the code and weights **never** touched.
- **Held-out test (n=20): NDCG@5 = 0.963, Recall@3 = 0.983, Recall@5 = 1.00, MRR = 0.95** (95% bootstrap CI on NDCG@5: [0.907, 1.00]). Even on **Hard** queries (n=8, 42% of the set): **NDCG@5 0.954, Recall@3 0.958** — the headline doesn't hide a Hard-query collapse.
- **Ablation proves every stage earns its place** (tune, NDCG@5): random 0.005 → BM25 0.861 → dense 0.881 → **hybrid 0.922** → **full + re-rank 0.959**. Hybrid beats *both* single retrievers; the re-ranker adds on top.
- All five gates green (real numbers):

| Gate | Metric | Result | Threshold |
|---|---|---|---|
| G1 | BM25 Recall@5 (tune) | **0.917** | ≥ 0.55 |
| G2 | hybrid NDCG@5 > max(BM25, dense) (tune) | **0.922 > 0.881** | > |
| G3 | full NDCG@5 / Recall@3 (**held-out test**) | **0.963 / 0.983** | ≥ 0.80 / ≥ 0.75 |
| G4 | warm p95 latency | **1.1 ms** | < 200 ms |
| G5 | robustness (60 eval + adversarial) | **138/138, 0 failures** | 100% |

- **Visual:** the ablation bar chart (4 rising bars) beside the G1–G5 gate table; "held-out test" stamped in red; the CI shown as a whisker on the 0.963 bar.

### Slide 7 — Hard constraints / correctness  · `D1 D2`
- Ranking is not enough — some queries demand a **correctness guarantee**, not just a good order.
- After the 8-signal re-rank, a **hard-constraint filter returns matches only** for the query's expressed constraints: **pure location** (district/city) → only POIs in that area; **subject** (distinctive content terms, e.g. "bún", "lẩu") → only POIs that actually have them; **pure category** → only that category (applied only when the parse is fully explained, so a mis-parse can't wrongly exclude).
- Safety valves so it never backfires: relaxes the most-specific constraint first if it would empty the list (G5), and an **anchor gate** floats near-anchor POIs to the top and demotes far ones.
- Measured, not just claimed: this is the design behind the fresh numbers on Slide 6 — it *raised* correctness on category/location/subject queries without hurting recall (Recall@5 stays 1.00 on test).
- **Visual:** "cafe" → semantic column is 100% Quán cà phê; "quan 1 tphcm" → 100% Quận 1 pins on the map. A red ✗ over any non-matching result to show none leak.

### Slide 8 — Integration-ready  · `D5`
- **Contract-exact `/v1/search`** per Tasco's API PDF: full param set (`q, lat, lon, radiusMeters, bbox, category, limit≤20, lang`), field-exact `PlaceResult` (stable `id` "poi:C001", WGS84 `coordinates`, `distanceMeters`, `score`, `tags`), contract `ErrorResponse` (`error.code/message`, `requestId`), Bearer / X-API-Key auth, `X-Request-Id`/`X-Locale`/`X-Timezone` headers, `/health`.
- **`openapi.json`** auto-generated + a **Dart client adapter** mapping `PlaceResult → SearchSuggestion` (the exact mapping table from the PDF).
- **Diacritics preserved in every output** (explicit Tasco compatibility requirement); folding lives only inside indexes.
- **"Integration is a base-URL change."** The extended `/v1/semantic-search` (adds `breakdown`, `reasons[]`, `intent`) is the transparency showcase — schema extension the organizers explicitly allow.
- Provenance + fallback story: **local `bge-m3` is the default the gates run on**; **Bedrock (`cohere.embed-multilingual-v3` / Titan v2, Claude parser) is a selectable, measured provider** for Built-with-AWS eligibility — never a runtime dependency.
- **Visual:** side-by-side JSON — Tasco's `PlaceResult` schema vs. our live response, identical fields highlighted; the one-line Dart base-URL swap.

### Slide 9 — Robustness  · `D4 D5`
- **Every eval query + adversarial inputs return HTTP 200 with ≥1 result and zero unhandled exceptions**: 138/138 checks pass (60 eval + 8 adversarial families).
- Adversarial families covered: empty string, emoji-only, all-caps no-diacritics, 200-char rambling text, pure-English, pure-address, coordinate-only, unknown city.
- **Empty-set backstop stays honest**: if retrieval + relaxation truly find nothing, fall back to top-by-popularity (or nearest to `lat`/`lon`) and stamp `meta.source = "fallback"` — a meaningless-but-present `q` always returns a result; only a *missing* `q` is a contract 400.
- Deterministic on identical input (injected reference time, not wall-clock) so demos and the eval are reproducible.
- **Visual:** a scrolling wall of adversarial inputs each with a green "200 ✓" and a result count; the robustness gate stamp 138/138.

### Slide 10 — Roadmap / what's next  · `D5`
- **Old ↔ new admin-name aliasing** (post-July-2025 restructuring): "quận 1, tphcm" ≡ "phường sài gòn" via a curated many-to-many alias table — mainstream map apps still fumble the new ward names; outputs keep the dataset's stable IDs/labels.
- **Coordinate-in-query anchors** ("10.7738, 106.704" → nearby-search) and ambiguous-name disambiguation (city/district context → request focus → popularity).
- **LLM intent parse on Bedrock (Claude)** layered over the rule parser (rules win on gazetteer-verified anchors; 800 ms timeout → rule fallback) + **Langfuse tracing** on LLM calls (sponsor awards judge-picked teams that use it).
- **Bedrock embeddings** (`cohere.embed-multilingual-v3` / Titan v2) as the measured cloud provider → **Built-with-AWS bonus-track eligibility**, with the local path always as fallback.
- **`freshness` signal** once a recency/last-verified field exists; OpenSearch/pgvector swap-in documented for production scale.
- **Visual:** a two-column "today vs. next" list; a small map showing an old district name and a new ward name resolving to the same pin.

### Slide 11 — Close: the ask + judging-dimension map  · `D1 D2 D3 D4 D5`
- **One-line ask:** *"A measured, explainable, contract-exact Vietnamese search engine — ready to drop into Tasco Maps today. We'd love the Mobility track."*
- We didn't assert quality — we **measured it on data we never trained on**, and matched your API to the field.
- Dimension map (what we showed for each):

| Judging dimension | What we showed | Slides |
|---|---|---|
| Search relevance & semantic understanding | intent queries with no category word land; measured, not asserted | 2, 4, 6 |
| Retrieval & ranking quality | hybrid > both single retrievers; 8-signal ranker (7 sponsor-aligned + category); ablation | 3, 5, 6 |
| Explainable ranking | per-signal breakdown + fact-derived Vietnamese reasons on every result | 5 |
| User experience | keyword-vs-semantic side-by-side, <200 ms warm, Vietnamese UI + map | 4, 9 |
| Technical design & production readiness | contract-exact `/v1/search` + OpenAPI + Dart adapter + fallback/latency notes | 8, 9, 10 |

- **Visual:** the five-row table on screen; final frame returns to the money-shot with the latency badge reading ~1 ms.

---

## Part B — Live demo script (the money shot, ~3.5 min, read verbatim)

**Setup before you speak:** app open at `http://127.0.0.1:8000/`, first chip already run (page loads with it), latency badge visible top-right. Left column = **Keyword · BM25**, right column = **Semantic · AI**.

> **0:00 — Step 1 · The core contrast (side-by-side)** *(chip)*
> **Do:** tap the chip **"quán cà phê yên tĩnh để làm việc"**.
> **Point at:** both columns at once — left (keyword) vs. right (semantic).
> **Say:** *"Same query, two engines. Keyword on the left matches words — it doesn't know what 'quiet café to work from' means. Our semantic engine on the right returns exactly that: quiet, work-friendly cafés — with the reasons right on the card."*

> **0:35 — Step 2 · Audit a result (breakdown + reasons)** *(no typing)*
> **Do:** on the **#1 semantic card** (e.g. *Tranquil Books & Coffee*), point to the reason line, then click **"▾ Xem đủ 9 tín hiệu"**.
> **Point at:** the reason chips **"✓ yên tĩnh, ✓ phù hợp làm việc · 4.7★ · mở đến 22:30"**, then the top-3 signal bars expanding to all 8.
> **Say:** *"Every result is explainable. These bars are our nine ranking signals — six mapped to Tasco's published list, plus category-fit and price-preference signals we added — and every reason is a real value, not generated prose. This is why it can't hallucinate: it only says what it can prove."*

> **1:10 — Step 3 · Intent with no category word** *(chip)*
> **Do:** tap the chip **"nơi hẹn hò lãng mạn có view đẹp"**.
> **Point at:** the semantic column (romantic restaurants / rooftops / cafés) vs. the weak keyword column; the map anchor panel.
> **Say:** *"There's no category word here — no 'nhà hàng', no 'cà phê'. Keyword search has nothing to grab. Ours reads the intent from the `lãng mạn` and `check-in` attributes and finds romantic spots anyway."*

> **1:45 — Step 4 · Location + distance (anchor + map)** *(chip)*
> **Do:** tap the chip **"cafe có wifi gần hồ gươm"**.
> **Point at:** the map anchor note ("neo: Hồ Gươm …"), and the reason **"cách Hồ Gươm 820 m"** on a top card.
> **Say:** *"'gần hồ gươm' isn't a filter keyword — it's a place. We resolve it to coordinates, so distance becomes a real ranking signal and the reason tells you how far each café is."*

> **2:15 — Step 5 · Hard-constraint correctness** *(type — short, low-risk)*
> **Do:** type **`cafe`** ↵, then type **`quan 1 tphcm`** ↵.
> **Point at:** for `cafe`, that **every** semantic result is *Quán cà phê*; for `quan 1 tphcm`, that **every** pin is in *Quận 1*.
> **Say:** *"For a pure category or a pure location, ranking isn't enough — you need a guarantee. After ranking we hard-filter to matches only: ask for cafés, you get only cafés; ask for Quận 1, nothing outside Quận 1 leaks in."*

> **2:50 — Step 6 · No diacritics + slang, and the speed** *(chip)*
> **Do:** tap the chip **"quan cafe yen tinh gan q1"**.
> **Point at:** results as strong as the fully-accented version; then the **latency badge** (~1 ms warm).
> **Say:** *"No accents, 'q1' slang — how Vietnamese people actually type. We fold diacritics and expand abbreviations, so 'q1' becomes Quận 1 and the results hold up. And look at the badge — about one millisecond, warm. Fast enough to feel instant."*

> **3:20 — Close** *(spoken, on the last screen)*
> **Say:** *"Everything you just saw ran on this laptop — local embeddings, no cloud in the loop. And it's the exact same engine behind the `/v1/search` contract Tasco published. Dropping it into their Flutter app is a one-line base-URL change."*

**Timing flex:** if short on time, cut Step 4 (location) — Steps 1, 2, 3, 5, 6 carry the story (contrast → explainability → intent → correctness → Vietnamese + speed). If you have extra time, add a mixed-language beat: type **`coffee shop with wifi to work`** to show VI/EN blending is handled by the same pipeline.

---

## Part C — Backup / rehearsal checklist

- **Boot the app:** `uv run uvicorn semsearch.api:create_app --factory --port 8000`, then open `http://127.0.0.1:8000/`.
- **Pre-warm before going on stage:** the model loads at server startup and the query cache pre-warms with the eval + rehearsed demo queries — but still **run all six chips once** manually before you present, so every demo query is warm (keeps the latency badge at ~1 ms, not a cold 20–100 ms).
- **Backup recorded video:** keep a screen-capture of the *exact* Part B script open in a second browser tab. If anything stalls on stage, switch to the video and narrate over it — this is also a required submission deliverable.
- **Wifi is not a dependency (per RUNBOOK contingency):** everything runs localhost — local `bge-m3` embeddings, cached parses, and OSM map tiles pre-cached during rehearsal. A venue wifi outage degrades nothing; **do not** mention the cloud/Bedrock path as live.
- **Pre-flight checks:** `uv run pytest -q` green; `scripts/robustness.py` → 138/138; latency badge renders; Vietnamese diacritics render correctly at 1080p (legible from 5 m). Freeze code after the last green run.
- **Dress-rehearsal reminders (do twice):**
  1. Run the full Part B script end-to-end against the *live* API twice, on the demo machine + demo display resolution — time it, confirm you land under ~3.5 min with the close.
  2. Deliberately rehearse the failure path: kill wifi mid-demo and confirm nothing changes; then practice the one-tap switch to the backup video so the recovery is smooth, not visible.
