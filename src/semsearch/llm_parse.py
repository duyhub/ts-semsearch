"""LLM query-intent parser (PRD FR-4), layered over the rule parser.

The rule parser (`parse.py`) runs unconditionally and owns precise, tested extraction.
This module OPTIONALLY enriches that parse with an LLM call that fills the gaps the
keyword rules missed — a category phrased unusually, an attribute the taxonomy map
didn't catch. It is OFF by default (NFR-5: `/v1/search` stays deterministic) and is
gated in the pipeline by `SEMSEARCH_LLM_PARSE=bedrock`.

Provider resolution (eager, at construction): Bedrock Claude (region chain x model-id
chain) -> OpenAI chat completions (only when every Bedrock candidate fails AND an API
key is discoverable) -> unavailable (rule parse only, pipeline warn-once). The
validation safety boundary below is PROVIDER-AGNOSTIC and applies identically to both.

HARD RULE: the OpenAI API key must NEVER appear in logs, exception messages, traces, or
test output — we log only the key's SOURCE ("env" / ".env file") and redact the key from
any error text we emit.

The safety boundary is OWNERSHIP + VALIDATION, not the model. The LLM contributes ONLY
category / attributes / price_pref / open_after — fields that feed soft ranking signals
or the coverage-gated category filter. Location (city/district) is deliberately NOT in
the contract: location is the gazetteer/rule parser's job, and it feeds the pipeline's
HARD location filter (`_constraint_filter`), where a hallucinated district would
destructively collapse recall. The prompt hands Claude the two CLOSED vocabularies
verbatim (12 categories, 10 taxonomy attributes) and demands STRICT JSON; `_validate`
then drops anything not exactly in those vocabularies (plus malformed times / price
directions), so a hallucinated value can never reach the ranker. Any failure — network
down, no creds, bad JSON — returns None, and the caller falls back to the rule intent
alone (HARD RULE: the demo never depends on the network).

Merge semantics (`merge_intent`): UNION with rules winning every conflict. The LLM only
fills a field the rules left empty; anchor / city / district / content_terms / residual
stay rule-owned.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import replace
from pathlib import Path

from . import tracing
from .data import QueryIntent
# CACHE_DIR / _safe are shared so the LLM disk cache lives beside the embedding caches and
# sanitizes provider/model ids into filesystem-safe path segments identically.
from .embeddings import CACHE_DIR, _safe, resolve_bedrock_regions  # FR-10 region chain
from .normalize import fold  # diacritic/punctuation folding — the corrected_query overlap guard
from .parse import ATTRIBUTE_KEYWORDS, CATEGORY_KEYWORDS

logger = logging.getLogger(__name__)

# PROMPT_VERSION salts the disk-cache key: bump it on ANY change to SYSTEM_PROMPT OR to the
# `_validate` validation semantics — the disk cache stores POST-validation output, so a
# validation-only edit (e.g. v3: case-only corrections are now rejected as no-ops) must
# invalidate old cache entries exactly like a reworded prompt would, or a stale pre-fix
# cache entry would keep serving a recapitalization "correction" forever.
PROMPT_VERSION = "v4-need-category"
# Disk cache for validated parses, keyed by (prompt version, provider, model, query). Lives
# under the embedding cache root; tests monkeypatch this to an isolated tmp dir.
LLMCACHE_DIR = CACHE_DIR / "llmcache"

# Model id: env override, else a fallback chain of ids for the same model. Accounts AND regions
# differ in which inference profiles exist (verified live on the AABW account: ap-southeast-1
# blocks Claude entirely; ap-northeast-1 serves it via the global. profile, with the apac.
# profile invalid and the plain id rejected for on-demand throughput). Construction resolves
# the working (region x id) combination once — regions outer, this id chain inner.
CLAUDE_MODEL_ENV = "SEMSEARCH_BEDROCK_CLAUDE"
DEFAULT_CLAUDE_MODEL = "apac.anthropic.claude-haiku-4-5-20251001-v1:0"
FALLBACK_CLAUDE_MODELS = (
    "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    "anthropic.claude-haiku-4-5-20251001-v1:0",
)

# HARD RULE (CLAUDE.md): Bedrock calls carry a timeout so a dead network fails fast, never
# hangs the demo. Parse sits in the request path, so the read timeout is short (~3s) and
# there are NO retries — on failure we degrade to the rule intent, we do not stall.
_CLAUDE_TIMEOUT = {"connect_timeout": 2, "read_timeout": 3, "retries": {"max_attempts": 1}}
_PING_MAX_TOKENS = 8  # a resolution ping just proves the (region, model) answers; keep it tiny

# ---- OpenAI fallback (used ONLY when every Bedrock candidate fails) ---------------------- #
# Claude access is blocked account-wide right now (the Anthropic use-case form), so an OpenAI
# chat-completions fallback keeps the FR-4 enrichment alive. Same timeout discipline as the
# Claude path (~2s connect / 3s read, no retries); httpx is already an installed dependency.
OPENAI_KEY_ENV = "OPENAI_API_KEY"
OPENAI_MODEL_ENV = "SEMSEARCH_OPENAI_MODEL"
# Cheapest model that RELIABLY fits the 3s parse read-timeout, measured live with our real
# SYSTEM_PROMPT (2026-07-11): gpt-4.1-nano $0.10/$0.40 per M at 1.1-1.7s. gpt-5-nano is
# nominally cheaper ($0.05/$0.40) but measured 2.1-3.1s — rides/exceeds the 3s budget, so it
# is NOT the default; override via SEMSEARCH_OPENAI_MODEL if latency tolerance differs.
DEFAULT_OPENAI_MODEL = "gpt-4.1-nano"
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
# Both observed spellings of the gitignored repo-root key file (never commit `.env/`).
OPENAI_KEY_FILENAMES = ("OPENAI-API-key.txt", "openai-api-key.txt")
_REPO_ROOT = Path(__file__).resolve().parents[2]  # src/semsearch/llm_parse.py -> repo root
_OPENAI_PING_MAX_TOKENS = 1  # the eager pin ping only proves key+model answer


def discover_openai_key() -> tuple[str, str] | None:
    """Find an OpenAI API key: env `OPENAI_API_KEY` first, else the gitignored repo-root
    `.env/` key file (both filename casings), whitespace-stripped. Returns (key, source)
    with source in {"env", ".env file"}, or None.

    HARD RULE: callers must log only the SOURCE — the key itself must never reach logs,
    exception messages, traces, or test output."""
    key = (os.environ.get(OPENAI_KEY_ENV) or "").strip()
    if key:
        return key, "env"
    for name in OPENAI_KEY_FILENAMES:
        try:
            key = (_REPO_ROOT / ".env" / name).read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if key:
            return key, ".env file"
    return None


def _openai_payload(model: str, messages: list[dict], max_tokens: int,
                    *, json_mode: bool = False) -> dict:
    """Build a chat-completions payload that works across OpenAI model families, so an env
    override never 400s: the gpt-5* reasoning family rejects non-default `temperature` (omit
    it) and the legacy `max_tokens` param (use `max_completion_tokens`); the non-reasoning
    default family (gpt-4.1-nano) takes `max_tokens` + `temperature: 0` (verified live)."""
    payload: dict = {"model": model, "messages": messages}
    if model.startswith("gpt-5"):
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["max_tokens"] = max_tokens
        payload["temperature"] = 0.0
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    return payload


def _redact(text: str, key: str | None) -> str:
    """Scrub the API key from any text we are about to log (HARD RULE: the key must never
    appear in logs, exception messages, traces, or test output). Defense in depth — our own
    call path never puts the key in error text, but upstream messages could."""
    return text.replace(key, "[REDACTED]") if key else text

# Closed vocabularies, derived from the SAME source of truth as the rule parser so the LLM
# can never widen them. dict.fromkeys dedupes the keyword-map values while preserving order.
CATEGORIES: tuple[str, ...] = tuple(dict.fromkeys(CATEGORY_KEYWORDS.values()))  # the 12
ATTRIBUTES: tuple[str, ...] = tuple(dict.fromkeys(ATTRIBUTE_KEYWORDS.values()))  # the 10
_CATEGORY_SET = frozenset(CATEGORIES)
_ATTRIBUTE_SET = frozenset(ATTRIBUTES)

_TIME_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")  # strict 24-hour HH:MM
# Strip a leading ```/```json fence and a trailing ``` fence (single fenced block).
_FENCE_RE = re.compile(r"^```(?:json)?[ \t]*\n?|\n?```$", re.IGNORECASE)
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)  # last-resort: first {...} span

SYSTEM_PROMPT = (
    "You extract structured search intent from a Vietnamese maps query. The user is "
    "searching for a place (POI) by need, not by name.\n\n"
    "Return ONLY a JSON object (no prose, no code fence) with EXACTLY these keys:\n"
    '  "corrected_query": the user\'s query with typos fixed and missing Vietnamese '
    "diacritics and tone marks restored to the MOST PLAUSIBLE common reading in a "
    'place-search context (e.g. "doi bung" -> "đói bụng" (hungry), NOT "đội bụng" (to '
    'carry on the head)); "minh"/"mình" is the pronoun "mình" (I/me), never the proper '
    "name Minh — do NOT capitalize it; preserve the meaning exactly; do NOT add, remove, "
    "or invent any place, need, or constraint; do NOT translate; do NOT expand abbreviations "
    'or shorthand (keep "q1", "hcm" as typed — deterministic code handles those); if the '
    "query is already clean, return it unchanged\n"
    '  "category": one of the allowed categories below, or null. When the query expresses a '
    "NEED or state rather than naming a place type, infer the single category that fulfills "
    "it (đói bụng/muốn ăn -> Nhà hàng; khát nước/muốn uống gì đó -> Quán cà phê, a drink "
    "place, NOT Trạm xăng; hết xăng -> Trạm xăng; cần rút tiền -> ATM); use null only when "
    "the need is genuinely ambiguous\n"
    '  "attributes": a list (possibly empty) of allowed attributes below\n'
    '  "price_pref": "cheap" | "expensive" | null\n'
    '  "open_after": the earliest opening time the user needs as "HH:MM" (24-hour), else null\n\n'
    "Do NOT extract locations (city, district, landmark) — they are handled elsewhere.\n\n"
    "ALLOWED CATEGORIES (use the EXACT string, or null — never invent one):\n"
    + "\n".join(f"  - {c}" for c in CATEGORIES)
    + "\n\nALLOWED ATTRIBUTES (use the EXACT strings; drop anything not listed):\n"
    + "\n".join(f"  - {a}" for a in ATTRIBUTES)
    + "\n\nOutput JSON only."
)


def _loads(raw: str) -> object:
    """Parse the model's JSON, tolerating ```json fences and leading/trailing prose.
    Returns the parsed object, or None if nothing parseable is found."""
    text = raw.strip()
    if text.startswith("```"):
        text = _FENCE_RE.sub("", text).strip()
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001 - fall through to the object-span heuristic
        m = _OBJECT_RE.search(text)
        if m is None:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:  # noqa: BLE001
            return None


def _validate_corrected_query(value: object, original: str, out: dict, dropped: dict) -> None:
    """Validate the LLM's `corrected_query` (its typo/diacritic repair of the user query),
    writing the accepted string to `out["corrected_query"]` or None with a `dropped` reason.

    An ABSENT/null key (legacy 4-key parse) is None WITHOUT a dropped reason — a missing
    correction is not an error. A PRESENT value is rejected (reason recorded) when it is not
    a string, is empty after cleanup, exceeds 200 chars, is case-only-different from the
    original (nothing WORTH correcting), or shares no folded token with the original
    (refusal/hallucination guard). Fold-equality is NOT itself a rejection: restoring
    diacritics onto an unaccented query changes the casefolded form too (different letters,
    not just different case) and is exactly what we want to keep. Never raises."""
    if value is None:  # key absent or explicit null -> no correction offered (not an error)
        out["corrected_query"] = None
        return
    if not isinstance(value, str):
        out["corrected_query"] = None
        dropped["corrected_query"] = "not-a-string"
        return
    cleaned = re.sub(r"\s+", " ", value).strip()  # strip + collapse newlines/runs to one space
    if not cleaned:
        reason = "empty"
    elif len(cleaned) > 200:
        reason = "too-long"
    elif cleaned.casefold() == original.casefold():
        # "no-op" also covers case-only differences (subsumes plain exact-equality: equal
        # strings are trivially casefold-equal too). A live A/B (2026-07-11) showed the LLM
        # "correcting" already-clean queries by ONLY recapitalizing proper nouns (e.g.
        # "gần hồ gươm" -> "gần hồ Gươm") — semantically nothing, since fold() already
        # lowercases for matching, but it perturbs the dense embedding with pure noise
        # (measured: -0.011 NDCG@5 on clean queries). casefold() (not lower()) is required
        # for Vietnamese case pairs: đ/Đ and accented uppercase (à/À, ơ/Ơ, ...) fold
        # correctly under casefold but not reliably under a naive .lower().
        reason = "no-op"
    elif not (set(fold(cleaned).split()) & set(fold(original).split())):
        reason = "no-overlap"  # zero shared folded tokens: a refusal/hallucination, not a fix
    else:
        out["corrected_query"] = cleaned
        return
    out["corrected_query"] = None
    dropped["corrected_query"] = reason


def _validate(raw: str, original: str) -> tuple[dict | None, dict]:
    """Validate the raw LLM output against the closed vocabularies. Returns
    (intent_dict, dropped): `intent_dict` is None only when the output is not a JSON
    object (a hard failure); otherwise every field is validated and out-of-vocab values
    are dropped. Location keys (city/district) are NOT part of the contract — if the
    model emits them anyway they are silently ignored here, so a hallucinated location
    can never reach the pipeline's hard location filter.

    `original` is the user's raw query, needed ONLY to validate `corrected_query` (the
    LLM's typo/diacritic repair of that query): it is kept only when it is a non-empty
    string, at most 200 chars, differs from the original by more than CASE after whitespace
    cleanup (case-only recapitalization is a no-op — it perturbs dense embeddings with zero
    semantic gain), and shares at least one folded token with the original (a
    refusal/hallucination sharing no token is dropped). Fold-equality is deliberately NOT a
    rejection on its own — restoring diacritics onto an unaccented query changes the
    casefolded form too and is the whole point. Never raises."""
    obj = _loads(raw)
    if not isinstance(obj, dict):
        return None, {"reason": "not-a-json-object"}

    dropped: dict = {}
    out: dict = {}

    _validate_corrected_query(obj.get("corrected_query"), original, out, dropped)

    cat = obj.get("category")
    if isinstance(cat, str) and cat in _CATEGORY_SET:
        out["category"] = cat
    else:
        out["category"] = None
        if cat not in (None, ""):
            dropped["category"] = cat

    attrs_in = obj.get("attributes")
    attrs_out: list[str] = []
    attrs_dropped: list = []
    if isinstance(attrs_in, list):
        for a in attrs_in:
            if isinstance(a, str) and a in _ATTRIBUTE_SET and a not in attrs_out:
                attrs_out.append(a)
            else:
                attrs_dropped.append(a)
    elif attrs_in not in (None, ""):
        dropped["attributes_type"] = attrs_in
    out["attributes"] = attrs_out
    if attrs_dropped:
        dropped["attributes"] = attrs_dropped

    price = obj.get("price_pref")
    if price in ("cheap", "expensive"):
        out["price_pref"] = price
    else:
        out["price_pref"] = None
        if price not in (None, ""):
            dropped["price_pref"] = price

    oa = obj.get("open_after")
    if isinstance(oa, str) and _TIME_RE.match(oa):
        out["open_after"] = oa
    else:
        out["open_after"] = None
        if oa not in (None, ""):
            dropped["open_after"] = oa

    return out, dropped


def _extract_text(resp: dict) -> str:
    """Concatenate the assistant text blocks from a Bedrock converse response."""
    blocks = resp["output"]["message"]["content"]
    return "".join(b.get("text", "") for b in blocks if isinstance(b, dict))


class LLMParser:
    """LLM intent parser. At CONSTRUCTION it resolves a working provider EAGERLY — Bedrock
    Claude first (region chain OUTER, model-id chain INNER, a cheap converse ping per
    candidate), then OpenAI (one tiny chat ping, only when all Bedrock candidates failed and a
    key is discoverable) — so the FIRST user query is a single pinned call (snappy) and the
    demo never re-resolves mid-flight. If nothing resolves, the parser is unavailable and
    `parse` returns None: the pipeline's warn-once + rule fallback then behaves exactly like
    today's parse-failure path. `parse` never raises."""

    def __init__(self, model_id: str | None = None, *, prefer: str = "auto") -> None:
        """`prefer='auto'` walks Bedrock then OpenAI; `prefer='openai'` skips every Bedrock
        candidate and pins OpenAI directly (SEMSEARCH_LLM_PARSE=openai — an operator with a
        key but no Bedrock shouldn't wait through 9 doomed converse probes)."""
        self._requested_model = model_id or os.environ.get(CLAUDE_MODEL_ENV) or DEFAULT_CLAUDE_MODEL
        self.model_id = self._requested_model  # the PINNED model after resolution (default until then)
        self._prefer = prefer  # 'auto' (Bedrock -> OpenAI) | 'openai' (skip Bedrock entirely)
        self._provider: str | None = None  # 'bedrock' | 'openai' after pinning; None => unavailable
        self._region: str | None = None  # pinned Bedrock region; always None for openai
        self._client = None  # pinned boto3 / httpx client; None => unavailable (parse returns None)
        self._openai_key: str | None = None  # NEVER logged/traced; used only in the auth header
        self._resolve()

    def _model_ids(self) -> list[str]:
        ids = [self._requested_model]
        # Only the default chain gets fallbacks; a user-set model id is used verbatim
        # (no guessing a fallback).
        if self._requested_model == DEFAULT_CLAUDE_MODEL:
            ids.extend(FALLBACK_CLAUDE_MODELS)
        return ids

    @staticmethod
    def _make_client(region: str):
        import boto3  # deferred: no import cost unless the LLM parser is actually used
        from botocore.config import Config

        return boto3.client(
            "bedrock-runtime", region_name=region, config=Config(**_CLAUDE_TIMEOUT)
        )

    def _ping(self, client, model_id: str) -> None:
        """Tiny converse proving THIS Claude id answers in THIS region. Raises on failure."""
        client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": "ping"}]}],
            system=[{"text": "ping"}],
            inferenceConfig={"maxTokens": _PING_MAX_TOKENS},
        )

    def _resolve(self) -> None:
        """Pin the first working provider. Bedrock first (unless prefer='openai'): (region,
        model-id) with regions OUTER, the id chain INNER — any candidate that raises
        (region-scoped use-case block, invalid profile, throttle, timeout) is skipped.
        Bedrock succeeding means OpenAI is NEVER contacted. All Bedrock candidates failing
        AND a key being discoverable -> ONE eager OpenAI ping pins provider='openai'.
        Nothing answering leaves the parser unavailable, which `parse` treats as today's
        parse-failure path."""
        if self._prefer != "openai":
            regions = resolve_bedrock_regions()
            model_ids = self._model_ids()
            for region in regions:
                try:
                    client = self._make_client(region)
                except Exception:  # noqa: BLE001 - client construction itself failed; next region
                    continue
                for mid in model_ids:
                    try:
                        self._ping(client, mid)
                    except Exception:  # noqa: BLE001 - (region, model) unavailable; try next
                        continue
                    self._client, self._region, self.model_id = client, region, mid
                    self._provider = "bedrock"
                    logger.info("LLM parser pinned to region %s via model %s.", region, mid)
                    return
        if self._resolve_openai():
            return
        logger.warning(
            "LLM parser: no working provider (%s); parse will degrade to the rule intent. "
            "Run scripts/check_bedrock.py to diagnose.",
            "OpenAI only, no key/ping" if self._prefer == "openai"
            else "Bedrock across all regions, then OpenAI",
        )

    def _resolve_openai(self) -> bool:
        """Try the OpenAI fallback: discover a key, ONE eager ping to pin it. Returns True when
        pinned. HARD RULE: only the key's SOURCE is ever logged, and error text is redacted."""
        found = discover_openai_key()
        if found is None:
            return False
        key, source = found
        model = os.environ.get(OPENAI_MODEL_ENV) or DEFAULT_OPENAI_MODEL
        logger.info("OpenAI key found (%s); trying the OpenAI fallback for the LLM parse.", source)
        client = self._make_openai_client()
        try:
            self._openai_post(
                client, key,
                _openai_payload(model, [{"role": "user", "content": "ping"}],
                                _OPENAI_PING_MAX_TOKENS),
            )
        except Exception as exc:  # noqa: BLE001 - ping failed -> unavailable (rule fallback)
            logger.warning(
                "OpenAI fallback ping failed (%s); LLM parse unavailable.",
                _redact(f"{type(exc).__name__}: {exc}", key),
            )
            try:
                client.close()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass
            return False
        self._client, self._provider = client, "openai"
        self._openai_key, self.model_id = key, model
        self._region = None
        logger.info("LLM parser pinned to OpenAI (%s).", model)
        return True

    @staticmethod
    def _make_openai_client():
        # httpx is ALREADY an installed dependency (no new dep). Same timeout discipline as
        # the Claude path: ~2s connect / 3s read, and NO retries (httpx does not retry).
        import httpx  # deferred: no import cost unless the OpenAI fallback is reached

        return httpx.Client(timeout=httpx.Timeout(3.0, connect=2.0))

    @staticmethod
    def _openai_post(client, key: str, payload: dict) -> dict:
        """POST one chat completion. The key travels ONLY in the auth header — httpx exceptions
        carry the URL, never request headers, so it cannot leak via raised errors."""
        resp = client.post(
            OPENAI_CHAT_URL, headers={"Authorization": f"Bearer {key}"}, json=payload
        )
        resp.raise_for_status()
        return resp.json()

    def _openai_chat(self, query: str) -> str:
        """Call the PINNED OpenAI model ONCE and return the raw assistant text. Reuses the same
        SYSTEM_PROMPT; `_validate` downstream applies the identical closed-vocab boundary."""
        data = self._openai_post(
            self._client, self._openai_key,
            _openai_payload(
                self.model_id,
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": query}],
                300,
                json_mode=True,
            ),
        )
        return data["choices"][0]["message"]["content"]

    def _converse(self, query: str) -> str:
        """Call the PINNED Bedrock (region, model) ONCE and return the raw assistant text. No
        mid-demo re-walk: a per-call failure raises here and `parse` degrades to the rule
        intent (determinism + latency)."""
        resp = self._client.converse(
            modelId=self.model_id,
            messages=[{"role": "user", "content": [{"text": query}]}],
            system=[{"text": SYSTEM_PROMPT}],
            inferenceConfig={"temperature": 0.0, "maxTokens": 300},
        )
        return _extract_text(resp)

    def _complete(self, query: str) -> str:
        """Dispatch to the pinned provider (bedrock is the default path — injected test fakes
        without a provider run the converse path)."""
        if self._provider == "openai":
            return self._openai_chat(query)
        return self._converse(query)

    def _cache_path(self, query: str) -> Path:
        """Disk-cache path for THIS parse, keyed by (prompt version, provider, model, query).
        LLMCACHE_DIR and PROMPT_VERSION are read as module globals at CALL time (not captured
        as defaults) so tests can monkeypatch them per case. A rewording of SYSTEM_PROMPT must
        bump PROMPT_VERSION so the old key never serves a stale parse."""
        key = hashlib.sha1(
            f"{PROMPT_VERSION}:{self._provider}:{self.model_id}:{query}".encode()
        ).hexdigest()
        bucket = f"{_safe(self._provider or 'bedrock')}.{_safe(self.model_id)}"
        return LLMCACHE_DIR / bucket / f"{key}.json"

    @staticmethod
    def _cache_read(path: Path) -> dict | None:
        """Return the cached validated dict, or None on a miss / unreadable / non-dict file.
        A corrupt cache entry must never break a query — it degrades to a fresh parse."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):  # missing file or malformed JSON -> treat as a miss
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _cache_write(path: Path, validated: dict) -> None:
        """Persist a validated (non-None) parse. Best-effort: a write failure is swallowed so
        it can never break a query. ensure_ascii=False keeps Vietnamese diacritics readable."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(validated, ensure_ascii=False), encoding="utf-8")
        except OSError:  # noqa: BLE001 - the cache is an optimization, never a hard dependency
            pass

    def parse(self, query: str, *, use_cache: bool = True) -> dict | None:
        """Run the LLM parse and return a validated intent dict, or None on ANY failure
        (including an unavailable parser, when construction resolved nothing). Emits one
        best-effort Langfuse generation (input query, raw output, model id, validated +
        dropped fields, latency).

        A disk cache (keyed by prompt version + provider + model + query) short-circuits the
        network call for a repeat query; only NON-None validated results are cached (failures
        are never cached, so a transient outage self-heals). `use_cache=False` bypasses both
        the read and the write."""
        if self._client is None:
            return None  # construction resolved nothing -> behave as today's parse-failure path
        cache_path = self._cache_path(query) if use_cache else None
        if cache_path is not None:
            cached = self._cache_read(cache_path)
            if cached is not None:
                return cached  # cache hit: skip the network call (and its span) entirely
        raw: str | None = None
        validated: dict | None = None
        dropped: dict = {}
        with tracing.traced(
            "llm_parse", kind="generation", model=self.model_id, input=query
        ) as span:
            try:
                raw = self._complete(query)
                validated, dropped = _validate(raw, query)
            except Exception as exc:  # noqa: BLE001 - never crash a query on an LLM failure
                logger.debug(
                    "LLM parse failed (%s); rule intent will be used.",
                    _redact(f"{type(exc).__name__}: {exc}", self._openai_key),
                )
                validated = None
            span.update(
                output=raw,
                metadata={
                    # pinned provider + model name in the trace — NEVER the key
                    "provider": self._provider or "bedrock",
                    "model_id": self.model_id,
                    "validated": validated,
                    "dropped": dropped or None,
                },
            )
        if cache_path is not None and validated is not None:
            self._cache_write(cache_path, validated)  # never cache None / failures
        return validated


def merge_intent(rule_intent: QueryIntent, llm_out: dict | None) -> QueryIntent:
    """UNION-merge the LLM output into the rule intent. Rules win every conflict (they are
    precise and tested); the LLM only fills a field the rules left empty. Returns a NEW
    QueryIntent — never mutates the input.

    OWNERSHIP: the LLM may contribute ONLY category / attributes / price_pref /
    open_after. Location (city/district) plus anchor / content_terms / has_residual /
    residual_terms / soft_prefs stay entirely rule-owned — city/district feed the
    pipeline's HARD location filter, and a hallucinated district there destructively
    collapses recall, so merge never reads them from the LLM output even if present."""
    if not llm_out:
        return rule_intent

    required = list(rule_intent.required_attrs)
    for attr in llm_out.get("attributes", []):
        if attr not in required:
            required.append(attr)

    return replace(
        rule_intent,
        category=rule_intent.category or llm_out.get("category"),
        required_attrs=required,
        price_pref=rule_intent.price_pref or llm_out.get("price_pref"),
        open_after=rule_intent.open_after or llm_out.get("open_after"),
    )
