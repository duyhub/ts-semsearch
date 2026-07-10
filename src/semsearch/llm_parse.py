"""LLM query-intent parser on Bedrock Claude (PRD FR-4), layered over the rule parser.

The rule parser (`parse.py`) runs unconditionally and owns precise, tested extraction.
This module OPTIONALLY enriches that parse with a Claude `converse` call that fills the
gaps the keyword rules missed — a category phrased unusually, an attribute the taxonomy
map didn't catch. It is OFF by default (NFR-5: `/v1/search` stays deterministic) and is
gated in the pipeline by `SEMSEARCH_LLM_PARSE=bedrock`.

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

import json
import logging
import os
import re
from dataclasses import replace

from . import tracing
from .data import QueryIntent
from .parse import ATTRIBUTE_KEYWORDS, CATEGORY_KEYWORDS

logger = logging.getLogger(__name__)

# Model id: env override, else the APAC cross-region inference profile. The account may
# not have that profile provisioned, so `_converse` retries once on the plain global id
# when converse rejects the profile with a ValidationException about the model id.
CLAUDE_MODEL_ENV = "SEMSEARCH_BEDROCK_CLAUDE"
DEFAULT_CLAUDE_MODEL = "apac.anthropic.claude-haiku-4-5-20251001-v1:0"
FALLBACK_CLAUDE_MODEL = "anthropic.claude-haiku-4-5-20251001-v1:0"

# HARD RULE (CLAUDE.md): Bedrock calls carry a timeout so a dead network fails fast, never
# hangs the demo. Parse sits in the request path, so the read timeout is short (~3s) and
# there are NO retries — on failure we degrade to the rule intent, we do not stall.
_CLAUDE_TIMEOUT = {"connect_timeout": 2, "read_timeout": 3, "retries": {"max_attempts": 1}}

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
    '  "category": one of the allowed categories below, or null\n'
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


def _validate(raw: str) -> tuple[dict | None, dict]:
    """Validate the raw LLM output against the closed vocabularies. Returns
    (intent_dict, dropped): `intent_dict` is None only when the output is not a JSON
    object (a hard failure); otherwise every field is validated and out-of-vocab values
    are dropped. Location keys (city/district) are NOT part of the contract — if the
    model emits them anyway they are silently ignored here, so a hallucinated location
    can never reach the pipeline's hard location filter. Never raises."""
    obj = _loads(raw)
    if not isinstance(obj, dict):
        return None, {"reason": "not-a-json-object"}

    dropped: dict = {}
    out: dict = {}

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


def _is_invalid_model_id(exc: Exception) -> bool:
    """True when a converse call was rejected because the model id is invalid (so we
    should retry the global id). Matches botocore's ValidationException about the model."""
    code = ""
    resp = getattr(exc, "response", None)
    if isinstance(resp, dict):
        code = str(resp.get("Error", {}).get("Code", ""))
    text = f"{code} {exc}".lower()
    return "validation" in text and "model" in text


class LLMParser:
    """Claude intent parser via Bedrock `converse`. Lazy boto3 client (no client / no
    credential lookup until the first `parse`), same region/timeout conventions as
    BedrockEmbedder. `parse` never raises — it returns a validated intent dict or None."""

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or os.environ.get(CLAUDE_MODEL_ENV) or DEFAULT_CLAUDE_MODEL
        self._client = None  # lazy: no boto3 import / cred lookup until first parse

    @staticmethod
    def _region() -> str:
        # Same precedence as BedrockEmbedder: explicit override, then AWS_*, then the event
        # region (ap-southeast-1, Singapore).
        return (
            os.environ.get("SEMSEARCH_BEDROCK_REGION")
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or "ap-southeast-1"
        )

    def _get_client(self):
        if self._client is None:
            import boto3  # deferred: no import cost unless the LLM parser is actually used
            from botocore.config import Config

            self._client = boto3.client(
                "bedrock-runtime", region_name=self._region(), config=Config(**_CLAUDE_TIMEOUT)
            )
        return self._client

    def _model_ids(self) -> list[str]:
        ids = [self.model_id]
        # Only the APAC profile has a distinct global fallback id; a user-set model id is
        # used verbatim (no guessing a fallback).
        if self.model_id == DEFAULT_CLAUDE_MODEL:
            ids.append(FALLBACK_CLAUDE_MODEL)
        return ids

    def _converse(self, query: str) -> str:
        """Call Claude and return the raw assistant text. Tries the APAC profile, then the
        global id on a ValidationException about the model id. Raises on real failure."""
        client = self._get_client()
        messages = [{"role": "user", "content": [{"text": query}]}]
        system = [{"text": SYSTEM_PROMPT}]
        inference = {"temperature": 0.0, "maxTokens": 300}
        model_ids = self._model_ids()
        last_exc: Exception | None = None
        for mid in model_ids:
            try:
                resp = client.converse(
                    modelId=mid, messages=messages, system=system, inferenceConfig=inference
                )
                return _extract_text(resp)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                # Fall through to the next id ONLY when the profile id itself was rejected.
                if mid is not model_ids[-1] and _is_invalid_model_id(exc):
                    continue
                raise
        raise last_exc  # pragma: no cover - loop always returns or raises above

    def parse(self, query: str) -> dict | None:
        """Run the LLM parse and return a validated intent dict, or None on ANY failure.
        Emits one best-effort Langfuse generation (input query, raw output, model id,
        validated + dropped fields, latency)."""
        raw: str | None = None
        validated: dict | None = None
        dropped: dict = {}
        with tracing.traced(
            "llm_parse", kind="generation", model=self.model_id, input=query
        ) as span:
            try:
                raw = self._converse(query)
                validated, dropped = _validate(raw)
            except Exception as exc:  # noqa: BLE001 - never crash a query on an LLM failure
                logger.debug(
                    "LLM parse failed (%s: %s); rule intent will be used.",
                    type(exc).__name__, exc,
                )
                validated = None
            span.update(
                output=raw,
                metadata={
                    "model_id": self.model_id,
                    "validated": validated,
                    "dropped": dropped or None,
                },
            )
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
