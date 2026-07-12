"""LLM query parser (FR-4) + Langfuse tracing + pipeline/API gate.

Every test here is OFFLINE. The Bedrock `converse` client is replaced by an in-process
fake (`FakeConverseClient`) injected onto `parser._client`, so no test constructs a real
bedrock-runtime client or touches the network — there are NO AWS credentials here, and the
suite MUST pass without any. Langfuse is exercised in its no-op (keys-absent) form AND in
its keys-present form via a fake `langfuse` module injected into sys.modules — never the
real SDK, never the network.

Contracts under test:
  - Validation is the safety boundary: out-of-vocab category/attribute, malformed time /
    price direction are DROPPED; a non-JSON output yields None; the parser never raises.
  - OWNERSHIP: location (city/district) is NOT in the LLM contract — even when the model
    emits it, it never reaches the intent. Location feeds the pipeline's HARD location
    filter, where a hallucinated district would destructively collapse recall.
  - ```json fences and surrounding prose are tolerated.
  - merge_intent is a UNION with rules winning conflicts; it returns a NEW QueryIntent.
  - The pipeline gate: OFF by default (LLMParser never constructed, behavior unchanged);
    ON (SEMSEARCH_LLM_PARSE=bedrock) enriches once per query and degrades to the rule
    intent on ANY failure, warning once.
  - The API resolves the intent ONCE: /v1/semantic-search's echo, reasons[] and ranking
    all reflect the SAME merged intent; gate-off API behavior is unchanged.
  - Tracing keys present: the generation/embedding emit carries the expected shape
    (model, input, validated/dropped metadata, latency); an emit failure is swallowed.
"""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from semsearch import llm_parse as L
from semsearch import tracing
from semsearch.api import create_app
from semsearch.data import QueryIntent, load_pois
from semsearch.pipeline import FullPipeline


# --------------------------------------------------------------------------- #
# Offline fake for bedrock-runtime.converse                                   #
# --------------------------------------------------------------------------- #
def _converse_response(text: str) -> dict:
    return {"output": {"message": {"content": [{"text": text}]}}}


class FakeConverseClient:
    """Records converse() calls and returns canned text (or raises). `per_model` maps a
    modelId to either a text response or an Exception, to exercise the id fallback."""

    def __init__(self, text: str | None = None, *, raise_exc: Exception | None = None,
                 per_model: dict | None = None):
        self.calls: list[dict] = []
        self._text = text
        self._raise_exc = raise_exc
        self._per_model = per_model

    def converse(self, *, modelId, messages, system, inferenceConfig):  # noqa: N803
        self.calls.append({"modelId": modelId, "messages": messages,
                           "system": system, "inferenceConfig": inferenceConfig})
        if self._per_model is not None:
            val = self._per_model.get(modelId)
            if isinstance(val, Exception):
                raise val
            if val is not None:
                return _converse_response(val)
        if self._raise_exc is not None:
            raise self._raise_exc
        return _converse_response(self._text or "")


def make_parser(client: FakeConverseClient, model_id: str | None = None) -> L.LLMParser:
    # LLMParser resolves its provider EAGERLY at construction; the autouse `_offline_llm`
    # fixture makes that resolution find nothing (no network, no key), then we inject the
    # fake so parse() runs against it (provider unset -> the bedrock/converse path).
    p = L.LLMParser(model_id=model_id)
    p._client = client  # inject the fake; a set _client marks the parser available
    return p


class _AlwaysFailConverse:
    """A stand-in bedrock-runtime client whose every call fails — models the offline /
    no-credentials default so an eager LLMParser resolution pins nothing."""

    def converse(self, **_kw):
        raise RuntimeError("offline: no bedrock reachable in tests")

    def invoke_model(self, **_kw):
        raise RuntimeError("offline: no bedrock reachable in tests")


def _no_openai_client():
    raise AssertionError("OpenAI must not be contacted in this test")


@pytest.fixture(autouse=True)
def _offline_llm(monkeypatch, tmp_path):
    """Default for EVERY test in this module: constructing an LLMParser resolves NOTHING with
    NO network — Bedrock fails offline, no OpenAI key is discoverable (env cleared; the repo
    root is pointed at an empty tmp dir so the developer's real gitignored `.env/` key file is
    invisible to tests), and any accidental OpenAI client construction fails loudly. Tests that
    exercise Bedrock/OpenAI resolution override these pieces explicitly."""
    monkeypatch.setattr("boto3.client", lambda *a, **k: _AlwaysFailConverse())
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv(L.OPENAI_MODEL_ENV, raising=False)
    # Clear the LLM degradation gate so every test defaults to "auto" (the degradation gate);
    # gate-specific tests set SEMSEARCH_LLM_GATE explicitly before constructing the pipeline.
    monkeypatch.delenv("SEMSEARCH_LLM_GATE", raising=False)
    monkeypatch.setattr(L, "_REPO_ROOT", tmp_path / "no-repo")
    # Isolate the LLM disk cache per test: module-scoped fixtures reuse the SAME query text
    # with DIFFERENT fake clients, and a shared cache would serve one test's parse to another.
    monkeypatch.setattr(L, "LLMCACHE_DIR", tmp_path / "llmcache")
    monkeypatch.setattr(L.LLMParser, "_make_openai_client", staticmethod(_no_openai_client))


def _llm_json(**overrides) -> str:
    base = {"category": None, "attributes": [], "price_pref": None, "open_after": None}
    base.update(overrides)
    return json.dumps(base)


# Includes city/district (NOT in the contract) + an out-of-vocab attribute, to prove
# both are rejected by validation.
VALID_JSON = _llm_json(
    category="Nhà hàng",
    attributes=["wifi", "yên tĩnh", "teleport"],  # 'teleport' is out-of-vocab
    city="Hà Nội", district="Quận 1",             # location: ignored (rule-owned)
    price_pref="cheap", open_after="18:00",
)


# --------------------------------------------------------------------------- #
# Closed vocabularies are exactly the rule parser's (12 categories, 10 attrs)  #
# --------------------------------------------------------------------------- #
def test_closed_vocab_sizes():
    assert len(L.CATEGORIES) == 12
    assert len(L.ATTRIBUTES) == 10
    assert "Quán cà phê" in L.CATEGORIES and "yên tĩnh" in L.ATTRIBUTES


def test_prompt_version_is_v4_need_category():
    # PROMPT_VERSION salts the disk llmcache; the v4 bump invalidates stale v3 entries so a
    # pre-fix "đội bụng" / null-category parse can never be re-served.
    assert L.PROMPT_VERSION == "v4-need-category"


def test_system_prompt_has_need_inference_and_diacritic_contrast():
    """v4 root-cause guards (cheap prompt-regression tripwire): the prompt must (a) instruct
    need -> category inference with inline examples, and (b) carry the đói/đội contrastive
    diacritic-restoration example that fixes the 'minh doi bung' mis-correction."""
    prompt = L.SYSTEM_PROMPT
    # (a) need-inference instruction + at least one hunger/thirst/fuel example mapping
    assert "NEED" in prompt and "infer the single category" in prompt
    assert "đói bụng/muốn ăn" in prompt  # a hunger NEED maps to a category, inline
    assert "hết xăng" in prompt          # a state -> Trạm xăng need example
    # (b) the đói (hungry) vs đội (carry) contrastive restoration example
    assert "đói bụng" in prompt and 'NOT "đội bụng"' in prompt


# --------------------------------------------------------------------------- #
# Validation drops out-of-vocab values; location is never emitted             #
# --------------------------------------------------------------------------- #
def test_valid_json_validated_and_invalid_dropped():
    out = make_parser(FakeConverseClient(VALID_JSON)).parse("q")
    assert out is not None
    assert out["category"] == "Nhà hàng"
    assert out["attributes"] == ["wifi", "yên tĩnh"]  # 'teleport' dropped, order kept
    assert out["price_pref"] == "cheap"
    assert out["open_after"] == "18:00"
    # location keys the model emitted anyway are IGNORED — not part of the contract
    assert "city" not in out and "district" not in out


def test_invalid_category_dropped_to_none():
    out = make_parser(FakeConverseClient(_llm_json(category="Bánh mì"))).parse("q")
    assert out["category"] is None  # not one of the 12 -> dropped


def test_malformed_time_and_price_dropped():
    raw = _llm_json(price_pref="moderate", open_after="25:99")
    out = make_parser(FakeConverseClient(raw)).parse("q")
    assert out["price_pref"] is None  # only cheap|expensive survive
    assert out["open_after"] is None  # not a valid HH:MM


def test_fenced_json_tolerated():
    body = _llm_json(category="ATM")
    for raw in (f"```json\n{body}\n```", f"```\n{body}\n```",
                f"Here is the intent:\n```json\n{body}\n```"):
        out = make_parser(FakeConverseClient(raw)).parse("q")
        assert out is not None and out["category"] == "ATM"


def test_garbage_output_returns_none():
    out = make_parser(FakeConverseClient("I think you want a quiet cafe, not JSON.")).parse("q")
    assert out is None  # nothing parseable -> hard failure -> None


def test_exception_returns_none_never_raises():
    out = make_parser(FakeConverseClient(raise_exc=RuntimeError("read timeout"))).parse("q")
    assert out is None  # a converse failure degrades to None, never raises


def test_converse_request_shape():
    client = FakeConverseClient(VALID_JSON)
    make_parser(client).parse("cà phê yên tĩnh")
    call = client.calls[-1]
    assert call["messages"][0]["content"][0]["text"] == "cà phê yên tĩnh"  # diacritics preserved
    assert call["inferenceConfig"]["temperature"] == 0.0
    assert call["inferenceConfig"]["maxTokens"] == 300
    prompt = call["system"][0]["text"]
    assert "ALLOWED CATEGORIES" in prompt
    assert "corrected_query" in prompt  # the typo/diacritic-repair key is now requested first
    # the prompt must NOT ask for location fields (ownership: gazetteer/rules only)
    assert '"city"' not in prompt and '"district"' not in prompt


# --------------------------------------------------------------------------- #
# corrected_query validation matrix: _validate(raw, original) — the LLM's      #
# typo/diacritic repair of the user query, guarded before it can reach the     #
# pipeline. Rejection -> None + a recorded dropped reason; never raises.        #
# --------------------------------------------------------------------------- #
def test_corrected_query_valid_correction_returned():
    raw = _llm_json(corrected_query="quán cà phê yên tĩnh wifi")
    out, _dropped = L._validate(raw, "quan cafe yen tinh wjfi")
    assert out["corrected_query"] == "quán cà phê yên tĩnh wifi"


def test_corrected_query_whitespace_and_newline_hygiene():
    raw = _llm_json(corrected_query="  quán cà phê\nyên tĩnh  ")
    out, _dropped = L._validate(raw, "quan ca phe yen tinh")
    assert out["corrected_query"] == "quán cà phê yên tĩnh"  # stripped, newline -> one space


def test_corrected_query_diacritic_only_restoration_kept():
    # fold-equal to the original but visually different: diacritic restoration IS the point,
    # so a fold-equal correction must NOT be treated as a no-op.
    raw = _llm_json(corrected_query="quán cà phê yên tĩnh")
    out, dropped = L._validate(raw, "quan ca phe yen tinh")
    assert out["corrected_query"] == "quán cà phê yên tĩnh"
    assert "corrected_query" not in dropped


def test_corrected_query_identical_to_original_is_noop():
    raw = _llm_json(corrected_query="quán cà phê yên tĩnh")
    out, dropped = L._validate(raw, "quán cà phê yên tĩnh")
    assert out["corrected_query"] is None            # nothing to correct -> no rewrite
    assert dropped["corrected_query"]                 # the reason is recorded for the trace


# --------------------------------------------------------------------------- #
# Case-only "corrections" are no-ops (live A/B, 2026-07-11): the LLM was       #
# observed recapitalizing a proper noun in an already-clean query — zero      #
# semantic change, but it perturbs dense-embedding retrieval and cost         #
# -0.011 NDCG@5. Test strings below are illustrative rewordings, not eval-    #
# query text verbatim (test_integrity.py::test_no_query_text_hardcoded_in_src #
# forbids that). casefold() (not lower()) is required for Vietnamese case     #
# pairs like đ/Đ and accented uppercase.                                      #
# --------------------------------------------------------------------------- #
def test_corrected_query_case_only_latin_is_noop():
    raw = _llm_json(corrected_query="quán nước gần chùa Trấn Quốc")
    out, dropped = L._validate(raw, "quán nước gần chùa trấn quốc")
    assert out["corrected_query"] is None             # capitalization alone is not a correction
    assert dropped["corrected_query"]                  # reason recorded for the trace


def test_corrected_query_case_only_vietnamese_uppercase_is_noop():
    # đ -> Đ and accented-vowel uppercase (à -> À etc.) are still case-only under casefold().
    raw = _llm_json(corrected_query="quán chè ngon ở Đông Anh")
    out, dropped = L._validate(raw, "quán chè ngon ở đông anh")
    assert out["corrected_query"] is None
    assert dropped["corrected_query"]


def test_corrected_query_typo_fix_still_accepted():
    raw = _llm_json(corrected_query="quan cafe wifi")
    out, dropped = L._validate(raw, "quan cafe wjfi")
    assert out["corrected_query"] == "quan cafe wifi"
    assert "corrected_query" not in dropped


def test_corrected_query_mixed_typo_and_recase_still_accepted():
    # recapitalizes AND fixes a typo -> differs beyond case -> kept, not a no-op.
    raw = _llm_json(corrected_query="Quán cà phê wifi ở Hà Nội")
    out, dropped = L._validate(raw, "quan ca phe wjfi o ha noi")
    assert out["corrected_query"] == "Quán cà phê wifi ở Hà Nội"
    assert "corrected_query" not in dropped


def test_corrected_query_too_long_dropped():
    raw = _llm_json(corrected_query="quán cà phê " * 30)  # > 200 chars after cleanup
    out, dropped = L._validate(raw, "quan cafe")
    assert out["corrected_query"] is None
    assert dropped["corrected_query"]


@pytest.mark.parametrize("bad", [42, ["quán", "cà", "phê"], {"q": 1}, True])
def test_corrected_query_non_string_dropped(bad):
    raw = _llm_json(corrected_query=bad)
    out, dropped = L._validate(raw, "quan cafe yen tinh")
    assert out["corrected_query"] is None
    assert dropped["corrected_query"]


@pytest.mark.parametrize("blank", ["", "   ", "\n\t "])
def test_corrected_query_empty_or_whitespace_dropped(blank):
    raw = _llm_json(corrected_query=blank)
    out, dropped = L._validate(raw, "quan cafe yen tinh")
    assert out["corrected_query"] is None
    assert dropped["corrected_query"]


def test_corrected_query_zero_token_overlap_dropped():
    # a refusal / hallucination shares no folded token with the query -> reject (guard rail)
    raw = _llm_json(corrected_query="xin lỗi, tôi không thể giúp")
    out, dropped = L._validate(raw, "quan cafe yen tinh")
    assert out["corrected_query"] is None
    assert dropped["corrected_query"]


def test_corrected_query_missing_key_is_none_legacy_json():
    # a legacy 4-key parse (no corrected_query) validates exactly as before; the new key is
    # simply absent -> None, and an absent key is NOT an error (no dropped reason recorded).
    raw = _llm_json(category="ATM")
    out, dropped = L._validate(raw, "quan cafe yen tinh")
    assert out["corrected_query"] is None
    assert "corrected_query" not in dropped
    assert out["category"] == "ATM"  # everything else validated as before


@pytest.mark.parametrize("raw", [
    '{"corrected_query": {"nested": "object"}}',   # nested object value
    '{"corrected_query": [1, 2, 3]}',              # array value
    '{"corrected_query": "!!!"}',                  # folds to empty -> zero-overlap guard
    '{"corrected_query": null, "category": "ATM"}',  # explicit JSON null
    'not json at all',                             # not a JSON object -> hard None
])
def test_corrected_query_validate_never_raises_on_garbage(raw):
    out, _dropped = L._validate(raw, "quan cafe yen tinh")  # must not raise on any of these
    assert out is None or out["corrected_query"] is None    # garbage never yields a correction


# --------------------------------------------------------------------------- #
# Eager (region x model-id) resolution at construction                        #
# --------------------------------------------------------------------------- #
_REGION_ENVS = (
    "SEMSEARCH_BEDROCK_REGION", "SEMSEARCH_BEDROCK_REGIONS", "AWS_REGION", "AWS_DEFAULT_REGION",
)


def _use_case_block() -> Exception:
    """The region-scoped Anthropic 'use case details form' block seen live in ap-southeast-1."""
    from botocore.exceptions import ClientError

    return ClientError(
        {"Error": {"Code": "ResourceNotFoundException",
                   "Message": "Could not resolve the model; complete the Anthropic use case form."}},
        "Converse",
    )


def _invalid_model() -> Exception:
    from botocore.exceptions import ClientError

    return ClientError(
        {"Error": {"Code": "ValidationException",
                   "Message": "The provided model identifier is invalid."}},
        "Converse",
    )


def _region_boto_factory(monkeypatch, per_region):
    """Monkeypatch boto3.client with a factory keyed on region_name (NO network).
    `per_region(region) -> FakeConverseClient`. Returns the dict of clients built."""
    built: dict = {}

    def factory(service, *, region_name=None, config=None):
        assert service == "bedrock-runtime"
        client = per_region(region_name)
        built[region_name] = client
        return client

    monkeypatch.setattr("boto3.client", factory)
    return built


def test_llm_parser_resolves_region_model_matrix(monkeypatch):
    """Walk regions OUTER, model-id chain INNER. Region A (ap-southeast-1) blocks Claude
    entirely (every id -> the use-case ResourceNotFoundException); region B (ap-northeast-1)
    serves it via the global. profile (apac id invalid, global id answers). The parser must
    pin B + the global id, with the converse-ping counts exactly as walked."""
    for var in _REGION_ENVS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv(L.CLAUDE_MODEL_ENV, raising=False)
    chain = [L.DEFAULT_CLAUDE_MODEL, *L.FALLBACK_CLAUDE_MODELS]  # apac, global, plain

    def per_region(region):
        if region == "ap-southeast-1":
            return FakeConverseClient(per_model={mid: _use_case_block() for mid in chain})
        # ap-northeast-1: apac id invalid, global id answers (plain never reached)
        return FakeConverseClient(per_model={chain[0]: _invalid_model(), chain[1]: "ok"})

    built = _region_boto_factory(monkeypatch, per_region)
    parser = L.LLMParser()

    assert parser._region == "ap-northeast-1"          # region A blocked -> pinned region B
    assert parser.model_id == chain[1]                  # the global. profile id
    assert parser._client is built["ap-northeast-1"]
    assert "us-west-2" not in built                     # stopped at the first working combo
    # ping counts: region A tried all 3 ids; region B tried apac (fail) + global (success)
    assert [c["modelId"] for c in built["ap-southeast-1"].calls] == chain
    assert [c["modelId"] for c in built["ap-northeast-1"].calls] == chain[:2]


def test_llm_parser_resolves_nothing_is_unavailable(monkeypatch):
    """No (region, model) answers -> the parser is unavailable and parse() returns None,
    exactly like today's parse-failure path (pipeline then serves the rule intent)."""
    for var in _REGION_ENVS:
        monkeypatch.delenv(var, raising=False)
    _region_boto_factory(
        monkeypatch, lambda region: FakeConverseClient(raise_exc=RuntimeError(f"{region} blocked"))
    )
    parser = L.LLMParser()
    assert parser._client is None and parser._region is None
    assert parser.parse("cà phê yên tĩnh") is None


def test_pinned_parser_does_not_rewalk_after_a_call_fails(monkeypatch):
    """Determinism/latency: once pinned, a per-call failure degrades to None; it MUST NOT
    construct new clients (no mid-demo region re-walk)."""
    for var in _REGION_ENVS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv(L.CLAUDE_MODEL_ENV, raising=False)
    built = _region_boto_factory(monkeypatch, lambda region: FakeConverseClient("ok"))
    parser = L.LLMParser()
    assert parser._region == "ap-southeast-1"  # closest region answers, pinned first
    n_clients = len(built)

    # the pinned call now starts failing (transient 5xx / creds expired mid-demo)
    parser._client = FakeConverseClient(raise_exc=RuntimeError("5xx mid-demo"))
    assert parser.parse("q") is None            # degrades to None
    assert len(built) == n_clients              # no new boto3.client constructed -> no re-walk


def test_non_model_error_returns_none_single_call():
    client = FakeConverseClient(raise_exc=RuntimeError("throttled"))
    out = make_parser(client).parse("q")
    assert out is None
    assert len(client.calls) == 1  # pinned model is called once; failure degrades, no re-walk


# --------------------------------------------------------------------------- #
# OpenAI fallback: Bedrock (all blocked) -> OpenAI (key discoverable) -> rules #
#                                                                             #
# HARD RULE under test throughout: the API key must NEVER appear in logs,     #
# exception messages, traces, or test output — assertions use FAKE keys and   #
# verify every captured log record is key-free.                               #
# --------------------------------------------------------------------------- #
FAKE_KEY = "sk-proj-FAKE-KEY-1234567890abcdefFAKE"


def _assert_no_key_in_logs(caplog):
    for rec in caplog.records:
        assert FAKE_KEY not in rec.getMessage(), "API key leaked into a log record"


def _chat_completion(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def _mock_openai(monkeypatch, handler):
    """Replace the OpenAI client factory with an httpx MockTransport (NO network).
    Returns the list of recorded httpx.Request objects."""
    import httpx

    calls: list = []

    def recording(request):
        calls.append(request)
        return handler(request)

    monkeypatch.setattr(
        L.LLMParser, "_make_openai_client",
        staticmethod(lambda: httpx.Client(transport=httpx.MockTransport(recording))),
    )
    return calls


def _openai_ok_handler(content: str = "{}"):
    import httpx

    def handler(request):
        return httpx.Response(200, json=_chat_completion(content))

    return handler


# ---- key discovery ---------------------------------------------------------- #
def test_openai_key_env_wins_over_file(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    (root / ".env").mkdir(parents=True)
    (root / ".env" / "OPENAI-API-key.txt").write_text("sk-file-key\n", encoding="utf-8")
    monkeypatch.setattr(L, "_REPO_ROOT", root)
    monkeypatch.setenv("OPENAI_API_KEY", "  sk-env-key\n")  # whitespace stripped for env too
    assert L.discover_openai_key() == ("sk-env-key", "env")


@pytest.mark.parametrize("name", ["OPENAI-API-key.txt", "openai-api-key.txt"])
def test_openai_key_file_both_casings_whitespace_stripped(monkeypatch, tmp_path, name):
    root = tmp_path / "repo"
    (root / ".env").mkdir(parents=True)
    (root / ".env" / name).write_text("  sk-file-key \n\n", encoding="utf-8")
    monkeypatch.setattr(L, "_REPO_ROOT", root)
    assert L.discover_openai_key() == ("sk-file-key", ".env file")


def test_openai_key_absent_returns_none():
    # autouse fixture: env cleared + repo root pointed at an empty tmp dir
    assert L.discover_openai_key() is None


# ---- resolution order ------------------------------------------------------- #
def test_bedrock_success_means_openai_never_contacted(monkeypatch):
    """Bedrock resolving first means OpenAI is NEVER contacted — the autouse factory raises
    AssertionError on any OpenAI client construction, so pinning bedrock proves it."""
    for var in _REGION_ENVS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)  # key present, must stay untouched
    monkeypatch.setattr("boto3.client", lambda *a, **k: FakeConverseClient("pong"))
    parser = L.LLMParser()
    assert parser._provider == "bedrock"
    assert parser._region == "ap-southeast-1"


def test_bedrock_blocked_with_key_pins_openai(monkeypatch, caplog):
    """All Bedrock candidates fail + a key exists -> ONE eager ping (max_tokens=1) pins
    provider='openai' with the default model; the source is logged, the key never is."""
    for var in _REGION_ENVS:
        monkeypatch.delenv(var, raising=False)
    calls = _mock_openai(monkeypatch, _openai_ok_handler())
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    with caplog.at_level("INFO"):
        parser = L.LLMParser()

    assert parser._provider == "openai"
    assert parser._client is not None
    assert parser.model_id == "gpt-4.1-nano"       # SEMSEARCH_OPENAI_MODEL default
    assert parser._region is None                   # no region concept for openai
    assert len(calls) == 1                          # exactly ONE eager construction ping
    ping = json.loads(calls[0].content)
    assert ping["max_tokens"] == 1                  # the ping is tiny
    assert calls[0].headers["authorization"] == f"Bearer {FAKE_KEY}"
    assert any("OpenAI key found (env)" in r.getMessage() for r in caplog.records)
    _assert_no_key_in_logs(caplog)


def test_openai_model_env_override(monkeypatch):
    for var in _REGION_ENVS:
        monkeypatch.delenv(var, raising=False)
    _mock_openai(monkeypatch, _openai_ok_handler())
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    monkeypatch.setenv(L.OPENAI_MODEL_ENV, "gpt-4.1-mini")
    parser = L.LLMParser()
    assert parser._provider == "openai" and parser.model_id == "gpt-4.1-mini"


def test_gpt5_override_payload_branching(monkeypatch):
    """A gpt-5* env override must not 400: that family rejects non-default `temperature`
    (omit it) and the legacy `max_tokens` (use `max_completion_tokens`). response_format
    json_object still applies on the parse call."""
    for var in _REGION_ENVS:
        monkeypatch.delenv(var, raising=False)
    import httpx

    def handler(request):
        body = json.loads(request.content)
        assert "temperature" not in body            # gpt-5* rejects non-default temperature
        assert "max_tokens" not in body             # legacy param rejected by the family
        assert "max_completion_tokens" in body
        content = "ok" if body["max_completion_tokens"] == 1 else VALID_JSON
        return httpx.Response(200, json=_chat_completion(content))

    calls = _mock_openai(monkeypatch, handler)
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    monkeypatch.setenv(L.OPENAI_MODEL_ENV, "gpt-5-nano")
    parser = L.LLMParser()
    assert parser._provider == "openai" and parser.model_id == "gpt-5-nano"
    out = parser.parse("cà phê wifi")
    assert out is not None and out["attributes"] == ["wifi", "yên tĩnh"]
    req = json.loads(calls[-1].content)
    assert req["max_completion_tokens"] == 300
    assert req["response_format"] == {"type": "json_object"}


def test_no_key_stays_unavailable_rule_intent_path():
    """Bedrock blocked + no key discoverable -> unavailable, exactly today's behavior."""
    parser = L.LLMParser()  # autouse defaults: bedrock fails, no key
    assert parser._client is None and parser._provider is None
    assert parser.parse("cà phê yên tĩnh") is None  # pipeline then serves the rule intent


# ---- OpenAI call path: request shape + provider-agnostic validation ---------- #
def test_openai_parse_request_shape_and_validation_identical(monkeypatch):
    """The OpenAI path reuses SYSTEM_PROMPT + _validate: out-of-vocab attrs and location keys
    coming back via OpenAI are dropped exactly as on the Bedrock path."""
    for var in _REGION_ENVS:
        monkeypatch.delenv(var, raising=False)
    import httpx

    # same canned parse as VALID_JSON, plus a corrected_query — it must flow through the
    # provider-agnostic _validate identically to the Bedrock path.
    valid_with_correction = _llm_json(
        category="Nhà hàng",
        attributes=["wifi", "yên tĩnh", "teleport"],
        city="Hà Nội", district="Quận 1",
        price_pref="cheap", open_after="18:00",
        corrected_query="quán cà phê wifi",
    )

    def handler(request):
        body = json.loads(request.content)
        if body.get("max_tokens") == 1:  # the construction ping
            return httpx.Response(200, json=_chat_completion("ok"))
        return httpx.Response(200, json=_chat_completion(valid_with_correction))

    calls = _mock_openai(monkeypatch, handler)
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    parser = L.LLMParser()
    out = parser.parse("cà phê wifi")

    # validation is the SAME safety boundary as bedrock: teleport + location dropped
    assert out is not None
    assert out["category"] == "Nhà hàng"
    assert out["attributes"] == ["wifi", "yên tĩnh"]
    assert out["corrected_query"] == "quán cà phê wifi"  # correction validated via OpenAI path
    assert "city" not in out and "district" not in out

    req = json.loads(calls[-1].content)
    assert req["model"] == "gpt-4.1-nano"
    assert req["temperature"] == 0.0
    assert req["max_tokens"] == 300
    assert req["response_format"] == {"type": "json_object"}
    assert req["messages"][0]["role"] == "system"
    assert "ALLOWED CATEGORIES" in req["messages"][0]["content"]  # SYSTEM_PROMPT reused
    assert req["messages"][1] == {"role": "user", "content": "cà phê wifi"}


# ---- failure modes: 401 / timeout; no key in logs; no re-resolution ---------- #
def test_openai_401_after_pin_degrades_no_key_in_logs(monkeypatch, caplog):
    """After pinning, a per-call 401 (body deliberately embeds the fake key, mimicking
    OpenAI's 'Incorrect API key provided' body) degrades to None — and NO log record may
    contain the key. No re-resolution happens (transport call count stays ping+1)."""
    for var in _REGION_ENVS:
        monkeypatch.delenv(var, raising=False)
    import httpx

    state = {"pinged": False}

    def handler(request):
        if not state["pinged"]:
            state["pinged"] = True
            return httpx.Response(200, json=_chat_completion("ok"))
        return httpx.Response(
            401, json={"error": {"message": f"Incorrect API key provided: {FAKE_KEY}"}}
        )

    calls = _mock_openai(monkeypatch, handler)
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    with caplog.at_level("DEBUG"):
        parser = L.LLMParser()
        assert parser._provider == "openai"
        assert parser.parse("q") is None            # 401 degrades to the rule-intent path
        assert parser._provider == "openai"          # still pinned: no mid-demo re-resolution
    assert len(calls) == 2                           # ping + ONE failed call, nothing more
    _assert_no_key_in_logs(caplog)


def test_openai_ping_timeout_leaves_parser_unavailable(monkeypatch, caplog):
    """Key exists but the eager ping times out -> unavailable (rule fallback), key-free logs."""
    for var in _REGION_ENVS:
        monkeypatch.delenv(var, raising=False)
    import httpx

    def handler(request):
        raise httpx.ConnectTimeout("connection timed out")

    _mock_openai(monkeypatch, handler)
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    with caplog.at_level("DEBUG"):
        parser = L.LLMParser()
    assert parser._client is None and parser._provider is None
    assert parser.parse("q") is None
    _assert_no_key_in_logs(caplog)


def test_openai_unavailable_serves_rule_intent_via_pipeline(monkeypatch, pois):
    """End-to-end wiring: gate ON, bedrock blocked, ping 401 -> resolve_intent falls back to
    the rule intent with the existing warn-once path."""
    import httpx

    for var in _REGION_ENVS:
        monkeypatch.delenv(var, raising=False)
    _mock_openai(monkeypatch, lambda request: httpx.Response(
        401, json={"error": {"message": f"Incorrect API key provided: {FAKE_KEY}"}}))
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("SEMSEARCH_LLM_PARSE", "bedrock")
    pipe = FullPipeline(pois, mode="local")
    assert pipe._llm_parser is not None and pipe._llm_parser._client is None
    q = "quán cà phê yên tĩnh"
    assert pipe.resolve_intent(q) == pipe.parser.parse(q)


# ---- tracing carries provider + model, never the key ------------------------- #
def test_openai_parse_traces_provider_and_model_never_key(fake_langfuse, monkeypatch):
    for var in _REGION_ENVS:
        monkeypatch.delenv(var, raising=False)
    import httpx

    def handler(request):
        body = json.loads(request.content)
        if body.get("max_tokens") == 1:
            return httpx.Response(200, json=_chat_completion("ok"))
        return httpx.Response(200, json=_chat_completion(VALID_JSON))

    _mock_openai(monkeypatch, handler)
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    parser = L.LLMParser()
    out = parser.parse("cà phê wifi")
    assert out is not None

    kw, span = fake_langfuse.last.observations[0]
    assert kw["name"] == "llm_parse"
    assert kw["model"] == "gpt-4.1-nano"             # pinned model name in the trace
    metas = [u["metadata"] for u in span.updates if "metadata" in u]
    assert any(m.get("provider") == "openai" for m in metas)  # pinned provider in metadata
    # the key appears NOWHERE in the emitted observation or updates
    emitted = json.dumps({"kw": {k: str(v) for k, v in kw.items()},
                          "updates": [str(u) for u in span.updates]})
    assert FAKE_KEY not in emitted


# --------------------------------------------------------------------------- #
# merge_intent: UNION, rules win, new object, location never merged           #
# --------------------------------------------------------------------------- #
def _rule(**kw) -> QueryIntent:
    base = dict(raw="q", normalized="q")
    base.update(kw)
    return QueryIntent(**base)


def test_merge_rules_win_llm_fills_gaps():
    rule = _rule(category="Quán cà phê", required_attrs=["yên tĩnh"], city=None, district=None)
    out = {"category": "Nhà hàng", "attributes": ["wifi", "yên tĩnh"],
           "price_pref": "cheap", "open_after": "18:00"}
    merged = L.merge_intent(rule, out)
    assert merged.category == "Quán cà phê"          # rule wins the conflict
    assert merged.required_attrs == ["yên tĩnh", "wifi"]  # union, no dup, rule-first
    assert merged.price_pref == "cheap" and merged.open_after == "18:00"
    # a NEW object; rule untouched
    assert merged is not rule
    assert rule.category == "Quán cà phê" and rule.required_attrs == ["yên tĩnh"]


def test_merge_never_accepts_location_even_if_present():
    """OWNERSHIP guard: even a dict that somehow still carries city/district (stale caller,
    hostile input) must not leak location into the intent — it feeds the HARD filter."""
    rule = _rule(category=None, city=None, district=None)
    merged = L.merge_intent(rule, {"category": "Nhà hàng", "attributes": [],
                                   "city": "TP.HCM", "district": "Quận 9",
                                   "price_pref": None, "open_after": None})
    assert merged.city is None and merged.district is None
    assert merged.category == "Nhà hàng"  # in-contract fields still merge


def test_merge_category_filled_only_when_rule_missing():
    out = {"category": "Khách sạn", "attributes": [], "price_pref": None, "open_after": None}
    assert L.merge_intent(_rule(category=None), out).category == "Khách sạn"
    assert L.merge_intent(_rule(category="ATM"), out).category == "ATM"  # rule wins


def test_merge_preserves_rule_owned_fields():
    from semsearch.data import Anchor
    a = Anchor(name="X", lat=1.0, lon=2.0)
    rule = _rule(category=None, anchor=a, city="Hà Nội", district="Quận 1",
                 content_terms=["bun", "cha"], residual_terms=["bun", "cha"],
                 has_residual=True)
    merged = L.merge_intent(rule, {"category": "Nhà hàng", "attributes": [],
                                   "price_pref": None, "open_after": None})
    assert merged.anchor is a  # anchor/location/content/residual stay rule-owned
    assert merged.city == "Hà Nội" and merged.district == "Quận 1"
    assert merged.content_terms == ["bun", "cha"] and merged.residual_terms == ["bun", "cha"]
    assert merged.has_residual is True


def test_merge_none_returns_rule_intent_unchanged():
    rule = _rule(category="ATM")
    assert L.merge_intent(rule, None) is rule
    assert L.merge_intent(rule, {}) is rule


def test_merge_ignores_corrected_query():
    """merge_intent must not read corrected_query — the merged intent is field-for-field
    identical whether or not the LLM output carries a correction (a later worker consumes
    corrected_query at the pipeline layer, not the intent merge)."""
    rule = _rule(category=None, required_attrs=["yên tĩnh"], city=None, district=None)
    without = {"category": "Nhà hàng", "attributes": ["wifi"],
               "price_pref": "cheap", "open_after": "18:00"}
    with_correction = {**without, "corrected_query": "quán cà phê yên tĩnh wifi"}
    assert L.merge_intent(rule, with_correction) == L.merge_intent(rule, without)


# --------------------------------------------------------------------------- #
# Disk cache: same (prompt version, provider, model, query) served from disk;  #
# failures never cached; prompt-version salting; use_cache=False bypass.        #
# --------------------------------------------------------------------------- #
def test_parse_caches_validated_result_client_called_once():
    client = FakeConverseClient(_llm_json(category="ATM"))
    parser = make_parser(client)
    first = parser.parse("cà phê")
    second = parser.parse("cà phê")           # served from disk, no second converse
    assert len(client.calls) == 1
    assert second == first and first is not None


def test_parse_never_caches_failures_and_reruns_client():
    client = FakeConverseClient("not json at all")  # _validate -> None (hard failure)
    parser = make_parser(client)
    assert parser.parse("cà phê") is None
    assert not L.LLMCACHE_DIR.exists()         # a None result writes NOTHING to the cache
    assert parser.parse("cà phê") is None      # so the next call must hit the client again
    assert len(client.calls) == 2


def test_cache_key_salts_on_prompt_version(monkeypatch):
    client = FakeConverseClient(_llm_json(category="ATM"))
    parser = make_parser(client)
    parser.parse("cà phê")
    assert len(client.calls) == 1
    monkeypatch.setattr(L, "PROMPT_VERSION", "v-reworded")  # a prompt edit re-keys the cache
    parser.parse("cà phê")
    assert len(client.calls) == 2              # new key -> cache miss -> client re-invoked


def test_use_cache_false_skips_read_and_write():
    client = FakeConverseClient(_llm_json(category="ATM"))
    parser = make_parser(client)
    parser.parse("cà phê", use_cache=False)
    parser.parse("cà phê", use_cache=False)
    assert len(client.calls) == 2              # no read: every call reaches the client
    assert not L.LLMCACHE_DIR.exists()          # no write: nothing persisted


# --------------------------------------------------------------------------- #
# Tracing is a silent no-op without Langfuse keys (no import, no network)      #
# --------------------------------------------------------------------------- #
def test_tracing_noop_without_keys(monkeypatch):
    monkeypatch.delenv(tracing.PUBLIC_KEY_ENV, raising=False)
    monkeypatch.delenv(tracing.SECRET_KEY_ENV, raising=False)
    monkeypatch.setattr(tracing, "_client", None)
    assert tracing.enabled() is False
    with tracing.traced("x", kind="generation", model="m", input="q") as h:
        h.update(output="o", metadata={"a": 1})  # must not raise
    tracing.flush()  # must not raise
    assert tracing._get_client() is None  # never built a client -> never touched langfuse/net


def test_llm_parse_runs_with_tracing_hook_noop(monkeypatch):
    monkeypatch.delenv(tracing.PUBLIC_KEY_ENV, raising=False)
    monkeypatch.delenv(tracing.SECRET_KEY_ENV, raising=False)
    monkeypatch.setattr(tracing, "_client", None)
    out = make_parser(FakeConverseClient(VALID_JSON)).parse("q")  # parse wraps in traced()
    assert out is not None and out["category"] == "Nhà hàng"


# --------------------------------------------------------------------------- #
# Tracing with keys present: fake langfuse module, assert the emit shape       #
# --------------------------------------------------------------------------- #
class _FakeSpan:
    raise_on_update = False  # class-level toggle set by tests

    def __init__(self):
        self.updates: list[dict] = []

    def update(self, **kw):
        if _FakeSpan.raise_on_update:
            raise RuntimeError("emit failed")
        self.updates.append(kw)


class _FakeObsCM:
    def __init__(self, span):
        self._span = span

    def __enter__(self):
        return self._span

    def __exit__(self, *args):
        return False


class _FakeLangfuse:
    last: "_FakeLangfuse | None" = None

    def __init__(self):
        self.observations: list[tuple[dict, _FakeSpan]] = []
        self.flushes = 0
        _FakeLangfuse.last = self

    def start_as_current_observation(self, **kw):
        span = _FakeSpan()
        self.observations.append((kw, span))
        return _FakeObsCM(span)

    def flush(self):
        self.flushes += 1


@pytest.fixture
def fake_langfuse(monkeypatch):
    """Keys present + a fake `langfuse` module in sys.modules: exercises the real emit
    path in tracing.py without the SDK or the network."""
    mod = types.ModuleType("langfuse")
    mod.Langfuse = _FakeLangfuse
    _FakeLangfuse.last = None
    _FakeSpan.raise_on_update = False
    monkeypatch.setitem(sys.modules, "langfuse", mod)
    monkeypatch.setenv(tracing.PUBLIC_KEY_ENV, "pk-test")
    monkeypatch.setenv(tracing.SECRET_KEY_ENV, "sk-test")
    monkeypatch.setattr(tracing, "_client", None)  # drop any cached client; restored on teardown
    return _FakeLangfuse


def test_traced_emits_observation_with_keys(fake_langfuse):
    with tracing.traced("llm_parse", kind="generation", model="m1", input="q") as h:
        h.update(output="raw-out")
    client = fake_langfuse.last
    assert client is not None, "keys present -> a client must be built"
    kw, span = client.observations[0]
    assert kw["name"] == "llm_parse" and kw["as_type"] == "generation"
    assert kw["model"] == "m1" and kw["input"] == "q"
    assert any(u.get("output") == "raw-out" for u in span.updates)
    assert any("latency_ms" in u.get("metadata", {}) for u in span.updates)  # latency recorded
    tracing.flush()
    assert client.flushes == 1


def test_llm_parse_emits_validated_and_dropped(fake_langfuse):
    out = make_parser(FakeConverseClient(VALID_JSON)).parse("cà phê wifi")
    kw, span = fake_langfuse.last.observations[0]
    assert kw["name"] == "llm_parse" and kw["input"] == "cà phê wifi"
    metas = [u["metadata"] for u in span.updates if "metadata" in u]
    validated = next(m["validated"] for m in metas if "validated" in m)
    assert validated == out  # the emitted validated dict IS what the parser returned
    dropped = next(m["dropped"] for m in metas if m.get("dropped"))
    assert "teleport" in dropped["attributes"]  # rejected values are visible in the trace


def test_bedrock_embed_traced_name_and_count_only(fake_langfuse):
    from semsearch import embeddings as E

    class _FakeInvoke:
        def invoke_model(self, *, modelId, body):  # noqa: N803
            n = len(json.loads(body)["texts"])

            class _Body:
                def __init__(self, data):
                    self._data = data

                def read(self):
                    return self._data

            payload = json.dumps({"embeddings": [[1.0] * E.EMBED_DIM] * n}).encode()
            return {"body": _Body(payload)}

    emb = E.BedrockEmbedder("bedrock-cohere")
    emb._client = _FakeInvoke()
    emb.embed(["a", "b", "c"])
    kw, _span = fake_langfuse.last.observations[0]
    assert kw["name"] == "bedrock_embed" and kw["as_type"] == "embedding"
    assert kw["metadata"]["count"] == 3
    assert kw.get("input") is None  # never the texts themselves


def test_emit_failure_swallowed(fake_langfuse):
    _FakeSpan.raise_on_update = True
    with tracing.traced("x", kind="generation", model="m", input="q") as h:
        h.update(output="o")  # raises inside the fake; must be swallowed
    # and the wrapped LLM parse still succeeds end-to-end
    out = make_parser(FakeConverseClient(VALID_JSON)).parse("q")
    assert out is not None and out["category"] == "Nhà hàng"


# --------------------------------------------------------------------------- #
# Pipeline gate                                                               #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def pois():
    return load_pois()


@pytest.fixture(scope="module")
def gate_pipe(pois):
    """A FullPipeline constructed with the LLM gate ON. Construction now resolves the parser
    EAGERLY, so we patch boto3.client to fail during construction (offline, no network): the
    parser ends up unavailable, and each test injects a FakeConverseClient onto
    pipe._llm_parser._client."""
    prev = os.environ.get("SEMSEARCH_LLM_PARSE")
    os.environ["SEMSEARCH_LLM_PARSE"] = "bedrock"
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("boto3.client", lambda *a, **k: _AlwaysFailConverse())
            mp.delenv("OPENAI_API_KEY", raising=False)
            # Force the LLM gate to "always" so these clean-query gate tests keep exercising a
            # real per-query LLM call — under the "auto" default they would be gated OFF (the
            # degradation gate is covered by its own tests below).
            mp.setenv("SEMSEARCH_LLM_GATE", "always")
            mp.setattr(L, "_REPO_ROOT", Path("/nonexistent-semsearch-tests"))  # hide real .env/
            mp.setattr(L.LLMParser, "_make_openai_client", staticmethod(_no_openai_client))
            pipe = FullPipeline(pois, mode="local")
    finally:
        if prev is None:
            os.environ.pop("SEMSEARCH_LLM_PARSE", None)
        else:
            os.environ["SEMSEARCH_LLM_PARSE"] = prev
    return pipe


def test_gate_off_never_constructs_llmparser(monkeypatch, pois):
    monkeypatch.delenv("SEMSEARCH_LLM_PARSE", raising=False)

    def _boom(*a, **k):
        raise AssertionError("LLMParser must not be constructed when the gate is off")

    monkeypatch.setattr("semsearch.pipeline.LLMParser", _boom)
    pipe = FullPipeline(pois, mode="local")
    assert pipe._llm_parser is None
    # gate-off intent is byte-identical to the plain rule parse (no new code path runs)
    q = "quán cà phê yên tĩnh"
    assert pipe.resolve_intent(q) == pipe.parser.parse(q)


def test_gate_on_constructs_parser_resolves_at_construction(gate_pipe):
    assert gate_pipe._llm_parser is not None
    # Eager resolution ran at construction but found nothing reachable (offline), so no
    # boto3 client persists — behaves as today's parse-failure path.
    assert gate_pipe._llm_parser._client is None
    assert gate_pipe._llm_parser._region is None


def test_gate_on_unavailable_serves_rule_intent(gate_pipe):
    """A construction that resolves nothing must behave exactly like the parse-failure path:
    resolve_intent is byte-identical to the plain rule parse. (Runs before any test injects a
    client onto the shared gate pipeline.)"""
    q = "quán cà phê yên tĩnh"
    assert gate_pipe._llm_parser._client is None  # unavailable at this point in the module
    assert gate_pipe.resolve_intent(q) == gate_pipe.parser.parse(q)


def test_resolve_intent_falls_back_to_rules_on_failure(gate_pipe):
    q = "quán cà phê yên tĩnh"
    expected = gate_pipe.parser.parse(q)
    gate_pipe._llm_parser._client = FakeConverseClient(raise_exc=RuntimeError("no creds"))
    got = gate_pipe.resolve_intent(q)
    assert got == expected  # rule intent used alone, unchanged


def test_resolve_intent_merges_llm_enhancement(gate_pipe):
    q = "cà phê"
    rule = gate_pipe.parser.parse(q)
    assert "wifi" not in rule.required_attrs  # the rule parse leaves a gap
    gate_pipe._llm_parser._client = FakeConverseClient(_llm_json(attributes=["wifi"]))
    got = gate_pipe.resolve_intent(q)
    assert "wifi" in got.required_attrs
    assert got.category == rule.category  # rule category preserved (LLM sent null)


def test_hallucinated_location_cannot_collapse_recall(gate_pipe):
    """The review's recall-collapse scenario: an LLM output carrying a hallucinated
    city/district must NOT reach the intent, so the pipeline's HARD location filter
    never fires on it and the lineup is exactly the rule-intent lineup."""
    q = "cà phê"
    rule = gate_pipe.parser.parse(q)
    baseline = [t[0] for t in gate_pipe.rank_scored(q, intent=rule)]
    gate_pipe._llm_parser._client = FakeConverseClient(
        _llm_json(city="TP.HCM", district="Quận 3"))  # hallucinated location keys
    intent = gate_pipe.resolve_intent(q)
    assert intent.city is None and intent.district is None  # location stays rule-owned
    assert [t[0] for t in gate_pipe.rank_scored(q, intent=intent)] == baseline


def test_resolve_intent_deterministic(gate_pipe):
    q = "cà phê"
    gate_pipe._llm_parser._client = FakeConverseClient(_llm_json(attributes=["wifi"]))
    assert gate_pipe.resolve_intent(q) == gate_pipe.resolve_intent(q)  # same query -> same intent


def test_llm_failure_warns_once(gate_pipe, caplog):
    gate_pipe._llm_warned = False
    gate_pipe._llm_parser._client = FakeConverseClient(raise_exc=RuntimeError("boom"))
    with caplog.at_level("WARNING"):
        gate_pipe.resolve_intent("cafe")
        gate_pipe.resolve_intent("nhà hàng")
    warned = [r for r in caplog.records if r.levelname == "WARNING" and "rule-parsed" in r.message]
    assert len(warned) == 1  # warned once, not per query
    gate_pipe._llm_warned = False  # reset for later tests in the module


def test_search_calls_llm_once_per_query(gate_pipe):
    client = FakeConverseClient(_llm_json(attributes=["wifi"]))
    gate_pipe._llm_parser._client = client
    intent, results = gate_pipe.search("cafe", k=3)
    assert results
    assert "wifi" in intent.required_attrs                 # enhancement reached the API surface
    assert len(client.calls) == 1  # resolved once, passed through to rank_scored (not twice)


# --------------------------------------------------------------------------- #
# Query rewrite: the LLM's corrected_query REPLACES the raw text for retrieval  #
# (rule parse + BM25 + dense + subject corroboration), while intent.raw stays   #
# the ORIGINAL. gate_pipe is module-scoped, so these build a fresh pipeline per  #
# case (mode='local' + SEMSEARCH_LLM_PARSE=bedrock, offline fake injected).      #
# --------------------------------------------------------------------------- #
def _pipe_llm_on(pois, monkeypatch, *, rewrite: str | None = None) -> FullPipeline:
    """A fresh LLM-on pipeline: the autouse `_offline_llm` fixture makes construction resolve
    nothing (boto3 fails, no key), so `_llm_parser._client` is None until the caller injects a
    FakeConverseClient. `rewrite` pins SEMSEARCH_QUERY_REWRITE; None leaves it at the default."""
    monkeypatch.setenv("SEMSEARCH_LLM_PARSE", "bedrock")
    if rewrite is None:
        monkeypatch.delenv("SEMSEARCH_QUERY_REWRITE", raising=False)
    else:
        monkeypatch.setenv("SEMSEARCH_QUERY_REWRITE", rewrite)
    return FullPipeline(pois, mode="local")


def test_query_rewrite_replaces_query_for_retrieval(pois, monkeypatch):
    """Rewrite ON (default): a corrected_query REPLACES the raw text for retrieval, so the
    top-k under the correcting LLM equals the top-k of the corrected text ranked with the LLM
    off — proof the corrected string genuinely drove BM25/dense/subject, not the raw typo."""
    raw = "quan cafe yen tinh wjfi"
    corrected = "quán cà phê yên tĩnh wifi"
    pipe = _pipe_llm_on(pois, monkeypatch)  # rewrite default -> True
    pipe._llm_parser._client = FakeConverseClient(_llm_json(corrected_query=corrected))

    intent = pipe.resolve_intent(raw)
    assert intent.corrected_query == corrected  # the correction is carried on the intent
    assert intent.raw == raw                     # the echo stays the ORIGINAL raw text
    got = pipe.rank_ids(raw)

    monkeypatch.delenv("SEMSEARCH_LLM_PARSE", raising=False)  # a plain, LLM-off pipeline
    plain = FullPipeline(pois, mode="local")
    assert plain._llm_parser is None
    assert got == plain.rank_ids(corrected)  # retrieval used the corrected string, not the raw


def test_query_rewrite_off_switch_is_inert_but_enrichment_works(pois, monkeypatch):
    """SEMSEARCH_QUERY_REWRITE=off: no replacement happens (corrected_query stays None and the
    ranking is the raw-text ranking) — but the rest of the LLM enrichment (merged attributes)
    still flows through, proving the off-switch disables ONLY the query replacement."""
    raw = "quan cafe yen tinh wjfi"
    corrected = "quán cà phê yên tĩnh wifi"
    pipe = _pipe_llm_on(pois, monkeypatch, rewrite="off")
    pipe._llm_parser._client = FakeConverseClient(
        _llm_json(corrected_query=corrected, attributes=["wifi"]))

    intent = pipe.resolve_intent(raw)
    assert intent.corrected_query is None        # off -> the correction never replaces the raw
    assert "wifi" in intent.required_attrs        # enrichment still merged

    merged = L.merge_intent(pipe.parser.parse(raw),
                            {"category": None, "attributes": ["wifi"],
                             "price_pref": None, "open_after": None})
    expected = [pid for pid, _, _ in pipe.rank_scored(raw, intent=merged)]
    assert pipe.rank_ids(raw) == expected         # retrieval ran on the RAW text, wifi merged


def test_query_rewrite_llm_off_is_rule_parse_with_no_correction(pois, monkeypatch):
    """LLM gate off (mode local, no env): resolve_intent is the plain rule parse on the raw
    text and corrected_query is None — byte-identical to today."""
    monkeypatch.delenv("SEMSEARCH_LLM_PARSE", raising=False)
    monkeypatch.delenv("SEMSEARCH_QUERY_REWRITE", raising=False)
    pipe = FullPipeline(pois, mode="local")
    assert pipe._llm_parser is None
    q = "quan cafe yen tinh wjfi"
    intent = pipe.resolve_intent(q)
    assert intent == pipe.parser.parse(q)
    assert intent.corrected_query is None


def test_query_rewrite_exactly_one_llm_call(pois, monkeypatch):
    """The new order preserves the one-LLM-call-per-query guarantee: search resolves the intent
    once (the single LLM call) and threads it into rank_scored."""
    pipe = _pipe_llm_on(pois, monkeypatch)
    client = FakeConverseClient(_llm_json(corrected_query="quán cà phê yên tĩnh wifi"))
    pipe._llm_parser._client = client
    intent, results = pipe.search("quan cafe yen tinh wjfi", k=3)
    assert results and intent.corrected_query == "quán cà phê yên tĩnh wifi"
    assert len(client.calls) == 1  # exactly one converse for the whole search


# --------------------------------------------------------------------------- #
# API path: /v1/semantic-search echo + reasons + results share ONE intent      #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def gate_app(pois):
    """create_app with the LLM gate ON; prewarm=False so no query (and no boto3 client)
    runs before a test injects the fake converse client."""
    prev = os.environ.get("SEMSEARCH_LLM_PARSE")
    os.environ["SEMSEARCH_LLM_PARSE"] = "bedrock"
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("boto3.client", lambda *a, **k: _AlwaysFailConverse())
            mp.delenv("OPENAI_API_KEY", raising=False)
            mp.setenv("SEMSEARCH_LLM_GATE", "always")  # clean-query API tests must still call
            mp.setattr(L, "_REPO_ROOT", Path("/nonexistent-semsearch-tests"))  # hide real .env/
            mp.setattr(L.LLMParser, "_make_openai_client", staticmethod(_no_openai_client))
            app = create_app(pois, prewarm=False, mode="local")
    finally:
        if prev is None:
            os.environ.pop("SEMSEARCH_LLM_PARSE", None)
        else:
            os.environ["SEMSEARCH_LLM_PARSE"] = prev
    return app


def test_api_echo_reasons_results_share_merged_intent(gate_app):
    """The review's finding-1 scenario: with the gate ON, the intent echo, reasons[],
    and the returned ranking must all reflect the SAME LLM-merged intent — no rule-only
    re-parse anywhere on the /v1/semantic-search path."""
    pipe = gate_app.state.pipeline
    llm = {"category": None, "attributes": ["wifi"], "price_pref": None, "open_after": None}
    fake = FakeConverseClient(json.dumps(llm))
    pipe._llm_parser._client = fake
    body = TestClient(gate_app).get("/v1/semantic-search", params={"q": "cà phê"}).json()

    # (a) the echo shows the merged intent (LLM contributed wifi)
    assert "wifi" in body["intent"]["requiredAttrs"]
    # (b) intent resolved exactly ONCE for echo + reasons + ranking
    assert len(fake.calls) == 1
    # (c) the returned ranking IS the ranking under that same merged intent
    merged = L.merge_intent(pipe.parser.parse("cà phê"), llm)
    expected_ids = [f"poi:{pid}" for pid, _, _ in pipe.rank_scored("cà phê", intent=merged)[:10]]
    assert [res["id"] for res in body["results"]] == expected_ids
    # (d) reasons[] are built from the merged intent: a wifi POI carries "✓ wifi"
    wifi_reasons = [reason for res in body["results"] for reason in res["reasons"]
                    if "wifi" in reason]
    assert wifi_reasons, "merged required_attrs must drive the explanations"


def test_api_gate_off_unchanged(pois, monkeypatch):
    """Gate OFF (default): the API serves exactly the rule-parsed intent and its ranking —
    today's deterministic behavior (NFR-5)."""
    monkeypatch.delenv("SEMSEARCH_LLM_PARSE", raising=False)
    app = create_app(pois, prewarm=False, mode="local")
    pipe = app.state.pipeline
    assert pipe._llm_parser is None
    body = TestClient(app).get("/v1/semantic-search", params={"q": "cà phê"}).json()
    rule = pipe.parser.parse("cà phê")
    assert body["intent"]["category"] == rule.category
    assert body["intent"]["requiredAttrs"] == rule.required_attrs
    expected_ids = [f"poi:{pid}" for pid, _, _ in pipe.rank_scored("cà phê", intent=rule)[:10]]
    assert [res["id"] for res in body["results"]] == expected_ids


# --------------------------------------------------------------------------- #
# API path: the corrected query surfaces additively as meta.correctedQuery,    #
# only when it differs from the raw echo; the `query` echo stays the ORIGINAL.  #
# --------------------------------------------------------------------------- #
def _app_llm_on(pois, monkeypatch, *, rewrite: str | None = None):
    """A fresh gate-on app (prewarm=False, mode='local'); autouse `_offline_llm` makes the
    parser resolve nothing, so the caller injects a FakeConverseClient onto its `_client`."""
    monkeypatch.setenv("SEMSEARCH_LLM_PARSE", "bedrock")
    if rewrite is None:
        monkeypatch.delenv("SEMSEARCH_QUERY_REWRITE", raising=False)
    else:
        monkeypatch.setenv("SEMSEARCH_QUERY_REWRITE", rewrite)
    return create_app(pois, prewarm=False, mode="local")


def test_api_corrected_query_surfaced_in_meta_both_endpoints(pois, monkeypatch):
    """A correcting fake: both /v1/search and /v1/semantic-search echo the ORIGINAL query and
    add meta.correctedQuery == the corrected string (additive, contract-exact otherwise)."""
    raw = "quan cafe yen tinh wjfi q1"
    corrected = "quán cà phê yên tĩnh wifi quận 1"
    app = _app_llm_on(pois, monkeypatch)  # rewrite default -> True
    app.state.pipeline._llm_parser._client = FakeConverseClient(
        _llm_json(corrected_query=corrected))
    client = TestClient(app)
    for path in ("/v1/search", "/v1/semantic-search"):
        body = client.get(path, params={"q": raw}).json()
        assert body["query"] == raw                         # echo stays the ORIGINAL
        assert body["meta"]["correctedQuery"] == corrected   # additive correction surface


def test_api_no_corrected_query_key_absent_both_endpoints(pois, monkeypatch):
    """A non-correcting fake (only an attribute merge): meta carries NO correctedQuery key on
    either endpoint — a clean-query response stays byte-identical to the contract shape."""
    app = _app_llm_on(pois, monkeypatch)
    app.state.pipeline._llm_parser._client = FakeConverseClient(_llm_json(attributes=["wifi"]))
    client = TestClient(app)
    for path in ("/v1/search", "/v1/semantic-search"):
        body = client.get(path, params={"q": "cà phê"}).json()
        assert "correctedQuery" not in body["meta"]


def test_api_keyword_lane_no_corrected_query_no_llm_call(pois, monkeypatch):
    """The keyword lane (engine=keyword) is rule-only: no correctedQuery key AND no LLM call,
    even when a correcting fake is wired in."""
    app = _app_llm_on(pois, monkeypatch)
    fake = FakeConverseClient(_llm_json(corrected_query="quán cà phê"))
    app.state.pipeline._llm_parser._client = fake
    body = TestClient(app).get(
        "/v1/search", params={"q": "quan cafe", "engine": "keyword"}).json()
    assert "correctedQuery" not in body["meta"]
    assert len(fake.calls) == 0  # the keyword column never pays for (or depends on) the LLM


def test_prefer_openai_skips_bedrock_entirely(monkeypatch):
    """LLMParser(prefer='openai') must pin OpenAI without a single Bedrock probe, even when
    Bedrock WOULD succeed (SEMSEARCH_LLM_PARSE=openai: operator has a key, no Bedrock)."""
    for var in _REGION_ENVS:
        monkeypatch.delenv(var, raising=False)

    def _no_boto(*a, **k):
        raise AssertionError("bedrock must not be probed with prefer='openai'")

    monkeypatch.setattr("boto3.client", _no_boto)
    calls = _mock_openai(monkeypatch, _openai_ok_handler())
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    parser = L.LLMParser(prefer="openai")
    assert parser._provider == "openai" and parser._client is not None
    assert len(calls) == 1  # the single eager ping; no bedrock walk happened


# --------------------------------------------------------------------------- #
# Degradation gate (SEMSEARCH_LLM_GATE): with the LLM parse ON, the "auto"      #
# default SKIPS the ~1.7s call for a CLEAN, in-vocab query (byte-identical to   #
# the LLM-off path) and fires ONLY for a degraded query — no diacritics, a      #
# long out-of-vocab token, or an English/mixed-language word. "always" forces   #
# the call for every query. Pure function of query text + the static lexicon    #
# (NFR-5 deterministic). gate defaults to "auto" (autouse `_offline_llm` clears  #
# SEMSEARCH_LLM_GATE); each pipeline is fresh via `_pipe_llm_on`.               #
# --------------------------------------------------------------------------- #
def test_gate_auto_clean_in_vocab_query_skips_llm(pois, monkeypatch):
    """A clean, diacritic'd, in-vocab query makes ZERO LLM calls under the auto gate: the
    intent is byte-identical to the plain rule parse (corrected_query None) and NOTHING is
    written to the LLM disk cache (no call happened at all)."""
    pipe = _pipe_llm_on(pois, monkeypatch)  # gate defaults to "auto"
    assert pipe._llm_gate == "auto"
    client = FakeConverseClient(_llm_json(attributes=["wifi"]))  # would merge wifi IF called
    pipe._llm_parser._client = client
    q = "cà phê yên tĩnh"                      # diacritics present, all long tokens in-vocab
    intent = pipe.resolve_intent(q)
    assert len(client.calls) == 0              # gated OFF: the LLM was never called
    assert intent == pipe.parser.parse(q)      # byte-identical to the rule parse
    assert intent.corrected_query is None
    assert "wifi" not in intent.required_attrs  # the injected enrichment never merged
    assert not L.LLMCACHE_DIR.exists()          # gated-off queries make no cache entries


def test_gate_auto_no_diacritics_calls_llm(pois, monkeypatch):
    """A query with NO Vietnamese diacritic at all (likely diacritic-stripped typing) trips
    the gate -> exactly one LLM call, and its enrichment merges through."""
    pipe = _pipe_llm_on(pois, monkeypatch)
    client = FakeConverseClient(_llm_json(attributes=["wifi"]))
    pipe._llm_parser._client = client
    intent = pipe.resolve_intent("quan cafe")   # no diacritic anywhere -> degradation signal
    assert len(client.calls) == 1
    assert "wifi" in intent.required_attrs       # the LLM ran and merged


def test_gate_auto_oov_long_token_calls_llm(pois, monkeypatch):
    """A diacritic'd query carrying a long out-of-vocab token ('wjfi', a typo) trips the
    gate even though diacritics are present."""
    pipe = _pipe_llm_on(pois, monkeypatch)
    client = FakeConverseClient(_llm_json())
    pipe._llm_parser._client = client
    pipe.resolve_intent("quán cà phê wjfi")     # 'wjfi' (len 4) is out-of-vocab
    assert len(client.calls) == 1


def test_gate_auto_mixed_language_calls_llm(pois, monkeypatch):
    """Mixed-language: an English out-of-vocab word ('laptop') trips the gate even with
    Vietnamese diacritics present elsewhere in the query."""
    pipe = _pipe_llm_on(pois, monkeypatch)
    client = FakeConverseClient(_llm_json())
    pipe._llm_parser._client = client
    pipe.resolve_intent("cà phê có laptop")   # diacritics present; 'laptop' is OOV English
    assert len(client.calls) == 1


def test_gate_always_forces_call_on_clean_query(pois, monkeypatch):
    """SEMSEARCH_LLM_GATE=always restores today's behavior: even a clean, in-vocab query pays
    the LLM call (useful when demoing the correction itself)."""
    monkeypatch.setenv("SEMSEARCH_LLM_GATE", "always")
    pipe = _pipe_llm_on(pois, monkeypatch)       # _pipe_llm_on leaves SEMSEARCH_LLM_GATE alone
    assert pipe._llm_gate == "always"
    client = FakeConverseClient(_llm_json(attributes=["wifi"]))
    pipe._llm_parser._client = client
    intent = pipe.resolve_intent("cà phê")       # clean query that "auto" would gate OFF
    assert len(client.calls) == 1
    assert "wifi" in intent.required_attrs


def test_gate_unknown_env_value_warns_and_uses_auto(pois, monkeypatch, caplog):
    """An unknown SEMSEARCH_LLM_GATE value logs a warning and behaves as 'auto' (never
    silently forces every call): a clean in-vocab query is still gated OFF."""
    monkeypatch.setenv("SEMSEARCH_LLM_GATE", "banana")
    with caplog.at_level("WARNING"):
        pipe = _pipe_llm_on(pois, monkeypatch)
    assert pipe._llm_gate == "auto"
    assert any("banana" in r.getMessage() for r in caplog.records)
    client = FakeConverseClient(_llm_json(attributes=["wifi"]))
    pipe._llm_parser._client = client
    pipe.resolve_intent("cà phê")                # clean -> gated OFF under the auto fallback
    assert len(client.calls) == 0


def test_gate_off_query_byte_identical_to_llm_off_pipeline(pois, monkeypatch):
    """A gated-off query's resolved intent equals the LLM-OFF pipeline's parse of the same
    text — the auto gate's skip path is the same code path as SEMSEARCH_LLM_PARSE=off."""
    pipe = _pipe_llm_on(pois, monkeypatch)       # LLM on, gate auto
    pipe._llm_parser._client = FakeConverseClient(_llm_json(attributes=["wifi"]))
    q = "cà phê yên tĩnh"                         # clean + in-vocab -> gated OFF

    monkeypatch.delenv("SEMSEARCH_LLM_PARSE", raising=False)
    off = FullPipeline(pois, mode="local")       # LLM parse entirely off
    assert off._llm_parser is None
    assert pipe.resolve_intent(q) == off.resolve_intent(q)


def test_gate_decision_deterministic(pois, monkeypatch):
    """`_needs_llm` is a pure function of the query text + the static lexicon: the same query
    always yields the same decision, and the two signals resolve as documented (NFR-5)."""
    pipe = _pipe_llm_on(pois, monkeypatch)
    for q in ("cà phê", "quan cafe", "quán cà phê wjfi", "cà phê có laptop"):
        assert pipe._needs_llm(q) is pipe._needs_llm(q)  # repeatable
    assert pipe._needs_llm("cà phê") is False            # clean + in-vocab -> skip
    assert pipe._needs_llm("quan cafe") is True          # no diacritics -> call
    assert pipe._needs_llm("quán cà phê wjfi") is True   # OOV long token -> call
    assert pipe._needs_llm("cà phê có laptop") is True   # OOV English word -> call


def test_claude_timeout_really_means_no_retries():
    """Same botocore legacy-mode trap as embeddings: max_attempts=1 was one RETRY
    (two attempts, 6.2s measured stall vs the intended 3s). 0 = single attempt."""
    assert L._CLAUDE_TIMEOUT["retries"]["max_attempts"] == 0
