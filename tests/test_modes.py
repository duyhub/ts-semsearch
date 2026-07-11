"""Deployment modes (src/semsearch/config.py): local / local-first / cloud.

Every test here is OFFLINE. boto3 is replaced with an always-failing (or canned) client,
sentence_transformers is replaced with a sys.modules fake where a test needs to prove it is
(or is NOT) touched, and OpenAI key discovery is pointed at an empty tmp dir. Contracts:

  - resolve_mode precedence: env SEMSEARCH_MODE > config.DEFAULT_MODE; invalid -> 'local'
    with a warning.
  - mode='local': exactly today's behavior — local embedder, local failure raises loudly,
    cloud never contacted.
  - mode='local-first': local probed FIRST; broken local -> loud warning + cloud chain
    (bedrock-cohere then bedrock-titan across the region chain); healthy local is
    byte-identical to mode='local'.
  - mode='cloud': local NEVER attempted (sentence_transformers must not import); all cloud
    providers failing -> BM25-only floor (dense=None) that still serves non-empty results
    with the semantic signal correctly calibrated to 1.0 for the top BM25 hit.
  - LLM parse default matrix: cloud -> ON, local/local-first -> OFF; explicit
    SEMSEARCH_LLM_PARSE ('bedrock' force-on / 'off' force-off) always wins.
  - /health reports mode + what actually resolved.
  - explicit `provider=` stays an expert override that skips mode resolution.
"""
from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from semsearch import api as A
from semsearch import config as C
from semsearch import embeddings as E
from semsearch import llm_parse as L
from semsearch import pipeline as P
from semsearch.api import create_app
from semsearch.data import load_pois

DIM = E.EMBED_DIM


@pytest.fixture(scope="module")
def pois():
    return load_pois()


class _AlwaysFailBoto:
    """bedrock-runtime stand-in whose every call fails (offline default)."""

    def invoke_model(self, **_kw):
        raise RuntimeError("offline: no bedrock reachable in tests")

    def converse(self, **_kw):
        raise RuntimeError("offline: no bedrock reachable in tests")


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """Every test: tmp caches, clean env, offline bedrock, no OpenAI key discoverable."""
    monkeypatch.setattr(E, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(E, "QCACHE_DIR", tmp_path / "qcache")
    for var in ("SEMSEARCH_MODE", "SEMSEARCH_QUERY_REWRITE", "SEMSEARCH_LLM_PARSE",
                C.LLM_GATE_ENV, "SEMSEARCH_BEDROCK_REGION", "SEMSEARCH_BEDROCK_REGIONS",
                "AWS_REGION", "AWS_DEFAULT_REGION", "OPENAI_API_KEY", L.OPENAI_MODEL_ENV,
                L.CLAUDE_MODEL_ENV):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(L, "_REPO_ROOT", tmp_path / "no-repo")  # hide the real .env/ key
    monkeypatch.setattr(L, "LLMCACHE_DIR", tmp_path / "llmcache")  # isolate the LLM disk cache
    monkeypatch.setattr("boto3.client", lambda *a, **k: _AlwaysFailBoto())


class UnitLocal:
    """Deterministic fake local embedder (bge-m3 stand-in)."""

    provider = "local"
    model_id = E.MODEL_IDS["local"]
    dim = DIM

    def __init__(self):
        self.calls = 0

    def embed(self, texts, *, input_type="search_document"):
        self.calls += 1
        return np.eye(len(texts), self.dim, dtype=np.float32)


class FakeCohereBoto:
    """Canned bedrock-runtime client: cohere invoke_model answers, everything else fails."""

    def invoke_model(self, *, modelId, body):  # noqa: N803 (boto3 kwarg name)
        if "cohere" not in modelId:
            raise RuntimeError(f"{modelId} not offered here")
        n = len(json.loads(body)["texts"])
        payload = json.dumps({"embeddings": [[1.0] * DIM] * n}).encode("utf-8")

        class _Body:
            def __init__(self, data):
                self._data = data

            def read(self):
                return self._data

        return {"body": _Body(payload)}

    def converse(self, **_kw):
        raise RuntimeError("claude blocked")


def _broken_sentence_transformers(monkeypatch, message: str):
    """Install a sys.modules fake whose SentenceTransformer constructor raises `message`."""
    fake = types.ModuleType("sentence_transformers")

    class _Boom:
        def __init__(self, *a, **k):
            raise RuntimeError(message)

    fake.SentenceTransformer = _Boom
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)


# --------------------------------------------------------------------------- #
# resolve_mode precedence: env > constant; invalid -> local + warning         #
# --------------------------------------------------------------------------- #
def test_resolve_mode_default_and_env(monkeypatch, caplog):
    # No env: resolve_mode returns the CONSTANT, whatever it is (proven both ways) — the
    # mechanic, not a hardcoded value that would break when DEFAULT_MODE is flipped.
    monkeypatch.setattr(C, "DEFAULT_MODE", "local")
    assert C.resolve_mode() == "local"
    monkeypatch.setattr(C, "DEFAULT_MODE", "cloud")  # flipped constant honored too
    assert C.resolve_mode() == "cloud"

    monkeypatch.setenv("SEMSEARCH_MODE", "cloud")
    assert C.resolve_mode() == "cloud"
    monkeypatch.setenv("SEMSEARCH_MODE", "local-first")
    assert C.resolve_mode() == "local-first"

    monkeypatch.setenv("SEMSEARCH_MODE", "hybrid-banana")  # unknown -> local + warning
    with caplog.at_level("WARNING"):
        assert C.resolve_mode() == "local"
    assert any("hybrid-banana" in r.getMessage() for r in caplog.records)


def test_resolve_query_rewrite_precedence(monkeypatch, caplog):
    monkeypatch.delenv("SEMSEARCH_QUERY_REWRITE", raising=False)
    # No env: resolve_query_rewrite returns the CONSTANT, whatever it is (proven both ways).
    monkeypatch.setattr(C, "DEFAULT_QUERY_REWRITE", True)
    assert C.resolve_query_rewrite() is True
    monkeypatch.setattr(C, "DEFAULT_QUERY_REWRITE", False)
    assert C.resolve_query_rewrite() is False
    monkeypatch.setattr(C, "DEFAULT_QUERY_REWRITE", True)  # restore for env-override checks

    for off in ("off", "0", "false", "no"):
        monkeypatch.setenv("SEMSEARCH_QUERY_REWRITE", off)
        assert C.resolve_query_rewrite() is False
    for on in ("on", "1", "true", "yes"):
        monkeypatch.setenv("SEMSEARCH_QUERY_REWRITE", on)
        assert C.resolve_query_rewrite() is True

    monkeypatch.setenv("SEMSEARCH_QUERY_REWRITE", "OFF")  # case-insensitive
    assert C.resolve_query_rewrite() is False

    monkeypatch.setenv("SEMSEARCH_QUERY_REWRITE", "maybe")  # unknown -> warn + constant default
    with caplog.at_level("WARNING"):
        assert C.resolve_query_rewrite() is True
    assert any("maybe" in r.getMessage() for r in caplog.records)


def test_resolve_llm_gate_precedence(monkeypatch, caplog):
    monkeypatch.delenv(C.LLM_GATE_ENV, raising=False)
    # No env: resolve_llm_gate returns the CONSTANT, whatever it is (proven both ways).
    monkeypatch.setattr(C, "DEFAULT_LLM_GATE", "auto")
    assert C.resolve_llm_gate() == "auto"
    monkeypatch.setattr(C, "DEFAULT_LLM_GATE", "always")
    assert C.resolve_llm_gate() == "always"
    monkeypatch.setattr(C, "DEFAULT_LLM_GATE", "auto")  # restore for env-override checks

    for val in ("auto", "always"):
        monkeypatch.setenv(C.LLM_GATE_ENV, val)
        assert C.resolve_llm_gate() == val
    monkeypatch.setenv(C.LLM_GATE_ENV, "ALWAYS")  # case-insensitive
    assert C.resolve_llm_gate() == "always"

    monkeypatch.setenv(C.LLM_GATE_ENV, "banana")  # unknown -> warn + constant default
    with caplog.at_level("WARNING"):
        assert C.resolve_llm_gate() == "auto"
    assert any("banana" in r.getMessage() for r in caplog.records)


def test_resolve_mode_constant_switch_env_still_wins(monkeypatch):
    monkeypatch.setattr(C, "DEFAULT_MODE", "cloud")  # editing the config.py line
    assert C.resolve_mode() == "cloud"
    monkeypatch.setenv("SEMSEARCH_MODE", "local")  # env override beats the constant
    assert C.resolve_mode() == "local"


# --------------------------------------------------------------------------- #
# mode=local: today's behavior — cloud never contacted, local failure raises  #
# --------------------------------------------------------------------------- #
def test_local_mode_never_constructs_cloud_and_local_failure_raises(pois, monkeypatch):
    class BrokenLocal(UnitLocal):
        def embed(self, texts, *, input_type="search_document"):
            raise RuntimeError("corrupt HF cache")

    def factory(provider="local"):
        assert provider == "local", "cloud must never be constructed in mode=local"
        return BrokenLocal()

    monkeypatch.setattr(P, "get_embedder", factory)
    with pytest.raises(RuntimeError, match="corrupt HF cache"):
        P.FullPipeline(pois, mode="local")


# --------------------------------------------------------------------------- #
# mode=local-first: healthy local is byte-identical to local; no cloud touch  #
# --------------------------------------------------------------------------- #
def test_local_first_healthy_local_identical_to_local(pois, monkeypatch):
    shared = UnitLocal()

    def factory(provider="local"):
        assert provider == "local", "cloud must not be constructed when local is healthy"
        return shared

    monkeypatch.setattr(P, "get_embedder", factory)

    def _no_boto(*a, **k):
        raise AssertionError("boto3 must not be constructed when local is healthy")

    monkeypatch.setattr("boto3.client", _no_boto)

    local_pipe = P.FullPipeline(pois, mode="local")
    lf_pipe = P.FullPipeline(pois, mode="local-first")
    assert lf_pipe.dense is not None and lf_pipe.dense.emb is shared
    q = "quán cà phê yên tĩnh"
    assert lf_pipe.rank_ids(q) == local_pipe.rank_ids(q)  # byte-identical ranking


def test_local_first_broken_local_falls_to_cloud(pois, monkeypatch, caplog):
    """SentenceTransformer raising at the construction probe -> loud warning -> the cloud
    chain pins bedrock-cohere in the first region (real BedrockEmbedder, mocked boto3)."""
    _broken_sentence_transformers(monkeypatch, "no local model on this host")
    monkeypatch.setattr("boto3.client", lambda *a, **k: FakeCohereBoto())

    with caplog.at_level("WARNING"):
        pipe = P.FullPipeline(pois, mode="local-first")

    assert pipe.dense is not None
    assert pipe.dense.emb.provider == "bedrock-cohere"
    assert pipe.dense.emb._region == "ap-southeast-1"  # walked the region chain, pinned first
    assert any("local" in r.getMessage().lower() and r.levelname == "WARNING"
               for r in caplog.records)  # the degradation is LOUD


# --------------------------------------------------------------------------- #
# mode=cloud: local NEVER attempted; all-fail -> BM25-only floor              #
# --------------------------------------------------------------------------- #
def test_cloud_mode_never_imports_local(pois, monkeypatch):
    _broken_sentence_transformers(
        monkeypatch, "sentence_transformers must never load in cloud mode")
    # trap: even asking the factory for 'local' is a failure
    requested: list[str] = []
    real = P.get_embedder

    def spy(provider="local"):
        requested.append(provider)
        return real(provider)

    monkeypatch.setattr(P, "get_embedder", spy)
    monkeypatch.setattr("boto3.client", lambda *a, **k: FakeCohereBoto())

    pipe = P.FullPipeline(pois, mode="cloud")
    assert "local" not in requested          # local embedder never requested
    assert pipe.dense.emb.provider == "bedrock-cohere"
    assert pipe.dense.emb._region == "ap-southeast-1"


def test_cloud_all_providers_fail_bm25_only_floor(pois, monkeypatch, caplog):
    """Every cloud provider failing in every region -> the pipeline constructs WITHOUT a
    dense index, still serves non-empty results (G5), and the semantic signal is correctly
    calibrated: the top BM25 hit reaches 1.0 (single-list RRF max is 1/(c+1), not 2/(c+1))."""
    _broken_sentence_transformers(
        monkeypatch, "sentence_transformers must never load in cloud mode")
    with caplog.at_level("WARNING"):
        pipe = P.FullPipeline(pois, mode="cloud")  # autouse boto3 always fails

    assert pipe.dense is None                     # BM25-only floor
    assert any("bm25" in r.getMessage().lower() for r in caplog.records)  # loud

    q = "quán cà phê yên tĩnh"
    ranked = pipe.rank_scored(q)
    assert ranked                                  # G5: non-empty
    sem = {pid: b["semantic"] for pid, _, b in ranked}
    bm25_top = pipe.bm25.rank_ids(q)[0]
    assert bm25_top in sem
    assert sem[bm25_top] == pytest.approx(1.0)     # calibration fix: no silent 0.5 cap
    intent, results = pipe.search(q, k=5)
    assert results                                 # end-to-end path also non-empty


def test_expert_provider_override_skips_mode(pois, monkeypatch):
    """Explicit provider= wins over any mode: provider='local' under SEMSEARCH_MODE=cloud
    still builds the local dense index (expert override, documented precedence)."""
    monkeypatch.setenv("SEMSEARCH_MODE", "cloud")
    shared = UnitLocal()
    monkeypatch.setattr(P, "get_embedder", lambda provider="local": shared)
    pipe = P.FullPipeline(pois, provider="local")
    assert pipe.dense is not None and pipe.dense.emb is shared
    assert pipe.mode == "cloud"  # mode still resolved (drives the LLM-parse default)


# --------------------------------------------------------------------------- #
# LLM parse default per mode: cloud ON, local/local-first OFF, env wins       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode,env,expect_on", [
    ("cloud", None, True),          # remote hosting implies network -> ON by default
    ("cloud", "off", False),        # explicit force-off wins
    ("local", None, False),         # today's determinism (NFR-5)
    ("local-first", None, False),
    ("local", "bedrock", True),     # explicit force-on wins
    ("cloud", "banana", False),     # unknown value keeps today's off semantics
])
def test_llm_parse_default_matrix(pois, monkeypatch, mode, env, expect_on):
    if env is not None:
        monkeypatch.setenv("SEMSEARCH_LLM_PARSE", env)
    monkeypatch.setattr(P, "get_embedder", lambda provider="local": UnitLocal())
    if mode == "cloud":  # cloud dense goes to the (failing) bedrock chain -> floor, fast
        monkeypatch.setattr(
            P, "get_embedder",
            lambda provider="local": (_ for _ in ()).throw(RuntimeError("cloud down")))
    pipe = P.FullPipeline(pois, mode=mode)
    assert (pipe._llm_parser is not None) is expect_on


# --------------------------------------------------------------------------- #
# /health: mode + what actually resolved                                      #
# --------------------------------------------------------------------------- #
def test_health_reports_local_mode(pois, monkeypatch):
    monkeypatch.setattr(P, "get_embedder", lambda provider="local": UnitLocal())
    app = create_app(pois, prewarm=False, mode="local")
    body = TestClient(app).get("/health").json()
    assert body == {"status": "ok", "pois": len(pois), "mode": "local",
                    "embeddings": "local", "llm_parse": "rules-only",
                    "query_rewrite": "off",  # LLM off in local mode -> rewrite can't fire
                    "llm_gate": "auto"}       # degradation gate default (env cleared -> auto)


def test_health_reports_query_rewrite_on_and_off(pois, monkeypatch):
    """query_rewrite is 'on' when the switch is on AND the LLM parse gate is on (the correction
    rides that parse), even if the parser resolved nothing offline; env off forces it off."""
    monkeypatch.setattr(P, "get_embedder", lambda provider="local": UnitLocal())
    monkeypatch.setenv("SEMSEARCH_LLM_PARSE", "bedrock")  # gate on -> _llm_parser is not None
    monkeypatch.delenv("SEMSEARCH_QUERY_REWRITE", raising=False)  # default True
    app_on = create_app(pois, prewarm=False, mode="local")
    assert TestClient(app_on).get("/health").json()["query_rewrite"] == "on"

    monkeypatch.setenv("SEMSEARCH_QUERY_REWRITE", "off")  # switch off wins even with the gate on
    app_off = create_app(pois, prewarm=False, mode="local")
    assert TestClient(app_off).get("/health").json()["query_rewrite"] == "off"


def test_health_reports_llm_gate(pois, monkeypatch):
    """/health surfaces the LLM degradation gate additively ('auto' default, 'always' when
    forced) — the existing llm_parse / query_rewrite strings are untouched."""
    monkeypatch.setattr(P, "get_embedder", lambda provider="local": UnitLocal())
    # default: autouse `_isolated` clears SEMSEARCH_LLM_GATE -> "auto"
    app_auto = create_app(pois, prewarm=False, mode="local")
    body = TestClient(app_auto).get("/health").json()
    assert body["llm_gate"] == "auto"
    assert body["llm_parse"] == "rules-only" and body["query_rewrite"] == "off"  # unchanged

    monkeypatch.setenv(C.LLM_GATE_ENV, "always")  # env forces every-query calls
    app_always = create_app(pois, prewarm=False, mode="local")
    assert TestClient(app_always).get("/health").json()["llm_gate"] == "always"


def test_health_reports_bm25_floor_in_cloud_mode(pois, monkeypatch):
    _broken_sentence_transformers(monkeypatch, "must not load")
    app = create_app(pois, prewarm=False, mode="cloud")  # offline -> floor + LLM unavailable
    body = TestClient(app).get("/health").json()
    assert body["mode"] == "cloud"
    assert body["embeddings"] == "bm25-only"
    assert body["llm_parse"] == "rules-only"   # LLM on-by-default but nothing resolved
    # G5 even on the floor: the contract endpoint still serves results
    r = TestClient(app).get("/v1/search", params={"q": "quán cà phê yên tĩnh"})
    assert r.status_code == 200 and r.json()["results"]


def test_health_status_strings_for_pinned_providers():
    """Unit-level: the health strings for pinned cloud providers (no network needed)."""
    emb = SimpleNamespace(provider="bedrock-cohere", _region="ap-southeast-1")
    pipe = SimpleNamespace(
        dense=SimpleNamespace(emb=emb),
        _llm_parser=SimpleNamespace(_client=object(), _provider="openai",
                                    model_id="gpt-4.1-nano"),
    )
    assert A._embeddings_status(pipe) == "bedrock-cohere@ap-southeast-1"
    assert A._llm_status(pipe) == "openai+gpt-4.1-nano"
    # bedrock parser
    pipe._llm_parser = SimpleNamespace(
        _client=object(), _provider="bedrock",
        model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0")
    assert A._llm_status(pipe) == "bedrock+global.anthropic.claude-haiku-4-5-20251001-v1:0"


# --------------------------------------------------------------------------- #
# Review findings: explicit mode pins fully; env value semantics; honesty     #
# --------------------------------------------------------------------------- #
def test_explicit_mode_argument_fully_pins(pois, monkeypatch):
    """An EXPLICIT mode= argument must skip env resolution entirely: under
    SEMSEARCH_MODE=cloud, mode='local' builds a local pipeline with the LLM default OFF."""
    monkeypatch.setenv("SEMSEARCH_MODE", "cloud")
    monkeypatch.setattr(P, "get_embedder", lambda provider="local": UnitLocal())
    pipe = P.FullPipeline(pois, mode="local")
    assert pipe.mode == "local"
    assert pipe.dense is not None and pipe.dense.emb.provider == "local"
    assert pipe._llm_parser is None  # nothing env-resolved leaked in


def test_explicit_invalid_mode_raises(pois, monkeypatch):
    """An invalid EXPLICIT mode is a programmer error and must fail loudly (the env path
    warns and falls back instead — operators get resilience, code gets correctness)."""
    monkeypatch.setattr(P, "get_embedder", lambda provider="local": UnitLocal())
    with pytest.raises(ValueError, match="mode"):
        P.FullPipeline(pois, mode="clouds")


def test_llm_parse_env_on_and_unknown_values(pois, monkeypatch, caplog):
    monkeypatch.setattr(P, "get_embedder", lambda provider="local": UnitLocal())
    # "on" is a force-on alias for "bedrock" (full Bedrock->OpenAI chain)
    monkeypatch.setenv("SEMSEARCH_LLM_PARSE", "on")
    assert P.FullPipeline(pois, mode="local")._llm_parser is not None
    # unknown value -> WARNING + off (never silently on)
    monkeypatch.setenv("SEMSEARCH_LLM_PARSE", "banana")
    with caplog.at_level("WARNING"):
        pipe = P.FullPipeline(pois, mode="cloud")
    assert pipe._llm_parser is None
    assert any("banana" in r.getMessage() for r in caplog.records)


def test_llm_parse_env_openai_skips_bedrock_probes(pois, monkeypatch):
    """SEMSEARCH_LLM_PARSE=openai: force-on, pin OpenAI DIRECTLY — an operator with a key
    but no Bedrock must not wait through 9 doomed Bedrock probes. Bedrock (boto3) must not
    be touched at all on this path."""
    import httpx

    monkeypatch.setattr(P, "get_embedder", lambda provider="local": UnitLocal())
    monkeypatch.setenv("SEMSEARCH_LLM_PARSE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-FAKE-modes-test")

    def _no_boto(*a, **k):
        raise AssertionError("bedrock must not be probed when SEMSEARCH_LLM_PARSE=openai")

    monkeypatch.setattr("boto3.client", _no_boto)
    monkeypatch.setattr(
        L.LLMParser, "_make_openai_client",
        staticmethod(lambda: httpx.Client(transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200, json={"choices": [{"message": {"content": "{}"}}]})))),
    )
    pipe = P.FullPipeline(pois, mode="local")
    assert pipe._llm_parser is not None
    assert pipe._llm_parser._provider == "openai"


def test_warn_once_message_reflects_how_llm_was_enabled(pois, monkeypatch, caplog):
    """A cloud operator who never set SEMSEARCH_LLM_PARSE must not be told about it: the
    warn-once message names the actual enabler (cloud-mode default vs the env value). The
    query is DEGRADED ('quan cafe' — no diacritics) so the degradation gate lets the (failing,
    offline) LLM call through to exercise the warn-once path; a clean query would be gated off."""
    _broken_sentence_transformers(monkeypatch, "must not load")
    with caplog.at_level("WARNING"):
        pipe = P.FullPipeline(pois, mode="cloud")  # LLM on via mode default; offline -> None
        pipe.resolve_intent("quan cafe")
    warned = [r.getMessage() for r in caplog.records if "rule-parsed" in r.getMessage()]
    assert warned and "cloud" in warned[0] and "SEMSEARCH_LLM_PARSE" not in warned[0]

    caplog.clear()
    monkeypatch.setenv("SEMSEARCH_LLM_PARSE", "bedrock")
    monkeypatch.setattr(P, "get_embedder", lambda provider="local": UnitLocal())
    with caplog.at_level("WARNING"):
        pipe = P.FullPipeline(pois, mode="local")
        pipe.resolve_intent("quan cafe")
    warned = [r.getMessage() for r in caplog.records if "rule-parsed" in r.getMessage()]
    assert warned and "SEMSEARCH_LLM_PARSE=bedrock" in warned[0]


# --------------------------------------------------------------------------- #
# No-credentials short-circuit: creds are account-wide, not regional          #
# --------------------------------------------------------------------------- #
class _CountingNoCreds:
    """invoke_model raises NoCredentialsError; the count proves the short-circuit."""

    attempts = 0  # class-level: shared across per-region client instances

    def invoke_model(self, **_kw):
        from botocore.exceptions import NoCredentialsError

        type(self).attempts += 1
        raise NoCredentialsError()

    def converse(self, **_kw):
        raise RuntimeError("claude blocked")


class _CountingNetworkFail:
    attempts = 0

    def invoke_model(self, **_kw):
        type(self).attempts += 1
        raise RuntimeError("connect timeout")

    def converse(self, **_kw):
        raise RuntimeError("claude blocked")


def test_cloud_no_creds_short_circuits_all_probes(pois, monkeypatch):
    """NoCredentialsError is account-wide: the FIRST probe failing that way must skip the
    remaining regions AND the remaining provider — exactly 1 embeddings probe total."""
    _CountingNoCreds.attempts = 0
    monkeypatch.setattr("boto3.client", lambda *a, **k: _CountingNoCreds())
    pipe = P.FullPipeline(pois, mode="cloud")
    assert pipe.dense is None                     # floor, as before
    assert _CountingNoCreds.attempts == 1         # not 6: creds don't vary by region/provider


def test_cloud_network_errors_keep_full_walk(pois, monkeypatch):
    """A network/model error IS potentially regional: the full walk must be preserved —
    cohere walks all 3 default regions; titan walks its own 2-region chain (not offered
    in ap-southeast-1, so its default chain skips Singapore) = 5 probes total."""
    _CountingNetworkFail.attempts = 0
    monkeypatch.setattr("boto3.client", lambda *a, **k: _CountingNetworkFail())
    pipe = P.FullPipeline(pois, mode="cloud")
    assert pipe.dense is None
    assert _CountingNetworkFail.attempts == 5
