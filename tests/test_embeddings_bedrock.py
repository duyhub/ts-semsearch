"""Bedrock embeddings provider (FR-10, Built-with-AWS core component).

Every test here is offline: the boto3 client is replaced by an in-process fake
(`FakeBedrockClient`) injected onto `emb._client`, so no test constructs a real
`bedrock-runtime` client or touches the network. There are NO AWS credentials on
CI/dev machines — the suite MUST pass without any. We assert the mock is the
active client to make that guarantee explicit (see `test_client_is_the_mock`).

Contracts under test (from the task spec):
  - Cohere request body {"texts", "input_type", "truncate":"END"}; input_type is
    'search_document' for docs and 'search_query' for queries; >96 texts batch.
  - Titan request body {"inputText", "dimensions", "normalize"}, one text/call.
  - We L2-normalize returned vectors ourselves (Cohere is NOT normalized), because
    DenseIndex treats cosine as a matvec over unit vectors.
  - resolve_provider preflight: any failure at construction -> 'local' (never mix
    vector spaces). Per-query failure after construction -> zero vector, not cached.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from semsearch import embeddings as E

DIM = E.EMBED_DIM


# --------------------------------------------------------------------------- #
# Offline fake for bedrock-runtime.invoke_model                               #
# --------------------------------------------------------------------------- #
class _FakeBody:
    """Mimics botocore's StreamingBody: a single .read() -> bytes."""

    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


class FakeBedrockClient:
    """Records invoke_model calls and returns canned payloads (or raises).

    `responder(body: dict) -> dict` builds the response payload from the parsed
    request body so batch sizes line up with the returned embedding count.
    """

    def __init__(self, responder=None, *, raise_exc: Exception | None = None):
        self.calls: list[dict] = []
        self._responder = responder
        self._raise_exc = raise_exc

    def invoke_model(self, *, modelId: str, body: str):  # noqa: N803 (boto3 kwarg name)
        parsed = json.loads(body)
        self.calls.append({"modelId": modelId, "body": parsed})
        if self._raise_exc is not None:
            raise self._raise_exc
        payload = self._responder(parsed)
        return {"body": _FakeBody(json.dumps(payload).encode("utf-8"))}


def cohere_responder(fill: float = 2.0):
    """Return one all-`fill` vector per input text (non-normalized on purpose)."""

    def _resp(body: dict) -> dict:
        n = len(body["texts"])
        return {"embeddings": [[fill] * DIM for _ in range(n)]}

    return _resp


def titan_responder(fill: float = 2.0):
    def _resp(body: dict) -> dict:
        assert "inputText" in body  # titan is one text per call
        return {"embedding": [fill] * DIM}

    return _resp


def make_cohere(responder=None, *, raise_exc=None):
    emb = E.BedrockEmbedder("bedrock-cohere")
    emb._client = FakeBedrockClient(responder or cohere_responder(), raise_exc=raise_exc)
    return emb


def make_titan(responder=None, *, raise_exc=None):
    emb = E.BedrockEmbedder("bedrock-titan")
    emb._client = FakeBedrockClient(responder or titan_responder(), raise_exc=raise_exc)
    return emb


# --------------------------------------------------------------------------- #
# get_embedder wiring                                                         #
# --------------------------------------------------------------------------- #
def test_get_embedder_returns_bedrock_providers():
    cohere = E.get_embedder("bedrock-cohere")
    titan = E.get_embedder("bedrock-titan")
    assert isinstance(cohere, E.BedrockEmbedder)
    assert cohere.provider == "bedrock-cohere"
    assert cohere.model_id == "cohere.embed-multilingual-v3"
    assert titan.model_id == "amazon.titan-embed-text-v2:0"


def test_get_embedder_local_unchanged():
    assert isinstance(E.get_embedder("local"), E.LocalEmbedder)


def test_get_embedder_unknown_still_fails_loudly():
    with pytest.raises(SystemExit):
        E.get_embedder("bedrock-nonsense")


def test_client_is_the_mock():
    """No test may touch the network: the active client must be the injected fake."""
    emb = make_cohere()
    assert isinstance(emb._get_client(), FakeBedrockClient)  # never a boto3 client


# --------------------------------------------------------------------------- #
# Cohere request shape: doc vs query input_type, batching                     #
# --------------------------------------------------------------------------- #
def test_cohere_doc_request_shape():
    emb = make_cohere()
    out = emb.embed(["a", "b", "c"])  # doc build path (default input_type)
    assert out.shape == (3, DIM)
    body = emb._client.calls[-1]["body"]
    assert body["texts"] == ["a", "b", "c"]
    assert body["input_type"] == "search_document"
    assert body["truncate"] == "END"
    assert emb._client.calls[-1]["modelId"] == "cohere.embed-multilingual-v3"


def test_cohere_query_request_uses_search_query(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "QCACHE_DIR", tmp_path / "qcache")
    emb = make_cohere()
    E.embed_query(emb, "cà phê yên tĩnh", use_cache=False)
    body = emb._client.calls[-1]["body"]
    assert body["input_type"] == "search_query"  # queries != docs (Cohere requires it)
    assert body["texts"] == ["cà phê yên tĩnh"]


def test_cohere_batches_over_96():
    emb = make_cohere()
    texts = [f"t{i}" for i in range(200)]
    out = emb.embed(texts)
    assert out.shape == (200, DIM)
    sizes = [len(c["body"]["texts"]) for c in emb._client.calls]
    assert sizes == [96, 96, 8]  # split into <=96-text batches


def test_cohere_empty_returns_zero_rows():
    emb = make_cohere()
    out = emb.embed([])
    assert out.shape == (0, DIM)
    assert emb._client.calls == []  # no API call for an empty batch


# --------------------------------------------------------------------------- #
# Titan request shape: one inputText per call                                 #
# --------------------------------------------------------------------------- #
def test_titan_per_text_shape():
    emb = make_titan()
    out = emb.embed(["a", "b"])
    assert out.shape == (2, DIM)
    assert len(emb._client.calls) == 2  # one invoke per text
    body = emb._client.calls[0]["body"]
    assert body["inputText"] == "a"
    assert body["dimensions"] == DIM
    assert body["normalize"] is True
    assert emb._client.calls[0]["modelId"] == "amazon.titan-embed-text-v2:0"


# --------------------------------------------------------------------------- #
# We normalize ourselves (do not trust API defaults)                          #
# --------------------------------------------------------------------------- #
def test_l2_normalization_applied_cohere():
    emb = make_cohere(cohere_responder(fill=2.0))  # raw norm = 2*sqrt(1024) = 64, not 1
    out = emb.embed(["x", "y"])
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    assert np.isclose(out[0, 0], 2.0 / 64.0, atol=1e-5)  # magnitude actually changed


def test_l2_normalization_applied_titan():
    emb = make_titan(titan_responder(fill=5.0))
    out = emb.embed(["x"])
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


# --------------------------------------------------------------------------- #
# Coherent fallback: construction preflight -> 'local' on ANY failure          #
# --------------------------------------------------------------------------- #
def test_resolve_provider_local_needs_no_preflight(monkeypatch):
    def _boom(_provider):
        raise AssertionError("get_embedder must not run for a non-bedrock provider")

    monkeypatch.setattr(E, "get_embedder", _boom)
    assert E.resolve_provider("local") == "local"


def test_resolve_provider_passthrough_on_success(monkeypatch):
    monkeypatch.setattr(E, "get_embedder", lambda p: make_cohere())
    assert E.resolve_provider("bedrock-cohere") == "bedrock-cohere"


def test_resolve_provider_falls_back_on_construction_failure(monkeypatch, caplog):
    """Timeout / no creds / no model access at preflight -> degrade to local, one warning."""

    class SimulatedTimeout(Exception):
        pass

    monkeypatch.setattr(E, "get_embedder",
                        lambda p: make_cohere(raise_exc=SimulatedTimeout("read timeout")))
    with caplog.at_level("WARNING"):
        assert E.resolve_provider("bedrock-cohere") == "local"
    assert any("local" in r.message.lower() for r in caplog.records)  # warned once, clearly


# --------------------------------------------------------------------------- #
# Per-query failure after construction: zero vector, never cached             #
# --------------------------------------------------------------------------- #
def test_embed_query_zero_vector_on_failure_and_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "QCACHE_DIR", tmp_path / "qcache")
    emb = make_cohere(raise_exc=RuntimeError("creds expired mid-demo"))
    vec = E.embed_query(emb, "nơi hẹn hò")
    assert vec.shape == (DIM,)
    assert np.all(vec == 0.0)  # dense sims -> 0, fused ranking degrades toward BM25
    # a transient failure must NOT poison the qcache
    cache_dir = E.QCACHE_DIR / f"{emb.provider}.{E._safe(emb.model_id)}"
    assert not cache_dir.exists() or not any(cache_dir.iterdir())


def test_embed_query_caches_only_success(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "QCACHE_DIR", tmp_path / "qcache")
    emb = make_cohere()
    v1 = E.embed_query(emb, "quán cà phê")
    assert np.allclose(np.linalg.norm(v1), 1.0, atol=1e-5)
    calls_after_first = len(emb._client.calls)
    v2 = E.embed_query(emb, "quán cà phê")  # served from disk cache -> no new API call
    assert len(emb._client.calls) == calls_after_first
    assert np.array_equal(v1, v2)


# --------------------------------------------------------------------------- #
# Degradation is BEDROCK-ONLY: a local embed failure must propagate loudly    #
# --------------------------------------------------------------------------- #
class BrokenLocal:
    """Simulates a broken local setup (corrupt HF cache, missing dep)."""

    provider = "local"
    model_id = E.MODEL_IDS["local"]
    dim = E.EMBED_DIM

    def embed(self, texts, *, input_type="search_document"):
        raise RuntimeError("corrupt HF cache")


def test_embed_query_local_failure_propagates(tmp_path, monkeypatch):
    """The zero-vector degradation is scoped to bedrock providers. A LOCAL embed
    failure is a setup bug the operator must see immediately — swallowing it would
    boot a 'healthy' server that silently serves BM25-only results."""
    monkeypatch.setattr(E, "QCACHE_DIR", tmp_path / "qcache")
    with pytest.raises(RuntimeError, match="corrupt HF cache"):
        E.embed_query(BrokenLocal(), "quán cà phê")
    with pytest.raises(RuntimeError, match="corrupt HF cache"):
        E.embed_query(BrokenLocal(), "quán cà phê", use_cache=False)


# --------------------------------------------------------------------------- #
# Zero query vector -> dense returns NO ranking (not dataset order)            #
# --------------------------------------------------------------------------- #
def _fake_pois(n: int):
    return [
        SimpleNamespace(
            poi_id=f"C{i:03d}", name=f"poi {i}", brand=None, category="Cafe",
            sub_category=None, district="Quận 1", city="TP.HCM", attributes=[],
            tags=[], description="",
        )
        for i in range(n)
    ]


def test_dense_search_empty_on_zero_query_vector(tmp_path, monkeypatch):
    """A degraded (zero) query vector means dense has NO opinion: search must
    return [] — an all-zero matvec argsorted would emit DATASET-ORDER ids, which
    would pollute RRF (reciprocal-rank votes for dataset order) and corrupt the
    subject-corroboration top-K."""
    monkeypatch.setattr(E, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(E, "QCACHE_DIR", tmp_path / "qcache")
    from semsearch.retrieve import DenseIndex

    def responder(body):
        if body["input_type"] == "search_query":
            raise RuntimeError("creds expired mid-demo")
        return {"embeddings": [[2.0] * DIM for _ in body["texts"]]}

    emb = make_cohere(responder)
    idx = DenseIndex(_fake_pois(3), emb)  # doc build succeeds
    assert idx.search("cà phê yên tĩnh") == []  # query embed fails -> no dense ranking
    assert idx.rank_ids("cà phê yên tĩnh") == []


def test_corroboration_sees_empty_dense_top():
    """With an empty dense ranking, NO subject term corroborates -> the pipeline's
    all-or-nothing rule discredits them and falls back to the category filter."""
    from semsearch.pipeline import FullPipeline

    fake_self = SimpleNamespace(_content={"C001": {"bun", "cha"}, "C002": {"pho"}})
    intent = SimpleNamespace(content_terms=["bun", "cha"])
    assert FullPipeline._corroborated_subjects(fake_self, intent, []) == set()
    # sanity: with a real dense top the same terms DO corroborate
    assert FullPipeline._corroborated_subjects(fake_self, intent, ["C001"]) == {"bun", "cha"}


def test_rrf_fuse_ignores_empty_dense_ranking():
    """rrf_fuse([bm25, []]) must reproduce the BM25 ordering exactly."""
    from semsearch.retrieve import rrf_fuse

    fused = rrf_fuse([["A", "B", "C"], []])
    assert [pid for pid, _ in fused] == ["A", "B", "C"]


# --------------------------------------------------------------------------- #
# Doc-build failure AFTER a passing preflight -> pipeline rebuilds on local    #
# --------------------------------------------------------------------------- #
class PreflightOnlyBedrock:
    """Preflight ping succeeds; the network then drops during the doc build."""

    provider = "bedrock-cohere"
    model_id = E.MODEL_IDS["bedrock-cohere"]
    dim = E.EMBED_DIM

    def __init__(self):
        self.calls = 0

    def embed(self, texts, *, input_type="search_document"):
        self.calls += 1
        if self.calls == 1:  # resolve_provider's one-string preflight
            return np.eye(len(texts), self.dim, dtype=np.float32)
        raise RuntimeError("network dropped during doc build")


class UnitLocal:
    provider = "local"
    model_id = E.MODEL_IDS["local"]
    dim = E.EMBED_DIM

    def embed(self, texts, *, input_type="search_document"):
        return np.eye(len(texts), self.dim, dtype=np.float32)


def test_pipeline_falls_back_to_local_on_doc_build_failure(tmp_path, monkeypatch, caplog):
    """The preflight only pings one string; if the network drops DURING the 111-doc
    matrix build, FullPipeline must still degrade to local (one warning), keeping
    the 'coherent for the entire run' guarantee instead of crashing construction."""
    from semsearch import pipeline as P
    from semsearch.data import load_pois

    monkeypatch.setattr(E, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(E, "QCACHE_DIR", tmp_path / "qcache")
    flaky, local = PreflightOnlyBedrock(), UnitLocal()

    def factory(provider="local"):
        return flaky if provider == "bedrock-cohere" else local

    monkeypatch.setattr(E, "get_embedder", factory)  # used by resolve_provider preflight
    monkeypatch.setattr(P, "get_embedder", factory)  # used by DenseIndex construction

    with caplog.at_level("WARNING"):
        pipe = P.FullPipeline(load_pois(), provider="bedrock-cohere")
    assert pipe.dense.emb is local  # rebuilt in the LOCAL vector space, not crashed
    assert flaky.calls >= 2  # preflight passed, doc build actually attempted
    assert any("local" in r.message.lower() for r in caplog.records)


def test_pipeline_local_doc_build_failure_still_propagates(tmp_path, monkeypatch):
    """The construction fallback is bedrock-only: a broken LOCAL doc build is a
    setup bug and must crash loudly, exactly as before."""
    from semsearch import pipeline as P
    from semsearch.data import load_pois

    monkeypatch.setattr(E, "CACHE_DIR", tmp_path)
    broken = BrokenLocal()
    monkeypatch.setattr(P, "get_embedder", lambda provider="local": broken)

    with pytest.raises(RuntimeError, match="corrupt HF cache"):
        P.FullPipeline(load_pois(), provider="local")


# --------------------------------------------------------------------------- #
# check_bedrock.py failure classification (network vs no-creds)                #
# --------------------------------------------------------------------------- #
def _load_check_bedrock():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "check_bedrock.py"
    spec = importlib.util.spec_from_file_location("check_bedrock", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_check_bedrock_classifies_sts_failures():
    """Absent credentials and an unreachable endpoint are DIFFERENT states and get
    different messages (both informational, exit 0); a rejected credential is a
    real failure."""
    from botocore.exceptions import (
        ClientError,
        EndpointConnectionError,
        NoCredentialsError,
        ReadTimeoutError,
        SSOTokenLoadError,
    )

    cb = _load_check_bedrock()
    assert cb.classify_sts_failure(NoCredentialsError()) == "no-credentials"
    assert cb.classify_sts_failure(SSOTokenLoadError(error_msg="expired")) == "no-credentials"
    assert cb.classify_sts_failure(
        EndpointConnectionError(endpoint_url="https://sts.example")) == "network"
    assert cb.classify_sts_failure(
        ReadTimeoutError(endpoint_url="https://sts.example")) == "network"
    denied = ClientError({"Error": {"Code": "AccessDenied", "Message": "no"}}, "GetCallerIdentity")
    assert cb.classify_sts_failure(denied) == "rejected"


# --------------------------------------------------------------------------- #
# Provider-stamped qcache key (A2): bedrock vs local never share a key         #
# --------------------------------------------------------------------------- #
def test_qcache_key_differs_across_providers():
    text = "cà phê yên tĩnh"
    k_local = E._qkey(E.LocalEmbedder(), text)
    k_cohere = E._qkey(E.BedrockEmbedder("bedrock-cohere"), text)
    k_titan = E._qkey(E.BedrockEmbedder("bedrock-titan"), text)
    assert len({k_local, k_cohere, k_titan}) == 3  # every vector space keyed apart


# --------------------------------------------------------------------------- #
# Region resolution precedence                                                #
# --------------------------------------------------------------------------- #
def test_region_precedence(monkeypatch):
    for var in ("SEMSEARCH_BEDROCK_REGION", "AWS_REGION", "AWS_DEFAULT_REGION"):
        monkeypatch.delenv(var, raising=False)
    assert E.BedrockEmbedder("bedrock-cohere")._region() == "ap-southeast-1"  # default

    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
    assert E.BedrockEmbedder("bedrock-cohere")._region() == "us-west-2"
    monkeypatch.setenv("AWS_REGION", "eu-central-1")
    assert E.BedrockEmbedder("bedrock-cohere")._region() == "eu-central-1"  # AWS_REGION wins
    monkeypatch.setenv("SEMSEARCH_BEDROCK_REGION", "ap-southeast-1")
    assert E.BedrockEmbedder("bedrock-cohere")._region() == "ap-southeast-1"  # override wins
