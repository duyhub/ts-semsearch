"""Tests for the deterministic pipeline `trace` on /v1/semantic-search (T4).

The trace powers the read-only /admin transparency view. It must be deterministic (NFR-5)
and must NOT leak onto the contract-exact /v1/search response.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from semsearch.api import create_app

app = create_app(prewarm=False)
client = TestClient(app)

_TRACE_KEYS = {"bm25Top", "denseTop", "anchorGateFired", "fallbackFired",
               "resultCount", "constraintsEngaged"}


def test_semantic_search_exposes_trace():
    d = client.get("/v1/semantic-search",
                   params={"q": "quán cà phê yên tĩnh để làm việc ở quận 1"}).json()
    tr = d["trace"]
    assert _TRACE_KEYS <= set(tr)
    assert isinstance(tr["bm25Top"], list) and isinstance(tr["denseTop"], list)
    assert isinstance(tr["anchorGateFired"], bool)
    assert isinstance(tr["fallbackFired"], bool)
    # a district query engages the location constraint (the anchor gate only *reorders*
    # when >=3 results survive within radius, so anchorGateFired may legitimately be False)
    assert "location" in tr["constraintsEngaged"]


def _structural(trace: dict) -> dict:
    # everything except wall-clock timings, which legitimately vary run to run
    return {k: v for k, v in trace.items() if k not in ("steps", "totalMs")}


def test_trace_structure_is_deterministic():
    params = {"q": "nơi hẹn hò lãng mạn"}
    a = client.get("/v1/semantic-search", params=params).json()["trace"]
    b = client.get("/v1/semantic-search", params=params).json()["trace"]
    assert _structural(a) == _structural(b)  # decisions identical (NFR-5)
    # step SEQUENCE is deterministic too; only the per-step ms differs
    assert [s["name"] for s in a["steps"]] == [s["name"] for s in b["steps"]]


def test_trace_reports_per_step_latency():
    d = client.get("/v1/semantic-search", params={"q": "quán cà phê yên tĩnh"}).json()
    steps = d["trace"]["steps"]
    names = [s["name"] for s in steps]
    # the core pipeline stages are timed, in execution order
    assert names[:4] == ["parse", "dense_retrieval", "lexical_fusion", "rank_signals"]
    assert "explain_serialize" in names
    assert all(isinstance(s["ms"], (int, float)) and s["ms"] >= 0 for s in steps)
    assert d["trace"]["totalMs"] >= 0


def test_contract_search_never_carries_trace():
    body = client.get("/v1/search", params={"q": "cà phê"}).json()
    assert "trace" not in body
    assert set(body) == {"query", "results", "meta"}  # contract shape unchanged
