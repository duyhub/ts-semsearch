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


def test_trace_is_deterministic():
    params = {"q": "nơi hẹn hò lãng mạn"}
    a = client.get("/v1/semantic-search", params=params).json()["trace"]
    b = client.get("/v1/semantic-search", params=params).json()["trace"]
    assert a == b  # identical request → identical trace (NFR-5)


def test_contract_search_never_carries_trace():
    body = client.get("/v1/search", params={"q": "cà phê"}).json()
    assert "trace" not in body
    assert set(body) == {"query", "results", "meta"}  # contract shape unchanged
