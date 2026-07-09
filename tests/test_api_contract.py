"""Contract tests for the Tasco /v1/search API (SPEC §9, §11; FR-11/12)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from semsearch.api import create_app
from semsearch.rank import SIGNALS, load_weights

app = create_app(prewarm=False)


def test_api_serves_tuned_weights():
    # the live API must rank with the tuned weights.json, not untuned DEFAULT_WEIGHTS,
    # so the demo matches the reported metrics.
    assert app.state.pipeline.ranker.weights == load_weights()
client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_search_shape_and_poi_prefix():
    r = client.get("/v1/search", params={"q": "quán cà phê yên tĩnh để làm việc"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"query", "results", "meta"}
    assert body["meta"].keys() >= {"count", "limitApplied", "tookMs", "source"}
    assert body["results"], "must return results for a valid query"
    res = body["results"][0]
    assert set(res) >= {"id", "type", "name", "label", "address", "category",
                        "coordinates", "score", "source", "tags"}
    assert res["id"].startswith("poi:")               # id prefixing (Phase-0 fact)
    assert set(res["coordinates"]) == {"lat", "lon"}
    assert isinstance(res["tags"], list)


def test_diacritics_preserved():
    r = client.get("/v1/search", params={"q": "cà phê"})
    names = " ".join(x["name"] for x in r.json()["results"])
    assert any(ord(c) > 127 for c in names)            # NFR-4: not ASCII-folded


def test_missing_q_is_contract_error():
    r = client.get("/v1/search")
    assert r.status_code == 400
    body = r.json()
    assert body["error"]["code"] == "invalid_request"
    assert "requestId" in body and body["requestId"]


def test_gibberish_q_still_returns_results():
    # present-but-meaningless q is a valid request -> >=1 result (C1 backstop, G5)
    r = client.get("/v1/search", params={"q": "asdfghjkl zzzz"})
    assert r.status_code == 200
    assert len(r.json()["results"]) >= 1


def test_limit_clamped_to_20():
    r = client.get("/v1/search", params={"q": "cà phê", "limit": 50})
    body = r.json()
    assert body["meta"]["limitApplied"] == 20
    assert len(body["results"]) <= 20


def test_bbox_malformed_is_400():
    r = client.get("/v1/search", params={"q": "cà phê", "bbox": "1,2,3"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_request"


def test_aliases_ok():
    for path in ("/search", "/v1/geocode-search"):
        assert client.get(path, params={"q": "cà phê"}).status_code == 200


def test_request_id_echoed():
    rid = "test-req-123"
    r = client.get("/v1/search", params={"q": "cà phê"}, headers={"X-Request-Id": rid})
    assert r.headers.get("X-Request-Id") == rid


def test_distance_meters_when_latlon_given():
    r = client.get("/v1/search", params={"q": "cà phê", "lat": 10.77, "lon": 106.70})
    assert all(x["distanceMeters"] is not None for x in r.json()["results"])


def test_category_filter():
    r = client.get("/v1/search", params={"q": "gần đây", "category": "Trạm xăng"})
    cats = {x["category"] for x in r.json()["results"]}
    assert cats == {"Trạm xăng"} or r.json()["meta"]["source"] == "fallback"


def test_keyword_engine_lane():
    r = client.get("/v1/search", params={"q": "quán cà phê yên tĩnh để làm việc", "engine": "keyword"})
    assert r.status_code == 200
    assert r.json()["results"]  # BM25-only lane still returns results


def test_root_serves_ui():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Tasco" in r.text


def test_semantic_search_has_breakdown_reasons_intent():
    r = client.get("/v1/semantic-search", params={"q": "quán cà phê yên tĩnh để làm việc"})
    assert r.status_code == 200
    body = r.json()
    assert "intent" in body and "requiredAttrs" in body["intent"]
    res = body["results"][0]
    assert len(res["breakdown"]) == len(SIGNALS)   # every signal exposed
    assert isinstance(res["reasons"], list) and res["reasons"]
