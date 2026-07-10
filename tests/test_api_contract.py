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
    r = client.get("/v1/search", params={"q": "quán cà phê yên tĩnh làm việc"})
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
    r = client.get("/v1/search", params={"q": "quán cà phê yên tĩnh làm việc", "engine": "keyword"})
    assert r.status_code == 200
    assert r.json()["results"]  # BM25-only lane still returns results


def test_root_serves_ui():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Tasco" in r.text


def test_semantic_search_has_breakdown_reasons_intent():
    r = client.get("/v1/semantic-search", params={"q": "quán cà phê yên tĩnh làm việc"})
    assert r.status_code == 200
    body = r.json()
    assert "intent" in body and "requiredAttrs" in body["intent"]
    res = body["results"][0]
    assert len(res["breakdown"]) == len(SIGNALS)   # every signal exposed
    assert isinstance(res["reasons"], list) and res["reasons"]


# --- Batch B: contract & API fixes ------------------------------------------


def test_error_codes_match_pdf_table():
    # tasco_api.pdf "Common error codes" table / SPEC §9 — assert the literal strings
    # (C13: 408 is 'timeout' not 'request_timeout'; 503 'service_unavailable' not 'unavailable').
    from semsearch.api import ERROR_CODES
    assert ERROR_CODES == {
        400: "invalid_request", 401: "unauthorized", 403: "forbidden", 404: "not_found",
        408: "timeout", 429: "rate_limited", 500: "internal_error", 503: "service_unavailable",
    }


def test_ocean_bbox_triggers_labeled_fallback():
    # C18: an impossible (mid-ocean) bbox has no in-window POI -> the never-empty
    # backstop fires and drops to the global popularity list, honestly labeled.
    r = client.get("/v1/search", params={"q": "cà phê", "bbox": "100,0,101,1"})
    assert r.status_code == 200
    body = r.json()
    assert body["results"], "G5: a valid query is never empty"
    assert body["meta"]["source"] == "fallback"


def test_fallback_honors_category_and_bbox():
    # C5/D3: 'quán cà phê' narrows the ranker to cafés; the gas-station + Hanoi-bbox
    # filter then rejects them all, firing the backstop. The honest fallback must
    # return gas stations INSIDE the bbox, not the global popularity list.
    bbox = "105.7,20.9,105.9,21.1"  # Hanoi window containing several Trạm xăng
    r = client.get("/v1/search",
                   params={"q": "quán cà phê", "category": "Trạm xăng", "bbox": bbox})
    assert r.status_code == 200
    body = r.json()
    assert body["meta"]["source"] == "fallback"
    assert body["results"]
    lo = [float(x) for x in bbox.split(",")]
    for x in body["results"]:
        assert x["category"] == "Trạm xăng"
        c = x["coordinates"]
        assert lo[0] <= c["lon"] <= lo[2] and lo[1] <= c["lat"] <= lo[3]


def test_impossible_category_bbox_falls_back_globally():
    # On-category POIs exist, but none inside the ocean bbox -> global fallback (G5 wins).
    r = client.get("/v1/search",
                   params={"q": "cà phê", "category": "Trạm xăng", "bbox": "100,0,101,1"})
    assert r.status_code == 200
    body = r.json()
    assert body["meta"]["source"] == "fallback"
    assert body["results"]


def test_semantic_search_threads_fallback_source():
    # C18 on the enriched endpoint too.
    r = client.get("/v1/semantic-search", params={"q": "cà phê", "bbox": "100,0,101,1"})
    assert r.status_code == 200
    assert r.json()["meta"]["source"] == "fallback"


def test_nonnumeric_limit_is_contract_400():
    # C19/C21: a non-numeric limit must return the contract ErrorResponse shape,
    # not FastAPI's raw 422 {detail:[...]}.
    r = client.get("/v1/search", params={"q": "cà phê", "limit": "abc"})
    assert r.status_code == 400
    body = r.json()
    assert body["error"]["code"] == "invalid_request"
    assert body.get("requestId")
    assert r.headers.get("X-Request-Id")


def test_nonnumeric_latlon_is_contract_400():
    r = client.get("/v1/search", params={"q": "cà phê", "lat": "abc", "lon": "def"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_request"


def test_nonnumeric_radius_is_contract_400():
    r = client.get("/v1/search", params={"q": "cà phê", "radiusMeters": "abc"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_request"


def test_error_response_echoes_request_id_in_body_and_header():
    # debt5/debt16: every error path carries X-Request-Id in BOTH body and header.
    rid = "err-req-999"
    r = client.get("/v1/search", headers={"X-Request-Id": rid})  # missing q -> 400
    assert r.status_code == 400
    assert r.json()["requestId"] == rid
    assert r.headers.get("X-Request-Id") == rid


def test_validation_error_echoes_request_id_header():
    rid = "val-req-777"
    r = client.get("/v1/search", params={"q": "cà phê", "limit": "abc"},
                   headers={"X-Request-Id": rid})
    assert r.status_code == 400
    assert r.json()["requestId"] == rid
    assert r.headers.get("X-Request-Id") == rid


def test_radius_zero_is_explicit_not_ignored():
    # D4: radiusMeters=0 is a degenerate-but-explicit constraint. No POI sits exactly
    # on the focus point, so the 0m window is empty and the honest backstop fires.
    # (If 0 were ignored, source would be 'semsearch' with real nearby results.)
    r = client.get("/v1/search",
                   params={"q": "cà phê", "lat": 10.0, "lon": 106.0, "radiusMeters": 0})
    assert r.status_code == 200
    body = r.json()
    assert body["results"]  # G5
    assert body["meta"]["source"] == "fallback"


def test_negative_radius_is_contract_400():
    r = client.get("/v1/search",
                   params={"q": "cà phê", "lat": 10.0, "lon": 106.0, "radiusMeters": -5})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_request"
