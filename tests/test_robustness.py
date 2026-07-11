"""Robustness / G5 guard (SPEC §11; PRD NFR-2).

Adversarial inputs + missing-q, through both endpoints. G5 rule: 200 with
>=1 result, OR a contract-valid 400 invalid_request. Never a 5xx, exception,
or empty 200. (The full 60-query sweep lives in scripts/robustness.py.)
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from semsearch.adversarial import ADVERSARIAL
from semsearch.api import create_app

client = TestClient(create_app(prewarm=False, mode="local"))
ENDPOINTS = ("/v1/search", "/v1/semantic-search")


def _g5_ok(status: int, body: dict) -> bool:
    if status == 200:
        return len(body.get("results", [])) >= 1
    if status == 400:
        return body.get("error", {}).get("code") == "invalid_request"
    return False


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize("name,text", ADVERSARIAL, ids=[n for n, _ in ADVERSARIAL])
def test_adversarial_inputs_satisfy_g5(endpoint, name, text):
    r = client.get(endpoint, params={"q": text})
    assert _g5_ok(r.status_code, r.json()), f"{name} on {endpoint}: status={r.status_code}"


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_missing_q_is_contract_400(endpoint):
    r = client.get(endpoint)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_request"


def test_non_empty_gibberish_returns_results_not_empty():
    # the C1 backstop must keep a present-but-meaningless query non-empty
    r = client.get("/v1/search", params={"q": "zzzz qwerty asdf"})
    assert r.status_code == 200 and len(r.json()["results"]) >= 1
