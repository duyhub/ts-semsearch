"""Tests for the read-only /admin transparency view (docs/plans/admin-dashboard-plan.md).

Guards the reframe's invariants: /admin is read-only (no mutating routes), it reports the
committed weights actually in use, and the contract-exact /v1/search is unchanged.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from semsearch.api import create_app
from semsearch.rank import SIGNALS, load_weights

app = create_app(prewarm=False)
client = TestClient(app)


def test_admin_page_serves_html():
    r = client.get("/admin")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Tasco" in r.text


def test_admin_config_reports_committed_weights():
    r = client.get("/admin/config")
    assert r.status_code == 200
    cfg = r.json()
    assert cfg["readOnly"] is True
    # every signal is present, exactly the committed weight the ranker runs with
    committed = load_weights()
    reported = {s["key"]: s["weight"] for s in cfg["signals"]}
    assert set(reported) == set(SIGNALS)
    for k in SIGNALS:
        assert reported[k] == round(committed[k], 4)


def test_admin_config_shares_sum_to_one():
    cfg = client.get("/admin/config").json()
    shares = [s["share"] for s in cfg["signals"]]
    assert abs(sum(shares) - 1.0) < 1e-3
    assert all(s["description"] for s in cfg["signals"])  # every signal glossed


def test_admin_config_is_read_only():
    # no mutating verbs exist on the admin surface (NFR-6 hard rule protection)
    assert client.put("/admin/config", json={"weights": {"semantic": 1.0}}).status_code == 405
    assert client.post("/admin/config", json={"weights": {"semantic": 1.0}}).status_code == 405


def test_admin_does_not_perturb_contract_search():
    # touching /admin must not change the contract-exact /v1/search response shape
    client.get("/admin")
    client.get("/admin/config")
    body = client.get("/v1/search", params={"q": "cà phê"}).json()
    assert set(body) == {"query", "results", "meta"}  # no leaked trace/admin fields
