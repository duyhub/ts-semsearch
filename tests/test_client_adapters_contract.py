"""Pins every client adapter in clients/ to the live /v1/search contract (FR-16).

The adapters (Dart, Go, Node, Python) are our "integration-ready" deliverable
(docs/tasco_api.pdf). Each reads a fixed set of PlaceResult keys. Only the Python one is
importable here, so a rename in the PlaceResult DTO could leave the others silently
broken while every other test stays green. These tests:

  1. derive the canonical read-set from the reference Dart adapter,
  2. assert it is valid against the Pydantic DTO,
  3. assert every other client references exactly the same fields (cross-client parity),
  4. assert a real /v1/search response carries those keys, and
  5. round-trip a real response through the importable Python client.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from semsearch.api import Coordinates, PlaceResult, create_app

CLIENTS = Path(__file__).resolve().parents[1] / "clients"


def _src(name: str) -> str:
    return (CLIENTS / name).read_text(encoding="utf-8")


def _bracket_keys(prefix: str, source: str) -> set[str]:
    # e.g. p['distanceMeters'] / c['lat'] — used by the Dart, Node and Python adapters.
    return set(re.findall(rf"""{prefix}\[['"]([A-Za-z]+)['"]\]""", source))


# Canonical contract, taken from the reference client (clients/tasco_adapter.dart).
_DART = _src("tasco_adapter.dart")
CANON_PLACE = _bracket_keys("p", _DART)
CANON_COORD = _bracket_keys("c", _DART)


def test_canonical_keys_are_valid_dto_fields():
    """The reference client's reads must be declared PlaceResult / Coordinates fields."""
    assert CANON_PLACE, "expected to derive PlaceResult keys from the Dart adapter"
    assert CANON_PLACE <= set(PlaceResult.model_fields), (
        "reference client reads PlaceResult keys the DTO no longer exposes: "
        f"{sorted(CANON_PLACE - set(PlaceResult.model_fields))}"
    )
    assert CANON_COORD == set(Coordinates.model_fields), (
        f"coordinate keys drifted: client={sorted(CANON_COORD)} "
        f"dto={sorted(Coordinates.model_fields)}"
    )


def test_bracket_clients_match_canonical():
    """Node and Python adapters must read exactly the same fields as the reference."""
    for name in ("tasco_adapter.mjs", "tasco_adapter.py"):
        src = _src(name)
        place, coord = _bracket_keys("p", src), _bracket_keys("c", src)
        assert place == CANON_PLACE, (
            f"{name} PlaceResult reads diverged from the contract: "
            f"missing={sorted(CANON_PLACE - place)} extra={sorted(place - CANON_PLACE)}"
        )
        assert coord == CANON_COORD, (
            f"{name} coordinate reads diverged: "
            f"missing={sorted(CANON_COORD - coord)} extra={sorted(coord - CANON_COORD)}"
        )


def test_go_struct_tags_cover_canonical():
    """The Go adapter binds via json struct tags; they must cover the whole contract.

    Subset (not equality): the file also carries ErrorResponse/results tags, which are
    legitimately extra. A DTO rename drops a contract field from CANON and fails this.
    """
    tags = set(re.findall(r'json:"([A-Za-z]+)"', _src("tasco_adapter.go")))
    contract = CANON_PLACE | CANON_COORD
    assert contract <= tags, (
        "tasco_adapter.go json tags no longer cover the contract: "
        f"missing={sorted(contract - tags)}"
    )


def _live_top_result():
    app = create_app(prewarm=False, mode="local")
    client = TestClient(app)
    r = client.get(
        "/v1/search",
        params={"q": "quán cà phê yên tĩnh làm việc", "lat": 10.7738, "lon": 106.7040},
    )
    assert r.status_code == 200
    results = r.json()["results"]
    assert results, "need at least one result to check the adapter contract"
    return client, results[0]


def test_live_search_response_satisfies_contract():
    """A real /v1/search result carries every key the adapters dereference.

    lat/lon are supplied so distanceMeters (Optional, anchor-only) is materialised.
    """
    _, top = _live_top_result()
    assert CANON_PLACE <= set(top), (
        f"live /v1/search result missing contract keys: {sorted(CANON_PLACE - set(top))}"
    )
    assert CANON_COORD <= set(top["coordinates"]), (
        f"live coordinates missing keys: {sorted(CANON_COORD - set(top['coordinates']))}"
    )
    assert top["distanceMeters"] is not None, "distanceMeters must be set when lat/lon given"


def _load_python_client():
    spec = importlib.util.spec_from_file_location(
        "tasco_adapter", CLIENTS / "tasco_adapter.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    # Register before exec: dataclass field-type resolution reads sys.modules[__module__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_python_client_round_trips_live_response():
    """Drive the importable Python client against the in-process app end-to-end."""
    client, _ = _live_top_result()
    mod = _load_python_client()

    def transport(url, headers, timeout):
        # Absolute URL -> path+query for the ASGI TestClient; headers/body preserved.
        parts = url.split("/v1/search", 1)
        path = "/v1/search" + (parts[1] if len(parts) > 1 else "")
        resp = client.get(path, headers=headers)
        return resp.status_code, dict(resp.headers), resp.text

    pyclient = mod.TascoSemanticClient(base_url="http://engine", transport=transport)
    suggestions = pyclient.search(
        "quán cà phê yên tĩnh làm việc", lat=10.7738, lon=106.7040
    )
    assert suggestions, "python client should map at least one suggestion"
    top = suggestions[0]
    assert top.id.startswith("poi:")
    assert top.label and top.description  # diacritics-bearing strings, non-empty
    assert isinstance(top.coordinates.lat, float) and isinstance(top.coordinates.lon, float)
    assert top.meta["distanceMeters"] is not None
    assert isinstance(top.meta["tags"], list)


def test_python_client_raises_structured_error_on_400():
    """A contract 400 (missing q) surfaces as a TascoApiError with code + request id."""
    client = TestClient(create_app(prewarm=False, mode="local"))
    mod = _load_python_client()

    def transport(url, headers, timeout):
        parts = url.split("/v1/search", 1)
        path = "/v1/search" + (parts[1] if len(parts) > 1 else "")
        resp = client.get(path, headers=headers)
        return resp.status_code, dict(resp.headers), resp.text

    pyclient = mod.TascoSemanticClient(base_url="http://engine", transport=transport)
    try:
        pyclient.search("")  # empty q -> contract 400 invalid_request
    except mod.TascoApiError as exc:
        assert exc.status == 400
        assert exc.code == "invalid_request"
        assert exc.request_id  # echoed from the response
    else:
        raise AssertionError("expected TascoApiError for missing q")
