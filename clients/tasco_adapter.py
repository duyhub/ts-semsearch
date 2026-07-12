"""Tasco Maps client adapter (PRD FR-16; tasco_api.pdf mapping table).

Drop-in for a Python service's search layer: point ``base_url`` at this engine's
/v1/search and map the contract-exact ``PlaceResult`` -> the app's ``SearchSuggestion``.
Integration is a base-URL change; no UI dependencies. Zero third-party deps (stdlib
``urllib``); the HTTP call is injectable via ``transport`` for testing.

    client = TascoSemanticClient(base_url="https://semsearch.example.com")
    suggestions = client.search(
        "quán cà phê yên tĩnh để làm việc", lat=10.7738, lon=106.7040
    )

PlaceResult -> SearchSuggestion mapping (per the PDF):
    id          -> id
    name/label  -> label
    category    -> meta["category"] ; type -> meta["type"]
    address     -> description
    coordinates -> coordinates (WGS84, unchanged)
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Optional

# A transport returns (status_code, response_headers, body_text). Injectable so tests can
# drive the client against an in-process ASGI app instead of a live socket.
Transport = Callable[[str, dict[str, str], float], "tuple[int, dict[str, str], str]"]


@dataclass(frozen=True)
class Coordinates:
    lat: float
    lon: float


@dataclass(frozen=True)
class SearchSuggestion:
    """Mirrors the app's existing SearchSuggestion DTO."""

    id: str
    label: str
    description: str
    coordinates: Coordinates
    meta: dict[str, Any]

    @classmethod
    def from_place_result(cls, p: dict[str, Any]) -> "SearchSuggestion":
        """Map a contract-exact PlaceResult object to a SearchSuggestion."""
        c = p["coordinates"]
        return cls(
            id=p["id"],  # stable, e.g. "poi:C001"
            label=p["label"] or p["name"],  # diacritics preserved
            description=p["address"],
            coordinates=Coordinates(float(c["lat"]), float(c["lon"])),  # WGS84, unchanged
            meta={
                "type": p["type"],
                "category": p["category"],
                "score": p["score"],
                "distanceMeters": p["distanceMeters"],
                "source": p["source"],
                "tags": p["tags"],
            },
        )


class TascoApiError(Exception):
    """Structured non-200 from the engine (contract ErrorResponse, or a body snippet)."""

    def __init__(
        self,
        status: int,
        code: Optional[str] = None,
        message: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> None:
        self.status = status
        self.code = code
        self.message = message
        self.request_id = request_id
        req = f" req={request_id}" if request_id else ""
        super().__init__(f"TascoApiError({status} {code or '-'}{req}): {message}")


def _urllib_transport(
    url: str, headers: dict[str, str], timeout: float
) -> "tuple[int, dict[str, str], str]":
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, dict(resp.headers), resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:  # non-2xx still carries a body
        return exc.code, dict(exc.headers or {}), exc.read().decode("utf-8", "replace")


@dataclass
class TascoSemanticClient:
    base_url: str
    api_key: Optional[str] = None  # optional; sent as Bearer when set
    timeout: float = 5.0
    transport: Transport = _urllib_transport

    def search(
        self,
        query: str,
        *,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        limit: int = 10,
        lang: str = "vi",
    ) -> list[SearchSuggestion]:
        params: dict[str, str] = {"q": query, "limit": str(limit), "lang": lang}
        if lat is not None:
            params["lat"] = repr(lat)
        if lon is not None:
            params["lon"] = repr(lon)
        url = f"{self.base_url}/v1/search?{urllib.parse.urlencode(params)}"

        headers = {"X-Request-Id": str(time.time_ns() // 1000)}
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key}"

        status, resp_headers, body_text = self.transport(url, headers, self.timeout)
        if status != 200:
            raise self._error_for(status, resp_headers, body_text)

        body = json.loads(body_text)
        return [SearchSuggestion.from_place_result(p) for p in body.get("results", [])]

    @staticmethod
    def _error_for(
        status: int, headers: dict[str, str], body_text: str
    ) -> TascoApiError:
        # Header lookup is case-insensitive per HTTP; normalise for safety.
        lowered = {k.lower(): v for k, v in headers.items()}
        request_id = lowered.get("x-request-id")
        try:
            err = json.loads(body_text)
            e = err.get("error")
            code = e.get("code") if isinstance(e, dict) else None
            message = e.get("message") if isinstance(e, dict) else None
            return TascoApiError(
                status, code, message, request_id or err.get("requestId")
            )
        except (ValueError, AttributeError):
            # Non-JSON body (e.g. an upstream proxy's HTML error page).
            snippet = body_text[:200] + "..." if len(body_text) > 200 else body_text
            return TascoApiError(status, None, snippet, request_id)
