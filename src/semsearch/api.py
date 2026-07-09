"""FastAPI service implementing Tasco's /v1/search contract (SPEC §9; FR-11/12).

- /v1/search           contract-exact PlaceResult[] (aliases /search, /v1/geocode-search)
- /v1/semantic-search  same + per-signal breakdown, reasons[], parsed intent echo
- /health              liveness

Determinism (NFR-5): the ranker's clock is fixed to DEFAULT_EVAL_NOW so identical
requests return identical results (demo + contract tests reproducible). Diacritics
are preserved in every field (NFR-4); poi ids are prefixed `poi:` on output only.
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import Optional

from pathlib import Path

from fastapi import FastAPI, Header, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

UI_INDEX = Path(__file__).resolve().parents[2] / "ui" / "index.html"

from .data import POI, load_pois
from .explain import generate_reasons
from .geo import haversine
from .pipeline import FullPipeline
from .rank import DEFAULT_EVAL_NOW

ERROR_CODES = {  # SPEC §9 error-code table
    400: "invalid_request", 401: "unauthorized", 403: "forbidden", 404: "not_found",
    408: "request_timeout", 429: "rate_limited", 500: "internal_error", 503: "unavailable",
}


# ---- DTOs (contract-exact) --------------------------------------------------
class Coordinates(BaseModel):
    lat: float
    lon: float


class PlaceResult(BaseModel):
    id: str
    type: str
    name: str
    label: str
    address: str
    category: str
    coordinates: Coordinates
    distanceMeters: Optional[int] = None
    score: float
    source: str
    tags: list[str]


class Meta(BaseModel):
    count: int
    limitApplied: int
    tookMs: float
    source: str = "semsearch"


class SearchResponse(BaseModel):
    query: str
    results: list[PlaceResult]
    meta: Meta


class SemanticPlaceResult(PlaceResult):
    breakdown: dict[str, float]
    reasons: list[str]


class IntentEcho(BaseModel):
    category: Optional[str] = None
    requiredAttrs: list[str] = []
    anchor: Optional[dict] = None
    city: Optional[str] = None
    openAfter: Optional[str] = None


class SemanticSearchResponse(BaseModel):
    query: str
    intent: IntentEcho
    results: list[SemanticPlaceResult]
    meta: Meta


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[dict] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
    requestId: str


def _error(status: int, message: str, request_id: str, details: dict | None = None) -> JSONResponse:
    body = ErrorResponse(error=ErrorDetail(code=ERROR_CODES[status], message=message, details=details),
                         requestId=request_id)
    return JSONResponse(status_code=status, content=body.model_dump())


def _distance_m(poi: POI, ref: Optional[tuple[float, float]]) -> Optional[int]:
    if ref is None:
        return None
    return int(round(haversine(ref[0], ref[1], poi.lat, poi.lon) * 1000))


def _parse_bbox(bbox: str) -> tuple[float, float, float, float]:
    parts = [float(x) for x in bbox.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be minLon,minLat,maxLon,maxLat")
    return tuple(parts)  # type: ignore[return-value]


def create_app(pois: Optional[list[POI]] = None, *, now: datetime = DEFAULT_EVAL_NOW,
               prewarm: bool = True) -> FastAPI:
    app = FastAPI(title="Tasco Semantic Search & Ranking", version="0.1.0")
    pipeline = FullPipeline(pois if pois is not None else load_pois(), now=now)
    if prewarm:  # P1: warm the embedding model at boot so the first live query is snappy.
        # Neutral warmup string (NOT an eval query — keeps eval text out of src, NFR-6).
        pipeline.rank_ids("khởi động hệ thống tìm kiếm")

    def _rank_filtered(q: str, *, limit: int, category: Optional[str],
                       ref: Optional[tuple[float, float]], radius: Optional[float],
                       bbox: Optional[tuple[float, float, float, float]], engine: str = "full"):
        intent = pipeline.parser.parse(q)
        if engine == "keyword":  # BM25-only lane for the demo's keyword column
            ranked = [(pid, 0.0, {}) for pid in pipeline.bm25.rank_ids(q)]
        else:
            ranked = pipeline.rank_scored(q)  # full corpus, (id, score, breakdown)
        picked = []
        for pid, score, breakdown in ranked:
            poi = pipeline.by_id[pid]
            if category and poi.category != category:
                continue
            if bbox and not (bbox[0] <= poi.lon <= bbox[2] and bbox[1] <= poi.lat <= bbox[3]):
                continue
            if ref and radius and haversine(ref[0], ref[1], poi.lat, poi.lon) * 1000 > radius:
                continue
            picked.append((poi, score, breakdown))
            if len(picked) >= limit:
                break
        source = "semsearch"
        if not picked:  # C1 backstop: valid query never returns empty
            top = sorted(pipeline.pois, key=lambda p: p.popularity, reverse=True)[:limit]
            picked = [(p, 0.0, {}) for p in top]
            source = "fallback"
        return intent, picked, source

    def _place(poi: POI, score: float, ref, source: str) -> PlaceResult:
        return PlaceResult(
            id=f"poi:{poi.poi_id}", type="poi", name=poi.name,
            label=f"{poi.name}, {poi.district}", address=poi.address, category=poi.category,
            coordinates=Coordinates(lat=poi.lat, lon=poi.lon),
            distanceMeters=_distance_m(poi, ref), score=round(score, 4),
            source=source, tags=poi.tags,
        )

    def _common(q, lat, lon, radiusMeters, bbox, category, limit, x_request_id):
        rid = x_request_id or str(uuid.uuid4())
        if q is None or not q.strip():
            return None, _error(400, "query parameter 'q' is required", rid)
        limit = max(1, min(int(limit), 20))  # default 10, max 20 (SPEC §9)
        ref = (lat, lon) if lat is not None and lon is not None else None
        box = None
        if bbox:
            try:
                box = _parse_bbox(bbox)
            except ValueError as e:
                return None, _error(400, str(e), rid)
        return (rid, q, limit, ref, radiusMeters, box, category), None

    @app.get("/", response_class=HTMLResponse)
    def index():
        if UI_INDEX.exists():
            return HTMLResponse(UI_INDEX.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>Tasco Semantic Search API</h1><p>UI not built. See /docs.</p>")

    @app.get("/health")
    def health():
        return {"status": "ok", "pois": len(pipeline.pois)}

    @app.get("/v1/search", response_model=SearchResponse)
    @app.get("/search", response_model=SearchResponse)
    @app.get("/v1/geocode-search", response_model=SearchResponse)
    def search(
        request: Request,
        q: Optional[str] = Query(None),
        lat: Optional[float] = None, lon: Optional[float] = None,
        radiusMeters: Optional[float] = None, bbox: Optional[str] = None,
        category: Optional[str] = None, limit: int = 10, lang: str = "vi",
        engine: str = "full",
        x_request_id: Optional[str] = Header(None),
    ):
        parsed, err = _common(q, lat, lon, radiusMeters, bbox, category, limit, x_request_id)
        if err:
            return err
        rid, q, limit, ref, radius, box, category = parsed
        t0 = time.perf_counter()
        _, picked, source = _rank_filtered(q, limit=limit, category=category, ref=ref,
                                           radius=radius, bbox=box,
                                           engine="keyword" if engine == "keyword" else "full")
        took = (time.perf_counter() - t0) * 1000
        results = [_place(p, s, ref, source) for p, s, _ in picked]
        resp = SearchResponse(query=q, results=results,
                              meta=Meta(count=len(results), limitApplied=limit, tookMs=round(took, 2)))
        return JSONResponse(content=resp.model_dump(), headers={"X-Request-Id": rid})

    @app.get("/v1/semantic-search", response_model=SemanticSearchResponse)
    def semantic_search(
        request: Request,
        q: Optional[str] = Query(None),
        lat: Optional[float] = None, lon: Optional[float] = None,
        radiusMeters: Optional[float] = None, bbox: Optional[str] = None,
        category: Optional[str] = None, limit: int = 10, lang: str = "vi",
        x_request_id: Optional[str] = Header(None),
    ):
        parsed, err = _common(q, lat, lon, radiusMeters, bbox, category, limit, x_request_id)
        if err:
            return err
        rid, q, limit, ref, radius, box, category = parsed
        t0 = time.perf_counter()
        intent, picked, source = _rank_filtered(q, limit=limit, category=category, ref=ref,
                                                radius=radius, bbox=box)
        took = (time.perf_counter() - t0) * 1000
        results = [
            SemanticPlaceResult(**_place(p, s, ref, source).model_dump(),
                                breakdown={k: round(v, 4) for k, v in b.items()},
                                reasons=generate_reasons(intent, p))
            for p, s, b in picked
        ]
        anchor = ({"name": intent.anchor.name, "lat": intent.anchor.lat, "lon": intent.anchor.lon}
                  if intent.anchor else None)
        echo = IntentEcho(category=intent.category, requiredAttrs=intent.required_attrs,
                          anchor=anchor, city=intent.city, openAfter=intent.open_after)
        resp = SemanticSearchResponse(query=q, intent=echo, results=results,
                                      meta=Meta(count=len(results), limitApplied=limit, tookMs=round(took, 2)))
        return JSONResponse(content=resp.model_dump(), headers={"X-Request-Id": rid})

    return app

# Serve with the factory (avoids building the pipeline at import time, so tests
# and `import semsearch.api` stay cheap):
#   uv run uvicorn semsearch.api:create_app --factory --port 8000
