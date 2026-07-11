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
from dataclasses import replace
from datetime import datetime
from typing import Optional

from pathlib import Path

from fastapi import FastAPI, Header, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

UI_DIR = Path(__file__).resolve().parents[2] / "ui"
UI_INDEX = UI_DIR / "index.html"

from .data import POI, Anchor, load_pois
from .explain import generate_reasons
from .geo import COORD_ANCHOR_NAME, haversine
from .pipeline import FullPipeline
from .rank import DEFAULT_EVAL_NOW, load_weights

ERROR_CODES = {  # SPEC §9 / tasco_api.pdf "Common error codes" table (verbatim strings)
    400: "invalid_request", 401: "unauthorized", 403: "forbidden", 404: "not_found",
    408: "timeout", 429: "rate_limited", 500: "internal_error", 503: "service_unavailable",
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
    # FR-4: the LLM-corrected query (typos fixed, diacritics restored) when it differs from the
    # `query` echo. Additive + optional — popped from clean-query responses so they stay
    # contract-exact. Declared here so it appears in the OpenAPI schema.
    correctedQuery: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    results: list[PlaceResult]
    meta: Meta


class PlaceDetail(BaseModel):
    """Extra POI fields for the UI's "Chi tiết" (details) panel — not part of the
    contract-exact /v1/search response, only the enriched /v1/semantic-search."""
    description: str = ""
    attributes: list[str] = []
    rating: float
    reviewCount: int
    priceLevel: Optional[int] = None
    openingHours: Optional[str] = None
    brand: Optional[str] = None
    subCategory: Optional[str] = None
    district: Optional[str] = None
    city: Optional[str] = None


class SemanticPlaceResult(PlaceResult):
    breakdown: dict[str, float]
    reasons: list[str]
    detail: PlaceDetail


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
    weights: dict[str, float] = {}  # the ranker's tuned per-signal weights (for the "Vì sao?" panel)


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
    # Every error path echoes X-Request-Id in the header too (matches the success paths).
    return JSONResponse(status_code=status, content=body.model_dump(),
                        headers={"X-Request-Id": request_id})


def _distance_m(poi: POI, ref: Optional[tuple[float, float]]) -> Optional[int]:
    if ref is None:
        return None
    return int(round(haversine(ref[0], ref[1], poi.lat, poi.lon) * 1000))


def _parse_bbox(bbox: str) -> tuple[float, float, float, float]:
    parts = [float(x) for x in bbox.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be minLon,minLat,maxLon,maxLat")
    return tuple(parts)  # type: ignore[return-value]


def _embeddings_status(pipeline) -> str:
    """/health string for what the embeddings actually resolved to: 'local',
    '<provider>@<region>' for a pinned cloud provider, or 'bm25-only' on the floor."""
    if pipeline.dense is None:
        return "bm25-only"
    emb = pipeline.dense.emb
    if emb.provider == "local":
        return "local"
    # a cached doc matrix defers the region pin to the first live query embed
    return f"{emb.provider}@{getattr(emb, '_region', None) or 'unpinned'}"


def _llm_status(pipeline) -> str:
    """/health string for the LLM parse: '<provider>+<model>' when a provider pinned at
    construction, else 'rules-only' (gate off, or nothing resolved)."""
    parser = pipeline._llm_parser
    if parser is None or parser._client is None:
        return "rules-only"
    return f"{parser._provider or 'bedrock'}+{parser.model_id}"


def _apply_corrected_query(payload: dict, intent, q: str) -> dict:
    """Surface the LLM-corrected query additively as meta.correctedQuery, in place, ONLY when a
    correction exists AND differs from the raw echo `q`. Otherwise the key is popped so a clean
    query (or the rule-only keyword lane) stays byte-identical to the contract shape (Meta
    declares correctedQuery, so model_dump() always emits it — the pop keeps it out)."""
    corrected = getattr(intent, "corrected_query", None)
    if corrected and corrected != q:
        payload["meta"]["correctedQuery"] = corrected
    else:
        payload["meta"].pop("correctedQuery", None)
    return payload


def create_app(pois: Optional[list[POI]] = None, *, now: datetime = DEFAULT_EVAL_NOW,
               prewarm: bool = True, mode: Optional[str] = None) -> FastAPI:
    app = FastAPI(title="Tasco Semantic Search & Ranking", version="0.1.0")
    if UI_DIR.exists():  # serve vendored assets (Leaflet js/css) offline-safe at /ui/*
        app.mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui")
    # Serve the TUNED weights (weights.json), so the live API matches the reported
    # metrics instead of falling back to untuned DEFAULT_WEIGHTS. `mode` defaults to the
    # env-resolved deployment mode (SEMSEARCH_MODE / config.DEFAULT_MODE) inside the
    # pipeline — no signature break for existing callers/tests.
    pipeline = FullPipeline(pois if pois is not None else load_pois(),
                            weights=load_weights(), now=now, mode=mode)
    app.state.pipeline = pipeline
    if prewarm:  # P1: warm the embedding model at boot so the first live query is snappy.
        # Neutral warmup string (NOT an eval query — keeps eval text out of src, NFR-6).
        pipeline.rank_ids("khởi động hệ thống tìm kiếm")

    def _rank_filtered(q: str, *, limit: int, category: Optional[str],
                       ref: Optional[tuple[float, float]], radius: Optional[float],
                       bbox: Optional[tuple[float, float, float, float]], engine: str = "full"):
        if engine == "keyword":  # BM25-only lane for the demo's keyword column
            # intent is unused by this lane's ranking and discarded by /v1/search — rule
            # parse only, so the keyword column never pays (or depends on) an LLM call.
            intent = pipeline.parser.parse(q)
            ranked = [(pid, 0.0, {}) for pid in pipeline.bm25.rank_ids(q)]
        else:
            # Resolve the intent ONCE and pass it through: ranking, the /v1/semantic-search
            # intent echo, and reasons[] must all read the SAME object. A rule-only re-parse
            # here + LLM-merged re-resolution inside rank_scored would contradict each other
            # whenever SEMSEARCH_LLM_PARSE=bedrock enriches the parse.
            intent = pipeline.resolve_intent(q)
            # A location explicitly named/pasted in the query is more specific than the
            # request focus. Otherwise, use the caller's current/selected location as the
            # distance anchor so proximity affects ranking, explanations, and the intent echo.
            if ref is not None and intent.anchor is None:
                intent = replace(
                    intent,
                    anchor=Anchor(name=COORD_ANCHOR_NAME, lat=ref[0], lon=ref[1]),
                )
            ranked = pipeline.rank_scored(q, intent=intent)  # full corpus, (id, score, breakdown)

        def _passes(poi: POI) -> bool:
            """The caller's explicit category / bbox / radius window (radius=0 is an
            explicit 0-metre constraint, not 'absent' — D4)."""
            if category and poi.category != category:
                return False
            if bbox and not (bbox[0] <= poi.lon <= bbox[2] and bbox[1] <= poi.lat <= bbox[3]):
                return False
            if ref is not None and radius is not None and \
                    haversine(ref[0], ref[1], poi.lat, poi.lon) * 1000 > radius:
                return False
            return True

        picked = []
        for pid, score, breakdown in ranked:
            poi = pipeline.by_id[pid]
            if not _passes(poi):
                continue
            picked.append((poi, score, breakdown))
            if len(picked) >= limit:
                break
        source = "semsearch"
        if not picked:  # C1/G5 backstop: a valid query never returns empty.
            # C5/D3: honour the caller's filters first — the fallback pool is the SAME
            # category/bbox/radius window, nearest-first when a request focus exists and
            # popularity-first otherwise. Only an impossible constraint (e.g. a mid-ocean
            # bbox) drops to the global list, still labelled source='fallback'.
            if ref is not None:
                fallback_order = sorted(
                    pipeline.pois,
                    key=lambda p: (
                        haversine(ref[0], ref[1], p.lat, p.lon),
                        -p.popularity,
                    ),
                )
            else:
                fallback_order = sorted(
                    pipeline.pois, key=lambda p: p.popularity, reverse=True
                )
            pool = [p for p in fallback_order if _passes(p)] or fallback_order
            picked = [(p, 0.0, {}) for p in pool[:limit]]
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
        if (lat is None) != (lon is None):
            missing = "lon" if lon is None else "lat"
            return None, _error(
                400,
                "lat and lon must be supplied together",
                rid,
                details={"field": missing},
            )
        if radiusMeters is not None and (lat is None or lon is None):
            return None, _error(
                400,
                "radiusMeters requires lat and lon",
                rid,
                details={"field": "radiusMeters"},
            )
        if radiusMeters is not None and radiusMeters < 0:  # D5: a radius is a distance, not signed
            return None, _error(400, "radiusMeters must be >= 0", rid,
                                details={"field": "radiusMeters"})
        limit = max(1, min(int(limit), 20))  # default 10, max 20 (SPEC §9)
        ref = (lat, lon) if lat is not None and lon is not None else None
        box = None
        if bbox:
            try:
                box = _parse_bbox(bbox)
            except ValueError as e:
                return None, _error(400, str(e), rid)
        return (rid, q, limit, ref, radiusMeters, box, category), None

    @app.exception_handler(RequestValidationError)
    def _on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # C19/C21: type-coercion failures (non-numeric limit/lat/lon/radiusMeters) reach
        # FastAPI as a raw 422 {detail:[...]}. Re-shape to the contract ErrorResponse
        # (400 invalid_request) and carry X-Request-Id in body + header like every path.
        rid = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        errors = exc.errors()
        first = errors[0] if errors else {}
        field = str(first.get("loc", ["", "?"])[-1])
        msg = first.get("msg", "invalid request parameter")
        return _error(400, f"invalid value for '{field}': {msg}", rid, details={"field": field})

    @app.get("/", response_class=HTMLResponse)
    def index():
        if UI_INDEX.exists():
            return HTMLResponse(UI_INDEX.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>Tasco Semantic Search API</h1><p>UI not built. See /docs.</p>")

    @app.get("/health")
    def health():
        # Ops visibility for remote hosting (NOT part of the Tasco /v1/search contract):
        # the deployment mode plus what embeddings/LLM-parse ACTUALLY resolved to.
        return {
            "status": "ok",
            "pois": len(pipeline.pois),
            "mode": pipeline.mode,
            "embeddings": _embeddings_status(pipeline),
            "llm_parse": _llm_status(pipeline),
            # query rewrite is 'on' only when it can actually fire: the switch is on AND the
            # LLM parse gate is on (the correction rides that parse).
            "query_rewrite": "on" if (pipeline._query_rewrite
                                      and pipeline._llm_parser is not None) else "off",
            # LLM latency gate: 'auto' (fire only for degraded queries) or 'always' (every query).
            "llm_gate": pipeline._llm_gate,
        }

    @app.get("/v1/search", response_model=SearchResponse)
    @app.get("/search", response_model=SearchResponse)
    @app.get("/v1/geocode-search", response_model=SearchResponse)
    def search(
        request: Request,
        q: Optional[str] = Query(None),
        lat: Optional[float] = Query(
            None, ge=-90, le=90, description="WGS84 latitude; supply together with lon"
        ),
        lon: Optional[float] = Query(
            None, ge=-180, le=180, description="WGS84 longitude; supply together with lat"
        ),
        radiusMeters: Optional[float] = Query(
            None, ge=0, description="Optional radius filter; requires both lat and lon"
        ),
        bbox: Optional[str] = None,
        category: Optional[str] = None, limit: int = 10, lang: str = "vi",
        engine: str = "full",
        x_request_id: Optional[str] = Header(None),
    ):
        parsed, err = _common(q, lat, lon, radiusMeters, bbox, category, limit, x_request_id)
        if err:
            return err
        rid, q, limit, ref, radius, box, category = parsed
        t0 = time.perf_counter()
        intent, picked, source = _rank_filtered(q, limit=limit, category=category, ref=ref,
                                                radius=radius, bbox=box,
                                                engine="keyword" if engine == "keyword" else "full")
        took = (time.perf_counter() - t0) * 1000
        results = [_place(p, s, ref, source) for p, s, _ in picked]
        resp = SearchResponse(query=q, results=results,
                              meta=Meta(count=len(results), limitApplied=limit,
                                        tookMs=round(took, 2), source=source))
        payload = _apply_corrected_query(resp.model_dump(), intent, q)  # query echo stays ORIGINAL
        return JSONResponse(content=payload, headers={"X-Request-Id": rid})

    @app.get("/v1/semantic-search", response_model=SemanticSearchResponse)
    def semantic_search(
        request: Request,
        q: Optional[str] = Query(None),
        lat: Optional[float] = Query(
            None, ge=-90, le=90, description="WGS84 latitude; supply together with lon"
        ),
        lon: Optional[float] = Query(
            None, ge=-180, le=180, description="WGS84 longitude; supply together with lat"
        ),
        radiusMeters: Optional[float] = Query(
            None, ge=0, description="Optional radius filter; requires both lat and lon"
        ),
        bbox: Optional[str] = None,
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
                                reasons=generate_reasons(intent, p),
                                detail=PlaceDetail(
                                    description=p.description, attributes=p.attributes,
                                    rating=p.rating, reviewCount=p.review_count,
                                    priceLevel=p.price_level, openingHours=p.opening_hours,
                                    brand=p.brand, subCategory=p.sub_category,
                                    district=p.district, city=p.city))
            for p, s, b in picked
        ]
        anchor = ({"name": intent.anchor.name, "lat": intent.anchor.lat, "lon": intent.anchor.lon}
                  if intent.anchor else None)
        echo = IntentEcho(category=intent.category, requiredAttrs=intent.required_attrs,
                          anchor=anchor, city=intent.city, openAfter=intent.open_after)
        resp = SemanticSearchResponse(query=q, intent=echo, results=results,
                                      meta=Meta(count=len(results), limitApplied=limit,
                                                tookMs=round(took, 2), source=source),
                                      weights={k: round(w, 4) for k, w in pipeline.ranker.weights.items()})
        payload = _apply_corrected_query(resp.model_dump(), intent, q)  # query echo stays ORIGINAL
        return JSONResponse(content=payload, headers={"X-Request-Id": rid})

    return app

# Serve with the factory (avoids building the pipeline at import time, so tests
# and `import semsearch.api` stay cheap):
#   uv run uvicorn semsearch.api:create_app --factory --port 8000
