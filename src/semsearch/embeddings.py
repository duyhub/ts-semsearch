"""Embeddings with a provider switch + provider-stamped caches (SPEC §4, D1, A2).

Local `bge-m3` is the primary provider (the build/tune/gates run against it).
Bedrock (cohere / titan) is selectable and *measured* but never the default, and
never required to run (NFR-3).

A2 (the silent-garbage guard): every cached vector is keyed by
`provider:model_id:text`, and the doc matrix is stamped with its
provider/model/dim in a manifest the loader asserts against. bge-m3, cohere-v3
and titan-v2 are all 1024-d, so a provider mismatch would otherwise return
noise, not an error. The doc-matrix cache path is ALSO namespaced by a corpus
fingerprint (a hash of the POI id sequence) — multiple corpora sharing one
provider/model (e.g. the official 111-POI set and a synthetic superset) never
collide at the same path, so indexing one can never overwrite another's cache.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from . import tracing
from .data import POI

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/derived")
QCACHE_DIR = CACHE_DIR / "qcache"

MODEL_IDS = {
    "local": "BAAI/bge-m3",
    "bedrock-cohere": "cohere.embed-multilingual-v3",
    "bedrock-titan": "amazon.titan-embed-text-v2:0",
}

# bge-m3, cohere-v3 and titan-v2 are all 1024-d — a provider mismatch would return
# noise, not a shape error, hence the A2 manifest/cache guards. Also the width of the
# zero-vector a failed per-query embed degrades to.
EMBED_DIM = 1024

BEDROCK_PROVIDERS = ("bedrock-cohere", "bedrock-titan")
COHERE_MAX_BATCH = 96  # Cohere embed on Bedrock accepts up to 96 texts/call
# HARD RULE (CLAUDE.md): Bedrock calls carry a timeout so a dead network fails fast
# (<= connect+read ≈ 12s) instead of hanging the demo. No retries — we degrade, not stall.
_BEDROCK_TIMEOUT = {"connect_timeout": 2, "read_timeout": 10, "retries": {"max_attempts": 1}}

# --------------------------------------------------------------------------- #
# Bedrock region-fallback chain (FR-10). Shared by BOTH consumers — embeddings #
# (here) and the Claude parse (llm_parse imports it) — and each resolves its    #
# OWN region independently by walking this chain until ITS model answers.       #
#                                                                              #
# EMPIRICAL, measured live on the AABW event account 2026-07-11:                #
#   ap-southeast-1 (Singapore, closest to the demo venue): cohere-v3 WORKS;      #
#       Claude WORKS (global. profile, after the Anthropic use-case form was     #
#       approved); titan-embed-v2 NOT OFFERED in-region (regional absence).      #
#   ap-northeast-1 (Tokyo): ALL THREE work.                                      #
#   us-west-2 (Oregon): ALL THREE work.                                          #
# Titan is the only capability that cannot use Singapore, so ONLY titan gets a   #
# per-model chain below; cohere and Claude pin venue-proximal Singapore.         #
DEFAULT_BEDROCK_REGIONS = ("ap-southeast-1", "ap-northeast-1", "us-west-2")
# Per-MODEL default chains, only for models MEASURED structurally absent from a default
# region (regional-absence — not an access gate that approval could lift): titan-v2 is not
# offered in ap-southeast-1, so its chain starts in Tokyo instead of burning a doomed probe
# on Singapore every run. Env overrides in resolve_bedrock_regions still replace ANY chain.
MODEL_DEFAULT_REGIONS: dict[str, tuple[str, ...]] = {
    MODEL_IDS["bedrock-titan"]: ("ap-northeast-1", "us-west-2"),
}
SEMSEARCH_BEDROCK_REGION_ENV = "SEMSEARCH_BEDROCK_REGION"    # singular: pin exactly one region
SEMSEARCH_BEDROCK_REGIONS_ENV = "SEMSEARCH_BEDROCK_REGIONS"  # plural (CSV): replace the whole chain


def is_no_credentials_error(exc: Exception) -> bool:
    """True when a Bedrock failure means 'no usable AWS credentials' (incl. broken/expired
    SSO tokens) — an ACCOUNT-WIDE state, not a regional one, so callers may short-circuit
    the remaining region/provider probes instead of burning one timeout per candidate.
    check_bedrock.py classifies STS failures with the same family."""
    try:
        from botocore.exceptions import (
            NoCredentialsError,
            PartialCredentialsError,
            SSOTokenLoadError,
            TokenRetrievalError,
            UnauthorizedSSOTokenError,
        )
    except ImportError:  # pragma: no cover - botocore absent -> nothing bedrock anyway
        return False
    return isinstance(exc, (NoCredentialsError, PartialCredentialsError, SSOTokenLoadError,
                            TokenRetrievalError, UnauthorizedSSOTokenError))


def resolve_bedrock_regions(model_id: str | None = None) -> tuple[str, ...]:
    """The ordered region chain a Bedrock consumer walks until ITS model works. Pass the
    consumer's model_id so a model with a measured per-model default chain (titan-v2, absent
    from Singapore) starts where it actually exists — only the DEFAULT is per-model.

    Precedence, highest first:
      1. SEMSEARCH_BEDROCK_REGION (singular) -> exactly one region (a chain of one; keeps the
         existing pin-one-region escape hatch as the strongest override).
      2. SEMSEARCH_BEDROCK_REGIONS (plural, comma-separated) -> replaces the whole chain.
      3. AWS_REGION / AWS_DEFAULT_REGION -> exactly one region (preserves today's semantics).
      4. MODEL_DEFAULT_REGIONS[model_id] if present, else DEFAULT_BEDROCK_REGIONS.

    Always non-empty (a blank plural value falls through), so a consumer never resolves zero
    regions."""
    single = os.environ.get(SEMSEARCH_BEDROCK_REGION_ENV)
    if single:
        return (single,)
    plural = os.environ.get(SEMSEARCH_BEDROCK_REGIONS_ENV)
    if plural:
        regions = tuple(r.strip() for r in plural.split(",") if r.strip())
        if regions:
            return regions
    aws = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if aws:
        return (aws,)
    return MODEL_DEFAULT_REGIONS.get(model_id or "", DEFAULT_BEDROCK_REGIONS)


def compose_doc_text(p: POI) -> str:
    """Embedding document text (SPEC §4). Shared by ingest and the dense index."""
    brand = p.brand or ""
    sub = p.sub_category or ""
    attrs = ", ".join(p.attributes)
    tags = ", ".join(p.tags)
    return (
        f"{p.name}. {brand}. {p.category} / {sub}. {p.district}, {p.city}. "
        f"Đặc điểm: {attrs}. {tags}. {p.description}"
    )


class Embedder(Protocol):
    provider: str
    model_id: str
    dim: int

    # input_type is a Bedrock/Cohere concern (docs vs queries embed differently);
    # symmetric encoders ignore it. Default = document side.
    def embed(self, texts: Sequence[str], *, input_type: str = "search_document") -> np.ndarray:
        ...  # (n, d), L2-normalized


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    """Row-wise unit-normalize (zero rows stay zero, no NaN). DenseIndex treats cosine
    as a plain matvec, so every provider's vectors must arrive at unit norm."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (arr / norms).astype(np.float32)


class LocalEmbedder:
    """bge-m3 via sentence-transformers. Model loads lazily (first embed) so
    importing this module is cheap and offline-safe."""

    provider = "local"
    model_id = MODEL_IDS["local"]
    dim = EMBED_DIM

    def __init__(self) -> None:
        self._model = None

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # heavy import, deferred

            self._model = SentenceTransformer(self.model_id)
        return self._model

    def embed(self, texts: Sequence[str], *, input_type: str = "search_document") -> np.ndarray:
        # bge-m3 is symmetric — input_type is accepted for a uniform Embedder API but ignored.
        model = self._ensure()
        vecs = model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float32)


class BedrockEmbedder:
    """Amazon Bedrock embeddings — the Built-with-AWS core-component path (FR-10).

    Two providers behind one class:
      * bedrock-cohere -> cohere.embed-multilingual-v3: batched (<=96 texts/call),
        body {"texts", "input_type", "truncate":"END"}; input_type is 'search_document'
        for docs and 'search_query' for queries (Cohere on Bedrock REQUIRES it).
      * bedrock-titan  -> amazon.titan-embed-text-v2:0: one inputText per call,
        body {"inputText", "dimensions", "normalize":true}.

    Design notes:
      * Lazy client: importing this module — and even *constructing* the embedder —
        pulls in no boto3 and triggers no credential lookup; the client is built on
        first embed. (resolve_provider's preflight is the only construction-time reach.)
      * Region fallback: first embed walks the region chain (resolve_bedrock_regions),
        probing one tiny embed per region; the FIRST region whose probe works is pinned
        for this instance's whole run (one vector space, A2 — never re-walked mid-run).
        All regions failing re-raises, so the caller degrades to local exactly as today.
      * We L2-normalize the returned vectors OURSELVES. Cohere embeddings are NOT
        normalized, and DenseIndex's cosine-as-matvec assumes unit norm — so we never
        trust the API defaults, keeping cohere/titan/bge-m3 interchangeable at the index.
      * Timeouts, not retries (see _BEDROCK_TIMEOUT): a dead network fails fast.
    """

    dim = EMBED_DIM

    def __init__(self, provider: str) -> None:
        if provider not in BEDROCK_PROVIDERS:
            raise ValueError(f"BedrockEmbedder got non-bedrock provider {provider!r}")
        self.provider = provider
        self.model_id = MODEL_IDS[provider]
        self._client = None  # lazy (no boto3 import / cred lookup until first embed)
        self._region: str | None = None  # pinned by _pin_region on first use (region walk)

    @staticmethod
    def _make_client(region: str):
        import boto3  # deferred: no import cost unless a bedrock provider is selected
        from botocore.config import Config

        return boto3.client(
            "bedrock-runtime", region_name=region, config=Config(**_BEDROCK_TIMEOUT)
        )

    def _probe(self, client) -> None:
        """One tiny invoke_model proving THIS embedding model works in THIS region. Raises on
        any failure — region unreachable, model not offered in-region, no model access."""
        if self.provider == "bedrock-cohere":
            body = json.dumps({"texts": ["ping"], "input_type": "search_query", "truncate": "END"})
        else:
            body = json.dumps({"inputText": "ping", "dimensions": self.dim, "normalize": True})
        client.invoke_model(modelId=self.model_id, body=body)

    def _pin_region(self):
        """Walk the region chain; pin the FIRST region whose preflight probe succeeds for this
        instance's whole run. Every region failing re-raises the last error so the caller (a
        preflight, or a per-query embed) degrades exactly as today.

        Tradeoff: a NO-CREDENTIALS failure short-circuits the walk after the first region —
        credentials are account-wide, not regional, so probing the remaining regions can only
        burn 2 more connect-timeouts for the same answer. Network/model errors keep the full
        per-region walk (an outage or a missing model CAN be regional)."""
        regions = resolve_bedrock_regions(self.model_id)
        last_exc: Exception | None = None
        for region in regions:
            try:
                client = self._make_client(region)
                self._probe(client)
            except Exception as exc:  # noqa: BLE001 - region/model unavailable; try the next
                last_exc = exc
                if is_no_credentials_error(exc):
                    logger.warning(
                        "Bedrock %s: no AWS credentials (%s) — skipping the remaining "
                        "regions (credentials are account-wide).",
                        self.provider, type(exc).__name__,
                    )
                    break
                logger.warning(
                    "Bedrock %s: region %s unavailable (%s: %s); trying the next region.",
                    self.provider, region, type(exc).__name__, exc,
                )
                continue
            self._client, self._region = client, region
            logger.info("Bedrock %s pinned to region %s.", self.provider, region)
            return self._client
        raise last_exc if last_exc is not None else RuntimeError("no bedrock regions configured")

    def _get_client(self):
        if self._client is None:
            self._pin_region()
        return self._client

    def embed(self, texts: Sequence[str], *, input_type: str = "search_document") -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        # Trace the batch by name + count only (never the 111 texts). No-op without keys.
        with tracing.traced(
            "bedrock_embed", kind="embedding", model=self.model_id,
            metadata={"provider": self.provider, "count": len(texts), "input_type": input_type},
        ):
            raw = (
                self._embed_cohere(texts, input_type)
                if self.provider == "bedrock-cohere"
                else self._embed_titan(texts)
            )
        return _l2_normalize(np.asarray(raw, dtype=np.float32))

    def _embed_cohere(self, texts: list[str], input_type: str) -> list[list[float]]:
        client = self._get_client()
        out: list[list[float]] = []
        for start in range(0, len(texts), COHERE_MAX_BATCH):
            batch = texts[start : start + COHERE_MAX_BATCH]
            body = json.dumps({"texts": batch, "input_type": input_type, "truncate": "END"})
            resp = client.invoke_model(modelId=self.model_id, body=body)
            out.extend(json.loads(resp["body"].read())["embeddings"])
        return out

    def _embed_titan(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        out: list[list[float]] = []
        for text in texts:  # Titan v2 embeds exactly one inputText per invoke
            body = json.dumps({"inputText": text, "dimensions": self.dim, "normalize": True})
            resp = client.invoke_model(modelId=self.model_id, body=body)
            out.append(json.loads(resp["body"].read())["embedding"])
        return out


def get_embedder(provider: str = "local") -> Embedder:
    if provider == "local":
        return LocalEmbedder()
    if provider in BEDROCK_PROVIDERS:
        return BedrockEmbedder(provider)
    raise SystemExit(
        f"provider {provider!r} not wired (supported: local, {', '.join(BEDROCK_PROVIDERS)}). "
        "See FR-10."
    )


def resolve_provider(requested: str = "local") -> str:
    """Resolve the *effective* embedding provider, degrading a broken Bedrock setup
    to 'local' BEFORE any index is built — the design decision that keeps vector spaces
    from ever mixing (A2). Called on the FullPipeline construction path.

    For a bedrock provider, run a cheap preflight (embed one short string, under the
    HARD-RULE timeout). On ANY failure — no credentials, no model access, region wrong,
    timeout — log one clear warning and return 'local', i.e. exactly today's offline
    behavior. A successful preflight means the whole doc matrix can be built in that
    space, so the choice is coherent for the entire run. Non-bedrock providers pass
    through untouched (no client, no preflight)."""
    if requested not in BEDROCK_PROVIDERS:
        return requested
    try:
        vec = get_embedder(requested).embed(["ping"], input_type="search_query")
        if vec.shape[0] != 1:  # pragma: no cover - defensive
            raise RuntimeError("preflight returned no embedding")
    except Exception as exc:  # noqa: BLE001 - ANY failure degrades to local (NFR-3)
        logger.warning(
            "Bedrock provider %r unavailable (%s: %s); falling back to local bge-m3. "
            "Run `python scripts/check_bedrock.py` to diagnose.",
            requested,
            type(exc).__name__,
            exc,
        )
        return "local"
    return requested


def _safe(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")


def _corpus_hash(poi_ids: Sequence[str]) -> str:
    """Short fingerprint of a POI id sequence, order-sensitive by design (row order is
    baked into the matrix, so a reordered corpus must not resolve to the same path as
    the original order). Used to namespace doc-matrix cache paths across corpora (see
    module docstring, A2) — NOT a substitute for the manifest's poi_ids equality check,
    which stays the authoritative guard; this is purely path namespacing."""
    return hashlib.sha1(",".join(poi_ids).encode("utf-8")).hexdigest()[:8]


def _matrix_path(emb: Embedder, poi_ids: Sequence[str]) -> Path:
    return CACHE_DIR / f"embeddings.{emb.provider}.{_safe(emb.model_id)}.c{_corpus_hash(poi_ids)}.npy"


def _manifest_path(emb: Embedder, poi_ids: Sequence[str]) -> Path:
    return (
        CACHE_DIR
        / f"embeddings.{emb.provider}.{_safe(emb.model_id)}.c{_corpus_hash(poi_ids)}.manifest.json"
    )


def build_doc_matrix(pois: Sequence[POI], emb: Embedder) -> np.ndarray:
    """Embed composed POI docs, write a provider+corpus-stamped matrix + manifest, return it.

    The cache path is namespaced by both provider/model AND a corpus fingerprint (a hash
    of the POI id sequence, see `_corpus_hash`) — two different corpora (e.g. the official
    111-POI set and a synthetic superset) resolve to different files, so building one can
    never overwrite another's cache. The manifest itself (written alongside) remains the
    authoritative A2 check; the path hash just keeps corpora from ever contending for it.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    poi_ids = [p.poi_id for p in pois]
    matrix = emb.embed([compose_doc_text(p) for p in pois])
    np.save(_matrix_path(emb, poi_ids), matrix)
    manifest = {
        "provider": emb.provider,
        "model_id": emb.model_id,
        "dim": int(matrix.shape[1]),
        "n_docs": int(matrix.shape[0]),
        "poi_ids": poi_ids,
    }
    with open(_manifest_path(emb, poi_ids), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    return matrix


def load_doc_matrix(emb: Embedder, expected_poi_ids: Sequence[str]) -> np.ndarray:
    """Load the cached matrix for THIS provider/model and THIS corpus (the path is
    resolved from a hash of expected_poi_ids, see `_matrix_path`), then assert the
    manifest matches the active provider/model and POI order exactly. Refuses a mismatch
    rather than returning garbage (A2). The corpus-hashed path makes a genuine cross-corpus
    collision structurally near-impossible, but the manifest check stays the authoritative
    guard — a corrupted/stale manifest landing at the right path must still be caught."""
    poi_ids = list(expected_poi_ids)
    mpath, jpath = _matrix_path(emb, poi_ids), _manifest_path(emb, poi_ids)
    if not mpath.exists() or not jpath.exists():
        raise FileNotFoundError(
            f"no cached matrix for provider={emb.provider} model={emb.model_id} "
            f"corpus={_corpus_hash(poi_ids)} ({len(poi_ids)} POIs); run ingest"
        )
    with open(jpath, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest["provider"] != emb.provider or manifest["model_id"] != emb.model_id:
        raise ValueError(
            f"doc-matrix provider/model mismatch: manifest={manifest['provider']}/{manifest['model_id']} "
            f"active={emb.provider}/{emb.model_id} (A2 guard)"
        )
    if manifest["poi_ids"] != poi_ids:
        raise ValueError("doc-matrix POI order does not match current dataset; rebuild (A2 guard)")
    return np.load(mpath)


def _qkey(emb: Embedder, text: str) -> str:
    return hashlib.sha1(f"{emb.provider}:{emb.model_id}:{text}".encode("utf-8")).hexdigest()


def _embed_query_vec(emb: Embedder, text: str) -> tuple[np.ndarray, bool]:
    """Embed one query on the SEARCH side, degrading a failed BEDROCK call to a zero
    vector. Returns (vec, ok).

    Rationale (A2 + NFR-3): a per-query bedrock failure AFTER construction — creds
    expired mid-demo, a transient 5xx — must never crash the request and must never
    substitute a *different* vector space (a bge-m3 vector in a cohere index is silent
    garbage, not an error, since both are 1024-d). A zero vector means dense has no
    opinion (DenseIndex returns an empty ranking for it), so RRF defers to BM25 for
    that one query. `ok=False` tells the caller NOT to cache it, so a transient
    failure can't poison the qcache for the rest of the demo.

    Scoped to BEDROCK_PROVIDERS only: a LOCAL bge-m3 failure (corrupt HF cache,
    missing dependency) is a setup bug — it must propagate loudly, not boot a
    'healthy' server that silently serves BM25-only results."""
    if emb.provider not in BEDROCK_PROVIDERS:
        return emb.embed([text], input_type="search_query")[0], True
    try:
        return emb.embed([text], input_type="search_query")[0], True
    except Exception as exc:  # noqa: BLE001 - never crash a query on a bedrock failure
        logger.warning(
            "query embed failed for provider %r (%s: %s); using zero vector "
            "(no dense ranking; falls back to BM25 for this query).",
            emb.provider,
            type(exc).__name__,
            exc,
        )
        return np.zeros(emb.dim, dtype=np.float32), False


def embed_query(emb: Embedder, text: str, *, use_cache: bool = True) -> np.ndarray:
    """Embed a single query, disk-cached by provider+model+text (A2). Returns (d,)."""
    if not use_cache:
        return _embed_query_vec(emb, text)[0]
    cache_dir = QCACHE_DIR / f"{emb.provider}.{_safe(emb.model_id)}"
    path = cache_dir / f"{_qkey(emb, text)}.npy"
    if path.exists():
        return np.load(path)
    vec, ok = _embed_query_vec(emb, text)
    if ok:  # cache successful embeds only — never persist a fallback zero vector
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(path, vec)
    return vec
