"""Embeddings with a provider switch + provider-stamped caches (SPEC §4, D1, A2).

Local `bge-m3` is the primary provider (the build/tune/gates run against it).
Bedrock (cohere / titan) is selectable and *measured* but never the default, and
never required to run (NFR-3).

A2 (the silent-garbage guard): every cached vector is keyed by
`provider:model_id:text`, and the doc matrix is stamped with its
provider/model/dim in a manifest the loader asserts against. bge-m3, cohere-v3
and titan-v2 are all 1024-d, so a provider mismatch would otherwise return
noise, not an error.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from .data import POI

CACHE_DIR = Path("data/derived")
QCACHE_DIR = CACHE_DIR / "qcache"

MODEL_IDS = {
    "local": "BAAI/bge-m3",
    "bedrock-cohere": "cohere.embed-multilingual-v3",
    "bedrock-titan": "amazon.titan-embed-text-v2:0",
}


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

    def embed(self, texts: Sequence[str]) -> np.ndarray:  # (n, d), L2-normalized
        ...


class LocalEmbedder:
    """bge-m3 via sentence-transformers. Model loads lazily (first embed) so
    importing this module is cheap and offline-safe."""

    provider = "local"
    model_id = MODEL_IDS["local"]

    def __init__(self) -> None:
        self._model = None

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # heavy import, deferred

            self._model = SentenceTransformer(self.model_id)
        return self._model

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        model = self._ensure()
        vecs = model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float32)


def get_embedder(provider: str = "local") -> Embedder:
    if provider == "local":
        return LocalEmbedder()
    raise SystemExit(
        f"provider {provider!r} not wired yet (local is the default; Bedrock is a later, "
        "credential-gated path). See FR-10."
    )


def _safe(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")


def _matrix_path(emb: Embedder) -> Path:
    return CACHE_DIR / f"embeddings.{emb.provider}.{_safe(emb.model_id)}.npy"


def _manifest_path(emb: Embedder) -> Path:
    return CACHE_DIR / f"embeddings.{emb.provider}.{_safe(emb.model_id)}.manifest.json"


def build_doc_matrix(pois: Sequence[POI], emb: Embedder) -> np.ndarray:
    """Embed composed POI docs, write a provider-stamped matrix + manifest, return it."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    matrix = emb.embed([compose_doc_text(p) for p in pois])
    np.save(_matrix_path(emb), matrix)
    manifest = {
        "provider": emb.provider,
        "model_id": emb.model_id,
        "dim": int(matrix.shape[1]),
        "n_docs": int(matrix.shape[0]),
        "poi_ids": [p.poi_id for p in pois],
    }
    with open(_manifest_path(emb), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    return matrix


def load_doc_matrix(emb: Embedder, expected_poi_ids: Sequence[str]) -> np.ndarray:
    """Load the cached matrix, asserting the manifest matches the active provider/model
    and POI order. Refuses a mismatch rather than returning garbage (A2)."""
    mpath, jpath = _matrix_path(emb), _manifest_path(emb)
    if not mpath.exists() or not jpath.exists():
        raise FileNotFoundError(f"no cached matrix for provider={emb.provider} model={emb.model_id}; run ingest")
    with open(jpath, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest["provider"] != emb.provider or manifest["model_id"] != emb.model_id:
        raise ValueError(
            f"doc-matrix provider/model mismatch: manifest={manifest['provider']}/{manifest['model_id']} "
            f"active={emb.provider}/{emb.model_id} (A2 guard)"
        )
    if manifest["poi_ids"] != list(expected_poi_ids):
        raise ValueError("doc-matrix POI order does not match current dataset; rebuild (A2 guard)")
    return np.load(mpath)


def _qkey(emb: Embedder, text: str) -> str:
    return hashlib.sha1(f"{emb.provider}:{emb.model_id}:{text}".encode("utf-8")).hexdigest()


def embed_query(emb: Embedder, text: str, *, use_cache: bool = True) -> np.ndarray:
    """Embed a single query, disk-cached by provider+model+text (A2). Returns (d,)."""
    if not use_cache:
        return emb.embed([text])[0]
    cache_dir = QCACHE_DIR / f"{emb.provider}.{_safe(emb.model_id)}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{_qkey(emb, text)}.npy"
    if path.exists():
        return np.load(path)
    vec = emb.embed([text])[0]
    np.save(path, vec)
    return vec
