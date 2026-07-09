"""RRF fusion + the A2 embedding-cache/matrix guards (SPEC §4-5, §11).

These stay model-free (no bge-m3 load, no network): RRF is pure, and the A2
guards are about keys/manifests, not vectors.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from semsearch import embeddings as E
from semsearch.retrieve import rrf_fuse


def test_rrf_fuse_orders_by_reciprocal_rank():
    # bm25=['A','B'], dense=['B','C']  (c=60)
    #   B = 1/62 + 1/61 (highest), A = 1/61, C = 1/62
    fused = rrf_fuse([["A", "B"], ["B", "C"]])
    assert [pid for pid, _ in fused] == ["B", "A", "C"]


def test_rrf_fuse_empty():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []


def test_query_cache_key_includes_provider_and_model():
    """Same text, different provider/model -> different cache key (A2)."""
    text = "cà phê yên tĩnh"
    local = SimpleNamespace(provider="local", model_id="BAAI/bge-m3")
    cohere = SimpleNamespace(provider="bedrock-cohere", model_id="cohere.embed-multilingual-v3")
    k_local = E._qkey(local, text)
    k_cohere = E._qkey(cohere, text)
    assert k_local != k_cohere                      # provider/model in the key
    assert k_local == E._qkey(local, text)          # stable for same inputs
    assert k_local != E._qkey(local, "khác")        # text still matters


def test_load_doc_matrix_refuses_provider_mismatch(tmp_path, monkeypatch):
    """A cached matrix whose manifest names a different provider must be refused, not
    silently used (A2 — all 1024-d, so a mismatch would return garbage)."""
    monkeypatch.setattr(E, "CACHE_DIR", tmp_path)
    local = E.LocalEmbedder()  # no model load; load_doc_matrix never calls embed
    poi_ids = ["C001", "C002"]

    # write a matrix at local's expected path but stamp the manifest as a DIFFERENT provider
    np.save(E._matrix_path(local), np.zeros((2, 1024), dtype=np.float32))
    import json
    with open(E._manifest_path(local), "w", encoding="utf-8") as fh:
        json.dump(
            {"provider": "bedrock-cohere", "model_id": "cohere.embed-multilingual-v3",
             "dim": 1024, "n_docs": 2, "poi_ids": poi_ids},
            fh,
        )

    with pytest.raises(ValueError, match="mismatch"):
        E.load_doc_matrix(local, poi_ids)


def test_load_doc_matrix_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "CACHE_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        E.load_doc_matrix(E.LocalEmbedder(), ["C001"])
