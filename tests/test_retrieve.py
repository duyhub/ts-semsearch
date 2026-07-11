"""RRF fusion + the A2 embedding-cache/matrix guards (SPEC §4-5, §11).

These stay model-free (no bge-m3 load, no network): RRF is pure, and the A2
guards are about keys/manifests, not vectors.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from semsearch import embeddings as E
from semsearch import retrieve as R
from semsearch.retrieve import DenseIndex, rrf_fuse


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
    silently used (A2 — all 1024-d, so a mismatch would return garbage). The manifest
    stays the authoritative check even though the corpus-hashed path makes a genuine
    cross-corpus collision structurally near-impossible (see build_doc_matrix docstring)."""
    monkeypatch.setattr(E, "CACHE_DIR", tmp_path)
    local = E.LocalEmbedder()  # no model load; load_doc_matrix never calls embed
    poi_ids = ["C001", "C002"]

    # write a matrix at local's hashed path but stamp the manifest as a DIFFERENT provider
    np.save(E._matrix_path(local, poi_ids), np.zeros((2, 1024), dtype=np.float32))
    with open(E._manifest_path(local, poi_ids), "w", encoding="utf-8") as fh:
        json.dump(
            {"provider": "bedrock-cohere", "model_id": "cohere.embed-multilingual-v3",
             "dim": 1024, "n_docs": 2, "poi_ids": poi_ids},
            fh,
        )

    with pytest.raises(ValueError, match="mismatch"):
        E.load_doc_matrix(local, poi_ids)


def test_load_doc_matrix_refuses_poi_order_mismatch(tmp_path, monkeypatch):
    """A cached matrix whose manifest lists different POI ids/order than the current
    dataset must be refused, not silently used against the wrong rows (A2/C24). This can
    no longer happen via normal build/load (different poi_ids hash to different paths),
    so we deliberately corrupt the manifest at the path the caller's poi_ids resolve to —
    proving the manifest check is still the authoritative guard, not just the hashed path."""
    monkeypatch.setattr(E, "CACHE_DIR", tmp_path)
    local = E.LocalEmbedder()  # no model load; load_doc_matrix never calls embed
    requested = ["C001", "C999"]

    # matrix + manifest are self-consistent (right provider/model) but stamped with a
    # DIFFERENT POI id set than the caller asks for, deliberately placed at the path
    # `requested` hashes to (simulating a corrupted/stale manifest, or a hash collision).
    np.save(E._matrix_path(local, requested), np.zeros((2, 1024), dtype=np.float32))
    with open(E._manifest_path(local, requested), "w", encoding="utf-8") as fh:
        json.dump(
            {"provider": local.provider, "model_id": local.model_id,
             "dim": 1024, "n_docs": 2, "poi_ids": ["C001", "C002"]},
            fh,
        )

    with pytest.raises(ValueError, match="POI order"):
        E.load_doc_matrix(local, requested)  # C999 != cached C002


def test_load_doc_matrix_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "CACHE_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        E.load_doc_matrix(E.LocalEmbedder(), ["C001"])


# --------------------------------------------------------------------------- #
# Corpus-discriminated cache paths (multi-corpus footgun: official 111 POIs   #
# vs. the 1000-POI synthetic superset must never share a doc-matrix path)     #
# --------------------------------------------------------------------------- #
class _FakeEmbedder:
    """Cheap, deterministic Embedder double — no model load, no network."""

    provider = "local"
    model_id = "BAAI/bge-m3"
    dim = 16

    def embed(self, texts, *, input_type="search_document"):
        n = len(texts)
        arr = np.zeros((n, self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            arr[i, i % self.dim] = 1.0 + (len(text) % 5)  # deterministic, text-dependent
        return arr


def _fake_pois(ids: list[str]):
    return [
        SimpleNamespace(
            poi_id=pid, name=f"poi {pid}", brand=None, category="Cafe",
            sub_category=None, district="Quận 1", city="TP.HCM", attributes=[],
            tags=[], description="",
        )
        for pid in ids
    ]


def test_matrix_path_distinct_for_different_corpora(tmp_path, monkeypatch):
    """Two different POI-id corpora must resolve to different cache paths — the bug this
    fix addresses: before this, both corpora shared one path keyed by provider+model only,
    so building over one corpus silently overwrote the other's cache."""
    monkeypatch.setattr(E, "CACHE_DIR", tmp_path)
    local = E.LocalEmbedder()
    ids_a = ["C001", "C002", "C003"]
    ids_b = ["D001", "D002"]
    assert E._matrix_path(local, ids_a) != E._matrix_path(local, ids_b)
    assert E._manifest_path(local, ids_a) != E._manifest_path(local, ids_b)


def test_matrix_path_distinct_for_reordered_same_ids(tmp_path, monkeypatch):
    """Same POI ids, different order -> different path. Row order is baked into the
    matrix, so an order change must not silently reuse another order's cache."""
    monkeypatch.setattr(E, "CACHE_DIR", tmp_path)
    local = E.LocalEmbedder()
    ids = ["C001", "C002", "C003"]
    reordered = ["C003", "C001", "C002"]
    assert E._matrix_path(local, ids) != E._matrix_path(local, reordered)
    assert E._manifest_path(local, ids) != E._manifest_path(local, reordered)


def test_corpus_hashed_paths_round_trip_for_two_corpora(tmp_path, monkeypatch):
    """build -> load round-trips correctly and independently for two distinct corpora
    sharing the same provider/model and the same tmp CACHE_DIR."""
    monkeypatch.setattr(E, "CACHE_DIR", tmp_path)
    emb = _FakeEmbedder()
    pois_a = _fake_pois(["C001", "C002", "C003"])
    pois_b = _fake_pois(["D001", "D002"])

    matrix_a = E.build_doc_matrix(pois_a, emb)
    matrix_b = E.build_doc_matrix(pois_b, emb)

    loaded_a = E.load_doc_matrix(emb, [p.poi_id for p in pois_a])
    loaded_b = E.load_doc_matrix(emb, [p.poi_id for p in pois_b])
    assert np.array_equal(loaded_a, matrix_a)
    assert np.array_equal(loaded_b, matrix_b)
    assert loaded_a.shape == (3, emb.dim)
    assert loaded_b.shape == (2, emb.dim)


def test_dense_index_switches_between_corpora_without_crashing(tmp_path, monkeypatch):
    """DenseIndex over corpus A, then over corpus B, using the same tmp CACHE_DIR: must
    NOT raise (this is exactly the bug — building B used to overwrite A's cache path,
    so a later A rebuild/load would trip the A2 poi-order ValueError uncaught)."""
    monkeypatch.setattr(E, "CACHE_DIR", tmp_path)
    emb = _FakeEmbedder()
    pois_a = _fake_pois(["C001", "C002"])
    pois_b = _fake_pois(["D001", "D002", "D003"])

    idx_a = DenseIndex(pois_a, emb)
    idx_b = DenseIndex(pois_b, emb)
    assert idx_a.poi_ids == ["C001", "C002"]
    assert idx_b.poi_ids == ["D001", "D002", "D003"]
    assert idx_a.matrix.shape == (2, emb.dim)
    assert idx_b.matrix.shape == (3, emb.dim)


def test_dense_index_reload_of_earlier_corpus_uses_its_own_cache(tmp_path, monkeypatch):
    """Re-creating a DenseIndex over corpus A after B has also been indexed must load A's
    OWN cached matrix (not rebuild, not B's) — proving the two corpora's caches coexist."""
    monkeypatch.setattr(E, "CACHE_DIR", tmp_path)
    emb = _FakeEmbedder()
    pois_a = _fake_pois(["C001", "C002"])
    pois_b = _fake_pois(["D001", "D002", "D003"])

    idx_a = DenseIndex(pois_a, emb)  # builds + caches A
    DenseIndex(pois_b, emb)          # builds + caches B (same CACHE_DIR, distinct path)

    def _boom(pois, e):
        raise AssertionError("must load A's cache, not rebuild")

    monkeypatch.setattr(R, "build_doc_matrix", _boom)
    idx_a2 = DenseIndex(pois_a, emb)  # re-create over corpus A: must hit A's cache
    assert np.array_equal(idx_a2.matrix, idx_a.matrix)
    assert idx_a2.poi_ids == ["C001", "C002"]


def test_dense_index_rebuilds_on_manifest_mismatch_instead_of_crashing(tmp_path, monkeypatch):
    """Defensive: DenseIndex.__init__ catches (FileNotFoundError, ValueError) from
    load_doc_matrix and rebuilds rather than propagating the A2 ValueError. With
    corpus-hashed paths a real mismatch is structurally near-impossible, but a stale/
    corrupted manifest landing at the hashed path (e.g. hand-edited, or a truncated-hash
    collision) must still degrade to a fresh, coherent-by-construction build — never a crash."""
    monkeypatch.setattr(E, "CACHE_DIR", tmp_path)
    emb = _FakeEmbedder()
    pois = _fake_pois(["C001", "C002"])
    poi_ids = [p.poi_id for p in pois]

    E.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(E._matrix_path(emb, poi_ids), np.zeros((2, emb.dim), dtype=np.float32))
    with open(E._manifest_path(emb, poi_ids), "w", encoding="utf-8") as fh:
        json.dump(
            {"provider": emb.provider, "model_id": emb.model_id, "dim": emb.dim,
             "n_docs": 2, "poi_ids": ["WRONG1", "WRONG2"]},
            fh,
        )

    idx = DenseIndex(pois, emb)  # must rebuild, not crash
    assert idx.poi_ids == ["C001", "C002"]
    assert idx.matrix.shape == (2, emb.dim)
