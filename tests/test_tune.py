"""Weight-tuning integrity + the extended (synth) pool (SPEC §6; PRD NFR-6, review C4).

`price` is a DELIBERATE FIXED preference weight — never eval-tuned (only 2/60 eval
queries express price). tune.py must therefore tune only the 8 non-price signals and
write exactly those 8 keys to data/weights.json; load_weights() re-supplies price via
its per-key default fallback. If tune.py ever coordinate-ascended `price`, a re-run
would silently write a 9-key weights.json and unfix the design.

`--pool extended` adds the 150 synthetic labelled queries (their own 1000-POI corpus)
to the 40 official tune queries and tunes on the mean NDCG@5 over all 190 (query,
home-corpus) pairs — but it is a CANDIDATE ONLY: it must NEVER write data/weights.json
(the committed weights stay reproducible from the frozen official protocol). These tests
lock: price exclusion, official default unchanged, the 190-query home-corpus mapping, the
mean-over-pairs objective, local pinning, the test-split firewall, and the never-clobber
write contract.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

from semsearch.rank import SIGNALS, WEIGHTS_PATH

ROOT = Path(__file__).resolve().parents[1]


def _load_tune_module():
    spec = importlib.util.spec_from_file_location("tune_script", ROOT / "scripts" / "tune.py")
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: tune.py's dataclasses use `from __future__ import annotations`,
    # so @dataclass resolves field annotations via sys.modules[cls.__module__] at class-def
    # time — an unregistered synthetic module would crash the import.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # module body is only imports + defs (main() gated on __main__)
    return mod


# --------------------------------------------------------------- price / key-set (C4)

def test_price_excluded_from_tunable_signals():
    tune = _load_tune_module()
    assert "price" not in tune.TUNABLE
    assert set(tune.TUNABLE) == set(SIGNALS) - {"price"}
    assert len(tune.TUNABLE) == 8


def test_write_seed_has_exactly_the_tunable_keys():
    tune = _load_tune_module()
    seed = tune.tunable_seed()
    assert "price" not in seed
    assert set(seed) == set(tune.TUNABLE)


def test_written_keyset_matches_committed_weights_json():
    tune = _load_tune_module()
    committed = json.loads(WEIGHTS_PATH.read_text(encoding="utf-8"))["weights"]
    # the key-set tune.py writes is invariant under coordinate ascent (values change,
    # keys don't), so the seed's key-set must equal the committed file's.
    assert set(tune.tunable_seed()) == set(committed)
    assert "price" not in committed


# ------------------------------------------------------------------ pool dispatch (CLI)

def test_default_pool_is_official(monkeypatch):
    tune = _load_tune_module()
    calls: list[str] = []
    monkeypatch.setattr(tune, "run_official", lambda: calls.append("official"))
    monkeypatch.setattr(tune, "run_extended", lambda: calls.append("extended"))
    tune.main([])  # no args -> default pool
    assert calls == ["official"]


def test_extended_pool_dispatches_to_extended_only(monkeypatch):
    tune = _load_tune_module()
    calls: list[str] = []
    monkeypatch.setattr(tune, "run_official", lambda: calls.append("official"))
    monkeypatch.setattr(tune, "run_extended", lambda: calls.append("extended"))
    tune.main(["--pool", "extended"])
    assert calls == ["extended"]


# ------------------------------------------------------------- extended pool loading

def test_extended_pool_loads_190_mapped_to_home_corpus():
    tune = _load_tune_module()
    official = tune.load_official_tune()
    synth = tune.load_synth()
    assert len(official) == 40
    assert len(synth) == 150
    assert len(official) + len(synth) == 190
    # Home corpora are distinct: the synth golds are SYN#### ids (only resolvable against
    # the 1000-POI synth corpus); the official golds are never SYN ids.
    syn_gold = {i for q in synth for i in q.expected_ids}
    off_gold = {i for q in official for i in q.expected_ids}
    assert any(i.startswith("SYN") for i in syn_gold)
    assert not any(i.startswith("SYN") for i in off_gold)
    # Query-id namespaces disjoint (no synth query leaks into the official arm).
    assert {q.query_id for q in official}.isdisjoint({q.query_id for q in synth})
    # The synth arm is pinned to the 1000-POI corpus xlsx.
    assert tune.SYNTH_CORPUS.name == "synth_dataset.xlsx"


# --------------------------------------------------------- mean-over-pairs objective

def test_mean_ndcg5_over_fake_pool():
    tune = _load_tune_module()
    # gold at rank 1 -> NDCG@5 = 1.0 ; gold at rank 2 -> gain-3 / log2(3) over idcg 3 = 1/log2(3)
    perfect = tune.QueryScorer(rank=lambda w: ["g1", "x", "y"], expected_ids=["g1"])
    rank2 = tune.QueryScorer(rank=lambda w: ["x", "g1", "y"], expected_ids=["g1"])
    got = tune.mean_ndcg5([perfect, rank2], {"semantic": 0.3})
    expected = (1.0 + 1.0 / math.log2(3)) / 2  # equal weight per pair
    assert abs(got - expected) < 1e-9
    assert tune.mean_ndcg5([], {"semantic": 0.3}) == 0.0


def test_mean_ndcg5_forwards_weights_to_each_scorer():
    tune = _load_tune_module()
    seen: list[dict] = []
    probe = tune.QueryScorer(rank=lambda w: seen.append(dict(w)) or ["g1"], expected_ids=["g1"])
    tune.mean_ndcg5([probe, probe], {"attributes": 0.4})
    assert seen == [{"attributes": 0.4}, {"attributes": 0.4}]


# -------------------------------------------------------------- shared ascent protocol

def test_ascend_drives_to_grid_max_and_leaves_price_out():
    tune = _load_tune_module()
    weights, best = tune.ascend(lambda w: w["attributes"], tune.tunable_seed(),
                                log=lambda *_: None)  # reward larger attributes only
    assert weights["attributes"] == max(tune.GRID)  # climbs to the 0.5 grid ceiling
    assert best == max(tune.GRID)
    assert "price" not in weights  # price never enters the ascent
    # signals the objective is indifferent to never move (margin gate)
    seed = tune.tunable_seed()
    for s in tune.TUNABLE:
        if s != "attributes":
            assert weights[s] == seed[s]


def test_ascend_margin_gate_blocks_sub_margin_gains():
    tune = _load_tune_module()
    seed = tune.tunable_seed()
    # max objective delta across the grid = (0.5-0.05)*1e-4 = 4.5e-5, far below MARGIN=1e-3
    weights, _ = tune.ascend(lambda w: w["review"] * 1e-4, seed, log=lambda *_: None)
    assert weights["review"] == seed["review"]


# ------------------------------------------------------ local pinning + test firewall

def test_build_local_pipeline_pins_provider_and_mode_local(monkeypatch):
    tune = _load_tune_module()
    captured: list[dict] = []

    class _FakePipe:
        def __init__(self, pois, **kw):
            captured.append(kw)

    monkeypatch.setattr(tune, "FullPipeline", _FakePipe)
    monkeypatch.setattr(tune, "load_pois", lambda *a, **k: [])
    tune.build_local_pipeline()                 # official corpus
    tune.build_local_pipeline(tune.SYNTH_CORPUS)  # synth corpus
    assert captured == [{"provider": "local", "mode": "local"},
                        {"provider": "local", "mode": "local"}]


def test_official_tune_never_reads_the_test_split():
    tune = _load_tune_module()
    from semsearch.split import load_split

    split = load_split()
    tune_ids = {q.query_id for q in tune.load_official_tune()}
    assert tune_ids == set(split["tune"])
    assert tune_ids.isdisjoint(set(split["test"]))  # NFR-6: test never enters tuning


# -------------------------------------------------- never-clobber write contract

def test_official_write_format_unchanged(tmp_path, monkeypatch):
    tune = _load_tune_module()
    out = tmp_path / "weights.json"
    monkeypatch.setattr(tune, "WEIGHTS_PATH", out)
    tune.write_official({"semantic": 0.3, "attributes": 0.05}, 0.9559)
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert list(saved) == ["weights", "tuned_ndcg5_tune", "grid", "margin"]
    assert saved["weights"] == {"semantic": 0.3, "attributes": 0.05}
    assert saved["tuned_ndcg5_tune"] == 0.9559
    assert saved["grid"] == tune.GRID
    assert saved["margin"] == tune.MARGIN


def test_candidate_write_path_is_derived_not_weights(tmp_path, monkeypatch):
    tune = _load_tune_module()
    assert tune.CANDIDATE_PATH != WEIGHTS_PATH
    assert "derived" in tune.CANDIDATE_PATH.parts  # gitignored
    guard = tmp_path / "weights.json"
    monkeypatch.setattr(tune, "WEIGHTS_PATH", guard)
    target = tmp_path / "cand.json"
    tune.write_candidate({"weights": {"semantic": 0.3}}, path=target)
    assert target.exists()
    assert not guard.exists()  # writing a candidate never touches weights.json


def test_run_extended_writes_candidate_and_never_committed_weights(monkeypatch):
    """Exercise the extended write dispatch with all heavy compute stubbed: it must call
    write_candidate and NEVER write_official."""
    tune = _load_tune_module()
    monkeypatch.setattr(tune, "build_local_pipeline", lambda *a, **k: object())
    monkeypatch.setattr(tune, "load_official_tune", lambda: [])
    monkeypatch.setattr(tune, "load_synth", lambda: [])
    monkeypatch.setattr(tune, "build_scorers", lambda pipe, qs: [])
    monkeypatch.setattr(tune, "ascend", lambda obj, seed, **k: (dict(seed), 0.0))
    monkeypatch.setattr(tune, "_eval_arm", lambda pipe, qs, w: (0.0, 0.0))
    monkeypatch.setattr(tune, "load_weights", lambda: {s: 0.1 for s in SIGNALS})
    wrote: dict = {}
    monkeypatch.setattr(tune, "write_candidate",
                        lambda payload, path=tune.CANDIDATE_PATH: wrote.setdefault("path", path))

    def _boom(*a, **k):
        raise AssertionError("run_extended must never write data/weights.json")

    monkeypatch.setattr(tune, "write_official", _boom)
    tune.run_extended()
    assert wrote["path"] == tune.CANDIDATE_PATH


# ----------------------------------------------- precompute path == full pipeline

def test_fast_rank_matches_full_pipeline(monkeypatch):
    """The extended objective re-scores WEIGHT-INDEPENDENT precomputed breakdowns; it must
    reproduce FullPipeline.rank_ids exactly (same order, filters, tie-breaks) so tuning on
    precomputed features == tuning on the live pipeline."""
    monkeypatch.setenv("SEMSEARCH_LLM_PARSE", "off")  # deterministic rules arm
    tune = _load_tune_module()
    pipe = tune.build_local_pipeline()  # official 111, local (warm dense cache)
    w1 = tune.tunable_seed()
    w2 = dict(w1)
    w2["attributes"], w2["semantic"] = 0.5, 0.3
    for q in tune.load_official_tune()[:6]:
        pc = tune.precompute(pipe, q.input_query)
        for w in (w1, w2):
            pipe.ranker.weights = w
            assert tune.fast_rank_ids(pipe, pc, w) == pipe.rank_ids(q.input_query)
