"""Weight-tuning integrity (SPEC §6; PRD NFR-6, review C4).

`price` is a DELIBERATE FIXED preference weight — never eval-tuned (only 2/60 eval
queries express price). tune.py must therefore tune only the 8 non-price signals and
write exactly those 8 keys to data/weights.json; load_weights() re-supplies price via
its per-key default fallback. If tune.py ever coordinate-ascended `price`, a re-run
would silently write a 9-key weights.json and unfix the design.

This test imports tune.py's tunable set + write-seed and asserts price is excluded and
the written key-set matches the committed weights.json — without running the tuner.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from semsearch.rank import SIGNALS, WEIGHTS_PATH

ROOT = Path(__file__).resolve().parents[1]


def _load_tune_module():
    spec = importlib.util.spec_from_file_location("tune_script", ROOT / "scripts" / "tune.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # module body is only imports + defs (main() gated on __main__)
    return mod


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
