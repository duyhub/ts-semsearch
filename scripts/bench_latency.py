"""Latency benchmark (SPEC §9, §11; PRD NFR-1; gate G4).

Reports COLD p95 (fresh query embedding — bge-m3 forward pass) and WARM p95
(query embedding cached), separately and honestly (eng-review P1). Gate G4 is
warm p95 < 200 ms over the 60 eval queries.

  python scripts/bench_latency.py
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semsearch import embeddings as E  # noqa: E402
from semsearch.data import load_eval, load_pois  # noqa: E402
from semsearch.pipeline import FullPipeline  # noqa: E402

REPORTS = Path("reports")
G4_WARM_P95_MS = 200.0


def _p(values: list[float], pct: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(round(pct / 100 * (len(s) - 1))))]


def main() -> None:
    pois = load_pois()
    queries = [q.input_query for q in load_eval()]
    # MEASUREMENT: pinned local (provider AND mode) — G4 latency is a gate on the local demo config.
    pipe = FullPipeline(pois, provider="local", mode="local")  # model loads here (boot cost, excluded from per-query timing)

    # COLD: clear the query-embed cache so each query pays a fresh bge-m3 forward
    shutil.rmtree(E.QCACHE_DIR, ignore_errors=True)
    cold = []
    for q in queries:
        t0 = time.perf_counter()
        pipe.rank_ids(q)
        cold.append((time.perf_counter() - t0) * 1000)

    # WARM: same queries, embeddings now cached
    warm = []
    for q in queries:
        t0 = time.perf_counter()
        pipe.rank_ids(q)
        warm.append((time.perf_counter() - t0) * 1000)

    report = {
        "n": len(queries),
        "cold_ms": {"p50": round(_p(cold, 50), 1), "p95": round(_p(cold, 95), 1)},
        "warm_ms": {"p50": round(_p(warm, 50), 1), "p95": round(_p(warm, 95), 1)},
        "gate_g4_warm_p95_ms": G4_WARM_P95_MS,
    }
    passed = report["warm_ms"]["p95"] < G4_WARM_P95_MS
    report["g4_passed"] = passed

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "latency.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nG4 {'PASSED' if passed else 'FAILED'}: warm p95 = {report['warm_ms']['p95']} ms "
          f"(threshold {G4_WARM_P95_MS} ms); cold p95 = {report['cold_ms']['p95']} ms")


if __name__ == "__main__":
    main()
