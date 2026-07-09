"""Robustness sweep (SPEC §11; PRD NFR-2; gate G5).

Every one of the 60 eval queries PLUS the adversarial set, through BOTH
/v1/search and /v1/semantic-search, must satisfy the G5 rule:

  PASS  = HTTP 200 with >=1 result   OR   HTTP 400 invalid_request (contract-valid)
  FAIL  = any 5xx / unhandled exception / 200 with 0 results

Writes reports/robustness.json and exits non-zero on any failure.

  python scripts/robustness.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from semsearch.adversarial import ADVERSARIAL  # noqa: E402
from semsearch.api import create_app  # noqa: E402
from semsearch.data import load_eval  # noqa: E402

REPORTS = Path("reports")
ENDPOINTS = ("/v1/search", "/v1/semantic-search")


def _ok(status: int, body: dict) -> bool:
    if status == 200:
        return len(body.get("results", [])) >= 1
    if status == 400:
        return body.get("error", {}).get("code") == "invalid_request"
    return False


def main() -> None:
    client = TestClient(create_app(prewarm=False))
    cases: list[tuple[str, str]] = [(f"eval:{q.query_id}", q.input_query) for q in load_eval()]
    cases += [(f"adv:{name}", text) for name, text in ADVERSARIAL]

    failures: list[dict] = []
    checked = 0
    for endpoint in ENDPOINTS:
        for label, q in cases:
            checked += 1
            try:
                r = client.get(endpoint, params={"q": q})
                body = r.json()
                if not _ok(r.status_code, body):
                    failures.append({"endpoint": endpoint, "case": label, "status": r.status_code,
                                     "n_results": len(body.get("results", []))})
            except Exception as exc:  # noqa: BLE001 - an exception here is itself a G5 failure
                failures.append({"endpoint": endpoint, "case": label, "exception": repr(exc)})

    # missing q (no param at all) must be a contract 400
    for endpoint in ENDPOINTS:
        checked += 1
        r = client.get(endpoint)
        if not (r.status_code == 400 and r.json().get("error", {}).get("code") == "invalid_request"):
            failures.append({"endpoint": endpoint, "case": "missing_q", "status": r.status_code})

    report = {
        "checked": checked,
        "eval_queries": len(load_eval()),
        "adversarial": len(ADVERSARIAL),
        "failures": failures,
        "g5_passed": not failures,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "robustness.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nG5 {'PASSED' if not failures else 'FAILED'}: {checked} checks, {len(failures)} failures")
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
