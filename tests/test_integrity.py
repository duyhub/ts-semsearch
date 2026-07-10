"""Evaluation-integrity guard (PRD NFR-6, eng-review T1).

The pitch rests on: no eval query is ever fitted to POI ids, and the test split
never leaks into code. This makes that a check, not a promise. If it fails, the
'honest metrics' story is compromised — treat a failure as a release blocker.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from semsearch.data import load_eval
from semsearch.split import make_split

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "semsearch"

# Everything that ships and could leak eval text: the library, the scripts (incl. the
# sample-query showcase), the demo UI (its CHIPS array) — and the tests themselves
# (a verbatim eval query in a test means behavior was fitted to eval text; reword the
# test's query, never weaken this guard). Illustrative demo queries are allowed — but
# must NOT be verbatim eval-query text (reword ours if they collide).
SCANNED = (
    list(SRC.rglob("*.py"))
    + list((ROOT / "scripts").glob("*.py"))
    + [ROOT / "ui" / "index.html"]
    + list((ROOT / "tests").glob("*.py"))
)


@pytest.fixture(scope="module")
def queries():
    return load_eval()


@pytest.fixture(scope="module")
def source_text():
    return "\n".join(p.read_text(encoding="utf-8") for p in SCANNED if p.exists())


def test_no_query_text_hardcoded_in_src(queries, source_text):
    """No eval query's raw text is embedded in shipped code/UI (would signal query-specific
    code, or an illustrative demo query accidentally reusing eval text verbatim)."""
    offenders = [q.query_id for q in queries if q.input_query and q.input_query in source_text]
    assert not offenders, f"eval query text hardcoded in src/scripts/ui: {offenders}"


def test_no_expected_id_mapping_hardcoded_in_src(queries, source_text):
    """No expected_top_poi_ids list is embedded in source (the core NFR-6 violation)."""
    offenders = []
    for q in queries:
        if len(q.expected_ids) >= 2:
            joined = ";".join(q.expected_ids)
            if joined in source_text:
                offenders.append(q.query_id)
    assert not offenders, f"expected poi-id mapping hardcoded in src/scripts/ui: {offenders}"


def test_tune_test_never_overlap(queries):
    split = make_split(queries)
    assert set(split["tune"]).isdisjoint(split["test"])
