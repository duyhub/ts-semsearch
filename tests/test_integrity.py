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


# --------------------------------------------------------------------------- #
# Deployment modes must NEVER leak into the measured paths (eval integrity)   #
# --------------------------------------------------------------------------- #
def test_eval_engines_immune_to_deployment_mode(monkeypatch, tmp_path):
    """REGRESSION GUARD: with SEMSEARCH_MODE=cloud exported AND a would-succeed cloud path
    (working fake Bedrock client + a discoverable OpenAI key), the eval engine factory must
    still build a fully-local pipeline — local embeddings, NO LLM parser. If this fails, the
    reported metrics (G3, weights.json) silently depend on network calls and the 'honest
    metrics' story is compromised. Treat a failure as a release blocker."""
    import semsearch.llm_parse as L
    import semsearch.pipeline as P
    from semsearch.data import load_pois
    from semsearch.engines import make_full_ranker

    monkeypatch.setenv("SEMSEARCH_MODE", "cloud")
    monkeypatch.delenv("SEMSEARCH_LLM_PARSE", raising=False)
    # a would-SUCCEED cloud path: if mode leaked, the LLM parser would pin bedrock here
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-FAKE-integrity-guard")
    monkeypatch.setattr(L, "_REPO_ROOT", tmp_path / "no-repo")

    class WorkingConverse:
        def converse(self, **_kw):
            return {"output": {"message": {"content": [{"text": "{}"}]}}}

        def invoke_model(self, **_kw):
            raise AssertionError("eval embeddings must stay local — bedrock was probed")

    monkeypatch.setattr("boto3.client", lambda *a, **k: WorkingConverse())

    captured: dict = {}
    real_pipeline = P.FullPipeline

    class Capture(real_pipeline):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            captured["pipe"] = self

    monkeypatch.setattr(P, "FullPipeline", Capture)

    rank = make_full_ranker(load_pois())  # exactly what run_eval/report_metrics build
    pipe = captured["pipe"]
    assert pipe._llm_parser is None, "mode leaked: eval queries would be LLM-enriched"
    assert pipe.dense is not None and pipe.dense.emb.provider == "local"
    assert pipe.mode == "local"  # measurement factories are local-by-definition
    assert rank(type("Q", (), {"input_query": "quán cà phê yên tĩnh"})())  # still ranks
