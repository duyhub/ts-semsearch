"""Bedrock preflight — credentials, region, and per-model access (FR-10, Built-with-AWS).

  python scripts/check_bedrock.py

Diagnostic, NOT a gate. Reports PASS/FAIL for AWS credentials and access to the
three models this entry can use on Bedrock:
  - cohere.embed-multilingual-v3         (embeddings, bedrock-cohere)
  - amazon.titan-embed-text-v2:0         (embeddings, bedrock-titan)
  - Claude Haiku 4.5                      (LLM query parse; apac inference profile,
                                           with a global-region fallback id)

With NO credentials it prints a friendly "running local-only" line and exits 0 —
the demo never depends on Bedrock (CLAUDE.md hard rule), so absence is informational.
When creds ARE present it exits non-zero if any model is unreachable, printing the
exact fix (aws configure / the Bedrock model-access console page), so it is usable
as a real setup check for the AWS bonus track.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from botocore.exceptions import (  # noqa: E402
    ClientError,
    NoCredentialsError,
    PartialCredentialsError,
    SSOTokenLoadError,
    TokenRetrievalError,
    UnauthorizedSSOTokenError,
)

from semsearch.embeddings import BedrockEmbedder  # noqa: E402  (shared region resolution)

# Truly-absent-credentials states (incl. expired/broken SSO tokens) — these mean
# "nothing usable configured", NOT "the network is down".
_NO_CREDS_ERRORS = (
    NoCredentialsError,
    PartialCredentialsError,
    SSOTokenLoadError,
    TokenRetrievalError,
    UnauthorizedSSOTokenError,
)


def classify_sts_failure(exc: Exception) -> str:
    """Classify an STS get-caller-identity failure; drives message + exit code.

    'no-credentials' : nothing configured -> friendly local-only info, exit 0.
    'rejected'       : credentials exist but STS refused them (ClientError) ->
                       actionable FAIL, exit 1.
    'network'        : credentials may exist but the endpoint/region is unreachable
                       (EndpointConnectionError, ReadTimeoutError, other
                       BotoCoreError) -> informational, exit 0.
    """
    if isinstance(exc, _NO_CREDS_ERRORS):
        return "no-credentials"
    if isinstance(exc, ClientError):
        return "rejected"
    return "network"

# Claude on Bedrock: APAC cross-region inference profile first, then the global id.
CLAUDE_MODEL_IDS = (
    "apac.anthropic.claude-haiku-4-5-20251001-v1:0",
    "anthropic.claude-haiku-4-5-20251001-v1:0",
)
COHERE_MODEL_ID = "cohere.embed-multilingual-v3"
TITAN_MODEL_ID = "amazon.titan-embed-text-v2:0"

MODEL_ACCESS_URL = "https://console.aws.amazon.com/bedrock/home#/modelaccess"


def _config():
    from botocore.config import Config

    from semsearch.embeddings import _BEDROCK_TIMEOUT

    return Config(**_BEDROCK_TIMEOUT)


def _check_embed_cohere(client) -> tuple[bool, str]:
    body = json.dumps({"texts": ["ping"], "input_type": "search_query", "truncate": "END"})
    resp = client.invoke_model(modelId=COHERE_MODEL_ID, body=body)
    n = len(json.loads(resp["body"].read())["embeddings"])
    return True, f"{n} embedding(s)"


def _check_embed_titan(client) -> tuple[bool, str]:
    body = json.dumps({"inputText": "ping", "dimensions": 1024, "normalize": True})
    resp = client.invoke_model(modelId=TITAN_MODEL_ID, body=body)
    dim = len(json.loads(resp["body"].read())["embedding"])
    return True, f"dim={dim}"


def _check_claude(client) -> tuple[bool, str]:
    """Tiny converse() against the APAC profile, then the global id (max_tokens 8)."""
    last_exc: Exception | None = None
    for model_id in CLAUDE_MODEL_IDS:
        try:
            resp = client.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": "ping"}]}],
                inferenceConfig={"maxTokens": 8},
            )
            stop = resp.get("stopReason", "ok")
            return True, f"{model_id} ({stop})"
        except Exception as exc:  # noqa: BLE001 - try the next id
            last_exc = exc
    raise last_exc  # type: ignore[misc]


def _run_check(label: str, model_hint: str, fn, client) -> bool:
    try:
        ok, detail = fn(client)
        print(f"  PASS  {label:<26} {detail}")
        return ok
    except Exception as exc:  # noqa: BLE001 - report, don't crash
        print(f"  FAIL  {label:<26} {type(exc).__name__}: {exc}")
        print(f"        fix: enable access to {model_hint} at {MODEL_ACCESS_URL}")
        return False


def main() -> int:
    import boto3
    from botocore.exceptions import BotoCoreError

    region = BedrockEmbedder._region()
    cfg = _config()

    print("Bedrock preflight (FR-10, Built-with-AWS)")
    print(f"  region: {region}  (override with SEMSEARCH_BEDROCK_REGION)")
    # Activation (FR-4 LLM parse + Langfuse tracing) — both OFF by default (NFR-5).
    print("  activate LLM query parse: SEMSEARCH_LLM_PARSE=bedrock "
          "(model override: SEMSEARCH_BEDROCK_CLAUDE)")
    print("  activate Langfuse tracing: LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY "
          "(unset = silent no-op)")

    # 1) credentials via STS get-caller-identity
    try:
        ident = boto3.client("sts", region_name=region, config=cfg).get_caller_identity()
    except (BotoCoreError, ClientError) as exc:
        kind = classify_sts_failure(exc)
        if kind == "rejected":
            print(f"  FAIL  credentials              {exc}")
            print("        fix: run `aws configure` (or set AWS_PROFILE / AWS_* env vars)")
            return 1
        if kind == "no-credentials":
            # No usable credentials is NOT a failure here — the demo runs local-only.
            print("  INFO  no AWS credentials — running local-only (bge-m3).")
            print("        Bedrock is optional; the demo never depends on it "
                  "(CLAUDE.md hard rule).")
            print("        To enable it: run `aws configure`, then re-run this check.")
            return 0
        # 'network': credentials may exist, but we can't reach AWS to find out.
        print(f"  INFO  AWS endpoint unreachable ({type(exc).__name__}: {exc}).")
        print("        Credentials may exist but the endpoint/region is unreachable — "
              "running local-only.")
        print(f"        Check network/VPN and the region ({region}), then re-run this check.")
        return 0
    print(f"  PASS  credentials              account={ident['Account']} arn={ident['Arn']}")

    # 2) per-model access
    runtime = boto3.client("bedrock-runtime", region_name=region, config=cfg)
    results = [
        _run_check("cohere.embed-multilingual", COHERE_MODEL_ID, _check_embed_cohere, runtime),
        _run_check("amazon.titan-embed-v2", TITAN_MODEL_ID, _check_embed_titan, runtime),
        _run_check("claude-haiku-4-5 (converse)", "Anthropic Claude Haiku 4.5", _check_claude, runtime),
    ]

    if all(results):
        print("\n  All checks passed — Bedrock is ready (bedrock-cohere / bedrock-titan / Claude).")
        return 0
    print("\n  Some models are unreachable. The demo still runs local-only; enable the models")
    print(f"  above at {MODEL_ACCESS_URL} to use the Bedrock path.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
