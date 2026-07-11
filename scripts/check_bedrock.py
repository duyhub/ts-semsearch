"""Bedrock preflight — credentials + a per-region model-access MATRIX (FR-10, Built-with-AWS).

  python scripts/check_bedrock.py

Diagnostic, NOT a gate. Walks the runtime's region-fallback chain and, for every region ×
each of the three models this entry can use, reports one of:
  - PASS              the model answered in this region
  - regional-absence  the model is not offered in this region (e.g. Titan v2 in Singapore)
  - access-block      the model exists but is gated here (e.g. an unsubmitted Anthropic
                      use-case form → ResourceNotFoundException for Claude, until approved)

The three models:
  - cohere.embed-multilingual-v3         (embeddings, bedrock-cohere)
  - amazon.titan-embed-text-v2:0         (embeddings, bedrock-titan)
  - Claude Haiku 4.5                      (LLM query parse; apac / global / plain id chain)

After the matrix it reports the OPENAI FALLBACK row — key present? (env vs .env file — the key
itself is NEVER printed), and, when the runtime would actually use it (Claude resolved nowhere),
a live ping. It closes with a `resolved:` line mirroring EXACTLY what the runtime would pin per
capability (embeddings resolve their own region; the LLM parse picks bedrock-claude first, else
openai) — because each consumer walks its own chain independently.

With NO credentials it prints a friendly "running local-only" line and exits 0 — the demo never
depends on Bedrock (CLAUDE.md hard rule), so absence is informational. When creds ARE present it
exits non-zero only if a REQUIRED capability resolves nowhere: cohere embeddings + an LLM parse
provider (claude OR openai); a model merely being absent in one region is expected, not a failure.
"""
from __future__ import annotations

import json
import os
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

from semsearch.embeddings import (  # noqa: E402  (shared region chain + model ids)
    MODEL_IDS,
    _BEDROCK_TIMEOUT,
    resolve_bedrock_regions,
)
from semsearch.llm_parse import (  # noqa: E402  (mirror the parser's chains exactly)
    CLAUDE_MODEL_ENV,
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_OPENAI_MODEL,
    FALLBACK_CLAUDE_MODELS,
    LLMParser,
    OPENAI_MODEL_ENV,
    _openai_payload,
    _redact,
    discover_openai_key,
)

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


# Claude on Bedrock: the SAME id chain the parser walks (env override first, used verbatim).
_env_claude = os.environ.get(CLAUDE_MODEL_ENV)
CLAUDE_MODEL_IDS = (_env_claude,) if _env_claude else (DEFAULT_CLAUDE_MODEL, *FALLBACK_CLAUDE_MODELS)
COHERE_MODEL_ID = MODEL_IDS["bedrock-cohere"]
TITAN_MODEL_ID = MODEL_IDS["bedrock-titan"]

MODEL_ACCESS_URL = "https://console.aws.amazon.com/bedrock/home#/modelaccess"


def _config():
    from botocore.config import Config

    return Config(**_BEDROCK_TIMEOUT)


def classify_model_failure(exc: Exception) -> str:
    """Map a per-model failure to the matrix vocabulary: an invalid/unknown model id means the
    model is not offered in this region (regional-absence); an access/enablement/use-case error
    means it exists but is gated here (access-block); anything else is a raw error."""
    text = str(exc).lower()
    if "identifier is invalid" in text or "invalid model" in text or "does not exist" in text:
        return "regional-absence"
    if ("resourcenotfound" in text or "accessdenied" in text or "not authorized" in text
            or "access" in text or "use case" in text):
        return "access-block"
    return f"error ({type(exc).__name__})"


def _probe_embed(client, model_id: str, body: str) -> tuple[bool, str]:
    try:
        resp = client.invoke_model(modelId=model_id, body=body)
        resp["body"].read()  # drain the streaming body
        return True, "PASS"
    except Exception as exc:  # noqa: BLE001 - classify, never crash the matrix
        return False, classify_model_failure(exc)


def _probe_claude(client) -> tuple[bool, str, str | None]:
    """Walk the Claude id chain in this region; PASS with the first id that answers. On total
    failure report the MOST INFORMATIVE classification across the chain — an access-block on the
    global. profile (the actionable "submit the Anthropic use-case form" gate) outranks the
    plain id's on-demand-throughput ValidationException, which would otherwise mask it."""
    seen: list[str] = []
    for model_id in CLAUDE_MODEL_IDS:
        try:
            client.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": "ping"}]}],
                inferenceConfig={"maxTokens": 8},
            )
            return True, "PASS", model_id
        except Exception as exc:  # noqa: BLE001 - collect the classification, try the next id
            seen.append(classify_model_failure(exc))
    for tier in ("access-block", "regional-absence"):  # priority: actionable first
        if tier in seen:
            return False, tier, None
    return False, seen[-1] if seen else "error", None


def _print_row(label: str, ok: bool, detail: str) -> None:
    print(f"    {'PASS' if ok else 'FAIL'}  {label:<28} {detail}")


def main() -> int:
    import boto3
    from botocore.exceptions import BotoCoreError

    regions = resolve_bedrock_regions()
    # Titan has its own default chain (not offered in ap-southeast-1 — regional absence,
    # measured). The matrix still PROBES titan in every displayed region (informative), but
    # the resolved line only counts regions the runtime would actually walk for titan.
    titan_chain = resolve_bedrock_regions(TITAN_MODEL_ID)
    cfg = _config()

    print("Bedrock preflight (FR-10, Built-with-AWS) — per-region model-access matrix")
    print(f"  region chain: {', '.join(regions)}")
    if titan_chain != regions:
        print(f"  titan-v2 chain: {', '.join(titan_chain)} (per-model default — titan-v2 is "
              "not offered in the skipped region(s))")
    print("    (SEMSEARCH_BEDROCK_REGION pins one region; SEMSEARCH_BEDROCK_REGIONS replaces the")
    print("     whole chain; else AWS_REGION / AWS_DEFAULT_REGION; else the venue-proximity default)")
    # Activation (FR-4 LLM parse + Langfuse tracing) — both OFF by default (NFR-5).
    print("  activate LLM query parse: SEMSEARCH_LLM_PARSE=on|bedrock (full chain) or =openai "
          "(pin OpenAI, skip Bedrock); =off disables; cloud mode defaults it on")
    print("    (model overrides: SEMSEARCH_BEDROCK_CLAUDE / SEMSEARCH_OPENAI_MODEL)")
    print("  activate Langfuse tracing: LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY "
          "(unset = silent no-op)")

    # 1) credentials via STS get-caller-identity (checked in the first region of the chain)
    sts_region = regions[0]
    try:
        ident = boto3.client("sts", region_name=sts_region, config=cfg).get_caller_identity()
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
        print(f"        Check network/VPN and the chain ({', '.join(regions)}), then re-run.")
        return 0
    print(f"  PASS  credentials              account={ident['Account']} arn={ident['Arn']}")

    # 2) per-region × per-model matrix; record the FIRST region each capability resolves to
    #    (exactly what the runtime pins, since every consumer walks the chain independently).
    cohere_body = json.dumps({"texts": ["ping"], "input_type": "search_query", "truncate": "END"})
    titan_body = json.dumps({"inputText": "ping", "dimensions": 1024, "normalize": True})
    resolved_cohere: str | None = None
    resolved_titan: str | None = None
    resolved_claude: tuple[str, str] | None = None

    for region in regions:
        print(f"\n  region {region}:")
        runtime = boto3.client("bedrock-runtime", region_name=region, config=cfg)

        ok, detail = _probe_embed(runtime, COHERE_MODEL_ID, cohere_body)
        _print_row("cohere.embed-multilingual", ok, detail)
        if ok and resolved_cohere is None:
            resolved_cohere = region

        ok, detail = _probe_embed(runtime, TITAN_MODEL_ID, titan_body)
        _print_row("amazon.titan-embed-v2", ok, detail)
        if ok and resolved_titan is None and region in titan_chain:
            resolved_titan = region

        ok, detail, claude_id = _probe_claude(runtime)
        _print_row("claude-haiku-4-5 (converse)", ok, detail)
        if ok and resolved_claude is None:
            resolved_claude = (region, claude_id or "")

    # 3) OpenAI fallback row — the runtime contacts OpenAI ONLY when Claude resolves nowhere.
    #    HARD RULE: only the key's SOURCE is printed; the key itself never appears anywhere.
    openai_model = os.environ.get(OPENAI_MODEL_ENV) or DEFAULT_OPENAI_MODEL
    resolved_openai: str | None = None
    found = discover_openai_key()
    print("\n  openai fallback (chat completions):")
    if found is None:
        print("    FAIL  openai key                   not found "
              "(env OPENAI_API_KEY or .env/OPENAI-API-key.txt)")
    else:
        key, source = found
        print(f"    PASS  openai key                   present ({source})")
        if resolved_claude is not None:
            print("    SKIP  openai ping                  not needed — claude resolved in "
                  f"{resolved_claude[0]}")
        else:
            client = LLMParser._make_openai_client()
            try:
                LLMParser._openai_post(
                    client, key,
                    _openai_payload(openai_model,
                                    [{"role": "user", "content": "ping"}], 1),
                )
                _print_row(f"openai ping ({openai_model})", True, "PASS")
                resolved_openai = openai_model
            except Exception as exc:  # noqa: BLE001 - report (redacted), never crash
                _print_row(f"openai ping ({openai_model})",
                           False, _redact(f"{type(exc).__name__}: {exc}", key))
            finally:
                client.close()

    # 4) resolved summary — mirrors EXACTLY what the runtime would pin per capability
    embed_region = resolved_cohere or "NONE"
    claude_str = f"{resolved_claude[0]}+{resolved_claude[1]}" if resolved_claude else "NONE"
    if resolved_claude:
        llm_str = f"bedrock:{resolved_claude[0]}+{resolved_claude[1]}"
    elif resolved_openai:
        llm_str = f"openai+{resolved_openai}"
    else:
        llm_str = "NONE"
    print(f"\n  resolved: embeddings→{embed_region}, claude→{claude_str}, llm-parse→{llm_str}")
    print(f"    bedrock-cohere → {resolved_cohere or 'NONE'}   "
          f"bedrock-titan → {resolved_titan or 'NONE'}")

    # 5) deployment modes — the active switch + what each mode WOULD resolve to right now
    from semsearch.config import MODE_ENV, resolve_mode  # noqa: PLC0415 (post-matrix import)

    active = resolve_mode()
    mode_src = "env" if os.environ.get(MODE_ENV) else "src/semsearch/config.py DEFAULT_MODE"
    if resolved_cohere:
        cloud_embed = f"bedrock-cohere@{resolved_cohere}"
    elif resolved_titan:
        cloud_embed = f"bedrock-titan@{resolved_titan}"
    else:
        cloud_embed = "bm25-only"
    llm_pick = llm_str if llm_str != "NONE" else "rules-only"
    print(f"\n  deployment modes (active: {active}, from {mode_src}; "
          f"switch: {MODE_ENV}=<mode> or edit config.py):")
    print("    local       → embeddings=local (bge-m3); llm-parse=rules-only "
          f"(SEMSEARCH_LLM_PARSE=bedrock would give {llm_pick})")
    print(f"    local-first → embeddings=local if bge-m3 loads, else {cloud_embed}; "
          "llm-parse=rules-only by default")
    print(f"    cloud       → embeddings={cloud_embed}; llm-parse={llm_pick} "
          "(ON by default; SEMSEARCH_LLM_PARSE=off disables)")

    # The demo path needs cohere embeddings + SOME LLM parse provider (claude OR openai);
    # a model being absent in one region (e.g. Titan in Singapore) is expected, not a failure.
    if resolved_cohere and (resolved_claude or resolved_openai):
        print("\n  Ready: bedrock-cohere embeddings + an LLM parse provider "
              f"({'claude' if resolved_claude else 'openai'}) both resolve.")
        return 0
    print("\n  A required capability resolves to NO provider. The demo still runs local-only;")
    print(f"  enable the missing models at {MODEL_ACCESS_URL}, add a region to the chain, or")
    print("  provide an OpenAI key (env OPENAI_API_KEY or .env/OPENAI-API-key.txt).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
