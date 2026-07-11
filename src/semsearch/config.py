"""Deployment-mode switch for model usage (embeddings + LLM parse defaults).

This is the ONE place to flip how the engine sources its models. Edit DEFAULT_MODE below,
or set the env var SEMSEARCH_MODE (env wins), and restart the server.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

MODE_ENV = "SEMSEARCH_MODE"
VALID_MODES = ("local", "local-first", "cloud")

# ============================================================================ #
#  DEPLOYMENT MODE — THE MANUAL SWITCH.                                        #
#                                                                              #
#  Edit this line (or set SEMSEARCH_MODE=<mode>) to switch the engine:         #
#                                                                              #
#  "local"        (default) Today's posture. Embeddings run on the local       #
#                 bge-m3 model; the cloud is never contacted for embeddings;   #
#                 a broken local setup fails LOUDLY at construction (it is a   #
#                 setup bug, not a runtime condition). LLM query parse is OFF  #
#                 by default (deterministic /v1/search, NFR-5).                #
#                                                                              #
#  "local-first"  Prefer local, degrade to the cloud. bge-m3 is probed at      #
#                 construction; if it fails (model missing on this host), a    #
#                 loud warning is logged and the cloud chain takes over        #
#                 (bedrock-cohere then bedrock-titan, each walking the region  #
#                 fallback chain). Everything failing lands on the BM25-only   #
#                 floor. LLM parse stays OFF by default.                       #
#                                                                              #
#  "cloud"        Remote hosting without the 2.3 GB local model: local is      #
#                 NEVER attempted (sentence_transformers is never imported).   #
#                 Embeddings walk bedrock-cohere -> bedrock-titan across the   #
#                 region chain; all failing -> BM25-only floor with a loud     #
#                 warning. LLM query parse is ON by default (remote hosting    #
#                 implies network) — SEMSEARCH_LLM_PARSE=off forces it off.    #
#                                                                              #
#  Precedence: env SEMSEARCH_MODE > this constant. An explicit `provider=`     #
#  passed to FullPipeline is an EXPERT override that skips mode resolution     #
#  for embeddings entirely (eval/gates use it to stay pinned to local).        #
# ============================================================================ #
DEFAULT_MODE = "local"


def resolve_mode() -> str:
    """The active deployment mode: env SEMSEARCH_MODE wins over DEFAULT_MODE; an unknown
    value logs a warning and falls back to 'local' (the safe, offline posture)."""
    mode = os.environ.get(MODE_ENV) or DEFAULT_MODE
    if mode not in VALID_MODES:
        logger.warning(
            "unknown %s value %r (valid: %s); using 'local'.",
            MODE_ENV, mode, ", ".join(VALID_MODES),
        )
        return "local"
    return mode
