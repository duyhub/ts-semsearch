#!/usr/bin/env sh
# Container entrypoint: guarantee the local embedding model is on disk (local
# modes only), then launch the API. The download is idempotent — a warm HF cache
# volume (/models) turns it into a no-op, so restarts stay fast.
set -eu

MODEL_ID="${SEMSEARCH_MODEL_ID:-BAAI/bge-m3}"
MODE="${SEMSEARCH_MODE:-local}"

case "$MODE" in
  cloud)
    # cloud mode never loads the 2.3 GB local model (embeddings come from Bedrock).
    echo "[entrypoint] SEMSEARCH_MODE=cloud — skipping local model download."
    ;;
  *)
    echo "[entrypoint] mode=$MODE — ensuring '$MODEL_ID' is cached (HF_HOME=$HF_HOME)..."
    # SentenceTransformer(...) downloads to HF_HOME if missing, no-op if present.
    python - "$MODEL_ID" <<'PY'
import sys
from sentence_transformers import SentenceTransformer
model_id = sys.argv[1]
SentenceTransformer(model_id)
print(f"[entrypoint] model ready: {model_id}")
PY
    ;;
esac

echo "[entrypoint] starting API on 0.0.0.0:8000 (mode=$MODE)"
exec uvicorn semsearch.api:create_app --factory --host 0.0.0.0 --port 8000
