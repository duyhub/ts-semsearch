#!/usr/bin/env bash
# Pre-download the bge-m3 embedding model into the hf_models volume BEFORE the
# first `docker compose up`. Optional — the container downloads it automatically
# on boot too — but running this first front-loads the ~2.3 GB pull so you can
# watch progress and keep the first real start fast.
#
# Usage (from the repo root on the VPS):
#   ./deploy/download-model.sh
set -Eeuo pipefail

cd "$(dirname "$0")/.."

MODEL_ID="${SEMSEARCH_MODEL_ID:-BAAI/bge-m3}"

echo "==> Building the image (first build installs torch + sentence-transformers)"
docker compose build semsearch

echo "==> Downloading '$MODEL_ID' into the hf_models volume"
# `run --rm` mounts the SAME hf_models volume the service uses, so the cache
# persists for the long-running container.
docker compose run --rm \
  -e SEMSEARCH_MODE=local \
  -e "SEMSEARCH_MODEL_ID=$MODEL_ID" \
  --entrypoint python \
  semsearch \
  -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('$MODEL_ID'); print('OK: $MODEL_ID cached to /models')"

echo "==> Done. Start the stack with:  docker compose up -d"
