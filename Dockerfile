# syntax=docker/dockerfile:1

# ─────────────────────────────────────────────────────────────────────────────
# Tasco Semantic Search & Ranking — VPS image (CPU-only, local bge-m3).
#
# The server builds its dense index in-memory at startup from data/raw/*.xlsx
# and the bge-m3 model, so the image carries the raw dataset + tuned weights +
# demo UI. The ~2.3 GB model is NOT baked in — it downloads once into a
# persistent volume on first boot (see deploy/entrypoint.sh + docker-compose.yml).
# ─────────────────────────────────────────────────────────────────────────────
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# UV_TORCH_BACKEND=cpu pins the CPU-only torch wheels (no CUDA) — the VPS has no
# GPU and this shaves ~1.5 GB off the image + skips the nvidia-cuda-* downloads.
# If `uv sync` ever errors on a torch/lock conflict, delete this one line: the
# build falls back to the default torch (larger, still CPU-functional).
ENV UV_TORCH_BACKEND=cpu \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    HF_HUB_DISABLE_TELEMETRY=1 \
    SEMSEARCH_MODE=local

WORKDIR /app

# 1) Dependencies first — this layer is cached until pyproject.toml/uv.lock change.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-install-project

# 2) App source + the exact runtime inputs the server reads at startup.
#    (data/derived is intentionally excluded — it is rebuilt in-memory / cached
#     to a volume at runtime; see .dockerignore.)
COPY src ./src
COPY scripts ./scripts
COPY ui ./ui
COPY data/raw ./data/raw
COPY data/weights.json ./data/weights.json
COPY data/eval_split.json ./data/eval_split.json
COPY openapi.json ./openapi.json

# 3) Install the project itself into the venv now that src/ is present.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev

COPY deploy/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
