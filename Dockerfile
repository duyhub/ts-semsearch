# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Tasco Semantic Search & Ranking — production image for the Railway demo.
#
# Runs in SEMSEARCH_MODE=cloud (the code default): embeddings come from Bedrock
# (cohere -> titan, region chain) and the LLM parse from Bedrock Claude / OpenAI.
# The local bge-m3 provider is NEVER imported in cloud mode (see
# src/semsearch/embeddings.py: `from sentence_transformers import ...` is lazy,
# inside LocalEmbedder._ensure), so this image deliberately EXCLUDES the entire
# ~5 GB local-embeddings stack (torch, sentence-transformers, CUDA, etc.) — see
# the filter step below. Result: a sub-1 GB image that boots without any GPU or
# model download.
# ---------------------------------------------------------------------------

# ===== Stage 1: builder — resolve + install only the runtime (cloud) deps =====
FROM python:3.11-slim AS builder

# uv (pinned-by-tag official image) provides the frozen-lock export + fast install.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Only the lockfile + project metadata are needed to export the dependency set
# (uv reads uv.lock directly under --frozen; src/ is not required here).
COPY pyproject.toml uv.lock ./

# Export the FULL non-dev, resolved dependency set from the committed lock...
RUN uv export --frozen --no-hashes --no-dev --no-emit-project -o requirements.full.txt

# ...then filter out EXACTLY the packages that exist only to serve the local
# bge-m3 embedder (SEMSEARCH_MODE=local / local-first), which cloud mode never
# imports. Dropping these removes ~5 GB of useless weight on linux:
#   sentence-transformers  the local embedder (direct dep, cloud never loads it)
#   torch                  its ONLY consumer is sentence-transformers
#   transformers/tokenizers/safetensors/regex   pulled in ONLY by sentence-transformers
#   huggingface-hub/hf-xet                      model download plumbing (ST-only)
#   scikit-learn/scipy/joblib/threadpoolctl/narwhals  ST-only numeric stack
#   typer/shellingham/rich/markdown-it-py/mdurl/pygments  transformers' CLI stack
#   jinja2/markupsafe/mpmath/networkx/sympy/setuptools    torch's build/runtime stack
#   triton, nvidia-*, cuda-*   linux CUDA wheels (torch GPU stack)
# Every one of these is a transitive-only dep of the ST/torch subtree; nothing on
# the cloud boot path imports them (proven by the /health boot test). httpx,
# pyarrow, langfuse+otel, boto3, pandas/numpy, openpyxl, uvicorn all STAY.
RUN grep -vE '^(sentence-transformers|torch|transformers|tokenizers|safetensors|regex|huggingface-hub|hf-xet|scikit-learn|scipy|joblib|threadpoolctl|narwhals|typer|shellingham|rich|markdown-it-py|mdurl|pygments|jinja2|markupsafe|mpmath|networkx|sympy|setuptools|triton|nvidia-[a-z0-9-]+|cuda-[a-z-]+)==' \
        requirements.full.txt > requirements.txt

# Install the filtered, fully-pinned set into a self-contained venv. --no-deps is
# safe and intended: the export is already a complete flat resolution, so no
# further resolution can re-introduce the filtered-out torch subtree.
RUN uv venv /opt/venv \
    && VIRTUAL_ENV=/opt/venv uv pip install --no-deps -r requirements.txt

# ===== Stage 2: runtime — slim image with just the venv + app code =====
FROM python:3.11-slim

# Non-root runtime user.
RUN useradd --create-home --uid 1000 app

WORKDIR /app

# The prebuilt venv (no uv, no build caches, no torch).
COPY --from=builder /opt/venv /opt/venv

# Runtime-only application payload (see .dockerignore for what stays out):
#   src/                 the package (imported via PYTHONPATH=/app/src)
#   ui/                  index.html + vendored Leaflet assets served at / and /ui/*
#   data/raw/            the sponsor xlsx (load_pois reads it directly at boot)
#   data/weights.json    the tuned ranker weights (load_weights reads it at boot)
# NOTE: data/eval_split.json is intentionally NOT copied — nothing on the serving
# path imports it (it is only used by scripts/eval, never at app boot).
COPY src/ ./src/
COPY ui/ ./ui/
COPY data/raw/ ./data/raw/
COPY data/weights.json ./data/weights.json
COPY pyproject.toml ./

# data/derived/ is written at runtime (dense doc-matrix + query caches). Create it
# and hand /app to the non-root user so those writes succeed.
RUN mkdir -p data/derived && chown -R app:app /app

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    SEMSEARCH_MODE=cloud

USER app

# Railway injects $PORT; default to 8000 for local `docker run`. `exec` so uvicorn
# is PID-forwarded SIGTERM for graceful shutdown. Binds 0.0.0.0 for the platform.
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn semsearch.api:create_app --factory --host 0.0.0.0 --port ${PORT:-8000}"]
