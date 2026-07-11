# Deploying to a VPS behind Caddy

This stack runs the FastAPI search engine in one container and lets your existing
Caddy reverse proxy terminate TLS and route a subdomain to it. Default mode is
fully local (`bge-m3`), so once the model is downloaded the demo has no hard
network dependency.

## What's here

| File | Purpose |
|---|---|
| `../Dockerfile` | CPU-only Python 3.11 image (uv). Carries src + raw dataset + tuned weights + UI. Model is **not** baked in. |
| `../docker-compose.yml` | The `semsearch` service, joined to the external `caddy` network. Named volumes for the model cache and query cache. |
| `../.dockerignore` | Keeps the build context lean; excludes `data/derived`, `.venv`, tests, docs. |
| `entrypoint.sh` | Ensures `bge-m3` is on disk (local modes), then starts uvicorn. |
| `download-model.sh` | Optional: front-load the ~2.3 GB model pull into the volume before first start. |
| `Caddyfile.snippet` | vhost block to paste into your Caddyfile. |
| `.env.example` | Copy to repo-root `.env` for cloud/Bedrock or Langfuse. Not needed for plain local. |

## Prerequisites on the VPS

- Docker Engine + Compose plugin (your `myvps/scripts/setup-ubuntu-vps.sh` installs these).
- The **Caddy stack running**, which creates the external Docker network named
  `caddy`. If Caddy isn't up yet: `docker network create caddy`.
- A **DNS record** for your chosen subdomain (e.g. `semsearch.ducanphatcons.com`)
  pointing at the VPS IP, so Caddy can auto-provision a TLS cert.
- **RAM: 4 GB minimum, 8 GB comfortable.** `bge-m3` resides at ~2.5 GB while serving.
  Cloud mode (below) removes this requirement entirely.

## Deploy (local mode — the default)

```bash
# 1. Get the repo onto the VPS (it needs the source + raw dataset to build).
git clone <your-repo-url> ts-semsearch && cd ts-semsearch

# 2. (Optional) front-load the model download so first start is fast.
./deploy/download-model.sh

# 3. Build + start.
docker compose up -d --build

# 4. Wire up Caddy: append the vhost to your Caddyfile and reload.
cat deploy/Caddyfile.snippet >> /Users/…/myvps/caddy/Caddyfile   # edit the hostname first
docker exec caddy caddy reload --config /etc/caddy/Caddyfile

# 5. Verify.
docker compose logs -f semsearch          # watch model download + boot
curl -s http://localhost:8000/health       # if you temporarily publish a port, else:
docker exec semsearch python -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8000/health').read().decode())"
# then open https://semsearch.ducanphatcons.com/
```

`/health` returns `{"status":"ok","pois":111,"mode":"local","embeddings":"local","llm_parse":"rules-only"}`
once the model is loaded.

### First boot is slow, later boots are fast

On the very first start the entrypoint downloads `bge-m3` (~2.3 GB) into the
`hf_models` volume — the healthcheck's `start_period` is 300 s to cover this.
The model and the query-embedding cache persist in named volumes, so restarts
and image rebuilds skip the download and stay snappy.

## Optional: cloud mode (AWS Bedrock, no local model)

Removes the 2.3 GB model and its RAM cost — embeddings come from Bedrock, so a
small VPS works. Needs AWS credentials and network.

```bash
cp deploy/.env.example .env
# edit .env: set SEMSEARCH_MODE=cloud and your AWS_* / region values
# then uncomment the matching env lines in docker-compose.yml so they pass through
docker compose up -d --build
```

Every Bedrock call has a timeout and degrades to the BM25-only floor rather than
hanging (see `src/semsearch/embeddings.py`). Diagnose credentials/regions with:
`docker exec semsearch python scripts/check_bedrock.py`.

## Operating it

```bash
docker compose ps                 # status + health
docker compose logs -f semsearch  # logs
docker compose up -d --build      # redeploy after a git pull
docker compose down               # stop (volumes, incl. the model, are kept)
docker compose down -v            # stop AND wipe volumes (re-downloads the model)
```

## How Caddy reaches the container

Both containers share the external `caddy` network. Caddy resolves the service
by its container name, so `reverse_proxy semsearch:8000` in the Caddyfile hits
this stack directly — no published host port required. Only Caddy binds 80/443
on the host; `semsearch` stays internal.
