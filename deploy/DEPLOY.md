# Free deployment guide

Zero-cost hosting for the rebuilt stack:

| Piece | Host | Cost |
| --- | --- | --- |
| Backend (FastAPI) | Hugging Face Spaces (Docker) | Free — 2 vCPU / 16 GB RAM, sleeps after ~48h idle |
| Frontend (Next.js) | Vercel Hobby | Free |
| Daily snapshot cron | GitHub Actions | Free |

The frontend calls the backend server-side through the Next.js rewrite
(`/api/:path*` → `BACKEND_URL`), so the browser only ever talks to the Vercel
origin. That keeps CORS a non-issue for normal use.

---

## 1. Backend on Hugging Face Spaces

The root `Dockerfile` builds the backend and listens on `${PORT:-7860}`.
`.dockerignore` keeps the image to `quant_analysis` + `backend` + `config.yaml`
+ committed `data/snapshots`.

### One-time setup

1. Create a **Docker** Space: https://huggingface.co/new-space (SDK = Docker).
2. Push this repo to the Space. Two options:
   - **Automatic (recommended):** add GitHub repo secrets `HF_TOKEN` (a write
     token from https://huggingface.co/settings/tokens) and `HF_SPACE`
     (`username/space-name`). The `deploy-hf.yml` workflow then pushes on every
     commit to `master`.
   - **Manual:** copy `deploy/huggingface/README.md` to the repo root as
     `README.md`, then
     `git push https://user:HF_TOKEN@huggingface.co/spaces/username/space-name HEAD:main`.
3. In **Space → Settings → Variables and secrets** add:
   - `TRADIER_TOKEN` — enables live option data (without it the API returns
     empty/degraded responses instead of erroring).
   - `OPENAI_API_KEY` — for AI narratives once that router lands.
   - `CORS_ORIGINS` — optional, JSON array e.g. `["https://your-app.vercel.app"]`.

Verify once the build turns green:
`https://<space-subdomain>.hf.space/api/health` → `{"status":"ok", ...}`.

---

## 2. Frontend on Vercel

1. Import the repo at https://vercel.com/new.
2. Set **Root Directory** to `frontend`.
3. Add an environment variable:
   - `BACKEND_URL` = `https://<space-subdomain>.hf.space`
4. Deploy. Vercel auto-detects Next.js; the rewrite proxies `/api/*` to the Space.

> Note: Vercel Hobby is licensed for non-commercial use. When you start
> charging (Phase 3), move the frontend to Cloudflare Pages or a Vercel Pro
> plan. Nothing in the code changes — only `BACKEND_URL`.

---

## 3. Daily snapshots (already wired)

`.github/workflows/snapshot.yml` runs weekdays after the US close, captures
option-chain snapshots, and commits them under `data/snapshots/`. It needs the
`TRADIER_TOKEN` repo secret and only fires on the default branch, so history
accrual starts once this work merges to `master`.

The backend reads that committed history for IV rank, OI change, and historical
GEX. Each Space rebuild picks up the latest committed snapshots automatically.

---

## Local dev

```bash
# Backend
uvicorn backend.app.main:app --reload --port 8000

# Frontend (separate shell)
cd frontend && npm install && npm run dev   # http://localhost:3000
```

Build the backend image locally to mirror the Space:

```bash
docker build -t gex-api .
docker run -p 7860:7860 -e TRADIER_TOKEN=... gex-api
curl localhost:7860/api/health
```
