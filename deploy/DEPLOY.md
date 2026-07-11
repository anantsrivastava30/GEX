# Free deployment guide

Zero-cost hosting for the legacy Streamlit app **and** the new stack, all from
one repo. The three products never share a runtime — only source and snapshot
data — so each can be developed and shipped independently.

| Piece | Host | Deploys from | Cost |
| --- | --- | --- | --- |
| Legacy Streamlit (`legacy/app.py`) | Streamlit Community Cloud | `streamlit-prod` branch | Free |
| Backend (FastAPI) | Hugging Face Spaces (Docker) | `master` (path-filtered) | Free — 2 vCPU / 16 GB RAM, sleeps after ~48h idle |
| Frontend (Next.js) | Vercel Hobby | `master`, root dir `frontend/` | Free |
| Daily snapshot cron | GitHub Actions | `master` | Free |

The frontend calls the backend server-side through the Next.js rewrite
(`/api/:path*` → `BACKEND_URL`), so the browser only ever talks to the Vercel
origin. That keeps CORS a non-issue for normal use.

## Keeping the two products separate

Everything lives on `master` (one shared `quant_analysis/`, no analytics drift).
Separation is enforced at the deploy layer:

- **Streamlit** deploys from a dedicated **`streamlit-prod`** branch, not
  `master`. New-stack pushes to `master` therefore never restart the Streamlit
  app. Ship a legacy release only when you intend to:
  `git checkout streamlit-prod && git merge --ff-only master && git push`.
  (`legacy/app.py` also stays frozen per the UW plan, so its output does not change.)
- **Backend Space** rebuilds only when its own files change — `deploy-hf.yml`
  is path-filtered to `backend/`, `quant_analysis/`, `Dockerfile`,
  `requirements.txt`, `config.yaml`. A frontend-only push does not touch it.
- **Frontend** on Vercel uses an *Ignored Build Step* so it rebuilds only when
  `frontend/` changes (command below).

Active development continues on feature branches → PR → `master`. `master` is
the integration branch for the new stack; `streamlit-prod` is the legacy
release pointer.

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
5. Optional isolation — **Settings → Git → Ignored Build Step**, command:
   `git diff --quiet HEAD^ HEAD -- .` (run from root dir `frontend/`) so Vercel
   only rebuilds when `frontend/` changed.
6. Set the **Production Branch** to `master`.

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

## 4. Legacy Streamlit on Streamlit Community Cloud

1. At https://share.streamlit.io create an app from this repo.
2. **Branch:** `streamlit-prod` · **Main file path:** `legacy/app.py`.
3. Add `TRADIER_TOKEN` / `OPENAI_API_KEY` under the app's **Secrets**.

Because it deploys from `streamlit-prod` (not `master`), day-to-day new-stack
work never disturbs it. Cut a legacy release intentionally:

```bash
git checkout streamlit-prod
git merge --ff-only master     # only pulls in already-reviewed master history
git push origin streamlit-prod
```

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
