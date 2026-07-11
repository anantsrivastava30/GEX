# Run the full stack locally with Docker

Runs the FastAPI backend and the Next.js frontend as two containers on your
own machine (Ubuntu or anything with Docker). No cloud host required.

## Quick start

```bash
cp .env.example .env          # then edit .env and set TRADIER_TOKEN
docker compose up --build
```

Open **http://localhost:3000**.

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000 (health: http://localhost:8000/api/health)

The frontend calls `/api/*` and Docker's internal network proxies that to the
backend container, so only port 3000 needs to be reachable.

Stop with `Ctrl-C`, or run detached with `docker compose up --build -d` and
stop with `docker compose down`.

## How it is wired

| Container | Image | Port | Notes |
| --- | --- | --- | --- |
| `backend` | root `Dockerfile` | 8000 | uvicorn; `PORT=8000`; reads `TRADIER_TOKEN` |
| `frontend` | `frontend/Dockerfile` | 3000 | Next standalone server |

The frontend's `/api` rewrite target is compiled in at **build** time, so
`BACKEND_URL=http://backend:8000` is passed as a build arg in
`docker-compose.yml`. To point a prebuilt frontend image at a different
backend, rebuild:

```bash
docker compose build --build-arg BACKEND_URL=http://my-host:8000 frontend
```

## Without Docker

See `deploy/DEPLOY.md` (Local dev section) for the venv + `npm run dev` flow.

## Exposing it as a public website (free)

Your box can serve a public HTTPS site with no port-forwarding or paid host
using a Cloudflare Tunnel pointed at the frontend:

```bash
cloudflared tunnel --url http://localhost:3000
```

That prints a public `https://*.trycloudflare.com` URL. For a permanent named
URL on your own domain, create a named tunnel in a free Cloudflare account.
`tailscale funnel 3000` and `ngrok http 3000` are equivalent alternatives.
