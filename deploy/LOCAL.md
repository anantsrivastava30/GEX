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
backend container. Backend port 8000 is bound to localhost only; port 3000 is
the public entry point.

Snapshot history and the SQLite fallback database persist under the host's
`data/` directory across image rebuilds and container replacement.

Watchlists, alert rules, and the in-app event inbox persist in
`data/gex_app.db`. Until Phase 3 authentication, this is one shared workspace
for every browser connected to the deployment. Set `WORKSPACE_PIN` (or reuse
`AI_PIN`) to authorize all watchlist and alert mutations.

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

## Optional alert delivery

In-app alerts require no additional configuration. Discord and email
destinations are server-owned and can be enabled in `.env`:

```text
ALERT_DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
ALERT_EMAIL_TO=operator@example.com
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USERNAME=
SMTP_PASSWORD=
SMTP_FROM=gex@example.com
SMTP_STARTTLS=true
```

Do not expose the shared watchlist/alert mutation API publicly until Phase 3
authentication is installed unless every viewer is trusted. The temporary
workspace PIN prevents unauthenticated mutation but does not provide per-user
visibility or ownership.

If an older Docker run created `data/*.db` as root and bare local Python reports
`attempt to write a readonly database`, correct the file ownership once before
mixing local and container runs.
