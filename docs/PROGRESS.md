# Overhaul Progress Tracker

Cross-session log for the Unusual Whales–parity rebuild. The full plan lives in
[`UW_PARITY_PLAN.md`](UW_PARITY_PLAN.md). **Every work session should update this
file**: tick checkboxes, append a session-log entry, and record what the next
session should pick up.

## How to resume work

1. Read `docs/UW_PARITY_PLAN.md` (architecture, phases, endpoint/function mappings).
2. Read the Status board and the last Session log entry below.
3. Work on the "Next up" items; keep `pytest` green; update this file before pushing.
4. Branch: `claude/unusual-whales-parity-fz318a` (until Phase 1 merges).

## Status board

### Phase 0 — Foundations (this preceded the shell on purpose)
- [x] Approved plan committed (`docs/UW_PARITY_PLAN.md`)
- [x] Cross-session tracker (this file)
- [x] Snapshot engine: per-contract chain snapshots + derived daily metrics to `data/snapshots/` (`quant_analysis/storage/snapshots.py`)
- [x] Snapshot CLI: `python -m quant_analysis.scripts.snapshot` (`quant_analysis/scripts/snapshot.py`)
- [x] GitHub Actions daily cron (`.github/workflows/snapshot.yml`) — **requires `TRADIER_TOKEN` repo secret; cron only fires on the default branch, so data accrual starts when this merges to `master`**
- [x] History consumers seeded: `compute_oi_change`, `compute_iv_rank` (powering Phase 2 features from our own data)
- [x] Tests for the snapshot engine (`tests/test_snapshots.py`)

### Phase 1 — UW-style shell + port (look & feel milestone)
- [x] P1.1 Backend skeleton: `backend/` FastAPI (config, deps w/ tenacity retry, `cache.py` TTL layer, `ratelimit.py` token bucket), routers `ticker` (snapshot/expirations/gex/skew/ratios), `market`, `news`, `flow` (unusual proxy), `history` (iv-rank/oi-change/metrics from our snapshots), pydantic schemas, TestClient tests w/ faked Tradier. *Still to add: `ai` router (port `render_ai_tab` pipeline), split exposure router when vanna/charm lands.*
- [ ] P1.2 Scheduler jobs in backend (APScheduler) calling the same snapshot engine intraday (gamma-gap every 30 min)
- [x] P1.3 Frontend shell: Next.js 16 + Tailwind v4 in `frontend/`, dark theme tokens (`globals.css`), `SidebarNav`/`TopBar` w/ ticker search, typed API client (`src/lib/api.ts`), `/api` rewrite to backend :8000. First pages live: `/market` (VIX/yields/futures), `/stock/[symbol]` (quote, expiration chips, net-GEX chart + table, gamma-gap signal card, interpretation), `/news`, `/flow` (live). *Still to add: shadcn/ui + TanStack Table for dense tables, more chart components, remaining pages.*
- [ ] P1.4 Page ports (one PR each): `/market`, `/stock/[symbol]` (+`/gex`, `/vol`), `/news`, `/ai`, `/tools/binomial`, `/flow` (basic unusual-spikes table), `/calendar` (iframe parity); resurrect intraday delta-projection on `/gex`
- [ ] P1.5 Legacy freeze: move `app.py` → `legacy/app.py`, `docker-compose.yml`, CI matrix (quant_analysis pytest + backend pytest + frontend build/Playwright), README update

### Phase 2 — Close the UW feature gap (free data)
- [ ] Vanna/Charm local Black-Scholes derivation (`quant_analysis/analytics/greeks.py`) + exposure endpoints/charts
- [ ] Max pain
- [ ] IV rank / percentile tiles + historical GEX charts (reads `data/snapshots/`)
- [ ] Term structure (fix disabled `compute_term_structure_slope`)
- [ ] Full proxy flow feed + hottest chains (vol/OI spikes + OI Δ from snapshots + IV move; filter chips)
- [ ] Congress trades (Senate eFD / House Clerk daily job)
- [ ] Native earnings/economic calendar
- [ ] Screener (presets + custom), server-persisted watchlists, alerts (in-app/Discord/email)

### Phase 3 — Monetization
- [ ] Supabase Auth + entitlements (replaces AI PIN)
- [ ] Stripe checkout/webhooks; Free / Pro $29–39 / Premium $79–99
- [ ] Landing page, `/pricing`, public gamma-gap track-record page
- [ ] Data-provider interface + migrate off Tradier retail (ToS blocker for charging)
- [ ] Delete `legacy/app.py`

### Phase 4 — Paid data & scale
- [ ] Real trade-tape flow feed (OPRA via provider interface), sweeps/blocks, Market Tide, 0DTE dashboards, SPX Periscope page
- [ ] Public API product (keys, metering); mobile polish

## Session log

### Session 1 — 2026-07-09
**Done:**
- Researched Unusual Whales (structure, features, pricing ~$48/$99, weaknesses: firehose w/o interpretation, weak computed dealer analytics, no track record) and inventoried this codebase (~5.6k LOC; strengths: gamma-gap engine w/ logged history, GEX/greeks analytics, AI narratives; gaps: no shell/caching/auth/flow data).
- Decisions locked: FastAPI + Next.js full rebuild · free data only for now · look & feel first · own our data history from day one.
- Wrote `docs/UW_PARITY_PLAN.md` + this tracker.
- Built Phase 0 snapshot engine: `quant_analysis/storage/snapshots.py` (capture → derive → write csv.gz per ticker/day + `daily_metrics.csv`; loaders + `compute_oi_change` + `compute_iv_rank`), CLI `quant_analysis/scripts/snapshot.py`, `snapshots:` config block, daily-cron workflow, tests.

- Scaffolded the FastAPI backend (P1.1): `backend/app/` with TTL cache, Tradier
  token-bucket rate limit + retry, routers for ticker/gex/skew/ratios, market
  overview, news, unusual-flow proxy, and history endpoints served from our own
  snapshot data (`/api/history/{sym}/iv-rank`, `/oi-change`, `/metrics`).
  Verified: 63 tests green, `uvicorn backend.app.main:app` boots, health +
  OpenAPI + graceful 404s confirmed live. CI installs backend deps.

- Built the frontend shell (P1.3): Next.js 16 + Tailwind v4, dark UW-style
  theme, sidebar nav + ticker search, `/market`, `/stock/[symbol]` (net-GEX
  table + gamma-gap card + interpretation), `/news`, `/flow` placeholder.
  Verified end-to-end: `npm run build` clean; `next start` + uvicorn smoke
  test shows / → /market redirect, page render, and `/api` rewrite proxying
  the backend.

**Next up (Session 2):**
1. Charts: load the `dataviz` skill, then add lightweight-charts candles on `/stock/[symbol]` and an ECharts horizontal strike-profile chart replacing the net-GEX table; stat-tile components for VIX/quote metrics.
2. Dense tables: TanStack Table wrapper; wire `/flow` to `/api/flow/unusual` with filter chips.
3. Add `ai` router to the backend (port `services/ai_analysis.py` pipeline off Streamlit).
4. P1.2 APScheduler jobs in the backend (intraday gamma-gap scoring reusing the snapshot engine).
5. Merge this branch early so the snapshot cron starts accruing history on `master` — every unmerged day is lost data.

**Dev run (current state):**
- Backend: `uvicorn backend.app.main:app --reload --port 8000` (needs `TRADIER_TOKEN` env for live data; degrades gracefully without).
- Frontend: `cd frontend && npm install && npm run dev` → http://localhost:3000.

**Open items for the human:**
- Add `TRADIER_TOKEN` as a GitHub Actions repo secret (Settings → Secrets → Actions) so the daily snapshot cron can run.
- Merge PR for this branch promptly to start data accrual.

### Session 2 — 2026-07-09
**Free deployment (no-cost hosting):** chose Hugging Face Spaces (Docker) for the
backend + Vercel Hobby for the frontend + the existing GitHub Actions cron for
snapshots. All $0; container cold starts are the only tradeoff.
- Root `Dockerfile` (+ `.dockerignore`): builds the FastAPI backend, listens on
  `${PORT:-7860}` (works on HF Spaces, Cloud Run, any Docker host). Installs the
  full root requirements because the shared `quant_analysis` lib hard-imports the
  scientific/plotting stack. Verified all COPY paths exist; the local Docker
  daemon was unavailable so the image was not run here.
- `deploy/huggingface/README.md` (Space config card) + `.github/workflows/deploy-hf.yml`
  (auto-push to the Space on `master`, needs `HF_TOKEN`/`HF_SPACE` secrets).
- `deploy/DEPLOY.md`: full step-by-step for all three pieces + local dev/docker.
- Added tracked `data/snapshots/` placeholder so the image builds before the cron
  has committed any history.

**Frontend (resumed Session-2 "Next up"):**
- **Root-cause fix:** `frontend/src/lib/api.ts` was silently gitignored (root
  `.gitignore` had an unanchored Python `lib/` rule matching `frontend/src/lib/`),
  so the typed API client from Session 1 was never committed and the frontend did
  not build. Anchored the rule to `/lib/` `/lib64/` and committed the client.
- Restored/typed `src/lib/api.ts` against the real backend schemas (ticker,
  gex, skew, ratios, flow/unusual, market, news, history/iv-rank).
- Added `components/charts/GexStrikeChart.tsx`: dependency-free horizontal
  diverging bar chart of net GEX by strike (polarity by both bar direction and
  color, centered zero baseline, spot marker, per-bar hover value). Loaded the
  `dataviz` skill first; reuses the app's positive/negative tokens.
- `/stock/[symbol]`: chart replaces the raw table (table kept behind a "Table
  view" details for accessibility).
- `/flow`: wired to `/api/flow/unusual` (symbol input → nearest 4 expirations →
  dense vol/OI table), replacing the placeholder.

**Verified:** `npm run build` clean (was broken before). Rendered the chart with
mock data via headless Chromium and eyeballed it (diverging bars, spot highlight,
hover value all correct). No backend code changed this session.

**Next up (Session 3):**
1. P1.2 APScheduler intraday jobs in the backend (gamma-gap every 30 min reusing
   the snapshot engine) — best-effort on HF Spaces since the container sleeps.
2. Port the `ai` router (`quant_analysis/services/ai_analysis.py`) — decouple from
   `st.secrets`, read `OPENAI_API_KEY` from env.
3. Stat-tile components for `/market` + `/stock` quote metrics; IV-rank tile.
4. TanStack Table + filter chips on `/flow`.

**Open items for the human:**
- Create a Hugging Face **Docker** Space and add repo secrets `HF_TOKEN` +
  `HF_SPACE`, then set `TRADIER_TOKEN` (and `OPENAI_API_KEY`) as Space secrets.
- On Vercel, import the repo with Root Directory = `frontend` and env
  `BACKEND_URL` = the Space URL. See `deploy/DEPLOY.md`.

### Session 2b — 2026-07-09 (UW look-and-feel pass)
Goal from the human: reach UW-style look and feel before deploying.
- Refined theme tokens (`globals.css`): near-black canvas, layered surfaces
  (`surface`/`surface-2`), `border-strong`, `faint` ink, `warning`, blue accent,
  14px base, thin scrollbars, selection color.
- Icon set `components/layout/icons.tsx`: inline feather-style SVGs (no dep).
- Reusable primitives: `components/ui/Panel.tsx` (titled card), `StatTile.tsx`
  (KPI with signed arrow delta), `Tabs.tsx` (underlined tab strip).
- `SidebarNav`: grouped into Markets/Research sections, SVG icons, whale logo mark.
- `TopBar`: search with leading icon + live US market-open/closed status + ET clock.
- `/market`: KPI stat-tile grid (VIX accent tile, 10Y, futures) with loading
  skeletons, replacing the two plain tables.
- `/stock/[symbol]`: ticker header (price, change badge, bid/ask/vol/52w strip),
  Overview/GEX/Volatility(soon)/Flow(soon) tabs, gamma-gap zone badge, panelized
  interpretation.

**Verified:** `npm run build` clean; captured the full shell (sidebar, top bar,
ticker header, stat tiles, GEX chart, gamma-gap + interpretation panels) and the
`/flow` page via headless Chromium at 1280px and eyeballed both — reads like a
UW-style terminal. Temp preview route removed before commit. No backend changes.

**Next up (Session 3):** unchanged from above (APScheduler, `ai` router,
TanStack Table + filter chips on `/flow`, IV-rank tile) plus: candlestick price
chart on `/stock` Overview, and port `/news` + `/flow` blocks onto `Panel`.

### Session 2c — 2026-07-09 (deploy isolation + charts)
Goal from the human: keep legacy Streamlit and the new stack hosted separately,
then keep building Phase 1.
- **Deploy isolation (a + b):** path-filtered `deploy-hf.yml` so the backend
  Space only rebuilds on `backend/`/`quant_analysis/`/`Dockerfile`/deps changes;
  created the `streamlit-prod` branch (Streamlit Cloud deploys from it, so
  master pushes never restart the legacy app). Strategy + steps documented in
  `deploy/DEPLOY.md` (one repo, one `quant_analysis`, separation at the deploy
  layer; Vercel Ignored Build Step for `frontend/`).
- **Candlestick chart:** new backend endpoint `GET /api/ticker/{sym}/candles`
  (`Candle` schema + `services.get_candles` reusing `TradierAPI.history`, cached
  1h). Dependency-free `CandlestickChart.tsx` (SVG OHLC candles, price
  gridlines, dashed last-close line, date axis, per-candle hover tooltip) on the
  `/stock` Overview tab.
- **Sortable flow table:** `/flow` rebuilt on `Panel` with click-to-sort columns
  (default Total vol/OI desc) and hover rows.
- `api.ts`: added `Candle` type + `api.candles()`.

**Verified:** `npm run build` clean; backend files `py_compile` clean (full
backend still not boot-tested here — heavy deps + no local Docker daemon).
Rendered the Overview candlestick + sortable flow table via headless Chromium at
1280px and eyeballed — reads like a UW ticker page. Temp preview removed. The
only console warning was preview-only (module-level `Math.random` mock data).

**Next up (Session 3, updated):** APScheduler intraday jobs; `ai` router (env
key, no `st.secrets`); IV-rank tile on `/stock` from `/api/history`; filter
chips on `/flow`; wire live candles once the backend runs on the Space.
