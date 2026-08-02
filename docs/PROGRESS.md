# Overhaul Progress Tracker

Cross-session log for the Unusual Whales–parity rebuild. The full plan lives in
[`UW_PARITY_PLAN.md`](UW_PARITY_PLAN.md). **Every work session should update this
file**: tick checkboxes, append a session-log entry, and record what the next
session should pick up.

## How to resume work

1. Read `docs/UW_PARITY_PLAN.md` (architecture, phases, endpoint/function mappings).
2. Read the Status board and the last Session log entry below.
3. Work on the "Next up" items; keep `pytest` green; update this file before pushing.
4. Active branch: `claude/streamlit-free-deployment-9r4x55`.

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
- [x] P1.1 Backend skeleton: `backend/` FastAPI (config, deps w/ tenacity retry, `cache.py` TTL layer, `ratelimit.py` token bucket), routers `ticker` (snapshot/expirations/gex/skew/ratios/max-pain/term-structure), `exposure` (vanna/charm), `market`, `news`, `flow` (unusual proxy), `history` (iv-rank/oi-change/metrics/gamma-gap from our own data), `ai` (analyze/analyses/status), pydantic schemas, TestClient tests w/ faked Tradier.
- [x] P1.2 Scheduler jobs in backend (APScheduler): intraday snapshot + gamma-gap scoring every 30 min during market hours (`backend/app/jobs.py`, `SCHEDULER_ENABLED`, on by default in compose)
- [x] P1.3 Frontend shell: Next.js 16 + Tailwind v4 in `frontend/`, dark theme tokens (`globals.css`), `SidebarNav`/`TopBar` w/ ticker search, typed API client (`src/lib/api.ts`), `/api` rewrite to backend :8000. First pages live: `/market` (VIX/yields/futures), `/stock/[symbol]` (quote, expiration chips, net-GEX chart + table, gamma-gap signal card, interpretation), `/news`, `/flow` (live). *Still to add: shadcn/ui + TanStack Table for dense tables, more chart components, remaining pages.*
- [x] P1.4 Page ports: `/market`, `/stock/[symbol]` (Overview/GEX/Volatility tabs incl. intraday delta-projection), `/news`, `/ai`, `/tools/binomial`, `/flow` (cached proxy feed), `/calendar` (iframe parity), `/track-record`
- [x] P1.5 Legacy freeze: `app.py` → `legacy/app.py`, `docker-compose.yml`, path-filtered CI matrix (analytics pytest + backend pytest + frontend build/Playwright e2e), README update

### Phase 2 — Close the UW feature gap (free data)
- [x] Vanna/Charm local Black-Scholes derivation (`quant_analysis/analytics/greeks.py`) + exposure endpoints/charts
- [x] Max pain (`compute_max_pain` + endpoint + tile on the GEX tab)
- [x] IV rank / percentile tiles + historical GEX charts (reads `data/snapshots/`) — tiles on `/stock` Overview + Volatility; historical net GEX on the GEX tab
- [x] Term structure (pure `compute_term_structure` + fixed `compute_term_structure_slope` + endpoint + chart)
- [x] Full proxy flow feed + hottest chains (`/api/flow/feed` + `/api/flow/hottest-chains` ranking vol/OI + OI Δ + IV Δ from snapshots; filters on `/flow`)
- [x] Congress trades (Senate eFD / House Clerk daily job) — *provider-pluggable feed (`/api/congress/trades` + `/congress` page). Live and current via `FMP_API_KEY` (verified Session 8: 156 filings, `stale=false`); the free mirrors remain a degraded fallback.*
- [x] Native earnings/economic calendar (yfinance earnings + curated FRED release dates, independently degradable sources)
- [x] Screener (presets + custom), server-persisted watchlists, alerts (in-app/Discord/email) - shared single-workspace persistence until Phase 3 auth

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

### Session 2d — 2026-07-11 (self-host Docker setup)
Hugging Face now gates Docker Spaces behind a paid PRO plan (Static-only free),
so the free HF backend path is dead. Pivoted to self-hosting the full stack via
Docker (user's Ubuntu box has far more RAM than any free tier anyway).
- `frontend/Dockerfile`: multi-stage, Next `output: "standalone"` (added to
  `next.config.ts`) → minimal `server.js` runtime image, non-root. `BACKEND_URL`
  is a build arg (Next bakes rewrites into the routes manifest at build time).
- `frontend/.dockerignore`.
- Root `docker-compose.yml`: `backend` (root Dockerfile, `PORT=8000`, healthcheck
  on `/api/health`) + `frontend` (waits for backend healthy). One command:
  `cp .env.example .env && docker compose up --build` → http://localhost:3000.
- `.env.example`, `deploy/LOCAL.md` (compose usage + Cloudflare Tunnel for a
  free public HTTPS URL from the box).

**Verified:** frontend standalone build clean; confirmed the `/api` rewrite is
baked into `routes-manifest.json` at build time (so the compose build arg is the
correct mechanism); `docker compose config` validates. Images not built here (no
Docker daemon in this session) — build/run happens on the user's box.

**Deploy status:** HF Spaces abandoned (now paid). Options documented: self-host
Docker (primary), or Render free / Cloud Run for cloud. `deploy-hf.yml` +
`deploy/huggingface/` remain in-tree but unused unless HF PRO.

### Session 3 — 2026-07-11 (P1 close-out + Phase 2 analytics)
Resumed toward Phase 2 per the plan. All work verified end to end against the
live backend (real Tradier data) plus mock-data chart previews via headless
Chrome.

**Phase 2 analytics (new `quant_analysis/analytics/greeks.py`):**
- Black-Scholes vanna (dDelta/dVol) and charm (dDelta/dTime) with dividend
  yield support; golden-value tests verify both against central finite
  differences of `bs_delta` plus a hand-checked ATM value (`tests/test_greeks.py`,
  23 tests).
- `compute_vanna_charm_exposures`: per-strike net exposure (calls minus puts,
  matching the net-GEX convention), `mid_iv` with `smv_vol` fallback, DTE floor
  of half a day for 0DTE.
- `compute_max_pain`: pure payout-minimisation over chain OI with the full
  pain curve for charting.
- `get_risk_free_rate`: ^IRX via yfinance with a safe default; cached 6h in
  the backend.
- `market_data.py`: pure `compute_term_structure` (ATM IV per expiration from
  pre-fetched chains); `compute_term_structure_slope` fixed (was requesting
  `greeks="false"` then reading greeks - dead since inception) and rebuilt on
  the pure function. Extracted `process_options_data` from `load_options_data`
  so cached-chain callers reuse the same math (public API unchanged).

**Backend:**
- New endpoints: `GET /api/ticker/{sym}/exposure` (vanna/charm by strike, own
  `exposure` router), `/max-pain` (strike + pain curve), `/term-structure`,
  `GET /api/history/gamma-gap` (logged scan rows - raw material for the
  track-record page).
- `ai` router (P1 leftover): `POST /api/ai/analyze` reusing the
  `build_analysis_payload`/`create_data_packet` pipeline headless
  (`backend/app/services_ai.py`), PIN-gated via `AI_PIN` env, saves to the
  existing `ai_analysis` store; `GET /api/ai/analyses`, `GET /api/ai/status`.
  `ai_analysis.build_analysis_payload` gained an optional `tradier_token`
  param (headless callers pass it; Streamlit behavior unchanged).
- P1.2 scheduler: `backend/app/jobs.py` (APScheduler BackgroundScheduler),
  intraday snapshots + gamma-gap rows every 30 min Mon-Fri 9:30-16:00 ET for
  the configured snapshot tickers, reusing `capture_ticker_snapshot` +
  `write_snapshot` (upsert) and `save_gamma_gap_results`. Gated by
  `SCHEDULER_ENABLED` (default off; on in docker-compose). Wired via FastAPI
  lifespan; `apscheduler` added to backend requirements.
- 9 new backend tests (exposure/max-pain/term-structure/gamma-gap-history/ai
  status + PIN gate). 91 tests pass; `test_missing_token_returns_503` fails
  locally only because a real `.env` exists (pydantic-settings reads the token
  from the file), passes in CI.

**Frontend:**
- Generic `StrikeBarChart` extracted from `GexStrikeChart` (now a wrapper);
  vanna + charm strike charts on the GEX tab with max-pain / gamma-gap-score /
  risk-free-rate tiles.
- Volatility tab is live: `SkewChart` (call/put IV by strike, legend + direct
  labels since green/red sits in the CVD floor band, crosshair tooltip, spot
  line), `TermStructureChart` (ATM IV per expiration, contango/inverted
  read), IV rank / percentile / ATM IV / term-slope tiles.
- IV rank + ATM IV tiles on Overview (graceful "accruing history" state while
  snapshots build up).
- `/flow`: UW-style filter chips (side dominance, vol/OI >= 2/5, total OI >=
  1k/10k) applied client-side on the sortable table.
- Fixed gamma-gap distance display (backend sends a fraction; UI now
  multiplies by 100).

**Verified:** `npm run build` clean; 91/92 pytest (see above); live end-to-end
against Tradier: max pain SPY 750 vs spot 754.95, contango term structure,
vanna/charm with live ^IRX 3.69%; `/api/ai/status` + `/api/ai/analyses` live;
scheduler boot + market-hours guard unit-smoked; chart components eyeballed at
1280px via headless Chrome (dataviz skill loaded; label collision fixed).

**Next up (Session 4):**
1. Historical GEX charts from `data/snapshots/` (last unchecked half of the
   IV-rank Phase 2 item) + surface `/api/history/gamma-gap` as a track-record
   page (hero differentiator).
2. `/ai` page in the frontend (the backend is ready; port the three-step flow).
3. Full proxy flow feed + hottest chains (vol/OI spikes + OI change from
   snapshots + IV move; scan a capped liquid universe instead of one symbol).
4. P1.4 leftovers: `/tools/binomial`, `/calendar` iframe parity; then P1.5
   legacy freeze (move `app.py` to `legacy/`, CI matrix, README).

**Open items for the human:**
- Merge to `main` so the snapshot cron accrues history (every unmerged day is
  lost data for IV rank).
- On the self-host box: `docker compose up --build` picks up the scheduler
  (`SCHEDULER_ENABLED=true` default) - snapshots then also accrue intraday.

### Session 3 recovery - 2026-07-11
Resumed after the prior session ended during review and corrected the concrete
site issues it left behind.

- Released a stale local Uvicorn process that was blocking port 8000.
- Keyed ticker-page state by symbol and added request cancellation so old
  symbol, expiration, and flow responses cannot overwrite newer selections.
- Added distinct unavailable states for IV rank, skew, and term structure;
  corrected max-pain wording and flow sorting/filter empty states.
- Made AI analysis fail closed when `AI_PIN` is absent, restricted requests to
  the configured model and bounded inputs, protected history with `X-AI-PIN`,
  and stopped returning provider exception details.
- Persisted snapshots and SQLite under the host `data/` mount in Compose and
  bound backend port 8000 to localhost only.
- Fixed option DTE date boundaries, scheduler write isolation, scheduler
  shutdown cleanup, and the existing frontend clock lint failure.

**Verified:** frontend lint and production build pass; Python compileall and
`docker compose config --quiet` pass; backend API tests are 16/17 locally. The
one failure is the known missing-token test because the real local `.env` is
loaded after the test removes only the process environment variable.

### Session 4 — 2026-07-11 (history and track record)
- Added a typed client for the existing `/api/history/metrics` and
  `/api/history/gamma-gap` endpoints.
- `/stock/[symbol]` now renders a historical net-GEX chart on the GEX tab from
  the project's daily snapshot metrics. It has clear loading and insufficient-
  history states so it does not imply data that has not accrued yet.
- Added `/track-record`: ticker-filterable gamma-gap signal log, summary tiles,
  and the scheduler capture history. It explicitly does not claim a hit rate
  until realized-outcome validation is implemented.
- Added Track Record to the live sidebar navigation.

**Verified:** `npm run lint` and `npm run build` in `frontend/` pass. The build
includes static `/track-record` and dynamic `/stock/[symbol]` routes.

**Next up:** port the AI frontend flow, then complete the multi-ticker proxy
flow feed and hottest-chains ranking. Add gamma-gap realized-outcome scoring
after enough snapshots exist to support a genuine hit-rate calculation.

### Session 5 — 2026-07-11 (P1 finish line + flow feed, screener, pages; two-agent handoff)
This session's work spans two agents that each hit usage limits mid-stream;
a third pass reconstructed the combined state, validated everything, and
fixed the handoff gaps. Combined delivery:

**Backend:**
- Cached proxy flow feed: `GET /api/flow/feed` (contract-level anomalies
  ranked by percentile-scored vol/OI + |OI Δ| + |IV Δ|) and
  `GET /api/flow/hottest-chains` (per-chain aggregates), both reading only
  persisted snapshots - no live Tradier scans. Snapshot layer gained
  `compute_contract_snapshot_diff` (OI + IV day-over-day diffs;
  `compute_oi_change` now delegates to it with an unchanged public frame).
- Snapshot-backed screener: `GET /api/screener` with `high_vol_oi`,
  `unusually_bullish`, `gamma_squeeze` presets, explicit methodology strings,
  and per-symbol staleness/unavailability reporting.
- Binomial tree: `GET /api/ticker/{sym}/binomial-tree` (CRR, market-calibrated
  rate/IV with manual overrides) reusing `generate_binomial_tree`.
- AI hardening: PIN gate now constant-time via `secrets.compare_digest`,
  required whenever configured (503 when unset), model allowlisted to the
  configured id, `/api/ai/analyses` PIN-gated via `X-AI-PIN` header.
- Scheduler: per-ticker pacing (`snapshots.refresh_pause_seconds`, config +
  env), per-ticker persistence error isolation, lifespan shutdown in
  try/finally.
- Snapshot universe expanded to 13 liquid tickers in `config.yaml`.

**Frontend:**
- `/flow` rebuilt as the cached positioning proxy: hottest-chains table +
  contract-level feed with symbol/side/vol-OI/OI-Δ/IV-Δ filters, staleness
  badges, and honest not-trade-tape labeling. AbortController on refetch.
- `/ai`: full three-step flow (status tiles, live expiration chips, PIN-gated
  review + send, protected history).
- `/tools/binomial`: CRR tree explorer against the new endpoint.
- `/calendar`: Investing.com iframe parity page (nav unflagged).
- `/track-record`: gamma-gap signal log + summary tiles (no hit-rate claim
  until realized-outcome scoring exists) - Session 4 item, now nav-linked.
- `HistoricalGexChart` on the GEX tab from `/api/history/metrics`.

**P1.5 legacy freeze:** `app.py` moved to `legacy/app.py`; path-filtered CI
matrix (analytics pytest / backend pytest / frontend build + Playwright);
README and deploy docs updated; Playwright smoke suite added
(`frontend/e2e/smoke.spec.ts`, fully API-mocked).

**Handoff fixes in the reconciliation pass:**
- Rewrote the stale flow e2e test for the rebuilt `/flow` page (old test
  targeted the removed per-symbol Scan UI) and added `/api/flow/feed` +
  `/api/flow/hottest-chains` mocks.
- Unflagged Calendar in the sidebar (page existed but nav still said SOON).

**Verified (combined):** analytics + backend pytest 93/94 (the one failure is
the known local-.env missing-token test); `npm run lint` + `npm run build`
clean; Playwright 4/4 against the standalone build; live end-to-end with real
Tradier data: snapshot capture (SPY/QQQ into a scratch dir) → `/api/flow/feed`,
`/api/flow/hottest-chains`, `/api/screener` all rank real contract anomalies
with correct staleness flags; binomial tree calibrates against live spot;
`/flow`, `/tools/binomial`, `/ai`, `/track-record` rendered and eyeballed via
headless Chrome against the live backend.

**Next up (Session 6):**
1. Screener frontend page (backend presets are live) + custom filter builder.
2. Resurrect intraday delta-projection on the GEX tab (last P1.4 item).
3. Gamma-gap realized-outcome scoring + hit-rate on `/track-record` once
   snapshot history accrues.
4. Congress trades daily job; native calendar.
5. Watchlists + alerts.

**Open items for the human:**
- Merge to `main` so the snapshot cron starts accruing the 13-ticker history
  (IV rank, OI Δ, flow feed, and the track record all depend on it).
- `docker compose up --build` on the box picks up the scheduler for intraday
  accrual.

### Session 6 — 2026-07-11 (screener page, realized track record, delta projection)
Closed the remaining P1.4 item and two Phase 2 items.

**`/screener` page:** preset chips (High Vol/OI, Unusually Bullish, Gamma
Squeeze Candidates), min vol/OI + min OI threshold filters, per-preset table
layouts (contract rows vs gamma-candidate rows), staleness/unavailability
banners, backend methodology string displayed under the table. Sidebar nav
unflagged. E2e coverage added.

**Gamma-gap realized outcomes (the hero differentiator):**
- `quant_analysis/analytics/track_record.py`: pure
  `evaluate_gamma_gap_outcome` (hit = magnet strike traded low<=magnet<=high
  within N sessions AFTER the signal date; the signal-day bar never counts;
  short histories stay pending, never miss) + `summarize_outcomes` (hit rate
  over decided signals only, avg sessions-to-hit, score-bucket breakdown).
  7 unit tests.
- `GET /api/history/gamma-gap/outcomes`: joins the signal log with cached
  Tradier daily candles per ticker and returns per-signal outcomes plus the
  summary. Endpoint test with faked candles.
- `/track-record` now shows the realized hit rate, decided counts, avg
  sessions to hit, score-bucket chips, and a per-signal Hit/Miss/Pending
  badge column. Live run against the 42 logged legacy signals: 57.1% hit
  rate, hits tagging the magnet in 1.4 sessions on average.

**Intraday delta projection (last P1.4 item):**
- Fixed the dead legacy pipeline: extracted pure
  `compute_delta_exposure_series` (one chain snapshot, exposure window
  sliding along the intraday path) from `get_delta_exposure_at_times`,
  which previously re-fetched the same chain for every bar; public
  signature preserved.
- `GET /api/ticker/{sym}/delta-projection`: yfinance 30-min bars (prev close
  prefixed, cached 5 min) + cached chain + linear exposure trend projected
  to the 16:00 close.
- `DeltaProjectionChart` on the GEX tab: two stacked charts sharing one
  slot-indexed time axis (price line above, exposure bars below - never a
  dual-axis chart), dashed trend extension to a hollow projected-close
  marker, per-bar tooltip.
- Fixed two `react-hooks/set-state-in-effect` lint errors by moving state
  resets into event handlers.

**Verified:** lint clean; `npm run build` clean (new `/screener` route);
Playwright 5/5 (new screener + delta-projection assertions); pytest 102/103
(known local-.env failure only); live against Tradier: outcomes endpoint
scored all 42 real signals in 2.6s, delta projection returned 13 Friday bars
with an EOD projection; screener + track-record + delta-projection chart all
rendered and eyeballed via headless Chrome.

**Next up (Session 7):**
1. Custom screener filter builder; server-persisted watchlists; alerts
   (in-app + Discord webhook + email) with an alert-evaluation job.
2. Congress trades daily job (Senate eFD / House Clerk, staleness flags).
3. Native earnings/econ calendar (yfinance earnings + FRED releases).
4. Public track-record page polish once intraday scheduler data accrues
   (top-bucket hit rates are the marketing asset).

**Open items for the human:** unchanged (merge to `main`; run compose with
the scheduler).

### Session 7 — 2026-07-11 (Congress trades feed)
Closed the first of the remaining Phase 2 UW-gap items: congressional
stock-trade disclosures, a marquee UW feature on free data. The `/congress`
nav entry was already stubbed (`soon`) with an icon but had no page or backend.

**Backend:**
- `backend/app/services_congress.py`: provider-pluggable, tolerant disclosure
  aggregator. A single `_normalise` collapses Senate/House/FMP record shapes
  into one row (probes many key aliases, parses 5 date formats, maps
  purchase/sale/exchange to buy/sell/exchange, resolves member names from
  `representative`/`senator`/`office`/first+last). Providers are tried in
  order and merged + de-duped: an optional `FMP_API_KEY` first, then the
  `congress.sources` JSON mirrors from `config.yaml`. Any provider failing is
  isolated into `unavailable_sources`; the response is flagged `stale` when the
  newest disclosure is > 14 days old or every provider failed. Cached 6h.
- `GET /api/congress/trades` (`ticker`/`chamber`/`days`/`limit` filters) +
  `CongressTrade`/`CongressTradesResponse` schemas; router registered in
  `main.py`.
- `config.py`: optional `fmp_api_key`. `config.yaml`: `congress.sources` list.

**Frontend:**
- `/congress`: chamber chips (Both/Senate/House), lookback windows
  (30/90/180/365d), ticker filter, dense disclosure table (traded/disclosed
  dates, member + owner, chamber, ticker, buy/sell-tinted side, dollar-range
  amount), staleness + unreachable-source badges, honest "PTRs filed on a lag,
  not real-time" labeling. AbortController on refetch. Sidebar `soon` flag
  removed. `api.ts`: `CongressTrade`/`CongressTradesResponse` + `congressTrades`.

**Data-source note (important):** the long-standing free stock-watcher S3
mirrors now return `403 AccessDenied`, and CapitolTrades' BFF is CloudFront/WAF
-blocked from this sandbox; FMP needs a key. So the default feed currently
returns an empty, `stale`, `unavailable_sources` result from here - which is
the intended graceful-degradation behavior (the plan flagged congress feeds as
scrape-fragile). Live data needs either a reachable mirror URL swapped into
`config.yaml` or an `FMP_API_KEY`. The parser is verified against real
Senate/House/FMP record shapes offline.

**Verified:** `backend/app/*` py_compile clean; `config.yaml` parses;
`/api/congress/trades` returns 200 and degrades gracefully (verified live:
`stale=true`, both mirrors in `unavailable_sources`); `_normalise` unit-checked
on real-shape House/Senate/FMP records (correct dates, sides, tickers, names);
frontend `npm run lint` + `npm run build` clean (new `/congress` route);
backend pytest 20/21 (only the known local-`.env` missing-token test fails).

**Next up (Session 8):**
1. Native earnings/econ calendar (yfinance earnings + FRED releases) to replace
   the Investing.com iframe.
2. Custom screener filter builder; then server-persisted watchlists + alerts
   (in-app + Discord webhook + email) - watchlists/alerts really want the
   Phase 3 auth layer, so consider sequencing auth first.
3. Swap in a reachable congress source (mirror URL or `FMP_API_KEY`) and add a
   scheduler job to warm the 6h cache.
4. Gamma-gap realized-outcome hit-rate polish once intraday snapshots accrue.

**Open items for the human:**
- To light up live Congress data: set `FMP_API_KEY` (free tier) as a backend
  env var, or replace the `congress.sources` URLs in `config.yaml` with a
  currently-reachable mirror. Without either, the page honestly shows an empty,
  stale feed.
- Still pending from prior sessions: merge to `main` for snapshot accrual; run
  compose with the scheduler.

### Session 8 - 2026-07-12 (seeded snapshots, admin capture, AI payload-only mode, Congress validation)

Goal: make the tool demonstrable now. The screener/flow pages were shipped but
empty because tracked `data/snapshots/` had never been seeded, and AI analysis
was unusable without OpenAI credentials.

**Snapshot seeding + weekend-aware dating:**
- `quant_analysis/storage/snapshots.py`: new `last_trading_date()` (rolls the
  US-market calendar date back over weekends). `capture_ticker_snapshot` now
  keys snapshots by it, and the staleness `today` comparisons in
  `backend/app/services.py` / `services_screener.py` use it, so weekend or
  pre-market reads of Friday data are not falsely flagged stale.
- `backend/app/jobs.py`: extracted `capture_universe()` from the scheduler job
  (returns a summary dict); `run_intraday_snapshots` keeps its token +
  market-open gates and delegates.
- New `POST /api/admin/capture` (`backend/app/routers/admin.py`), gated by the
  same `X-AI-PIN`: on-demand full-universe refresh, works off-hours (closed
  chains carry the last session's OI/volume). `CaptureResponse` schema.
- Seeded real snapshots for the 13-ticker universe (keyed 2026-07-13, ET
  pre-market). Screener now returns 50 live candidates with `stale=false`;
  gamma_squeeze honestly returns 0 (top score 34.8 < the 50 threshold). Flow
  feed serves volume rows now and gains OI/IV deltas when a second distinct
  trading day accrues.

**AI payload-only mode (usable without any agent/key):**
- `backend/app/services_ai.py`: extracted `_build_symbol_packet` (data fetch +
  compact payload + prompt packet + token estimate) shared by the paid path;
  new `get_ai_payload` and `POST /api/ai/payload` - no PIN, no OpenAI call.
- `/ai` page: "Build payload only" button (enabled even when OpenAI/PIN are
  missing), payload panel with Copy prompt / Copy payload JSON, prompt preview
  and payload JSON accordions, prompt-token count.
- `api.ts`: `AIPayloadRequest/Response` + `aiPayload`.

**Congress handoff validated (Session 7 work stream):**
- `FMP_API_KEY` is now set in the local `.env`; `/api/congress/trades` is live
  and current (`as_of` 2026-07-07, `stale=false`, no unavailable sources) and
  the `/congress` page renders 156 real filings. Committed as part of this
  session.

**Verified:** backend+analytics pytest 102/103 (only the known local-`.env`
missing-token test fails; passes in CI); frontend `npm run lint` +
`npm run build` clean; live smokes against the real backend: screener presets,
flow feed, `/api/admin/capture` (401 without PIN, full capture with it),
`/api/ai/payload` (1,769-token SPY prompt), congress trades; headless-browser
screenshots of `/screener`, `/ai` payload flow, `/congress` all render real
data.

**Next up (Session 9):**
1. Native earnings/econ calendar (yfinance earnings + FRED releases).
2. Custom screener filter builder; then watchlists + alerts (consider
   sequencing Phase 3 auth first).
3. Scheduler job to warm the 6h congress cache.
4. Track-record polish as intraday snapshots accrue on the default branch.

### Session 9 - 2026-08-02 (Phase 2 completion + snapshot recovery)

**Snapshot root cause and recovery:**
- Confirmed all 17 scheduled GitHub Actions runs reached the capture step but
  exited with code 2 because the repository Actions secret `TRADIER_TOKEN` is
  absent. Added an explicit workflow preflight that reports the exact setup
  path instead of failing after dependency installation.
- The latest seed/scheduler work is still unmerged: local HEAD is eight commits
  ahead of `master`, and the latest seed commit is one commit ahead of its
  remote feature branch. `master` therefore has no committed snapshot history.
- Attempted a bounded Sunday capture, then removed it during review because the
  provider had already rolled its expiration universe; labeling that chain as
  Friday would create false OI changes. Capture entry points now skip weekends,
  holidays, and premarket instead of overwriting the previous session.
- Added standard US market-holiday handling and consecutive-session checks.
  OI/IV changes are unavailable across date gaps rather than treating every
  newly listed contract as fresh OI. `/api/health` now reports the latest date,
  history-day count, and whether current consecutive history is ready. Local
  history remains the single 2026-07-13 seed until a valid market-session run.
- Found a separate local gamma-gap persistence issue: `data/ai_analysis.db` is
  owned by `root:root` from an earlier Docker run, so bare local Python cannot
  append. The database needs a one-time
  ownership correction before local gamma-gap logging uses that path.

**Completed Phase 2:**
- Replaced the calendar iframe with `GET /api/calendar` and a native page:
  yfinance earnings for a bounded company universe plus curated FRED release
  dates, with independent source status and honest provider limitations.
- Added field metadata and `POST /api/screener/query`: contract/ticker scopes,
  whitelisted typed fields/operators, AND conditions, sorting, bounded output,
  snapshot staleness, and no live Tradier reads. `/screener` now includes a
  reusable custom filter builder while preserving all three presets.
- Added shared SQLite watchlist CRUD under `data/gex_app.db`, bounded to the
  configured snapshot universe, plus `/watchlists` management UI.
- Added persisted alert-rule CRUD, in-app event inbox, daily deduplication, and
  scheduled evaluation 10 minutes after each intraday snapshot. Stale or
  incomplete snapshots never emit. Optional Discord and SMTP delivery uses
  server-owned environment settings; API callers cannot provide destinations.
- Added a constant-time workspace PIN gate for every watchlist/alert mutation,
  outbound delivery retry (three attempts), and retained event history that
  prevents deleting a rule after it has emitted.
- Added the pending weekday congress-feed cache warm-up job at 07:15 ET.
- Added `/alerts`, `/watchlists`, responsive desktop/mobile navigation, and
  accessible custom-filter controls. The UI labels the state as a shared
  server workspace pending Phase 3 authentication.

**Verified:** backend compile/OpenAPI generation clean (40 routes); lightweight
API smoke covered health, watchlist initialization, custom ticker query, alert
rule create/list/evaluate/delete, and event persistence. Final safety smoke
confirmed unauthenticated mutation returns 401, valid PIN mutation returns 201,
stale history emits zero events, retained event history blocks rule deletion
with 409, and Sunday capture performs zero provider calls. Frontend
`npm run lint` and `npm run build` pass; the production route list includes
`/alerts`, `/calendar`, `/screener`, and `/watchlists`. `docker compose config
--quiet` passes. Per repository policy, no test files were changed and broad
test suites were not run.

**Next up:**
1. Add `TRADIER_TOKEN` under GitHub Settings > Secrets and variables > Actions,
   manually dispatch Daily market snapshot, and verify a bot data commit.
2. Push the local feature commit, merge the feature branch to `master`, and
   advance `streamlit-prod` intentionally.
3. Correct ownership of `data/ai_analysis.db` or migrate the root database into
   the canonical Compose data path.
4. Begin Phase 3 with Supabase Auth so watchlists and alerts gain user ownership.
