# Overhaul Progress Tracker

Cross-session log for the Unusual Whales–parity rebuild. The full plan lives in
[`UW_PARITY_PLAN.md`](UW_PARITY_PLAN.md). **Every work session should update this
file**: tick checkboxes, append a session-log entry, and record what the next
session should pick up.

## How to resume work

1. Read `docs/UW_PARITY_PLAN.md` (architecture, phases, endpoint/function mappings).
2. Read the Status board and the last Session log entry below.
3. Work on the "Next up" items; keep `pytest` green; update this file before pushing.
4. Active branch: `feature/multi-expiration-gex`.

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

### Session 10 - 2026-08-02 (multi-expiration GEX comparison)

- Added multi-expiration selection to the ticker GEX tab. A normal click keeps
  the existing single-expiration behavior; Ctrl-click or Cmd-click toggles
  comparison expirations. Touch devices get an explicit Compare mode.
- Selected GEX profiles load concurrently through the existing per-expiration
  endpoint. Partial failures leave successful profiles visible and report how
  many expirations were unavailable.
- Added a same-lane overlapping diverging chart with one color per expiration,
  a shared magnitude scale, zero baseline, spot-strike highlight,
  keyboard/touch labels, and a merged strike table. Expiration values are
  independent and never summed.
- Kept one highlighted primary expiration for max pain, vanna/charm, IV skew,
  and intraday delta projection so those existing calculations remain
  unambiguous.
- Created `feature/multi-expiration-gex` from the committed Phase 2 tip after
  preserving the `.env.example` workspace settings from the prior stash.
- Fixed the native economic calendar timeout: FRED's bulk release-date endpoint
  does not filter by calendar range and timed out while loading the full feed.
  The backend now fetches only the 15 curated release calendars concurrently;
  a live 14-day smoke returned six scheduled US macro releases.
- Fixed empty earnings results by merging Yahoo's forward quote calendar when
  its detailed earnings-history endpoint is stale. Date-only estimates are
  labeled time TBD; a live 14-day smoke returned AMD on 2026-08-04.
- Replaced the configured-ticker earnings limitation with Nasdaq's market-wide
  daily calendar. The page now supports Past or Upcoming 7/14/30-day windows;
  an empty ticker field means all companies and entered symbols are only a
  filter. A live Past 14-day smoke returned 1,325 earnings rows with reported
  EPS, forecast, surprise, session, company, and market-cap metadata.
- Added a reusable accessible top-bar info control with route-specific usage,
  methodology, freshness, and limitation guidance for Market, Flow, Screener,
  Watchlists, Alerts, News, Track Record, AI, Calendar, Congress, Binomial, and
  all ticker tabs.

**Verified:** `npm run lint`, `npm run build`, and `git diff --check` pass. No
test files changed.

### Session 11 - 2026-08-02 (flow clarity + custom snapshot symbols)

- Renamed Hottest Chains to Hottest Expiration Chains and documented that each
  row aggregates every call/put strike for one symbol and expiration. Rows now
  fetch and scroll to a symbol/expiration-filtered strike-level contract feed.
- Added INTC, MU, SNDK, PENG, SMH, SOXL, AVGO, and the explicitly requested
  MRVL symbol to the configured snapshot baseline.
- Replaced the fixed Watchlists symbol picker with comma/space ticker entry,
  removable chips, scheduled-symbol suggestions, and a 50-unique-symbol server
  capacity. Symbols receive strict path-safe syntax validation but are not
  silently corrected or assumed to be provider-valid.
- The backend snapshot universe is now the configured baseline plus unique
  symbols from persisted shared watchlists. Snapshot capture, cached Flow,
  screeners, and alerts use that same dynamic universe. Removing a custom
  symbol from every watchlist stops future captures without deleting history.
- Added Watchlists warnings for history accrual, alert skips caused by an
  unavailable symbol, disabled scheduling, and missing Tradier credentials.
  Increased backend capture pacing to six seconds between symbols so the
  expanded universe leaves more provider headroom.
- Documented that server-custom symbols require the backend scheduler; the
  GitHub Actions snapshot job still sees only the static config baseline.

**Verified:** frontend `npm run lint` and `npm run build`, scoped Python
`py_compile`, `docker compose config --quiet`, and `git diff --check` pass. No
test files changed or broad test suites run.

### Session 12 - 2026-08-03 (loading and empty-state clarity)

- Audited every frontend route for initial loading, successful emptiness,
  request errors, and controls that require an explicit user action.
- Calendar and Congress now explain which controls load immediately versus
  which ticker fields require Apply. Requests show loading rows and ellipsis
  counts instead of blank tables or false zero results, and stale rows are
  cleared when filters change.
- Screener tabs now clear prior-mode results, distinguish field loading from
  field failure, explain preset/threshold/custom execution behavior, and show
  separate pre-run, loading, and no-match result states.
- Flow shows initial chain/feed loading rows, labels its reload as a cached-data
  request, and gives selected-chain requests an isolated loading/error state.
- Watchlists, Alerts, News, and Track Record now distinguish initial loading
  from a valid empty workspace/feed. Alerts no longer report scheduler and
  integration settings as disabled before status loading finishes.
- Ticker pages now distinguish expiration loading, no expirations, and request
  failure. Gamma Gap no longer remains on Loading when a valid profile has no
  signal, and max-pain/history panels distinguish loading from unavailable.
- Added request identity guards where Strict Mode or superseded requests could
  otherwise clear loading early or overwrite newer data.

**Verified:** frontend `npm run lint`, `npm run build`, and `git diff --check`
pass. No test files changed.

### Session 13 - 2026-08-03 (mobile readability pass)

- Mobile pass scoped entirely below the `md` breakpoint, so the desktop shell
  renders identically:
  - `globals.css`: root font 14px -> 16px on phones, and inputs/selects/
    textareas pinned to 16px so Safari stops force-zooming the page on focus.
    Written as `@media (width < 48rem)` to match the breakpoint syntax
    Tailwind v4 emits; a `max-width: 767px` block is dropped by the compiler.
  - Phone tab bar rebuilt: four primary destinations (Market, Flow, Tickers,
    Screener) plus a More sheet holding the other eight. The previous bar put
    all twelve in one horizontal scroller, hiding half of them off-screen.
  - Raised the `text-[9px]`/`text-[10px]`/`text-[11px]` floors on phones only
    in `Tabs`, `StatTile`, `/alerts`, `/tools/binomial`.
- Recorded in `deploy/DEPLOY.md` that Hugging Face now requires a paid plan to
  create Docker Spaces, so the backend path described there is no longer free.
  No replacement host has been chosen.
- Documented `WORKSPACE_PIN` in `.env.example`.
- Review follow-ups on PR #42: primary tab links now close the More sheet
  (it previously stayed over the newly selected page, since `SidebarNav` is
  mounted by the persistent root layout), and the sheet now moves focus to
  its first entry on open, traps Tab while open, restores focus to the
  trigger on close, and keeps the backdrop out of the tab order.
- Fixed two stale `e2e/smoke.spec.ts` assertions, with the user's explicit
  approval for test work. PR #41 renamed the flow panels to "Hottest
  Expiration Chains" and "Strike-Level Contract Feed" without updating the
  test, leaving master red on `test (frontend)` before this branch existed.

**Verified:** `npx tsc --noEmit`, `npm run lint`, and `npm run build` (14
routes) pass. Measured in headless Chromium at iPhone 13 width: root and input
font-size both 16px, zero horizontal page overflow on `/market`, `/stock/SPY`,
`/flow`, `/screener`, `/news`, `/track-record`, and sub-12px text nodes down
from 17-41 per page to 8-9 (the rest are chart SVG labels, left alone).
Desktop at 1440px re-measured unchanged: root 14px, sidebar visible, phone tab
bar hidden. No test files changed and no broad test suites run.

**Next up:**
1. Verify the newly configured GitHub `TRADIER_TOKEN` with the next Daily market
   snapshot run and confirm the first bot data commit. `data/snapshots/` still
   holds only the seeded 2026-07-13 date, so IV rank, OI change, and historical
   GEX have no accumulated history yet.
2. Decide where the backend is hosted now that HF Spaces requires a paid plan.
3. Chart SVG labels render at roughly 5px on a phone because they scale with
   the viewBox; needs a separate pass with live data to avoid label collisions.
4. Begin Phase 3 with Supabase Auth so watchlists and alerts gain user
   ownership.

### Session 14 - 2026-08-03 (per-ticker proxy flow tab + flow data doc)

- Enabled the ticker page Flow tab (was a disabled "soon" placeholder). It
  renders the existing cached proxy feed filtered to the viewed symbol via
  `GET /api/flow/feed?ticker=X`, which the backend already supported. New
  presentational `components/flow/ProxyFeedTable.tsx` mirrors the /flow
  columns minus the redundant Symbol column; the tab hides the expiration
  selector (the feed spans all expirations), refetches on each tab entry
  (cache-only endpoint, no provider calls), and carries an explicit
  disclaimer that this is a snapshot-derived proxy, not a live trade tape.
  Loading, error, and empty states follow the Session 12 conventions; the
  empty state explains the snapshot-universe coverage rule.
- Added `docs/FLOW_DATA.md`: how the proxy feed is computed, what aggregates
  can and cannot reveal (direction, sweeps, blocks, per-trade premium,
  multi-leg, 0DTE), what paid OPRA data adds in Phase 4, and vendor/licensing
  notes. Linked from the Phase 4 bullet in `UW_PARITY_PLAN.md`.

**Verified:** `npx tsc --noEmit`, `npm run lint`, `npm run build` (14 routes)
pass. Headless Chromium against the production build with mocked API, desktop
and iPhone 13 widths: tab enabled without the soon badge, panel and
disclaimer render, status chips show, call and put rows render with correct
strikes, row count matches, expiration chips absent on the tab, no horizontal
overflow. Per repository policy, no test files were changed.

**Next up:** unchanged from Session 13, plus consider surfacing the hottest
chains ranking on the ticker Flow tab once real data needs emerge.

### Session 15 - 2026-08-03 (append-only intraday chain archive)

- Preserved the existing daily CSV/upsert behavior for all current consumers
  while adding immutable, timestamped Parquet captures for every successful
  backend scheduler or admin run.
- Added a versioned Arrow schema with UTC capture timing, shared run ID,
  per-ticker capture ID, market session, spot, provider/producer, requested
  expirations, deployment revision, and contract identifiers.
- Added atomic no-replace publication and a per-run JSON manifest describing
  expected tickers, successes, failures, timing, and complete/partial status.
  A ticker's daily file advances only after its intraday archive succeeds.
- Added an in-process lock so scheduled and manual captures cannot overlap in
  the current single-worker Compose deployment.
- Persisted the archive through the existing `./data:/app/data` volume at
  `data/intraday_snapshots/`, excluded it from Git and image builds, and left
  the GitHub Actions daily workflow unchanged.
- Added `docs/INTRADAY_DATA.md` covering data layout, provenance, retention,
  backup requirements, and the Tradier redistribution/licensing constraint.

**Verified:** scoped Python compilation, the eight existing snapshot tests, an
append/load Parquet smoke, `docker compose config --quiet`, and
`git diff --check` pass. No test files changed.

### Session 16 - 2026-08-03 (O'Neil CAN SLIM feature plan)

- Planning session only, no code changes. Wrote
  `docs/ONEIL_CANSLIM_PLAN.md`: feasibility tiering of the seven CAN SLIM
  criteria against the free data budget, UX design (market-direction panel on
  `/market` with follow-through-day and distribution-day states, per-ticker
  CAN SLIM report card, `oneil_leaders` screener preset, alert rule types,
  AI packet regime field), architecture split across `quant_analysis`
  analytics modules and backend adapters, four small phases, and the honesty
  requirements (configurable FTD thresholds, per-criterion unavailable
  states, no advice language).
- Key call: the "M" market-direction engine ships first and alone - pure
  OHLCV math, no new data dependencies, and FTD signals are loggable into the
  existing track-record outcome-scoring pattern.

**Next up:** implement Phase A of the plan (market-direction engine +
`GET /api/market/direction` + `/market` panel) once the plan is approved.

### Session 17 - 2026-08-03 (O'Neil market direction shipped, single PR)

Owner redirected the plan: one PR, algorithms as the heart, applied to the
indices that govern stock groups. Delivered end to end:

- `quant_analysis/analytics/market_direction.py` (pure, no I/O): rally-attempt
  / follow-through-day / distribution-day state machine with dated event log,
  EMA (SMA-seeded) and Wilder RSI, EMA touch edge detection, index-adapted
  CAN SLIM scorecard (C/A trend, N 52w-high proximity, S up/down volume, L RS
  vs SPY with universe rank, I accumulation-vs-distribution days, M benchmark
  state), narrative and signal-text builders. All thresholds configurable and
  surfaced, never hidden.
- Config: `market_direction` block in `config.yaml` - 16 index ETFs
  (SPY/QQQ/DIA/IWM broad; SMH/XLK/XLF/XLE/XLV/XLI/XLY/XLP/XLU/XLB/XBI/XLRE
  sectors), FTD 1.25% day-4+ on rising volume, distribution 0.2% decline on
  rising volume with 25-session window and 5% recovery expiry, pressure at 4,
  correction at 6 or 8% drawdown, EMA 20/50/200 with 0.4% touch band, RSI 14.
- Backend: `services_direction.py` (Tradier candles with yfinance fallback,
  cached 30 min; overview with breadth + benchmark bottom-durability checks;
  per-index detail with trimmed candles, aligned EMA series, event markers;
  signal evaluation on completed sessions only - intraday bars never persist),
  `direction_signals` table (unique symbol+type+date, insert-or-ignore),
  routers `/api/direction` + `/{symbol}` + `/signals`, scheduler job 16:20 ET
  weekdays plus an idempotent lazy evaluation on overview reads so signals
  accrue without the scheduler. Optional Discord/email delivery reuses the
  alert transports (extracted `post_discord_message`/`send_email_message`;
  gated by new `DIRECTION_ALERT_DISCORD`/`DIRECTION_ALERT_EMAIL`, in
  `.env.example`).
- Frontend: `/direction` page - benchmark hero (state pill, narrative, RSI
  buy/wait chip, breadth line, three-check bottom-durability card), grouped
  index card grid (state pill, RSI zone chip, 20/50/200 EMA above/below chips,
  seven-letter scorecard dots, met count), detail panel with an annotated
  candle + EMA + volume chart (FTD guide line, marker glyph key, EMA palette
  validated against the dark surface via the dataviz checks, collision-
  resolved line-end labels, crosshair tooltip incl. markers), full scorecard
  table, persisted signal feed, thresholds footnote. Sidebar Direction entry,
  ContextHelp section, market-page link card. Loading/error/empty states per
  Session 12 conventions; state clearing kept out of effects per the repo lint
  rule.
- Docs: revision note in `docs/ONEIL_CANSLIM_PLAN.md` (per-stock fundamental
  letters remain future work).
- Follow-up: each index carries a configurable `domain` tag naming the part
  of the market it rules (e.g. SMH "Chipmakers and semi equipment"), rendered
  full-width on the index cards and beside the benchmark hero title.

### Session 18 - 2026-08-03 (per-stock CAN SLIM layer: which stocks to buy)

Answers "which stocks are flagged to buy": the full seven-letter stock scan
at O'Neil's published thresholds, gated by the market-direction state.

- `quant_analysis/integrations/fundamentals.py`: yfinance fundamentals with
  per-field graceful degradation and a `missing` list (quarterly EPS growth
  via reported-EPS history with income-statement fallback, quarterly revenue,
  annual EPS growth averaged across fiscal years, ROE, float, institutional
  percent). Loss-to-profit bases report None rather than fake percentages.
- `quant_analysis/analytics/canslim.py`: C (quarterly EPS 25%+ YoY, sales as
  context), A (annual EPS 25%+/yr and ROE 17%+), N (52-week-high breakout on
  1.4x average volume; near-pivot is watch, and the qualitative "new" is
  declared not computable), S (up/down volume plus float), L (weighted
  12-month RS percentile, recent quarter double-weighted), I (13F ownership
  with over-owned and undiscovered bands), M (the follow-through-day state).
  Readiness ladder: buy_candidate (fresh breakout + gate open + score),
  near_pivot, wait_market (qualified but the market gate is closed - the
  O'Neil no-buys-in-a-correction rule made explicit), not_ready,
  insufficient_data. Every flag carries the 7-8% stop discipline.
- Backend `services_canslim.py`: scan universe = snapshot + watchlist symbols
  minus configured ETFs (config `canslim.exclude`), fundamentals cached 24h
  with pacing, scan cached 15 min; `GET /api/canslim` + `/api/canslim/{sym}`;
  fresh completed-session breakouts persist as `stock_breakout` rows in the
  shared direction signal feed (idempotent, delivered over the same
  Discord/email channels), evaluated by the 16:20 ET job and lazily on
  leaders-page reads.
- Frontend `/leaders`: market-gate banner (green/amber/red by state), ranked
  table (readiness badge, letter dots, met count, RS, qtr/annual EPS, ROE,
  off-high, institutional, breakout chip), full letter table + narrative for
  the selected stock, methodology and exclusions footer. Sidebar Leaders
  entry, help section, cross-link from the Direction page.

**Verified:** TestClient smoke with mocked bars and fundamentals: NVDA
(strong growth + fresh 3x-volume breakout) flags buy_candidate 7/7 with a
persisted signal carrying the stop price; TSLA (negative growth) rejected
with C not_met; missing fundamentals score honestly (3/4 technicals only);
ETF exclusion, 404s, and signal idempotency all pass. Backend pytest 20/21
(known pre-existing sandbox failure only). Frontend tsc/lint/build clean (16
routes); headless screenshots at 1280px/390px: zero overflow, no console
errors. Live Yahoo fundamentals could not be exercised here (proxy blocks
Yahoo); every field degrades to a labeled unavailable state by design.

### Session 19 - 2026-08-03 (entry discipline + realized signal outcomes)

Closes the gap the owner identified: a confirmed uptrend was being read as a
buy signal, so a user could enter extended and be stopped out by an ordinary
pullback while the system was technically right.

- **Entry vs regime (`entry_assessment`)**: indices now report buyable
  (within 5% of the follow-through close), pullback entry (testing a rising
  20/50-day EMA), extended (past the chase limit), wait (rally attempt), or
  no entry (correction). Rendered as an `EntryChip` on every index card and
  the benchmark hero, with the reasoning inline.
- **Stock entry discipline**: `evaluate_stock_canslim` returns pivot, buy
  limit (pivot + 5%), stop price, and extension percent; a qualified stock
  past the chase limit becomes readiness `extended` ("Extended - do not
  chase") instead of a buy flag. `/leaders` gained an Entry vs pivot column.
  Fixed a real flaw found by the smoke: `detect_breakout` returned the most
  recent new high, which re-anchored the pivot every session during a run-up
  and reported a stock 12.6% extended as sitting at its pivot. It now walks
  back to the origin of the advance, so extension and stop are measured from
  the price O'Neil would have bought. Stock breakout signals additionally
  require `sessions_ago == 0` so a mid-run stock cannot re-alert, and a name
  stretched above its 20-day EMA can no longer read as "near pivot".
- **Signal text reframed**: the follow-through message now says it is
  permission to start buying rather than a signal to buy the index, names
  the pilot-position practice, and gives the rally-attempt low as the
  invalidation level while stating that index exposure is not managed by the
  7-8% stock stop. EMA touches inside an uptrend are labeled the standard
  second-chance entry.
- **Realized outcomes**: `evaluate_direction_signal_outcome` /
  `summarize_direction_outcomes` in `track_record.py` score each logged
  follow-through and breakout against its own invalidation level, plus max
  gain, max drawdown, and whether a mechanical 8% stop would have fired.
  New `GET /api/direction/outcomes` and an `OutcomesPanel` on
  `/track-record` (renamed from the gamma-gap-only page) publish hold rate,
  average drawdown, and stop-fire rate, so the whipsaw cost becomes a
  measured number as history accrues.

**Verified:** dedicated smokes for all four items - entry states across
buyable/pullback/extended/wait/no-entry, reframed FTD and EMA-touch text,
persisted-then-scored outcomes (18 signals: 17 held, 1 pending, avg drawdown
-0.4%, stop never fired), and the pivot-anchor fix (NVDA buyable at 0%, AMD
extended at +12.6% from the same pivot). Prior smokes re-run green; backend
pytest 20/21 (known pre-existing failure); frontend tsc/lint/build clean;
screenshots of `/direction`, `/leaders`, `/track-record` at 1280px and
`/leaders` at 390px with zero overflow and no console errors.

### Session 20 - 2026-08-03 (market-wide candidate universe)

The CAN SLIM scan only saw watchlist symbols, so leaders in sections the
user did not already follow were invisible. Candidates now come from the
whole tracked market.

- **Universe from ETF holdings**: `fetch_etf_holdings` reads top holdings
  per ETF (yfinance `funds_data`, cached 24h) for every `market_direction`
  index; unreachable providers fall back to `canslim.fallback_constituents`
  in `config.yaml` (a hand-maintained, editable snapshot) and the UI states
  which source is in use. Watchlist symbols are unioned in. ETFs themselves
  are excluded as candidates. 125 candidates across 15 groups in the smoke,
  versus 15 before.
- **Leading-group attribution**: ETFs are ranked by relative strength, and a
  stock held by several is credited to its strongest sector group (a sector
  always beats the broad market), carrying that group's rank - O'Neil bought
  leaders of leading groups. Universe capping keeps leading-group names
  first.
- **Two-stage funnel**: bars for the whole universe are fetched concurrently
  (`fetch_workers`, cached per symbol), ranked technically (RS + breakout +
  pivot proximity), and company fundamentals are fetched only for the top
  `fundamentals_top_n` (25). Remaining rows are labeled technical-only with
  "Not fetched" reasons rather than implying data was missing. `_cached_scan`
  now backs both the leaders endpoint and signal evaluation, removing a
  duplicate full scan per request.
- **UI**: group filter chips with counts, an Actionable-only filter, a Group
  column with the group's RS rank, dot markers for technical-only columns,
  and a coverage line reporting scanned/universe counts, fundamentals depth,
  holdings source, and capacity drops.

**Verified:** new universe smoke (125 candidates, 15 groups, fundamentals
capped at 25 and always covering the breakout names, NVDA credited to
Semiconductors rather than SPY/QQQ, ETFs never candidates, technical-only
rows carrying the "Not fetched" reason, detail endpoint keeping attribution);
prior entry/outcome smokes still green; backend pytest 20/21 (known
pre-existing failure); frontend tsc/lint/build clean; `/leaders` screenshot
at 1280px and 390px with zero overflow and no console errors. Live ETF
holdings could not be exercised here (sandbox proxy blocks Yahoo), so the
smoke ran through the configured-fallback path and the response correctly
reported `holdings_source: configured`.

**Verified:** synthetic-series unit smoke (correction -> rally day 1 -> FTD on
day 5 -> confirmed uptrend, truncations land in each intermediate state; EMA
and RSI seed boundaries checked); FastAPI TestClient smoke over the real app
(overview 16 indices, detail markers, 404 on unknown symbol, signal insert
idempotency: 16 inserted once, rerun 0); backend pytest 20/21 (the flow
snapshot-date test also fails on the unmodified base tree in this sandbox -
pre-existing, environment-dependent); frontend `tsc`, lint, and build clean
(15 routes incl. `/direction`); headless Chromium screenshots of the
production build at 1280px and 390px against mocked APIs - zero horizontal
overflow, no console errors, chart/labels eyeballed after fixing an EMA
label collision. Live provider data could not be exercised here: the sandbox
proxy blocks Yahoo (CONNECT 403) and no Tradier token is present; the
endpoints degrade exactly as designed (200 with symbols listed unavailable).

**Next up:**
1. Exercise `/direction` against live data on the self-host box (scheduler on)
   and confirm the first persisted real signals.
2. Track-record integration: score follow-through-day outcomes like gamma-gap
   signals once real FTDs accrue.
3. Per-stock CAN SLIM fundamentals (yfinance C/A/ROE/13F) per the plan doc.
