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
- [x] P1.3 Frontend shell: Next.js 16 + Tailwind v4 in `frontend/`, dark theme tokens (`globals.css`), `SidebarNav`/`TopBar` w/ ticker search, typed API client (`src/lib/api.ts`), `/api` rewrite to backend :8000. First pages live: `/market` (VIX/yields/futures), `/stock/[symbol]` (quote, expiration chips, net-GEX table, gamma-gap signal card, interpretation), `/news`, `/flow` placeholder. *Still to add: shadcn/ui + TanStack Table for dense tables, lightweight-charts + ECharts chart components (load the `dataviz` skill first), remaining pages.*
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
