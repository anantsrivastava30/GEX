# GEX → Unusual Whales–Class Platform: Rebuild & Overhaul Plan

## Context

GEX today is a ~5,600-LOC single-user Streamlit dashboard (`app.py`, 2,648 lines, 8 tabs) with genuinely strong dealer-positioning analytics (net GEX, gamma-gap scoring engine with logged history, greeks exposures, IV skew, binomial pricing, AI trade narratives) but no product shell: no multi-page navigation, no dense real-time tables, no auth, no billing, no caching, no background jobs, and no trade-level data. The goal is to overhaul it into an Unusual Whales–style product — same look (dark, left-sidebar nav, dense tables, TradingView-style charts), same page structure, and as many of the same features as free/cheap data allows — while positioning it to be **better than UW where UW is weak** (interpretation, computed dealer analytics, verifiable signal track record). End state: a presentable, advertisable, monetizable platform.

**User decisions (fixed):**
1. **Full rebuild: FastAPI backend + Next.js/React frontend** (Streamlit kept frozen as legacy during transition, deleted at Phase 3).
2. **Data budget: free/cheap only for now** — Tradier + yfinance + FRED + free public feeds. Real OPRA flow data slots in later behind a provider interface.
3. **Priority: look & feel first** — new UW-style shell + port existing features before net-new features.
4. **Own our data history** — start capturing daily snapshots (chains, OI, IV, GEX profiles, gamma-gap scores) to git-versioned local storage immediately, so history-dependent features (IV rank, OI change, historical GEX, signal track record) run off our own accumulated dataset instead of paid APIs. Free sources only for now.

**Cross-session tracking:** progress across work sessions is logged in [`docs/PROGRESS.md`](PROGRESS.md) — update it at the end of every session.

---

## Unusual Whales: what we're matching

**Their structure:** left sidebar nav → Flow Feed (filterable live trade tape: sweeps/blocks, premium, filters by expiry/OI/volume/sector/trade-type), Market Tide/Net Flow, Hottest Chains, 0DTE dashboards, Options Screener (presets like "Unusually Bullish" + custom filters), per-Ticker pages (overview, charting, **GEX/DEX/Vanna/Charm strike charts**, OI/volume, max pain, IV term structure), SPX "Periscope" dealer-exposure page, Dark Pool feed, Congress/insider trades, earnings/economic calendars, news, alerts (in-app/Discord), watchlists, custom dashboards, portfolio, API, "Mr. Whale" AI. Pricing: free delayed tier → ~$48/mo standard → ~$99/mo Retail Pro.

**Their weaknesses (our wedge):**
- **Firehose without interpretation** — widely criticized as overwhelming; raw data, weak guidance.
- **Weak computed dealer analytics beyond basic GEX** (no gap-fill scoring, no posture synthesis, limited vol analytics vs SpotGamma).
- **No verifiable signal track record.**

**How we beat them, not just copy them:**
1. **Interpretation as a first-class layer** — every chart ships with the plain-English read (`interpret_net_gex`, `describe_gamma_gap`, three-pillar posture) that UW makes users derive themselves.
2. **Gamma Gap Radar as hero feature** — a scored, logged gap-fill signal UW has no equivalent of; publish the hit-rate as a public **track-record page** (we already persist every scan in `gamma_gap_analysis`).
3. **AI narrative built-in** (existing OpenAI pipeline) — curated trade theses, not a chatbot bolt-on.
4. **Curated, not firehose** — UW's #1 complaint is cognitive overload; we default to opinionated dashboards with drill-down, priced under UW ($29–39 vs $48).

---

## Architecture: monorepo layout

Restructure in place on branch `claude/unusual-whales-parity-fz318a`. `quant_analysis/` stays as the shared analytics library (its tests keep passing untouched). `app.py` moves to `legacy/app.py`, frozen for parity checks, deleted in Phase 3.

```
GEX/
├── quant_analysis/          # UNCHANGED analytics lib (all compute_* fns are UI-free & portable)
├── legacy/app.py            # frozen Streamlit app for numeric parity checks
├── backend/
│   ├── pyproject.toml       # fastapi, uvicorn, pydantic v2, apscheduler, cachetools, httpx, tenacity
│   ├── app/
│   │   ├── main.py          # app factory, CORS, routers, scheduler startup
│   │   ├── config.py        # pydantic-settings wrapping root config.yaml + env
│   │   ├── deps.py          # shared TradierAPI singleton + tenacity retry wrapper
│   │   ├── cache.py         # get_or_compute(key, ttl, fn) over cachetools.TTLCache (Redis-swappable)
│   │   ├── ratelimit.py     # token bucket (~120 req/min Tradier), jobs get low-priority budget
│   │   ├── schemas/         # pydantic models: ticker, gex, screener, market, news, ai, calendar, congress
│   │   ├── routers/         # one per domain (mapping below)
│   │   ├── services/        # adapters: call quant_analysis fns → shape into schemas
│   │   ├── jobs/            # APScheduler: snapshot_job (daily OI/GEX/IV), gamma_gap_job (30-min scoring)
│   │   └── storage/models.py # new tables: chain_snapshots, oi_history, iv_history, watchlists, alerts
│   └── tests/               # pytest + httpx AsyncClient, mocked Tradier fixtures
├── frontend/
│   ├── package.json         # next 14+, tailwind, shadcn/ui, @tanstack/react-table + react-query, lightweight-charts, echarts
│   ├── src/app/             # App Router pages (route map below)
│   ├── src/components/      # layout/ (SidebarNav, TopBar, TickerSearch), charts/, tables/DataTable, ui/
│   ├── src/lib/             # api.ts typed client, format.ts ($, %, compact numbers)
│   ├── src/styles/theme.css # dark tokens
│   └── e2e/                 # Playwright smoke tests
├── docker-compose.yml       # backend :8000 + frontend :3000 dev parity
└── docs/, tests/, config.yaml, setup.py   # existing, unchanged
```

**Cache TTLs:** quotes 15s · chains 60s · expirations 24h · macro 5min · news 10min · congress/calendar 6h · candles 15min.

### Backend endpoints → existing function reuse

| Endpoint | Reuses (quant_analysis) |
|---|---|
| `GET /api/ticker/{sym}/snapshot` | `get_stock_quote`, `get_price_stats`, `get_liquidity_metrics` (market_data.py) |
| `GET /api/ticker/{sym}/candles` | `TradierAPI.history`; intraday via `get_intraday_prices_with_prev_close` (visualization.py:514) |
| `GET /api/ticker/{sym}/gex` | `compute_net_gamma_exposure` + `interpret_net_gex` (visualization.py:61,18) |
| `GET /api/ticker/{sym}/greeks` | `compute_greek_exposures` (market_data.py:292) |
| `GET /api/ticker/{sym}/gamma-gap` (+ `/history`) | `compute_gamma_gap_metrics`, `describe_gamma_gap`; `load_gamma_gap_history` (db.py:258) |
| `GET /api/ticker/{sym}/skew` | `compute_iv_skew`, `compute_risk_reversal`, `compute_butterfly_skew` |
| `GET /api/ticker/{sym}/ratios` | `compute_put_call_ratios` |
| `GET /api/ticker/{sym}/binomial-tree` | `generate_binomial_tree` (visualization.py:648) — JSON node grid, render client-side |
| `GET /api/flow/unusual` | `compute_unusual_spikes` (market_data.py:241) across watchlist; +OI-change in P2 |
| `GET /api/market/overview` | `get_market_snapshot`, `get_vix_info`, `get_futures_quotes`, `get_bond_yield_info`, `get_bid_to_cover` |
| `POST /api/ai/analyze` | `services/ai_analysis.py` pipeline, `save_analysis` — PIN-gated now, auth-gated in P3 |

DataFrames convert at the service boundary into typed pydantic models. Plotly/Streamlit `plot_*`/render functions are NOT ported — the frontend renders from JSON.

### Frontend: UW-style shell & routes

**Look:** background `#0b0e14`, surface `#11151f`, border `#1e2433`, text `#e6e9f0` / muted `#8b93a7`, calls green `#22c55e`, puts red `#ef4444`, accent `#6366f1`, `tabular-nums` monospace numerals in all tables. Fixed collapsible left `SidebarNav`, `TopBar` with global ticker-search autocomplete, watchlist strip, market-status pill. Port the *intent* of `inject_global_styles()` (app.py:455-833), not the CSS.

**Chart stack:** lightweight-charts for candles (native TradingView look), ECharts for strike-profile bars / term structure / gauges / binomial tree (canvas perf on many-bar charts, native dark theme), TanStack Table for dense sortable/filterable tables. Load the `dataviz` skill before building chart components.

| Route | Content (source tab) |
|---|---|
| `/` → `/market` | Market overview: indices, VIX, futures, yields, 3-pillar posture (Overview + Sentiment tabs) |
| `/flow` | Proxy flow feed: unusual vol/OI anomalies + hottest chains, UW-style filter chips |
| `/stock/[symbol]` | Ticker overview: candles, quote stats, liquidity, P/C ratios |
| `/stock/[symbol]/gex` | Net GEX by strike, greeks exposures, gamma-gap score + history; intraday delta projection (resurrects dead `plot_price_and_delta_projection` data fns) |
| `/stock/[symbol]/vol` | IV skew, risk reversal, butterfly; +term structure & IV rank (P2) |
| `/stock/[symbol]/oi` | (P2) OI/volume heatmap, OI change, max pain |
| `/tools/binomial` | Binomial tree explorer |
| `/screener` | (P2) presets + custom filters |
| `/calendar` | Earnings/econ calendar (P1 iframe parity → P2 native) |
| `/congress` | (P2) Senate/House disclosure trades |
| `/news` | RSS news feed |
| `/ai` | AI analysis |
| `/watchlists`, `/alerts` | (P2) |
| `/pricing`, `/login`, `/settings` | (P3) |

---

## Phases

### Phase 1 — UW-style shell + full port (self-contained; the "look & feel" milestone)
1. **Backend skeleton**: scaffold `backend/`, infra (`cache.py`, `ratelimit.py`, `deps.py` w/ tenacity retry), routers `ticker`/`exposure`/`market`/`news`/`ai` wired per the mapping table; pytest+httpx tests with mocked Tradier fixtures (reuse patterns from `tests/test_market_data.py`).
2. **Snapshot jobs start day one**: APScheduler jobs writing daily chain/OI/ATM-IV snapshots + 30-min gamma-gap scores to new tables — Phase 2's IV rank, OI-change, and historical GEX charts need this history accruing ASAP.
3. **Frontend shell**: Next.js scaffold, theme tokens, SidebarNav/TopBar/TickerSearch, typed API client, DataTable + chart wrappers.
4. **Page ports** (one PR each): `/market`, `/stock/[symbol]` + `/gex` + `/vol`, `/news`, `/ai`, `/tools/binomial`, `/flow` (basic), `/calendar` (iframe).
5. **Legacy freeze**: move `app.py` → `legacy/app.py`; add `docker-compose.yml`; extend CI (path-filtered matrix: quant_analysis pytest + backend pytest + frontend build/Playwright); update README.

### Phase 2 — Close the UW feature gap (free data)
- **Vanna/Charm exposure** (UW headline feature; confirmed derivable): Tradier greeks lack vanna/charm but supply `mid_iv` — new `quant_analysis/analytics/greeks.py` computes Black-Scholes vanna/charm (r from yfinance `^IRX`; fall back to `smv_vol` when `mid_iv` null on illiquid strikes). Golden-value unit tests.
- **Max pain**: pure function over chain OI — trivial from existing data.
- **IV rank/percentile + historical GEX charts** from accrued snapshots ("rank since inception" labeling until history matures).
- **Term structure**: fix and re-enable `compute_term_structure_slope` (market_data.py:563, currently disabled).
- **Full proxy flow feed + Hottest Chains**: rank by vol/OI spike + OI Δ (snapshot diffs) + IV move; UW-style filter chips. Cap scan universe to ~50–100 liquid tickers, driven from cached chains only.
- **Congress trades**: daily job scraping Senate eFD / House Clerk feeds (or free `senate-stock-watcher` mirrors); tolerant parsing + staleness flags.
- **Native earnings/econ calendar**: yfinance earnings dates + FRED release calendar.
- **Screener** (presets: "Unusually Bullish", "High Vol/OI", "Gamma Squeeze Candidates" + custom filters), **server-persisted watchlists**, **alerts** (in-app + Discord webhook + email; alert-evaluation job).

### Phase 3 — Monetization
- Supabase Auth (project already exists) → FastAPI JWT dependency + Next.js middleware; replace AI PIN with entitlements.
- Stripe Checkout + webhooks (`routers/billing.py`); tiers: **Free** (delayed, 3 tickers) / **Pro $29–39** (full watchlist, intraday, alerts, history) / **Premium $79–99** (SPX/0DTE, API) — undercutting UW's $48/$99.
- Marketing landing page + `/pricing` + public **track-record page** (gamma-gap hit-rate from logged history — the differentiator UW can't match).
- **Build the data-provider interface at the start of this phase** and migrate off Tradier retail (Polygon/ThetaData) — Tradier's retail ToS prohibits redistribution, hard blocker to charging.
- Delete `legacy/app.py`.

### Phase 4 — Paid data & scale
- Real trade-tape Flow Feed via the provider interface (OPRA feed), sweep/block detection, Market Tide, 0DTE dashboards, SPX Periscope-style dealer exposure page.
- Public API product with keys + usage metering; responsive/mobile polish.

---

## Verification

- **Every phase**: existing `pytest` suite for `quant_analysis` stays green; `backend/tests` via httpx `AsyncClient` with mocked Tradier fixtures; `npm run build` + Playwright smoke (each route renders, one table + one chart visible).
- **Phase 1 parity check**: run `legacy/app.py` and new stack side-by-side for SPY — GEX profile, gamma-gap score, P/C ratios must match exactly (same functions; any drift = serialization bug).
- **Phase 2**: golden-value tests for vanna/charm/max pain; snapshot-job integration test on SQLite.
- **Phase 3**: Stripe test-mode webhook tests; Playwright auth flows.
- **Dev run**: `docker compose up`, or `uvicorn app.main:app --reload` (:8000) + `npm run dev` (:3000, `/api` proxied).

## Key risks
- **Tradier ToS** blocks redistribution → provider migration is a Phase 3 entry gate, not an afterthought.
- **Snapshot data debt** — IV rank ideally wants ~1yr history; mitigated by starting jobs in Phase 1 and honest labeling.
- **Rate limits** — screener/flow scans multiply Tradier calls; token bucket + capped universe + cache-only scans.
- **Congress feeds are scrape-fragile** — isolate in `congress_service.py` with staleness flags.

## Critical existing files
`quant_analysis/services/market_data.py` · `quant_analysis/analytics/visualization.py` · `quant_analysis/integrations/tradier.py` · `quant_analysis/storage/db.py` · `quant_analysis/services/ai_analysis.py` · `app.py` · `docs/MONETIZATION_AND_EXPANSION_PLAN.md` (update to reference this plan)
