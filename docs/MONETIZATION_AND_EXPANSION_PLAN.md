# GEX: Expansion & Monetization Plan

This document lays out a realistic path from the current single-user Streamlit dashboard to a revenue-generating product, in phases that each build on what already exists in this repository.

## 1. What we have today (the asset)

- A working **dealer-positioning analytics engine**: net GEX by strike, gamma flip detection, magnet/gap-fill scoring (`compute_gamma_gap_metrics`), unusual volume/OI spike detection, IV skew, put/call ratios.
- A **multi-source data pipeline**: Tradier (chains + greeks), yfinance (macro), FRED, RSS news.
- An **AI analysis layer** that packages positioning + macro context into a prompt and returns structured trade ideas, with persistence (Supabase/SQLite) and token accounting already in place.
- A **signal-history log** (`gamma_gap_analysis` table) — the seed of a track record, which is the single most valuable marketing asset a trading-signals product can have.

The comparable market is real and paying: SpotGamma ($40–$150/mo), Menthor Q, Tier1Alpha, Unusual Whales ($48/mo), GEXBot ($20–$60/mo). Retail options flow is a proven subscription niche; the moat is signal quality, latency, and trust (published track record).

## 2. Product thesis

**Sell "dealer positioning, explained and actionable" to retail options traders**, at a lower price than SpotGamma, differentiated by:

1. **Gamma Gap Radar as the hero feature** — a scored, ranked "which tickers are likely to pin/mean-revert today" list is more actionable than raw GEX charts, and nobody in the budget tier ships it well.
2. **AI narrative layer** — turning positioning data into plain-English daily briefs is cheap for us (the prompt pipeline exists) and hugely valued by less technical traders.
3. **Verifiable track record** — we already log every scan; publishing hit-rates builds the trust competitors buy with ads.

## 3. Phased roadmap

### Phase 0 (weeks 1–4): Hardening — prerequisite for charging anyone
- Test coverage of the analytics core (started in this PR), CI on every push (fixed in this PR).
- Move off `st.secrets` sprawl into a single config/credentials layer; add caching (`st.cache_data`) around Tradier calls to cut latency and rate-limit pressure.
- Background **snapshot jobs** (cron/Cloud Run) that persist GEX profiles per ticker every 15–30 min, so the product has history and doesn't recompute everything per page load. The `gamma_gap_analysis` table schema is already designed for this.
- Track outcomes: for every logged magnet score, record whether spot tagged the magnet before expiry. This is the **hit-rate dataset** the whole business stands on.

### Phase 1 (months 1–3): Free audience builder
- **Daily "Gamma Brief"** — automated post (email + X/Twitter + Substack) for SPY/QQQ/NVDA/TSLA: magnet levels, flip zones, gap-fill scores, one-paragraph AI narrative. Fully automatable with the existing `describe_gamma_gap` + AI pipeline.
- Public **free tier**: 3 tickers, end-of-day data, delayed scans.
- Goal: 1–2k email subscribers. This costs almost nothing (Tradier sandbox/base plan + one small VM) and validates demand before any billing work.

### Phase 2 (months 3–6): Paid subscriptions
- **Pro tier ($29–39/mo)**: full watchlist, intraday refresh, full Gamma Gap Radar with alerts (Discord/Telegram/email when a score crosses a threshold or a flip zone breaks), AI daily brief per ticker, historical GEX charts.
- **Premium tier ($79–99/mo)**: SPX/0DTE focus, intraday snapshot history, custom watchlists, on-demand AI deep dives, early access to new signals.
- Implementation: Stripe + simple auth. Streamlit supports this at small scale (`streamlit-authenticator` or an Auth0/Supabase-auth wrapper); plan a Next.js/FastAPI split only when MRR justifies it (see §4).
- Publish the **track record page** (hit rate of gap-fill scores by score bucket) — the conversion engine.
- Realistic target: 100–300 paying users → **$3k–$15k MRR**.

### Phase 3 (months 6–12): Expand surface area
- **API product** ($99–299/mo): the GEX/gap-score data as JSON for algo traders — the `TradierAPI`-style wrapper inverted. Low support cost, high willingness to pay.
- **Discord community** (included in Pro) — retention lever; alerts post into it, traders discuss levels. Communities cut churn dramatically in this niche.
- **More signals** from data already fetched: vanna/charm exposure (greeks are in the chain payload), IV rank/percentile, term-structure slope (function exists, currently disabled), put-wall/call-wall levels, 0DTE intraday flip tracking.
- **Broker integrations** (read-only first): position-aware briefs — "you hold NVDA calls; dealer positioning turned adverse."

### Phase 4 (year 2): B2B and scale
- White-label the analytics for RIAs/prop-trading education firms.
- Sell historical GEX snapshot datasets (institutional quant teams buy positioning history).
- Newsletter sponsorships once the free list is >10k.

## 4. Technical evolution required

| Stage | Architecture |
|---|---|
| Now | Streamlit monolith, per-session data fetch |
| Phase 1 | + scheduled snapshot worker writing to Supabase/Postgres; Streamlit reads from DB |
| Phase 2 | + auth/billing (Stripe), alert dispatcher (worker consuming score events) |
| Phase 3 | Split: FastAPI backend (also serves the paid API) + web frontend; Streamlit remains the internal research tool |

Key cost lines: Tradier market-data agreement for redistribution (must upgrade from personal API terms — **do this before charging**), OpenAI usage (~$0.01–0.05 per AI brief with gpt-4o-mini-class models; batch and cache), hosting (<$100/mo until Phase 3).

## 5. Legal & compliance (non-negotiable before charging)

- **Data redistribution**: Tradier's retail API terms don't cover republishing derived quotes to third parties — negotiate a vendor/redistribution agreement or switch to a data vendor licensed for it (Polygon, databento, ThetaData).
- **Not investment advice**: publisher's exemption generally covers impersonal, subscription-based commentary (no individualized advice). Keep all output impersonal, add prominent disclaimers (added to README in this PR), consult a securities attorney before launch, and never auto-execute trades for users.
- **AI output**: label AI-generated content clearly; keep the human-readable data packet visible (already the pattern in the AI tab).

## 6. What to do first (concrete next steps)

1. Ship the snapshot worker + outcome tracking (turns the existing log into a track record).
2. Automate the daily Gamma Brief for 4 tickers and start the free list.
3. Fix data licensing for redistribution.
4. Add Stripe + auth behind a `Pro` flag on the Gamma Gap Radar and alerts.
5. Publish the track record and launch Pro at $29/mo.

The sequencing principle: **audience → trust (track record) → paywall → API/B2B**. Each phase is funded by the previous one and reuses the code that already exists in this repo.
