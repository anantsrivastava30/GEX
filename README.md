# GEX — Options Analytics Dashboard

An options analytics platform for dealer gamma exposure (GEX), delta exposure, open interest, volume, macro context, and AI trade narratives. The FastAPI and Next.js stack is the active rebuild; the original Streamlit dashboard is frozen under `legacy/` for parity checks during the transition.

## Features

- **Overview Metrics** — net GEX by strike, gamma flip zones, peak-GEX "magnet" strikes, IV skew (put − call), put/call ratios, liquidity snapshot, and a three-pillar "holistic trade posture" signal (dealer magnet, flow & liquidity, macro & sentiment).
- **Options Positioning** — strike- and expiration-lens drilldowns of gamma/delta exposure, OI, and volume, split by calls/puts.
- **Gamma Gap Radar** — scans a watchlist for positive-gamma magnets near spot and scores the likelihood of a gap fill; results are logged for later verification.
- **Binomial Tree** — CRR tree pricing calibrated with live rates (10Y yield) and market implied vol.
- **Market Sentiment** — futures board (ES/NQ/YM/RTY/CL/GC), VIX, 10Y yield, and auction bid-to-cover.
- **Market News** — RSS headlines filtered to macro/options-relevant topics.
- **AI Analysis** — builds a data packet (positioning, skew, ratios, spikes, snapshot, headlines), estimates token cost, and (after PIN confirmation) asks an OpenAI model for trade ideas across 1–2 week, 1–2 month, and 6–12 month horizons. Analyses are persisted to Supabase with a SQLite fallback.

## Project Structure

```
legacy/app.py                       # Frozen Streamlit entrypoint
backend/                             # FastAPI service
frontend/                            # Next.js frontend
config.yaml                         # API endpoints, RSS feeds, topics, DB names
quant_analysis/
├── config.py                       # Loads config.yaml
├── analytics/visualization.py      # GEX math, gamma-gap scoring, plots, binomial tree
├── integrations/tradier.py         # Tradier REST API wrapper
├── services/market_data.py         # Chains, quotes, ratios, spikes, macro data, RSS
├── services/ai_analysis.py         # OpenAI model discovery, prompt building, querying
├── storage/db.py                   # Supabase persistence with SQLite fallback
└── scripts/sync.py                 # One-off SQLite → Supabase migration
tests/                              # Unit tests (pytest)
```

## Data Sources

| Source | Used for | Credential |
|---|---|---|
| [Tradier](https://tradier.com/) | Option chains, greeks, quotes, history | `TRADIER_TOKEN` |
| Yahoo Finance (`yfinance`) | Futures, VIX, Treasury yields | none |
| [FRED](https://fred.stlouisfed.org/) | Auction bid-to-cover | `FRED_API_KEY` |
| RSS feeds (config.yaml) | Curated market news | none |
| OpenAI | AI trade analysis | `OPENAI_API_KEY` |
| Supabase (optional) | Persisting analyses | `SUPABASE_URL`, `SUPABASE_KEY` |

## Setup

1. Install dependencies (Python **3.10+**):
   ```bash
   pip install -r requirements.txt
   ```
2. Configure secrets in `.streamlit/secrets.toml` (never commit this file):
   ```toml
   TRADIER_TOKEN = "..."
   OPENAI_API_KEY = "..."       # required only for the AI tab
   AI_PIN = "1234"              # confirmation PIN before sending AI requests
   FRED_API_KEY = "..."         # optional
   SUPABASE_URL = "..."         # optional — SQLite fallback used if absent
   SUPABASE_KEY = "..."
   ```

## Running the legacy dashboard

```bash
streamlit run legacy/app.py
```

Run this command from the repository root so the frozen app can import the shared
`quant_analysis` package. New feature work belongs in the FastAPI and Next.js stack.

## Running the new stack

```bash
uvicorn backend.app.main:app --reload --port 8000
cd frontend && npm install && npm run dev
```

The frontend is available at `http://localhost:3000` and proxies `/api/*` to the
backend. Deployment instructions, including the separate legacy release branch,
are in [deploy/DEPLOY.md](deploy/DEPLOY.md).

## Testing

```bash
pytest -q tests
pytest -q backend/tests
cd frontend && npm run build && npm run test:e2e
```

CI (`.github/workflows/ci.yml`) selects analytics, backend, and frontend checks
from changed paths. Frontend smoke tests intercept every API request and do not
require Tradier, OpenAI, or any other credentials.

## Daily Data Snapshots (own-data history)

The project accumulates its own market history so that history-dependent
features (IV rank, OI-change tracking, historical GEX, gamma-gap track record)
don't require paid data APIs:

```bash
python -m quant_analysis.scripts.snapshot            # config.yaml watchlist
python -m quant_analysis.scripts.snapshot --tickers SPY,QQQ --expirations 4
```

Each run writes per-contract chain snapshots to `data/snapshots/YYYY-MM-DD/{TICKER}.csv.gz`
plus derived per-day metrics (ATM IV, put/call ratios, net GEX, gamma-gap score)
to `data/snapshots/daily_metrics.csv`. `.github/workflows/snapshot.yml` runs this
every weekday after the close and commits the results — add a `TRADIER_TOKEN`
repository secret to enable it. Consumers live in `quant_analysis/storage/snapshots.py`
(`compute_oi_change`, `compute_iv_rank`, `load_contract_history`).

## Roadmap

The project is being rebuilt into an Unusual Whales–style platform (FastAPI +
Next.js). See [docs/UW_PARITY_PLAN.md](docs/UW_PARITY_PLAN.md) for the
architecture and phases, and [docs/PROGRESS.md](docs/PROGRESS.md) for
cross-session progress tracking. The original monetization strategy is in
[docs/MONETIZATION_AND_EXPANSION_PLAN.md](docs/MONETIZATION_AND_EXPANSION_PLAN.md).

## Disclaimer

This tool is for research and education. Nothing it produces — including AI-generated trade ideas — is financial advice. Options involve substantial risk.
