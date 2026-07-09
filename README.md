# GEX — Options Analytics Dashboard

A Streamlit dashboard for options traders that visualizes **dealer gamma exposure (GEX)**, delta exposure, open interest, and volume across strikes and expirations, and layers macro context (VIX, Treasury yields, futures, curated news) on top. An optional AI tab sends a structured market snapshot to OpenAI models to generate directional option trade ideas.

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
app.py                              # Streamlit entrypoint (run this)
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

## Running

```bash
streamlit run app.py
```

## Testing

```bash
pytest
```

CI (`.github/workflows/ci.yml`) runs the test suite on every push and pull request to `master`.

## Roadmap

See [docs/MONETIZATION_AND_EXPANSION_PLAN.md](docs/MONETIZATION_AND_EXPANSION_PLAN.md) for the product expansion and monetization strategy.

## Disclaimer

This tool is for research and education. Nothing it produces — including AI-generated trade ideas — is financial advice. Options involve substantial risk.
