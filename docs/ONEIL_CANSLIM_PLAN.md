# O'Neil CAN SLIM Integration Plan

Plan for bringing William O'Neil's framework (CAN SLIM criteria, rally-attempt /
follow-through-day market timing, distribution-day counting) into the FastAPI +
Next.js stack. Written from the uploaded framework summary; scoped to free data
per the platform rules in `UW_PARITY_PLAN.md`.

## Revision (2026-08-03): delivered as one PR at the index level

The owner redirected the scope: one PR, algorithms first, applied to the
indices that govern stock groups (SPY/QQQ/DIA/IWM plus SMH and the SPDR
sectors) rather than per-stock fundamentals. Delivered:

- `quant_analysis/analytics/market_direction.py`: the state machine
  (correction, rally attempt, follow-through day, distribution days), EMA and
  Wilder RSI math, EMA touch detection, the index-adapted CAN SLIM scorecard,
  and narrative/signal text builders.
- Backend: `GET /api/direction` (+ `/{symbol}`, `/signals`), the
  `direction_signals` SQLite log, a 16:20 ET scheduler evaluation with an
  idempotent lazy fallback, optional Discord/email delivery, config block in
  `config.yaml`.
- Frontend: `/direction` page (benchmark hero with the bottom-durability
  checklist, index card grid, annotated candle/EMA/volume chart, scorecard
  table, signal feed), sidebar entry, page help, market-page link.

The per-stock fundamental letters (C/A earnings, ROE, institutional 13F) from
the original phasing below remain future work; the index adaptation maps every
letter to a price/volume proxy and says so in the UI.

## Verdict: feasible, in three tiers

Not every letter of CAN SLIM is equally computable from our data budget
(Tradier + yfinance + own snapshots). Honest tiering up front:

| Criterion | Feasibility | Data source |
|---|---|---|
| M - Market direction (FTD, distribution days) | High - pure OHLCV math | yfinance index candles (^GSPC/^IXIC carry volume), Tradier SPY/QQQ fallback |
| L - Leader or laggard (relative strength) | High | Tradier `history` (already wrapped) vs SPY |
| N - New price highs | High (price part only) | 52w high already in the quote payload |
| S - Supply and demand (volume confirmation) | High | daily candles; float via yfinance `info` best-effort |
| C - Current quarterly earnings growth | Medium | yfinance quarterly income statement; Nasdaq calendar already gives reported EPS + surprise |
| A - Annual earnings growth + ROE | Medium | yfinance annual statements + `info` ROE |
| I - Institutional sponsorship | Low-Medium | yfinance holders (13F-derived, quarterly, stale by nature) |
| N - "something new" (products, management) | Not computable | surfaced qualitatively via news feed + AI narrative instead |

The "M" engine is the highest-value piece: it is the letter O'Neil himself
weighted most, it needs zero new data dependencies, and it lands squarely on
this project's stated wedge (interpretation as a first-class layer - UW has no
market-direction verdict at all). It ships first and alone.

## Why this fits the product

- The platform's differentiator is plain-English interpretation
  (`interpret_net_gex`, `describe_gamma_gap`, posture synthesis). A market
  state machine that says "Rally attempt, day 4 - watching for a
  follow-through day" is the same product idea applied to the index level.
- The gamma-gap track record is the hero credibility feature. Follow-through
  days are a loggable, verifiable signal with a decades-old public definition:
  log every detected FTD and score it later exactly like
  `evaluate_gamma_gap_outcome` scores gap signals. Two signal families on one
  public track-record page beats one.
- O'Neil is momentum/trend oriented and struggles in chop. The dealer-gamma
  lens is strongest exactly there (positive-gamma pinning regimes). The two
  frameworks are complementary, and the UI should say so.

## UX design (the core of this plan)

Guiding rules, matching existing conventions (Sessions 10 and 12):

1. Never show a letter grade without the sentence explaining it.
2. Every criterion renders one of four states: Met / Not met / Borderline /
   Data unavailable. Unavailable is a first-class state with a reason
   ("institutional ownership data is quarterly 13F data and may lag").
3. No "buy signal" language anywhere. The card reports criteria status, the
   disclaimer stays, and the state names mirror O'Neil's own vocabulary so
   users can cross-check any IBD-style reference.
4. Progressive disclosure: a compact verdict first, the mechanics one click
   deeper via the existing `PageHelp` / expandable panel patterns.

### 1. Market Direction panel on `/market` (hero)

A `Panel` at the top of the market page:

- Status pill with the four O'Neil states:
  - `Confirmed Uptrend` (green)
  - `Uptrend Under Pressure` (amber, driven by distribution-day count)
  - `Rally Attempt - Day N` (blue, with the day counter live)
  - `Correction` (red)
- One plain-English line under the pill, generated from the state, e.g.
  "Day 5 of a rally attempt on the S&P 500. A follow-through day (a 1%+ gain
  on rising volume) between days 4 and 7 would confirm a new uptrend. The
  attempt fails if the index undercuts the day-1 low."
- Distribution-day counter chip per tracked index (S&P 500, Nasdaq), styled
  like the existing zone badges: 0-1 quiet, 2-3 caution, 4+ pressure.
- Annotated candlestick strip: reuse `CandlestickChart` with a small marker
  layer - day-1 rally low marked, FTD candle highlighted, distribution days
  dotted red. Volume bars underneath since volume is the whole game here.
- A "How this works" section in the page's existing `PageHelp` explaining
  rally attempt, follow-through day, and distribution days in three short
  paragraphs, including the honest caveat that FTDs fail regularly and the
  thresholds used (configurable, shown, not hidden).

### 2. CAN SLIM report card on `/stock/[symbol]` Overview

A `CanSlimCard` panel: seven rows, one per letter.

- Each row: letter chip, criterion name, status chip, measured value, one
  sentence. Example: `C | Quarterly EPS growth | Met | +34% YoY | Latest
  quarter EPS grew 34% vs a year ago (O'Neil looked for 20-25%+).`
- Header shows "5 of 7 met, 1 borderline, 1 unavailable" - never a
  percentage score, which would imply false precision.
- The M row is the shared market state (same API as the market panel), so the
  ticker page always carries the "three out of four stocks follow the market"
  context.
- Rows with unavailable data say why and what would light them up.
- A short footer line ties into the house analytics when relevant:
  "Breakout volume confirmation pairs with the flow feed's vol/OI spikes."

### 3. Screener preset: `oneil_leaders`

New preset alongside `high_vol_oi` / `unusually_bullish` / `gamma_squeeze` in
`services_screener.py`, technical criteria only at first (RS percentile vs the
snapshot universe, proximity to 52w high, up/down volume ratio), with the
methodology string the screener page already renders. Fundamentals join the
preset only after Phase C proves the data reliable.

### 4. Alerts

Extend the alert-rule vocabulary in `services_alerts.py` with market-level
events, evaluated by the existing scheduler cadence:

- Follow-through day detected (index-level, not per-ticker)
- Rally attempt failed (day-1 low undercut)
- Distribution-day count crossed a threshold

Per-position 7-8% stop-loss alerts are deferred: they need entry prices, which
means portfolio state, which is Phase 3 auth territory.

### 5. AI narrative integration

Add the market-direction state to `_build_symbol_packet` in
`backend/app/services_ai.py` so AI theses are conditioned on regime, which is
exactly how O'Neil said to use it.

## Architecture

Pure math lives in `quant_analysis`, adapters in `backend/app`, rendering in
the frontend - same shape as vanna/charm and max pain.

### New analytics modules

`quant_analysis/analytics/market_direction.py` (pure, no I/O):

- `classify_market_state(ohlcv: pd.DataFrame, config) -> dict` - a
  deterministic state machine over daily OHLCV:
  - Correction: trailing decline past a configurable threshold.
  - Rally attempt day 1: close up off the low after decline; count advances
    while the day-1 low holds, resets on undercut.
  - Follow-through: day 4-7 (configurable window), gain >= threshold
    (default 1.25%, configurable - the PDF notes modern practitioners want
    more than the historical 1%), volume above the prior day.
  - Distribution days: decline >= 0.2% on higher volume; each expires after
    25 sessions; 4+ within the window degrades Confirmed Uptrend to Under
    Pressure; clustering after an FTD flags the bottom as suspect.
- Returns state, day counters, key dates/levels (day-1 low, FTD date), and
  the annotation list the chart needs. Everything explainable from the
  return value alone.

`quant_analysis/analytics/canslim.py` (pure evaluators):

- One function per computable letter, each returning
  `{status, value, detail, source, as_of}` so the UI can label staleness
  per-criterion.
- Thresholds (20-25% growth, 17% ROE, RS percentile cutoffs) in one config
  block, never inline.

`quant_analysis/integrations/fundamentals.py`:

- Thin yfinance wrapper for quarterly/annual statements, ROE, float, holder
  counts. Every field independently optional; failures degrade that one
  criterion to unavailable, mirroring how `services_congress.py` isolates
  flaky sources. Cached 24h in the backend.

### Backend

- `GET /api/market/direction` - state for tracked indices, cached ~15 min,
  index candles via yfinance (^GSPC/^IXIC include volume; Tradier SPY/QQQ
  history as fallback).
- `GET /api/ticker/{sym}/canslim` - full report card; price-derived rows
  computed fresh from cached candles, fundamental rows from the 24h cache.
- Screener preset + alert rule types as above.
- Log detected FTDs to a small table beside `gamma_gap_analysis` for the
  track-record page (append-only, evaluated later).

### Frontend

- `components/market/MarketDirectionPanel.tsx` on `/market`.
- `components/stock/CanSlimCard.tsx` on the ticker Overview tab.
- Marker/annotation support added to `CandlestickChart` (kept optional so
  existing usages are untouched).
- `api.ts` types for both endpoints; `PageHelp` entries for both surfaces.

## Phasing

- **Phase A - Market Direction (M).** Engine + endpoint + `/market` panel +
  help copy. Self-contained, no new dependencies, immediately demonstrable.
  This phase alone delivers the most-emphasized part of the framework.
- **Phase B - Technical letters (L, N, S).** Per-ticker RS / new-high /
  volume evaluators from candle data; `CanSlimCard` ships here in partial
  mode with C, A, I explicitly marked "coming - fundamental data";
  `oneil_leaders` screener preset (technical criteria).
- **Phase C - Fundamental letters (C, A, I).** The fundamentals wrapper with
  per-field degradation; card completes; fundamentals join the preset if
  live data quality proves acceptable.
- **Phase D - Signal plumbing.** FTD/distribution alert rules, AI packet
  regime field, FTD logging + outcome scoring on `/track-record`.

Each phase is a small, reversible PR in the repo's usual style.

## Risks and honesty requirements

- **FTD parameters are contested.** The 1% threshold is historical; the PDF
  itself notes practitioners now demand more. Ship configurable thresholds,
  display them in the help panel, and never present the state as more than a
  rules-based classification.
- **yfinance fundamentals are fragile and unofficial.** Per-field graceful
  degradation is a hard requirement, not polish. If C/A/I prove too flaky,
  the card stays honest with three unavailable rows rather than guessing.
- **Not every FTD works and O'Neil said so.** The help copy must carry his
  own framing: a failed follow-through is a cost of doing business. The
  track-record logging turns that honesty into a feature.
- **Volume data quality.** ^GSPC volume from Yahoo is consolidated-tape data
  and adequate for higher/lower-than-prior-day comparisons; document that SPY
  volume is the fallback comparator.
- **No advice language.** Status vocabulary only ("criteria met", "confirmed
  uptrend per the O'Neil definition"), existing disclaimer unchanged.
- **Tests.** Golden-fixture tests for the state machine (a synthetic
  correction-bottom-FTD series) are strongly recommended but are test work,
  which per repo rules waits for explicit approval.

## Dependencies on existing work

- None on snapshot history: candles come live from providers, so unlike IV
  rank this feature has no accrual wait.
- Track-record integration reuses the `evaluate_*_outcome` /
  `summarize_outcomes` pattern from
  `quant_analysis/analytics/track_record.py`.
- Alert delivery reuses the existing rule/event/delivery machinery untouched.
