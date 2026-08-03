# Flow data: the snapshot proxy vs a real trade tape

This app shows options "flow" in two places: the global `/flow` page and the
Flow tab on each ticker page. Both are powered by the **positioning proxy
feed**, not a real trade tape. This doc explains what that proxy is, what it
can and cannot tell you, and what the paid upgrade path (OPRA data, Phase 4)
would add.

## What the proxy feed is

There is no live trade data in the free data budget (Tradier REST + yfinance +
FRED). Instead, the app builds its own history: the snapshot engine captures
full option chains for the snapshot universe every 30 minutes during market
hours (backend scheduler, `SCHEDULER_ENABLED`) and once daily after the close
(GitHub Actions cron). The proxy feed is computed from those snapshots:

- **Volume/OI ratio** per contract - today's cumulative volume far above open
  interest flags unusual activity.
- **OI change** day over day - positions actually opened or closed overnight.
- **IV change** - demand shifts in the option's price.
- A combined **score** ranks contracts across the universe.

Endpoints: `GET /api/flow/feed` (per-contract, optional `ticker` and
`expiration_date` filters) and `GET /api/flow/hottest-chains` (per-expiration
aggregates). Both are cache-only: they read persisted snapshots and never call
the data provider, so they are cheap to query but only as fresh as the last
snapshot.

Coverage: only symbols in the snapshot universe (the config watchlist plus
server-persisted watchlists) with at least one captured snapshot. Other
symbols return empty results.

## What it can and cannot tell you

The proxy sees aggregates, so it can say **where** unusual positioning
accumulated. It cannot say anything about the individual trades behind an
aggregate:

| Question | Proxy feed | Trade tape |
| --- | --- | --- |
| Did unusual volume hit this contract? | yes | yes |
| Were positions opened or closed? | next day, via OI change | inferred same day |
| Buyer or seller aggression (direction)? | no | inferred per trade vs NBBO |
| Sweeps (urgent multi-exchange orders)? | no | yes |
| Blocks (single large institutional prints)? | no | yes |
| Dollar premium per trade ("only trades > $500k")? | no | yes |
| Spread legs vs naked directional bets? | no | via condition codes |
| Intraday timing of the activity? | 30-minute buckets | millisecond |
| 0DTE activity? | no - same-day contracts never enter OI | yes |

The 0DTE row matters most for this app's analytics: OI-based GEX is
structurally blind to same-day positioning, which is a large share of SPX/SPY
volume.

## What paid OPRA data adds (Phase 4)

OPRA (Options Price Reporting Authority) is the consolidated tape of every
options trade print across all US options exchanges, in real time, with
millisecond timestamps, size, price, exchange, and condition codes. That
enables the classic flow-feed features: per-trade premium, aggressor-side
inference (print price vs the NBBO at that instant), sweep and block
detection, multi-leg identification, net call/put premium ("market tide"),
and 0DTE dashboards.

Two honest caveats: the tape itself does not carry aggressor side or
open-vs-close - vendors infer both (NBBO comparison, next-day OI
reconciliation), and the inference is good but not perfect.

Access is through licensed vendors (for example Polygon, Databento, Intrinio,
ThetaData), roughly $50-200+/mo for personal real-time use. Redistributing
the data to paying subscribers triggers OPRA redistribution agreements and
fees, which is why `docs/UW_PARITY_PLAN.md` puts this behind a provider
interface in Phase 4 rather than wiring in a vendor directly.

## Where it surfaces in the UI

- `/flow` - the full proxy feed with filters, plus hottest expiration chains.
- `/stock/[symbol]` Flow tab - the same feed filtered to that ticker. The tab
  is labeled as a snapshot-derived proxy; it becomes a real per-trade tape
  only when Phase 4 lands.
