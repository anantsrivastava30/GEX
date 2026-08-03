# Intraday Snapshot Archive

The self-hosted backend preserves every successful scheduler and admin capture
without changing the existing daily-history contract.

## Storage layout

```text
data/
|-- snapshots/
|   |-- YYYY-MM-DD/{TICKER}.csv.gz
|   `-- daily_metrics.csv
`-- intraday_snapshots/
    `-- YYYY-MM-DD/
        |-- _runs/{UTC_TIMESTAMP}-{RUN_ID}.json
        `-- {TICKER}/{UTC_TIMESTAMP}-{CAPTURE_ID}.parquet
```

The daily CSV remains one latest capture per ticker and session. Existing IV
rank, OI-change, Flow, Screener, and alert code continues reading only that
layout. The Parquet archive is append-only and is intended for future intraday
research.

GitHub Actions runs the daily CLI and does not create intraday files. The local
backend scheduler and the PIN-protected admin capture endpoint create both
formats.

## Capture provenance

Every Parquet row includes:

- Shared run ID and per-ticker capture ID
- Capture start and completion timestamps in UTC
- US market session date
- Ticker, accompanying spot, and requested expirations
- Provider and producer (`scheduler` or `admin`)
- Schema version and optional deployment source revision
- OCC contract symbol and root symbol when supplied by the provider
- The persisted quote, volume, OI, IV, Greek, and exposure fields

Each completed universe run also writes an immutable JSON manifest containing
the expected ticker universe, archived/daily successes, failures, timing, and
complete/partial status. Files without a corresponding completed manifest came
from an interrupted run and should be treated as incomplete.

## Reliability semantics

The archive file is published before the daily file is refreshed. If archival
fails, that ticker's daily file is not advanced to an unarchived state. Files
are written to a temporary path and atomically linked to a destination that may
not already exist, so retries cannot silently replace prior captures.

Scheduler and admin captures share an in-process lock. The current Docker image
runs one Uvicorn process, so they cannot overlap. Multiple backend workers or
replicas would require a filesystem or distributed lock before sharing this
archive.

## Retention and backups

There is intentionally no automatic deletion policy yet. At the current schema
and schedule, storage is expected to grow by roughly 12 MB per market session
for the 21-symbol baseline and up to roughly 28 MB per session at the 50-symbol
cap. Actual chain density varies.

The Compose bind mount (`./data:/app/data`) preserves files across container
rebuilds, but it is not a backup. Monitor free disk space and replicate
`data/intraday_snapshots/` to durable object storage before relying on it as a
long-term research asset.

## Licensing

Possessing captured files does not grant redistribution rights. The current
Tradier retail data arrangement should be treated as internal-research use only.
Do not sell or redistribute raw chains, quotes, Greeks, or a reconstructable
historical feed without a written commercial market-data agreement and legal
review. Future commercial datasets should be collected under a provider license
that explicitly permits the intended derived product or redistribution.
