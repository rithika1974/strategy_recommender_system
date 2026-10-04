# Multi-Agent Strategy Recommender

This repository implements Dataset 1: raw historical market data for a 30-stock Indian equity development universe.

## Historical Data Setup

1. Create an Upstox Developer App.
2. Generate an access token for the app.
3. Put the token in `.env` as `UPSTOX_ACCESS_TOKEN`.
4. Install the project dependencies.
5. The configured universe is resolved and verified against Upstox's official NSE equity instrument master.
6. Run:

```bash
python scripts/download_data.py
```

7. The pipeline is:

```text
Upstox Historical Candle V3
        ↓
Python downloader
        ↓
normalization and validation
        ↓
SQLite database
```

Data is saved in:

```text
data/market_data.db
```

The database contains `stocks` metadata and `daily_prices` records. Its
`(symbol, timestamp)` primary key prevents duplicate daily records, so the
downloader can be run repeatedly. Use `refresh=True` from Python when a full
redownload is explicitly required.

## Dataset 2

Run the raw market/index ingestion first, then build deterministic features:

```powershell
python scripts/ingest_market_macro.py
python scripts/build_features.py
```

The ingestion currently retrieves NIFTY 50 and India VIX through the official
Upstox instrument master and Historical Candle V3 API. USD/INR and Brent
features are supported when available. Repo-rate and CPI columns remain in the
schema for compatibility but are intentionally NULL and unused in V1.

For an incremental replacement or addition, download and build only the
requested symbol; existing symbols and their features are preserved:

```powershell
python -c "from src.data.downloader import download_all_stocks; download_all_stocks(symbols=['ALKEM'])"
python -c "from scripts.build_features import build_features; build_features(symbols=['ALKEM'])"
```

Stocks that cannot be verified from the official instrument master are reported as failures and
are not downloaded using guessed identifiers.

Do not commit any real credentials.

## Rolling-window augmentation

Build compact chronological training and evaluation window references after
Dataset 2 has been created:

```powershell
python scripts/build_augmentation.py
```

The command reads the existing `features` table and rebuilds the
`augmentation_windows` metadata table. It does not copy feature values or
create tensor data. Each reference identifies one symbol, split, start date,
end date, window length, stride, and the market regimes at the end of the
window. A future training loader can retrieve the referenced 60 rows from
`features` and materialize `(batch, 60 days, features)` tensors on demand.

The frozen baseline configuration is in `configs/augmentation.yaml`. Training
uses 60-row windows with stride 30. Validation and test use stride 60 and do
not overlap. Windows containing NULL active V1 features are rejected; the
intentionally unused `repo_rate` and `cpi` compatibility columns are ignored.

## Quantitative strategy signals

Generate a read-only summary of the four deterministic V1 strategy families:

```powershell
python scripts/run_strategies.py
```

The strategies read the unified `features` table and return compact
`symbol`, `date`, `strategy`, and `signal` rows in memory. Signals are `long`,
`short`, or `flat`; required NULL indicators produce `flat`. No signal table,
feature copy, portfolio result, or trade execution data is created.

The baseline rules and parameters are defined in `configs/strategies.yaml`:

- momentum uses the configured trailing return and symmetric threshold;
- trend following uses price versus SMA, the SMA crossover, and optional
  bull/bear regime confirmation;
- mean reversion uses RSI and Bollinger percent-B extremes;
- breakout uses the lookahead-safe prior-range features with volume-ratio
  confirmation.

These are research baselines, not optimized trading rules. A `flat` signal is
the strategy layer's exit/no-position instruction; portfolio accounting and
execution remain outside this layer.

## Quantitative backtesting

Run each enabled strategy independently across the frozen train, validation,
and test periods:

```powershell
python scripts/run_backtest.py
```

Signals calculated from the close of date `t` execute at the next available
trading day's open. Buy fills receive upward slippage and sell fills receive
downward slippage. Proportional transaction costs are charged on every entry
and exit. The baseline portfolio opens at most five positions, targets 20% of
current equity per new position, processes exits before entries, and uses
alphabetical symbol order when more signals are available than slots.

Each split starts with fresh cash and cannot inherit a signal or position from
another split. Any position still open at a split boundary is liquidated using
the final available close with the same adverse slippage and transaction-cost
rules. Backtest results, trade ledgers, and equity curves are produced in
memory only; the `features` and `augmentation_windows` tables are not changed.

The reported cumulative return, CAGR, volatility, Sharpe ratio, maximum
drawdown, win rate, and trade count are deterministic research measurements.
No parameter optimization or strategy selection is performed.

## Strategy parameter optimization

Run the small deterministic parameter search using TRAIN and VALIDATION only:

```powershell
python scripts/run_optimization.py
```

The optimizer ranks every candidate on TRAIN using:

```text
Sharpe + 0.5 × cumulative return − maximum drawdown
```

The top three TRAIN candidates plus the baseline are evaluated on VALIDATION.
The selected configuration has the highest equal-weight mean of its TRAIN and
VALIDATION objective. The optimizer never loads or evaluates the 2026 TEST
split, and it does not replace the baseline settings in `strategies.yaml`.

The complete compact candidate report is written to
`data/strategy/performance/optimization_results.json`. It contains parameters
and metrics only; no feature, price, augmentation, equity-curve, or tensor rows
are duplicated. The intentionally small grids are defined in
`configs/optimization.yaml` and are research baselines rather than exhaustive
or optimal ranges.

## Strategy robustness diagnostics

Run the approved temporal, parameter, stock-breadth, and regime diagnostics:

```powershell
python scripts/run_robustness.py
```

This command reuses the selected configurations in
`data/strategy/performance/optimization_results.json` without re-optimizing or
changing them. The 2023, 2024, and 2025 folds measure fixed-parameter temporal
stability; they are not new unseen out-of-sample validation. Cross-stock
results are breadth diagnostics rather than cross-sectional generalization.

Regime attribution uses the existing market and volatility labels from the
previous available trading day. Predeclared sensitivity and sample-size
thresholds are stored in `configs/robustness.yaml`. The compact report is
written to `data/strategy/performance/robustness_results.json`; the command
opens SQLite read-only and never loads the frozen 2026 TEST period.

## Quantitative service boundary

The framework-neutral service facade in `src/services/` exposes stable,
JSON-compatible operations corresponding to:

- `GET /stocks`
- `GET /features/{stock}`
- `GET /market-context/{stock}`
- `GET /strategies/{stock}`
- `GET /backtest/{stock}`
- `GET /optimization/{stock}`
- `GET /robustness/{stock}`

Stock-specific operations accept the shared request contract:

```json
{"stock": "RELIANCE", "horizon": "intraday"}
```

Supported horizon values are `intraday`, `short_term`, `swing`, `medium_term`,
and `long_term`. The horizon is validated but does not change V1 strategy
rules. Current quantitative inputs are daily, so responses explicitly report
that their parameters are not intraday- or horizon-calibrated.

`QuantitativeAnalysisService` is the reusable Python boundary. `QuantitativeAPI`
maps the endpoint shapes to it without introducing a web framework. A later
HTTP application can adapt this interface without exposing SQLite or internal
calculation functions to agents. Historical access is behind the
`AnalysisDataSource` protocol, allowing a future live-data adapter without
changing the analysis service.

All strategies expose the same structured evidence fields: strategy, signal,
parameters, entry, exit, stop loss, take profit, performance, and risk metrics.
V1 does not calculate executable entry/exit prices or stop/target levels, so
those fields are explicitly `null` rather than fabricated. Repo rate and CPI
remain nullable compatibility columns and are excluded from active V1 output.

## Project layout

- `src/data/` contains the Upstox ingestion, normalization, validation, and SQLite storage logic.
- `data/market_data.db` stores stock metadata and validated daily prices.
- `configs/` stores configuration for the selected universe and data source.
- `tests/data/` stores unit tests for normalization, validation, SQLite storage, and mapping.

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/download_data.py
```

On Windows PowerShell, use:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/download_data.py
```
