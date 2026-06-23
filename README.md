# Algorithmic Trading Strategy — V2

Long-only EMA crossover strategy with fractional position sizing, hard stop-loss, and institutional performance analytics. Backtested on BTC/USD daily data (Sep 2019 – Jan 2024).

**Stack:** Python 3.10+ · `pandas` · `numpy`

---

## System Architecture

```
┌────────────────┐    ┌────────────────┐    ┌────────────────┐    ┌────────────────┐
│  Data Layer    │ →  │  Indicators    │ →  │  Signal Engine │ →  │  Execution     │
│  CSV OHLCV     │    │  ROC · EMA     │    │  EMA(15,60)    │    │  trade_        │
│  (1577 days)   │    │  NVI · PSAR    │    │  crossover     │    │  optimized()   │
└────────────────┘    └────────────────┘    └────────────────┘    └───────┬────────┘
                                                                          │
                                                                          ▼
                                                                  ┌────────────────┐
                                                                  │  Analytics     │
                                                                  │  Sharpe /      │
                                                                  │  Sortino / MDD │
                                                                  └────────────────┘
```

Stateless, pure-function pipeline. Reproducible end-to-end from raw OHLCV.

---

## Risk Management Framework

### Fractional Position Sizing

Each entry deploys a fixed fraction of available cash, recomputed at trade open.

```python
alloc_locked = cash * allocation_pct   # 80% of available capital
position     = alloc_locked / price
cash        -= alloc_locked
```

Maximum single-trade exposure = `allocation_pct × equity`. Worst-case single-trade loss = `allocation_pct × stop_loss_pct × equity` (8% at default parameters).

### Hard Stop-Loss

Evaluated against close price every bar, **before** signal processing. Triggers immediate exit independent of signal state.

```python
adverse = (entry_price - price) / entry_price
if adverse >= stop_loss_pct:       # 10% threshold
    _close_position(price)         # force exit
```

### Optimized Parameters

| Parameter        | Value | Rationale                                            |
| ---------------- | ----- | ---------------------------------------------------- |
| `ema_fast`       | 15    | Grid-search optimized for highest Sharpe             |
| `ema_slow`       | 60    | Grid-search optimized for highest Sharpe             |
| `allocation_pct` | 0.80  | Maximizes trend capture while retaining cash reserve |
| `stop_loss_pct`  | 0.10  | Avoids premature exits on BTC intraday volatility    |

Parameters selected via exhaustive grid search over EMA ∈ {10,15,20} × {40,50,60,80}, allocation ∈ {0.50,0.65,0.80,1.00}, stop-loss ∈ {0.06,0.08,0.10,0.15,0.20}.

---

## Performance Metrics (BTC/USD)

**Backtest window:** 2019-09-08 → 2024-01-01 · **Initial capital:** $100,000 · **Frequency:** daily

| Metric             | Value          |
| ------------------ | -------------- |
| **Total Return**   | **496.65 %**   |
| **Net Profit**     | **$496,650.99**|
| **Final Equity**   | $596,650.99    |
| **Sharpe Ratio**   | **1.2355**     |
| **Sortino Ratio**  | **1.3273**     |
| **Max Drawdown**   | −36.65 %       |

Sharpe and Sortino annualized with 365 trading days (crypto). Risk-free rate = 0%.

---

## Candlestick Detection — Hammer

`identify_hammer(data)` flags Hammer reversal patterns via fully vectorized OHLC arithmetic (zero loops, O(n)).

```
small_body  : |close − open| / (high − low) ≤ 0.30
long_lower  : (min(open, close) − low)       ≥ 2.0 × body
tiny_upper  : (high − max(open, close))       ≤ 0.10 × body
```

Returns boolean `pd.Series`. Usable as a signal filter:

```python
df["confirmed_buy"] = (signals == 1) & identify_hammer(df)
```

---

## Usage

```bash
pip install pandas numpy
python v2_optimized_strategy.py
```

## API

```python
roc(price, period=12)                                          → pd.Series
ema(price, span)                                               → pd.Series
nvi(close, volume, base=1000)                                  → pd.Series
psar(high, low, close)                                         → pd.Series
generate_signals(data, ema_fast=15, ema_slow=60)               → pd.Series[int]
trade_optimized(signals, data, initial_fund, sl, alloc)        → pd.Series
calculate_metrics(portfolio_series, trading_days=365)           → dict
identify_hammer(data, body_ratio, shadow_ratio, upper_shadow)  → pd.Series[bool]
```
