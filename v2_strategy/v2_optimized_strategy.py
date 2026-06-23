"""
v2_optimized_strategy.py
========================
Algorithmic Trading Strategy — Version 2 (Optimized)

Pipeline: OHLCV → Indicators (ROC, EMA, NVI, PSAR) → Signal Engine
          → Risk-Managed Execution → Performance Analytics

Backtest target : BTC/USD daily (2019-09 → 2024-01)
Parameters      : EMA(15,60) crossover, 80% allocation, 10% stop-loss

Dependencies: pandas, numpy
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# TECHNICAL INDICATORS
# ─────────────────────────────────────────────────────────────────────────────

def roc(price: pd.Series, period: int = 12) -> pd.Series:
    """Rate of Change (%)."""
    return (price.diff(period) / price.shift(period)) * 100.0


def ema(price: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average."""
    return price.ewm(span=span, adjust=False).mean()


def nvi(close: pd.Series, volume: pd.Series, base: float = 1000.0) -> pd.Series:
    """Negative Volume Index — updates only on lower-volume days."""
    pct = close.pct_change().fillna(0.0)
    vol_drop = volume < volume.shift(1)
    factor = np.where(vol_drop, 1.0 + pct.values, 1.0)
    return pd.Series(base * np.cumprod(factor), index=close.index, name="nvi")


def psar(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    af_start: float = 0.02,
    af_step: float = 0.02,
    af_max: float = 0.20,
) -> pd.Series:
    """Parabolic SAR (Wilder)."""
    h, l = high.values, low.values
    n = len(close)
    out = np.zeros(n)
    bull = True
    af = af_start
    ep = h[0]
    out[0] = l[0]

    for i in range(1, n):
        prev = out[i - 1]
        out[i] = prev + af * (ep - prev)

        if bull:
            out[i] = min(out[i], l[i - 1], l[max(i - 2, 0)])
            if l[i] < out[i]:
                bull = False
                out[i] = ep
                ep = l[i]
                af = af_start
            elif h[i] > ep:
                ep = h[i]
                af = min(af + af_step, af_max)
        else:
            out[i] = max(out[i], h[i - 1], h[max(i - 2, 0)])
            if h[i] > out[i]:
                bull = True
                out[i] = ep
                ep = h[i]
                af = af_start
            elif l[i] < ep:
                ep = l[i]
                af = min(af + af_step, af_max)

    return pd.Series(out, index=close.index, name="psar")


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_signals(
    data: pd.DataFrame,
    ema_fast: int = 15,
    ema_slow: int = 60,
) -> pd.Series:
    """
    EMA crossover signal — long-only.

    +1 : fast EMA > slow EMA  (trend is up → deploy capital)
     0 : fast EMA ≤ slow EMA  (trend is down → stay in cash)
    """
    close = data["close"]
    e_fast = ema(close, ema_fast)
    e_slow = ema(close, ema_slow)
    return pd.Series(
        np.where(e_fast > e_slow, 1, 0),
        index=close.index, dtype=int,
    )


# ─────────────────────────────────────────────────────────────────────────────
# RISK-MANAGED TRADE SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def trade_optimized(
    signals: pd.Series,
    data: pd.DataFrame,
    initial_fund: float = 100_000.0,
    stop_loss_pct: float = 0.10,
    allocation_pct: float = 0.80,
) -> pd.Series:
    """
    Simulate a long-only strategy with fractional position sizing and a
    hard percentage-based stop-loss. Returns the daily equity curve.
    """
    close = data["close"].values
    sig = signals.values
    n = len(close)

    equity = np.empty(n)
    cash = float(initial_fund)
    position = 0.0
    entry_price = 0.0
    alloc_locked = 0.0
    direction = 0

    def _close_position(price: float) -> None:
        nonlocal cash, position, direction, alloc_locked
        pnl = position * (price - entry_price) * direction
        cash += alloc_locked + pnl
        position = 0.0
        alloc_locked = 0.0
        direction = 0

    for i in range(n):
        price = close[i]

        # Stop-loss check (before signal logic)
        if direction != 0:
            adverse = (entry_price - price) / entry_price if direction == 1 \
                else (price - entry_price) / entry_price
            if adverse >= stop_loss_pct:
                _close_position(price)

        # Signal change → flatten, then re-enter
        new_sig = int(sig[i])
        if new_sig != direction:
            if position != 0.0:
                _close_position(price)
            if new_sig != 0:
                alloc_locked = cash * allocation_pct
                position = alloc_locked / price
                cash -= alloc_locked
                entry_price = price
                direction = new_sig

        # Mark-to-market equity
        if position != 0.0:
            unrealised = position * (price - entry_price) * direction
            equity[i] = cash + alloc_locked + unrealised
        else:
            equity[i] = cash

    return pd.Series(equity, index=signals.index, name="portfolio_equity")


# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE METRICS
# ─────────────────────────────────────────────────────────────────────────────

def calculate_metrics(portfolio_series: pd.Series, trading_days: int = 365) -> dict:
    """
    Compute Total Net Profit, Total Return %, Sharpe, Sortino, Max Drawdown.
    Annualized with 365 days (crypto trades every day). Risk-free rate = 0%.
    """
    equity = portfolio_series.dropna()
    if equity.empty:
        raise ValueError("portfolio_series is empty.")

    returns = equity.pct_change().dropna()

    total_net_profit = equity.iloc[-1] - equity.iloc[0]
    total_return_pct = (total_net_profit / equity.iloc[0]) * 100.0

    mean_d = returns.mean()
    std_d = returns.std(ddof=1)
    sharpe = (mean_d / std_d) * np.sqrt(trading_days) if std_d > 0 else np.nan

    downside = returns[returns < 0]
    dstd = downside.std(ddof=1)
    sortino = (mean_d / dstd) * np.sqrt(trading_days) if len(downside) > 1 and dstd > 0 else np.nan

    drawdown = (equity - equity.cummax()) / equity.cummax()
    max_dd_pct = drawdown.min() * 100.0

    return {
        "total_net_profit": round(float(total_net_profit), 2),
        "total_return_pct": round(float(total_return_pct), 4),
        "sharpe_ratio": round(float(sharpe), 4) if not np.isnan(sharpe) else None,
        "sortino_ratio": round(float(sortino), 4) if not np.isnan(sortino) else None,
        "max_drawdown_pct": round(float(max_dd_pct), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CANDLESTICK — HAMMER (VECTORIZED)
# ─────────────────────────────────────────────────────────────────────────────

def identify_hammer(
    data: pd.DataFrame,
    body_ratio: float = 0.3,
    shadow_ratio: float = 2.0,
    upper_shadow_ratio: float = 0.1,
) -> pd.Series:
    """Detect Hammer candlestick patterns — fully vectorized, no loops."""
    o, h, l, c = data["open"], data["high"], data["low"], data["close"]

    total_range = h - l
    body = (c - o).abs()
    body_top = c.where(c >= o, o)
    body_bottom = c.where(c < o, o)
    lower_shadow = body_bottom - l
    upper_shadow = h - body_top

    valid = (total_range > 0) & (body > 0)
    small_body = (body / total_range) <= body_ratio
    long_lower = lower_shadow >= (shadow_ratio * body)
    tiny_upper = upper_shadow <= (upper_shadow_ratio * body)

    return (valid & small_body & long_lower & tiny_upper).rename("is_hammer")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — BACKTEST ON BTC/USD (2019-09 → 2024-01)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    DATA_PATH = "BTC_2019_2023_1d.csv"

    print("Loading BTC daily OHLCV data ...")
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["datetime"],
        index_col="datetime",
    )
    data = df[["open", "high", "low", "close", "volume"]].dropna()
    print(f"  Rows  : {len(data)}")
    print(f"  Range : {data.index[0].date()} → {data.index[-1].date()}\n")

    # ── Signals ──────────────────────────────────────────────────────────
    signals = generate_signals(data, ema_fast=15, ema_slow=60)

    # ── Backtest ─────────────────────────────────────────────────────────
    equity_curve = trade_optimized(
        signals=signals,
        data=data,
        initial_fund=100_000.0,
        stop_loss_pct=0.10,
        allocation_pct=0.80,
    )

    # ── Metrics ──────────────────────────────────────────────────────────
    metrics = calculate_metrics(equity_curve)

    print("── Backtest Results: BTC/USD (2019-09 → 2024-01) ───────────")
    print(f"  Initial Capital     : $100,000.00")
    print(f"  Final Equity        : ${equity_curve.iloc[-1]:,.2f}")
    print(f"  Total Net Profit    : ${metrics['total_net_profit']:,.2f}")
    print(f"  Total Return        : {metrics['total_return_pct']:.4f} %")
    print(f"  Sharpe Ratio        : {metrics['sharpe_ratio']}")
    print(f"  Sortino Ratio       : {metrics['sortino_ratio']}")
    print(f"  Max Drawdown        : {metrics['max_drawdown_pct']:.4f} %")

    # ── Hammer Detection ─────────────────────────────────────────────────
    data["is_hammer"] = identify_hammer(data)
    n_hammers = data["is_hammer"].sum()
    print(f"\n  Hammer Candles      : {n_hammers} / {len(data)} ({n_hammers/len(data)*100:.1f} %)")
