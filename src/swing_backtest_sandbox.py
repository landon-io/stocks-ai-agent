"""
Swing Backtest Sandbox
----------------------
Leaderboard rotation with automatic QQQ cash parking.

Entry rules (both required):
  - RSI(7) < oversold threshold (cross-sectional bottom rank)
  - Macro trend filter: Adj Close > SMA_200 (buy dips only in structural uptrends)

QQQ is park-only — never in the swing universe (avoids QQQ/QQQM wash trades).

Run a single backtest via main(), or import StrategyConfig + run_simulation()
for parameter sweeps (see grid_search_optimizer.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta

from analyzer import resolve_simulation_range
from ticker_config import (
    ANALYSIS_LOOKBACK_MONTHS,
    MAXIMUM_SWING_POSITIONS,
    PARK_TICKER,
    RSI_OVERSOLD_THRESHOLD,
    RSI_PERIOD,
    SLIPPAGE_PCT,
    SMA_TREND,
    STARTING_CAPITAL,
    STOP_LOSS_PCT,
    SWING_ALLOCATION,
    SWING_SECTORS,
    TAKE_PROFIT_RSI,
    swing_tickers,
)

# Re-export for grid_search_optimizer and legacy imports
SECTORS = SWING_SECTORS
SWING_TICKERS = [t for t in SECTORS.values() if t != PARK_TICKER]
TICKERS = swing_tickers()
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class StrategyConfig:
    """Tunable strategy parameters for backtests and grid search."""

    rsi_period: int = RSI_PERIOD
    rsi_oversold_threshold: float = RSI_OVERSOLD_THRESHOLD
    stop_loss_pct: float = STOP_LOSS_PCT
    take_profit_rsi: float = TAKE_PROFIT_RSI
    sma_trend: int = SMA_TREND
    swing_allocation: float = SWING_ALLOCATION
    maximum_swing_positions: int = MAXIMUM_SWING_POSITIONS
    slippage_fee: float = SLIPPAGE_PCT
    starting_capital: float = STARTING_CAPITAL
    lookback_months: int | None = ANALYSIS_LOOKBACK_MONTHS


DEFAULT_CONFIG = StrategyConfig()


# =============================================================================
# Data structures
# =============================================================================
@dataclass
class ParkPosition:
    ticker: str
    shares: float
    cost_basis: float
    entry_idx: int


@dataclass
class SwingPosition:
    sector: str
    ticker: str
    shares: float
    entry_price: float
    cost_basis: float
    entry_idx: int
    stop_price: float


@dataclass
class PendingSwingBuy:
    sector: str
    ticker: str
    signal_idx: int
    park_extract_pct: float


@dataclass
class PendingSwingSell:
    sector: str
    reason: str
    signal_idx: int
    repark_to_qqq: bool = True


@dataclass
class ClosedTrade:
    sector: str
    ticker: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    pnl: float
    return_pct: float
    holding_days: int
    exit_reason: str


@dataclass
class SimulationState:
    cash: float = 0.0
    park: ParkPosition | None = None
    swing_positions: dict[str, SwingPosition] = field(default_factory=dict)
    pending_swing_sells: list[PendingSwingSell] = field(default_factory=list)
    pending_swing_buys: list[PendingSwingBuy] = field(default_factory=list)
    closed_trades: list[ClosedTrade] = field(default_factory=list)


@dataclass
class BacktestResult:
    config: StrategyConfig
    state: SimulationState
    dates: pd.DatetimeIndex
    ending_equity: float
    total_return_pct: float
    park_benchmark_return_pct: float
    alpha_vs_qqq_pct: float
    sharpe_ratio: float
    qqq_bh_sharpe_ratio: float
    sharpe_alpha: float
    win_rate_pct: float
    num_trades: int
    swing_pnl: float
    daily_equity: pd.Series


# =============================================================================
# Data loading
# =============================================================================
def load_and_prepare(
    tickers: list[str],
    config: StrategyConfig,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Load CSVs, compute RSI + SMA, and align on a shared timeline."""
    frames: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        path = DATA_DIR / f"{ticker}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing data file: {path}")

        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        close = df["Adj Close"]
        df["RSI"] = ta.rsi(close, length=config.rsi_period)
        df["SMA_200"] = ta.sma(close, length=config.sma_trend)
        frames[ticker] = df

    master_index = frames[tickers[0]].index
    for ticker in tickers[1:]:
        master_index = master_index.intersection(frames[ticker].index)

    aligned = {ticker: frames[ticker].loc[master_index].copy() for ticker in tickers}
    panel = pd.concat(
        {ticker: aligned[ticker][["Open", "High", "Low", "Close", "Adj Close", "RSI", "SMA_200"]]
         for ticker in tickers},
        axis=1,
    )
    return panel, aligned


# =============================================================================
# Cross-sectional ranking
# =============================================================================
def build_rsi_leaderboard(
    aligned: dict[str, pd.DataFrame],
    day_idx: int,
) -> list[tuple[str, str, float]]:
    rows: list[tuple[str, str, float]] = []
    for sector, ticker in SECTORS.items():
        if ticker == PARK_TICKER:
            continue
        rsi_val = aligned[ticker].iloc[day_idx]["RSI"]
        if pd.isna(rsi_val):
            continue
        rows.append((sector, ticker, float(rsi_val)))
    rows.sort(key=lambda item: item[2], reverse=True)
    return rows


def _passes_macro_trend_filter(
    aligned: dict[str, pd.DataFrame],
    ticker: str,
    day_idx: int,
) -> bool:
    """Macro trend filter: only buy dips when price is above SMA 200."""
    row = aligned[ticker].iloc[day_idx]
    sma = row["SMA_200"]
    if pd.isna(sma):
        return False
    return float(row["Adj Close"]) > float(sma)


def _passes_oversold_entry(
    aligned: dict[str, pd.DataFrame],
    ticker: str,
    day_idx: int,
    rsi: float,
    config: StrategyConfig,
) -> bool:
    """RSI oversold + macro uptrend — both required for a clean swing entry."""
    if rsi >= config.rsi_oversold_threshold:
        return False
    return _passes_macro_trend_filter(aligned, ticker, day_idx)


def bottom_n_oversold_sectors(
    leaderboard: list[tuple[str, str, float]],
    n: int,
    aligned: dict[str, pd.DataFrame],
    day_idx: int,
    config: StrategyConfig,
) -> list[tuple[str, str, float]]:
    """Lowest-RSI sectors passing RSI oversold and SMA-200 macro trend filter."""
    eligible = [
        item
        for item in reversed(leaderboard)
        if _passes_oversold_entry(aligned, item[1], day_idx, item[2], config)
    ]
    return eligible[:n]


def _bottom_n_sectors(
    leaderboard: list[tuple[str, str, float]],
    n: int,
    aligned: dict[str, pd.DataFrame],
    day_idx: int,
    config: StrategyConfig,
) -> list[tuple[str, str, float]]:
    return bottom_n_oversold_sectors(leaderboard, n, aligned, day_idx, config)


# =============================================================================
# Pricing helpers
# =============================================================================
def _buy_fill_price(raw_open: float, config: StrategyConfig) -> float:
    return raw_open * (1 + config.slippage_fee)


def _sell_fill_price(raw_open: float, config: StrategyConfig) -> float:
    return raw_open * (1 - config.slippage_fee)


def _adj_open(row: pd.Series) -> float:
    if row["Close"] == 0:
        return float(row["Adj Close"])
    return float(row["Adj Close"] * (row["Open"] / row["Close"]))


def portfolio_value(
    state: SimulationState,
    aligned: dict[str, pd.DataFrame],
    day_idx: int,
) -> float:
    total = state.cash
    if state.park is not None:
        total += state.park.shares * float(aligned[state.park.ticker]["Adj Close"].iloc[day_idx])
    for pos in state.swing_positions.values():
        total += pos.shares * float(aligned[pos.ticker]["Adj Close"].iloc[day_idx])
    return total


def _held_swing_sectors(state: SimulationState) -> set[str]:
    return set(state.swing_positions.keys())


def _pending_swing_sell_sectors(state: SimulationState) -> set[str]:
    return {order.sector for order in state.pending_swing_sells}


def _queued_swing_buy_sectors(state: SimulationState) -> set[str]:
    return {order.sector for order in state.pending_swing_buys}


def compute_sharpe_ratio(daily_equity: pd.Series, trading_days: int = 252) -> float:
    """Annualized Sharpe from daily equity curve (risk-free rate = 0)."""
    returns = daily_equity.pct_change().dropna()
    return _annualized_sharpe_from_returns(returns, trading_days)


def _annualized_sharpe_from_returns(returns: pd.Series, trading_days: int = 252) -> float:
    """Sharpe = (mean / std) * sqrt(252), assuming risk-free rate = 0."""
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(trading_days))


def compute_qqq_buy_hold_sharpe(
    aligned: dict[str, pd.DataFrame],
    ticker: str = PARK_TICKER,
    trading_days: int = 252,
    start_idx: int = 0,
    end_idx: int | None = None,
) -> float:
    """Annualized Sharpe for QQQ buy & hold using daily close-to-close returns."""
    closes = aligned[ticker]["Adj Close"].iloc[start_idx : end_idx + 1 if end_idx is not None else None]
    daily_returns = closes.pct_change().dropna()
    return _annualized_sharpe_from_returns(daily_returns, trading_days)


# =============================================================================
# Park & swing execution
# =============================================================================
def initialize_park(
    state: SimulationState,
    aligned: dict[str, pd.DataFrame],
    day_idx: int,
    config: StrategyConfig,
) -> None:
    row = aligned[PARK_TICKER].iloc[day_idx]
    fill = _buy_fill_price(_adj_open(row), config)
    shares = config.starting_capital / fill
    state.cash = 0.0
    state.park = ParkPosition(
        ticker=PARK_TICKER,
        shares=shares,
        cost_basis=config.starting_capital,
        entry_idx=day_idx,
    )


def _repark_cash(
    state: SimulationState,
    aligned: dict[str, pd.DataFrame],
    day_idx: int,
    cash_amount: float,
    config: StrategyConfig,
) -> None:
    if cash_amount <= 0:
        return
    fill = _buy_fill_price(_adj_open(aligned[PARK_TICKER].iloc[day_idx]), config)
    if fill <= 0:
        state.cash += cash_amount
        return
    shares = cash_amount / fill
    if state.park is None:
        state.park = ParkPosition(ticker=PARK_TICKER, shares=shares, cost_basis=cash_amount, entry_idx=day_idx)
    else:
        state.park.shares += shares
        state.park.cost_basis += cash_amount
    state.cash -= cash_amount
    if state.cash < 1e-9:
        state.cash = 0.0


def _extract_from_park(
    state: SimulationState,
    aligned: dict[str, pd.DataFrame],
    day_idx: int,
    extract_pct: float,
    config: StrategyConfig,
) -> float:
    if state.park is None or state.park.shares <= 0:
        return 0.0
    row = aligned[PARK_TICKER].iloc[day_idx]
    open_px = _adj_open(row)
    park_value = state.park.shares * open_px
    fill = _sell_fill_price(open_px, config)
    shares_to_sell = min(state.park.shares, (park_value * extract_pct) / fill)
    if shares_to_sell <= 0:
        return 0.0
    proceeds = shares_to_sell * fill
    cost_removed = state.park.cost_basis * (shares_to_sell / state.park.shares)
    state.park.shares -= shares_to_sell
    state.park.cost_basis -= cost_removed
    state.cash += proceeds
    return proceeds


def _open_swing_position(
    state: SimulationState,
    *,
    sector: str,
    ticker: str,
    day_idx: int,
    fill: float,
    cash_to_deploy: float,
    config: StrategyConfig,
) -> bool:
    if sector in state.swing_positions:
        return False
    if len(state.swing_positions) >= config.maximum_swing_positions:
        return False
    if cash_to_deploy < fill:
        return False
    shares = cash_to_deploy / fill
    cost = shares * fill
    state.cash -= cost
    state.swing_positions[sector] = SwingPosition(
        sector=sector,
        ticker=ticker,
        shares=shares,
        entry_price=fill,
        cost_basis=cost,
        entry_idx=day_idx,
        stop_price=fill * (1 - config.stop_loss_pct),
    )
    if state.cash < 1e-9:
        state.cash = 0.0
    return True


def _rebalance_swing_extract_pcts(
    state: SimulationState,
    config: StrategyConfig,
) -> None:
    """
    Split the total swing pool equally across buys executing this open.

    1 signal → full SWING_ALLOCATION (e.g. 80%).
    N signals → SWING_ALLOCATION / N each (e.g. 40% + 40%).
    """
    available_slots = config.maximum_swing_positions - len(state.swing_positions)
    eligible = [
        order
        for order in state.pending_swing_buys
        if order.sector not in state.swing_positions
    ][:available_slots]
    if not eligible:
        return
    per_signal_pct = config.swing_allocation / len(eligible)
    for order in eligible:
        order.park_extract_pct = per_signal_pct


def execute_pending_orders(
    state: SimulationState,
    aligned: dict[str, pd.DataFrame],
    day_idx: int,
    dates: pd.DatetimeIndex,
    config: StrategyConfig,
) -> None:
    for order in state.pending_swing_sells:
        if order.sector not in state.swing_positions:
            continue
        pos = state.swing_positions.pop(order.sector)
        fill = _sell_fill_price(_adj_open(aligned[pos.ticker].iloc[day_idx]), config)
        proceeds = pos.shares * fill
        state.closed_trades.append(
            ClosedTrade(
                sector=order.sector,
                ticker=pos.ticker,
                entry_date=dates[pos.entry_idx],
                exit_date=dates[day_idx],
                entry_price=pos.entry_price,
                exit_price=fill,
                pnl=proceeds - pos.cost_basis,
                return_pct=(fill / pos.entry_price - 1) * 100,
                holding_days=day_idx - pos.entry_idx,
                exit_reason=order.reason,
            )
        )
        if order.repark_to_qqq:
            state.cash += proceeds
            _repark_cash(state, aligned, day_idx, proceeds, config)
        else:
            state.cash += proceeds
    state.pending_swing_sells.clear()

    # Split total swing pool across all buys filling this open (balanced sizing)
    _rebalance_swing_extract_pcts(state, config)

    state.pending_swing_buys.sort(
        key=lambda o: float(aligned[o.ticker].iloc[o.signal_idx]["RSI"]),
    )
    for order in list(state.pending_swing_buys):
        if order.sector in state.swing_positions:
            continue
        if len(state.swing_positions) >= config.maximum_swing_positions:
            break
        proceeds = _extract_from_park(state, aligned, day_idx, order.park_extract_pct, config)
        if proceeds <= 0:
            continue
        fill = _buy_fill_price(_adj_open(aligned[order.ticker].iloc[day_idx]), config)
        _open_swing_position(
            state,
            sector=order.sector,
            ticker=order.ticker,
            day_idx=day_idx,
            fill=fill,
            cash_to_deploy=proceeds,
            config=config,
        )
    state.pending_swing_buys.clear()


def queue_swing_take_profits(
    state: SimulationState,
    aligned: dict[str, pd.DataFrame],
    day_idx: int,
    config: StrategyConfig,
) -> None:
    """Exit when RSI reaches the take-profit threshold at close."""
    pending = _pending_swing_sell_sectors(state)
    for sector, pos in list(state.swing_positions.items()):
        if sector in pending:
            continue
        rsi = aligned[pos.ticker].iloc[day_idx]["RSI"]
        if pd.isna(rsi) or float(rsi) < config.take_profit_rsi:
            continue
        state.pending_swing_sells.append(
            PendingSwingSell(
                sector=sector,
                reason="take_profit_rsi",
                signal_idx=day_idx,
                repark_to_qqq=True,
            )
        )
        pending.add(sector)


def queue_stop_losses(
    state: SimulationState,
    aligned: dict[str, pd.DataFrame],
    day_idx: int,
) -> None:
    pending = _pending_swing_sell_sectors(state)
    for sector, pos in list(state.swing_positions.items()):
        if sector in pending:
            continue
        if float(aligned[pos.ticker].iloc[day_idx]["Adj Close"]) <= pos.stop_price:
            state.pending_swing_sells.append(
                PendingSwingSell(sector=sector, reason="stop_loss", signal_idx=day_idx, repark_to_qqq=True)
            )
            pending.add(sector)


def queue_swing_entries(
    state: SimulationState,
    aligned: dict[str, pd.DataFrame],
    day_idx: int,
    leaderboard: list[tuple[str, str, float]],
    config: StrategyConfig,
) -> None:
    pending_buys = list(state.pending_swing_buys)
    open_slots = (
        config.maximum_swing_positions
        - len(state.swing_positions)
        + len(state.pending_swing_sells)
        - len(pending_buys)
    )
    if open_slots <= 0:
        return

    held = _held_swing_sectors(state)
    queued = _queued_swing_buy_sectors(state)
    candidates = _bottom_n_sectors(leaderboard, config.maximum_swing_positions, aligned, day_idx, config)

    batch: list[tuple[str, str]] = []
    for sector, ticker, _ in candidates:
        if open_slots <= 0:
            break
        if sector in held or sector in queued:
            continue
        batch.append((sector, ticker))
        open_slots -= 1

    if not batch:
        return

    # Placeholder — final split applied at execution open via _rebalance_swing_extract_pcts
    per_signal_pct = config.swing_allocation / len(batch)
    for sector, ticker in batch:
        state.pending_swing_buys.append(
            PendingSwingBuy(
                sector=sector,
                ticker=ticker,
                signal_idx=day_idx,
                park_extract_pct=per_signal_pct,
            )
        )


def queue_signals_at_close(
    state: SimulationState,
    aligned: dict[str, pd.DataFrame],
    day_idx: int,
    leaderboard: list[tuple[str, str, float]],
    config: StrategyConfig,
) -> None:
    queue_swing_take_profits(state, aligned, day_idx, config)
    queue_stop_losses(state, aligned, day_idx)
    queue_swing_entries(state, aligned, day_idx, leaderboard, config)


def run_simulation(
    config: StrategyConfig = DEFAULT_CONFIG,
    aligned: dict[str, pd.DataFrame] | None = None,
) -> BacktestResult:
    """Run the full backtest; pass pre-loaded `aligned` data to skip I/O."""
    if aligned is None:
        _, aligned = load_and_prepare(TICKERS, config)
    full_dates = next(iter(aligned.values())).index
    start_idx, last_idx, window_start, window_end = resolve_simulation_range(
        full_dates, config.lookback_months
    )
    dates = full_dates[start_idx : last_idx + 1]

    state = SimulationState()
    daily_values: list[float] = []

    initialize_park(state, aligned, day_idx=start_idx, config=config)
    daily_values.append(portfolio_value(state, aligned, start_idx))

    for day_idx in range(start_idx + 1, last_idx + 1):
        execute_pending_orders(state, aligned, day_idx, full_dates, config)
        daily_values.append(portfolio_value(state, aligned, day_idx))
        if day_idx < last_idx:
            leaderboard = build_rsi_leaderboard(aligned, day_idx)
            queue_signals_at_close(state, aligned, day_idx, leaderboard, config)

    for sector in list(state.swing_positions):
        pos = state.swing_positions.pop(sector)
        fill = _sell_fill_price(float(aligned[pos.ticker]["Adj Close"].iloc[last_idx]), config)
        proceeds = pos.shares * fill
        state.cash += proceeds
        state.closed_trades.append(
            ClosedTrade(
                sector=sector,
                ticker=pos.ticker,
                entry_date=full_dates[pos.entry_idx],
                exit_date=full_dates[last_idx],
                entry_price=pos.entry_price,
                exit_price=fill,
                pnl=proceeds - pos.cost_basis,
                return_pct=(fill / pos.entry_price - 1) * 100,
                holding_days=last_idx - pos.entry_idx,
                exit_reason="end_of_data",
            )
        )
        _repark_cash(state, aligned, last_idx, proceeds, config)

    ending_equity = portfolio_value(state, aligned, last_idx)
    daily_equity = pd.Series(daily_values, index=dates)
    park_bh_return = _benchmark_buy_hold_return(
        aligned, PARK_TICKER, config.starting_capital, start_idx, last_idx
    )
    strategy_sharpe = compute_sharpe_ratio(daily_equity)
    qqq_bh_sharpe = compute_qqq_buy_hold_sharpe(
        aligned, PARK_TICKER, start_idx=start_idx, end_idx=last_idx
    )
    total_return_pct = (ending_equity / config.starting_capital - 1) * 100
    trades = state.closed_trades
    winners = sum(1 for t in trades if t.pnl > 0)

    return BacktestResult(
        config=config,
        state=state,
        dates=dates,
        ending_equity=ending_equity,
        total_return_pct=total_return_pct,
        park_benchmark_return_pct=park_bh_return,
        alpha_vs_qqq_pct=total_return_pct - park_bh_return,
        sharpe_ratio=strategy_sharpe,
        qqq_bh_sharpe_ratio=qqq_bh_sharpe,
        sharpe_alpha=strategy_sharpe - qqq_bh_sharpe,
        win_rate_pct=(winners / len(trades) * 100) if trades else 0.0,
        num_trades=len(trades),
        swing_pnl=sum(t.pnl for t in trades),
        daily_equity=daily_equity,
    )


def _benchmark_buy_hold_return(
    aligned: dict[str, pd.DataFrame],
    ticker: str,
    starting_cash: float,
    start_idx: int,
    end_idx: int,
) -> float:
    start_px = float(aligned[ticker]["Adj Close"].iloc[start_idx])
    end_px = float(aligned[ticker]["Adj Close"].iloc[end_idx])
    return (starting_cash * (end_px / start_px) / starting_cash - 1) * 100


def print_report(result: BacktestResult) -> None:
    """Print a single-run performance report."""
    cfg = result.config
    state = result.state
    dates = result.dates
    w = 72

    print()
    print("=" * w)
    print("SWING BACKTEST SANDBOX — QQQ PARK + LEADERBOARD ROTATION")
    print("=" * w)
    print(f"{'Park asset':<28} {PARK_TICKER}  (passive only — not in swing pool)")
    swing_list = ", ".join(SECTORS.values())
    print(f"{'Swing universe':<28} {swing_list}")
    print(f"{'Entry filters':<28} RSI < {cfg.rsi_oversold_threshold} AND Close > SMA_{cfg.sma_trend}")
    lookback = cfg.lookback_months
    lookback_label = f"{lookback} months" if lookback else "full history"
    print(f"{'Analysis window':<28} {lookback_label}")
    print(f"{'Date range':<28} {dates[0].date()} → {dates[-1].date()} ({len(dates)} sessions)")
    print()
    print(f"{'Starting capital':<28} ${cfg.starting_capital:>14,.2f}")
    print(f"{'Ending portfolio value':<28} ${result.ending_equity:>14,.2f}")
    print(f"{'Strategy return':<28} {result.total_return_pct:>+14.2f}%")
    print(f"{'QQQ B&H return':<28} {result.park_benchmark_return_pct:>+14.2f}%")
    print(f"{'Alpha vs QQQ':<28} {result.alpha_vs_qqq_pct:>+14.2f}%")
    print()
    print("RISK-ADJUSTED PERFORMANCE (annualized, risk-free rate = 0)")
    print("-" * w)
    print(f"{'Strategy Sharpe ratio':<28} {result.sharpe_ratio:>14.2f}")
    print(f"{'QQQ B&H Sharpe ratio':<28} {result.qqq_bh_sharpe_ratio:>14.2f}")
    print(f"{'Sharpe alpha':<28} {result.sharpe_alpha:>+14.2f}")
    print()
    print(f"{'Swing overlay P&L':<28} ${result.swing_pnl:>14,.2f}")
    print(f"{'Total swing trades':<28} {result.num_trades:>14}")
    print(f"{'Win rate':<28} {result.win_rate_pct:>13.1f}%")
    print()
    print("-" * w)
    print("ACTIVE HYPERPARAMETERS")
    print("-" * w)
    print(f"  RSI_PERIOD              = {cfg.rsi_period}")
    print(f"  RSI_OVERSOLD_THRESHOLD  = {cfg.rsi_oversold_threshold}")
    print(f"  TAKE_PROFIT_RSI         = {cfg.take_profit_rsi}")
    print(f"  STOP_LOSS_PCT           = {cfg.stop_loss_pct:.1%}")
    print(f"  SWING_ALLOCATION        = {cfg.swing_allocation:.0%}  (total pool; split across same-day entries)")
    print(f"  MAX_SWING_POSITIONS     = {cfg.maximum_swing_positions}")
    print(f"  LOOKBACK_MONTHS         = {cfg.lookback_months or 'all'}")
    print("=" * w)

    trades = state.closed_trades
    if trades:
        print()
        print("Per-sector swing breakdown:")
        print(f"{'Sector':<16} {'Ticker':<6} {'Trades':>7} {'Win%':>7} {'Total P&L':>12}")
        print("-" * 52)
        for sector, ticker in SECTORS.items():
            if ticker == PARK_TICKER:
                continue
            subset = [t for t in trades if t.sector == sector]
            if not subset:
                continue
            wins = sum(1 for t in subset if t.pnl > 0)
            pnl = sum(t.pnl for t in subset)
            wr = wins / len(subset) * 100
            print(f"{sector:<16} {ticker:<6} {len(subset):>7} {wr:>6.1f}% ${pnl:>10,.0f}")
        print("=" * w)


def main() -> None:
    """Production backtest entry point (also: python src/analyzer.py)."""
    print_report(run_simulation(DEFAULT_CONFIG))


if __name__ == "__main__":
    main()
