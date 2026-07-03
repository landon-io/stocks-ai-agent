"""
Cross-sector rotation strategy for multi-ETF swing trading.

For daily pre-market trade planning and paper trading, run pre_market_scanner.py.
This module is the historical backtester only.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta

from ticker_config import (
    ANALYSIS_LOOKBACK_MONTHS,
    BENCHMARK_SPLIT,
    LEVERAGED_EXEC,
    TICKER_CONFIG,
    all_tickers,
    theme_for_ticker,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Indicators
EMA_FAST = 5
SMA_TREND = 200
RSI_LENGTH = 7

# Sentiment thresholds
RSI_OVERSOLD = 35
RSI_DEEP_BUY = 30
RSI_OVEREXTENDED = 75

# Portfolio constraints
STARTING_CASH = 10_000.0
MAX_POSITIONS = 2
POSITION_ALLOC_PCT = 0.35
MIN_CASH_PCT = 0.30
STOP_LOSS_PCT = 0.015
SLIPPAGE_PCT = 0.0005


@dataclass
class Trade:
    sector: str
    theme: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: float
    allocated_cash: float
    pnl: float
    exit_reason: str
    holding_days: int


@dataclass
class OpenPosition:
    sector: str
    theme: str
    entry_date: pd.Timestamp
    entry_idx: int
    shares: float
    entry_price: float
    cost_basis: float
    stop_raw: float


@dataclass
class RotationBacktestResult:
    starting_cash: float
    ending_equity: float
    trades: list[Trade]
    skipped_entries: int
    benchmark_ending: float
    benchmark_return_pct: float
    window_start: pd.Timestamp
    window_end: pd.Timestamp

    @property
    def cumulative_return_pct(self) -> float:
        return (self.ending_equity / self.starting_cash - 1) * 100

    @property
    def alpha_vs_benchmark_pct(self) -> float:
        return self.cumulative_return_pct - self.benchmark_return_pct

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate_pct(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.pnl > 0)
        return wins / len(self.trades) * 100

    @property
    def avg_holding_days(self) -> float:
        if not self.trades:
            return 0.0
        return sum(t.holding_days for t in self.trades) / len(self.trades)


def _adj_open(row: pd.Series) -> float:
    if row["Close"] == 0:
        return float(row["Adj Close"])
    return float(row["Adj Close"] * (row["Open"] / row["Close"]))


def _apply_entry_slippage(raw_price: float) -> float:
    return raw_price * (1 + SLIPPAGE_PCT)


def _apply_exit_slippage(raw_price: float) -> float:
    return raw_price * (1 - SLIPPAGE_PCT)


def load_raw_prices(ticker: str) -> pd.DataFrame:
    csv_path = DATA_DIR / f"{ticker}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No data file found for {ticker}: {csv_path}")
    return pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")


def _indicator_series(close: pd.Series, values, index: pd.Index) -> pd.Series:
    """Normalize pandas-ta output; ta may return None on thin history."""
    if values is None:
        return pd.Series(np.nan, index=index, dtype=float)
    return values


def enrich_sector(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI, EMA 5, SMA 200, and sentiment flags."""
    close = df["Adj Close"]
    out = df.copy()
    out["EMA_5"] = _indicator_series(close, ta.ema(close, length=EMA_FAST), out.index)
    out["SMA_200"] = _indicator_series(close, ta.sma(close, length=SMA_TREND), out.index)
    out["RSI_7"] = _indicator_series(close, ta.rsi(close, length=RSI_LENGTH), out.index)

    above_trend = close > out["SMA_200"]
    out["buy_pullback"] = (out["RSI_7"] < RSI_OVERSOLD) & above_trend
    out["deep_buy"] = (out["RSI_7"] < RSI_DEEP_BUY) & above_trend
    out["overextended"] = out["RSI_7"] > RSI_OVEREXTENDED
    out["rs_score"] = out.apply(_relative_strength_score, axis=1)
    return out


def _relative_strength_score(row: pd.Series) -> float:
    """Point-in-time RS score for a single row (scanner display)."""
    if pd.isna(row["SMA_200"]) or pd.isna(row["RSI_7"]):
        return -999.0

    close = float(row["Adj Close"])
    sma_200 = float(row["SMA_200"])
    ema_5 = float(row["EMA_5"])
    rsi = float(row["RSI_7"])

    if close <= sma_200:
        return -999.0

    score = 0.0
    if rsi < RSI_DEEP_BUY:
        score += 100.0
    elif rsi < RSI_OVERSOLD:
        score += 60.0
    else:
        score += max(0.0, (50.0 - rsi) * 0.5)

    score += max(0.0, RSI_OVERSOLD - rsi) * 2.0

    if close < ema_5:
        score += 15.0

    if rsi > RSI_OVEREXTENDED:
        score -= 80.0

    return score


def _equity(
    cash: float,
    positions: dict[str, OpenPosition],
    universe: dict[str, pd.DataFrame],
    day_idx: int,
) -> float:
    total = cash
    index = next(iter(universe.values())).index
    date = index[day_idx]
    for sector, pos in positions.items():
        mark = float(universe[sector].loc[date, "Adj Close"])
        total += pos.shares * mark
    return total


def _close_position(
    sector: str,
    position: OpenPosition,
    *,
    day_idx: int,
    index: pd.DatetimeIndex,
    raw_exit_price: float,
    exit_reason: str,
    cash: float,
    trades: list[Trade],
) -> float:
    exit_price = _apply_exit_slippage(raw_exit_price)
    proceeds = position.shares * exit_price
    pnl = proceeds - position.cost_basis
    trades.append(
        Trade(
            sector=sector,
            theme=position.theme,
            entry_date=position.entry_date,
            exit_date=index[day_idx],
            entry_price=position.entry_price,
            exit_price=exit_price,
            shares=position.shares,
            allocated_cash=position.cost_basis,
            pnl=pnl,
            exit_reason=exit_reason,
            holding_days=day_idx - position.entry_idx,
        )
    )
    return cash + proceeds


def _entry_priority(
    sector: str,
    day_idx: int,
    universe: dict[str, pd.DataFrame],
    positions: dict[str, OpenPosition],
) -> float:
    """Higher score = higher priority to receive capital (relative strength)."""
    row = universe[sector].iloc[day_idx]
    if not bool(row["buy_pullback"]):
        return -1.0

    score = float(row["rs_score"])

    others_overextended = any(
        bool(universe[other].iloc[day_idx]["overextended"])
        for other in universe
        if other != sector
    )
    if others_overextended:
        score += 25.0

    held_themes = {pos.theme for pos in positions.values()}
    sector_theme = theme_for_ticker(sector)
    if sector_theme and any(
        theme != sector_theme
        and bool(universe[held].iloc[day_idx]["overextended"])
        for held, theme in ((h, positions[h].theme) for h in positions)
    ):
        score += 15.0

    return score


def _held_themes(positions: dict[str, OpenPosition]) -> set[str]:
    return {pos.theme for pos in positions.values()}


def _max_new_allocation(cash: float, equity: float, open_slots: int) -> float:
    """Respect 35% per slot and 30% minimum cash reserve."""
    if open_slots <= 0:
        return 0.0
    min_cash = MIN_CASH_PCT * equity
    spendable = max(0.0, cash - min_cash)
    target = POSITION_ALLOC_PCT * equity
    return min(target, spendable)


def backtest_rotation(
    universe: dict[str, pd.DataFrame],
    starting_cash: float = STARTING_CASH,
    lookback_months: int | None = ANALYSIS_LOOKBACK_MONTHS,
) -> RotationBacktestResult:
    """
    Cross-sector rotation backtest on a shared cash wallet.

    - Max 2 positions at 35% each across all groups; one ticker per theme.
    - Relative strength scoring ranks pullback candidates within and across themes.
    - Force exit overextended holdings when another sector hits deep buy (RSI < 30).
    """
    index = next(iter(universe.values())).index
    start_idx, last_idx, window_start, window_end = resolve_simulation_range(
        index, lookback_months
    )
    cash = starting_cash
    positions: dict[str, OpenPosition] = {}
    trades: list[Trade] = []
    skipped_entries = 0

    for day_idx in range(start_idx, last_idx + 1):
        date = index[day_idx]

        # --- Forced rotation: exit overextended winner to fund deep-buy sector ---
        deep_buy_sectors = [
            s for s in universe if bool(universe[s].iloc[day_idx]["deep_buy"])
        ]
        if deep_buy_sectors:
            for held in list(positions):
                if held in deep_buy_sectors:
                    continue
                held_row = universe[held].iloc[day_idx]
                if bool(held_row["overextended"]):
                    target = min(
                        (s for s in deep_buy_sectors if s not in positions),
                        key=lambda s: float(universe[s].iloc[day_idx]["RSI_7"]),
                        default=None,
                    )
                    if target is not None:
                        pos = positions.pop(held)
                        raw_exit = float(held_row["Adj Close"])
                        cash = _close_position(
                            held,
                            pos,
                            day_idx=day_idx,
                            index=index,
                            raw_exit_price=raw_exit,
                            exit_reason=f"rotation_to_{target}",
                            cash=cash,
                            trades=trades,
                        )

        # --- Standard exits (stop loss + RSI take profit) ---
        for held in list(positions):
            pos = positions[held]
            if day_idx <= pos.entry_idx:
                continue

            row = universe[held].iloc[day_idx]
            adj_close = float(row["Adj Close"])
            prev_rsi = float(universe[held].iloc[day_idx - 1]["RSI_7"])
            rsi = float(row["RSI_7"])

            raw_exit: float | None = None
            exit_reason = ""

            if adj_close <= pos.stop_raw:
                raw_exit = pos.stop_raw
                exit_reason = "stop_loss"
            elif prev_rsi <= RSI_OVEREXTENDED < rsi:
                raw_exit = adj_close
                exit_reason = "take_profit_rsi"

            if raw_exit is not None:
                positions.pop(held)
                cash = _close_position(
                    held,
                    pos,
                    day_idx=day_idx,
                    index=index,
                    raw_exit_price=raw_exit,
                    exit_reason=exit_reason,
                    cash=cash,
                    trades=trades,
                )

        # --- Entries: signal yesterday -> enter today open ---
        if day_idx == 0:
            continue

        open_slots = MAX_POSITIONS - len(positions)
        if open_slots <= 0:
            continue

        equity = _equity(cash, positions, universe, day_idx)
        candidates: list[tuple[str, float]] = []

        for sector in universe:
            if sector in positions:
                continue
            theme = theme_for_ticker(sector)
            if theme and theme in _held_themes(positions):
                continue
            signal_yesterday = bool(universe[sector].iloc[day_idx - 1]["buy_pullback"])
            if not signal_yesterday:
                continue
            priority = _entry_priority(sector, day_idx - 1, universe, positions)
            if priority >= 0:
                candidates.append((sector, priority))

        candidates.sort(key=lambda item: item[1], reverse=True)

        for sector, _ in candidates:
            if open_slots <= 0:
                break

            equity = _equity(cash, positions, universe, day_idx)
            allocation = _max_new_allocation(cash, equity, open_slots)
            entry_row = universe[sector].iloc[day_idx]
            raw_entry = _adj_open(entry_row)
            entry_price = _apply_entry_slippage(raw_entry)

            if allocation < entry_price:
                skipped_entries += 1
                continue

            shares = allocation / entry_price
            cost_basis = shares * entry_price
            cash -= cost_basis

            positions[sector] = OpenPosition(
                sector=sector,
                theme=theme_for_ticker(sector) or sector,
                entry_date=date,
                entry_idx=day_idx,
                shares=shares,
                entry_price=entry_price,
                cost_basis=cost_basis,
                stop_raw=raw_entry * (1 - STOP_LOSS_PCT),
            )
            open_slots -= 1

    # Close remaining positions at last close of analysis window
    for held, pos in list(positions.items()):
        raw_exit = float(universe[held].iloc[last_idx]["Adj Close"])
        cash = _close_position(
            held,
            pos,
            day_idx=last_idx,
            index=index,
            raw_exit_price=raw_exit,
            exit_reason="end_of_data",
            cash=cash,
            trades=trades,
        )
        positions.pop(held)

    benchmark_ending, benchmark_return = _benchmark_split_hold(
        universe, starting_cash, BENCHMARK_SPLIT, start_idx, last_idx
    )

    return RotationBacktestResult(
        starting_cash=starting_cash,
        ending_equity=cash,
        trades=trades,
        skipped_entries=skipped_entries,
        benchmark_ending=benchmark_ending,
        benchmark_return_pct=benchmark_return,
        window_start=window_start,
        window_end=window_end,
    )


def _benchmark_split_hold(
    universe: dict[str, pd.DataFrame],
    starting_cash: float,
    tickers: tuple[str, str],
    start_idx: int,
    end_idx: int,
) -> tuple[float, float]:
    """50/50 buy & hold on two sector ETFs over the analysis window."""
    a, b = tickers
    start_a = float(universe[a]["Adj Close"].iloc[start_idx])
    end_a = float(universe[a]["Adj Close"].iloc[end_idx])
    start_b = float(universe[b]["Adj Close"].iloc[start_idx])
    end_b = float(universe[b]["Adj Close"].iloc[end_idx])

    half = starting_cash / 2
    ending = half * (end_a / start_a) + half * (end_b / start_b)
    return ending, (ending / starting_cash - 1) * 100


def format_price(value: float) -> str:
    return f"${value:,.2f}"


def _sentiment_label(row: pd.Series) -> str:
    if bool(row["deep_buy"]):
        return "DEEP BUY (RSI<30, above SMA200)"
    if bool(row["buy_pullback"]):
        return "Pullback ready (RSI<35, above SMA200)"
    if bool(row["overextended"]):
        return "Overextended (RSI>75) — source of funds"
    return "Neutral"


def print_sector_matrix(universe: dict[str, pd.DataFrame]) -> None:
    """Print relative strength matrix grouped by theme."""
    index = next(iter(universe.values())).index
    report_date = index[-1]

    print("=" * 100)
    print("CROSS-SECTOR RELATIVE STRENGTH SCANNER")
    print("=" * 100)
    print(f"Report date: {report_date.date()}")
    print()

    # Global RS ranks across all loaded tickers
    rs_ranking = sorted(
        universe.keys(),
        key=lambda t: float(universe[t].loc[report_date, "rs_score"]),
        reverse=True,
    )
    global_rank = {ticker: i + 1 for i, ticker in enumerate(rs_ranking)}

    headers = ["Theme", "Ticker", "RS Rank", "RS Score", "RSI 7", "EMA 5", "SMA 200", "Sentiment"]
    rows: list[list[str]] = []

    for theme, tickers in TICKER_CONFIG.items():
        theme_loaded = [t for t in tickers if t in universe]
        theme_ranking = sorted(
            theme_loaded,
            key=lambda t: float(universe[t].loc[report_date, "rs_score"]),
            reverse=True,
        )
        theme_rank = {ticker: i + 1 for i, ticker in enumerate(theme_ranking)}

        for ticker in tickers:
            if ticker not in universe:
                rows.append([theme, ticker, "—", "—", "—", "—", "—", "NO DATA"])
                continue
            row = universe[ticker].loc[report_date]
            rows.append(
                [
                    theme,
                    ticker,
                    f"{theme_rank[ticker]}/{len(theme_loaded)} (global #{global_rank[ticker]})",
                    f"{float(row['rs_score']):.1f}",
                    f"{float(row['RSI_7']):.1f}",
                    format_price(float(row["EMA_5"])),
                    format_price(float(row["SMA_200"])),
                    _sentiment_label(row),
                ]
            )

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def fmt(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(cells)) + " |"

    print(fmt(headers))
    print("|" + "|".join("-" * (w + 2) for w in col_widths) + "|")
    for row in rows:
        print(fmt(row))

    print()
    print("Top 3 global pullback candidates (by RS score):")
    for ticker in rs_ranking[:3]:
        row = universe[ticker].loc[report_date]
        theme = theme_for_ticker(ticker) or "?"
        print(
            f"  #{global_rank[ticker]} {ticker} [{theme}] "
            f"RS={float(row['rs_score']):.1f} RSI={float(row['RSI_7']):.1f} — {_sentiment_label(row)}"
        )

    print()
    print("Backtest universe: all loaded TICKER_CONFIG groups | max 2 positions | 1 per theme")
    print("Portfolio rules: max 2 positions @ 35% each | min 30% cash | rotate into pullbacks")
    print("=" * 100)


def print_rotation_backtest(result: RotationBacktestResult) -> None:
    print()
    print("=" * 80)
    print("CROSS-SECTOR ROTATION BACKTEST")
    print("=" * 80)
    print(f"Analysis window        : {result.window_start.date()} → {result.window_end.date()}")
    print(f"Starting cash          : {format_price(result.starting_cash)}")
    print(f"Ending equity          : {format_price(result.ending_equity)}")
    print(f"Rotation return        : {result.cumulative_return_pct:+.2f}%")
    print()
    print(f"Benchmark (50/50 {BENCHMARK_SPLIT[0]}/{BENCHMARK_SPLIT[1]} B&H)")
    print(f"  Ending value         : {format_price(result.benchmark_ending)}")
    print(f"  Benchmark return     : {result.benchmark_return_pct:+.2f}%")
    print(f"  Alpha vs benchmark   : {result.alpha_vs_benchmark_pct:+.2f}%")
    print()
    print(f"Total trades           : {result.total_trades}")
    print(f"Skipped entries        : {result.skipped_entries}")
    print(f"Win rate               : {result.win_rate_pct:.1f}%")
    print(f"Avg hold               : {result.avg_holding_days:.1f} days")
    print(f"Slippage               : {SLIPPAGE_PCT:.2%} per side")
    print()

    if result.trades:
        by_sector: dict[str, list[Trade]] = {}
        for trade in result.trades:
            by_sector.setdefault(trade.sector, []).append(trade)

        headers = ["Theme", "Ticker", "Trades", "Realized P&L", "Win Rate", "Avg Hold"]
        rows = []
        traded_tickers = sorted(by_sector.keys())
        for sector in traded_tickers:
            sector_trades = by_sector[sector]
            pnl = sum(t.pnl for t in sector_trades)
            wins = sum(1 for t in sector_trades if t.pnl > 0)
            wr = wins / len(sector_trades) * 100
            avg_hold = sum(t.holding_days for t in sector_trades) / len(sector_trades)
            theme = sector_trades[0].theme if sector_trades else "?"
            rows.append(
                [
                    theme,
                    sector,
                    str(len(sector_trades)),
                    f"{pnl:+,.0f}",
                    f"{wr:.1f}%",
                    f"{avg_hold:.1f}d",
                ]
            )

        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(cell))

        def fmt(cells: list[str]) -> str:
            return "| " + " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(cells)) + " |"

        print("Per-sector trade breakdown")
        print(fmt(headers))
        print("|" + "|".join("-" * (w + 2) for w in col_widths) + "|")
        for row in rows:
            print(fmt(row))

    print("=" * 80)


def resolve_simulation_range(
    index: pd.DatetimeIndex,
    lookback_months: int | None,
) -> tuple[int, int, pd.Timestamp, pd.Timestamp]:
    """
    Map a lookback window onto daily bar indices.

    Full history is still used upstream for indicator warm-up (e.g. SMA 200);
    only the returned index range is traded and reported.
    """
    end_idx = len(index) - 1
    window_end = index[end_idx]
    if lookback_months is None:
        return 0, end_idx, index[0], window_end

    window_start = window_end - pd.DateOffset(months=lookback_months)
    start_idx = int(index.searchsorted(window_start, side="left"))
    start_idx = min(start_idx, end_idx)
    return start_idx, end_idx, index[start_idx], window_end


def find_missing_tickers(tickers: list[str]) -> list[str]:
    """Return tickers with no CSV file in /data."""
    return sorted(t for t in tickers if not (DATA_DIR / f"{t}.csv").exists())


def print_missing_data_report(missing: list[str]) -> None:
    """Print a theme-grouped report of missing ETF downloads."""
    if not missing:
        return

    missing_set = set(missing)
    print()
    print("=" * 72)
    print("MISSING ETF DATA")
    print("=" * 72)
    print(f"The following {len(missing)} ticker(s) have no CSV in data/:")
    print()

    for theme, theme_tickers in TICKER_CONFIG.items():
        theme_missing = [t for t in theme_tickers if t in missing_set]
        if theme_missing:
            print(f"  {theme:<16} {', '.join(theme_missing)}")

    exec_only = [t for t in missing if t in set(LEVERAGED_EXEC.values())]
    if exec_only:
        parents = [
            f"{parent}→{child}"
            for parent, child in LEVERAGED_EXEC.items()
            if child in exec_only
        ]
        print()
        print("  Leveraged execution (needed for live fills):")
        print(f"    Missing: {', '.join(exec_only)}")
        if parents:
            print(f"    Used when signaling: {', '.join(parents)}")

    print()
    print("  Fix: python src/download_data.py")
    print("=" * 72)
    print()


def load_available_universe(
    tickers: list[str],
    *,
    report_missing: bool = True,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Load tickers that exist on disk; return (universe, missing_tickers)."""
    missing = find_missing_tickers(tickers)
    if missing and report_missing:
        print_missing_data_report(missing)

    universe: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        if ticker in missing:
            continue
        universe[ticker] = enrich_sector(load_raw_prices(ticker))
    return universe, missing


def align_universe(universe: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Align all loaded ETFs on a shared date index."""
    if not universe:
        return {}
    tickers = list(universe.keys())
    master_index = universe[tickers[0]].index
    for ticker in tickers[1:]:
        master_index = master_index.intersection(universe[ticker].index)
    return {ticker: universe[ticker].loc[master_index].copy() for ticker in tickers}


def main() -> None:
    print("Historical backtester — for daily pre-market signals, run: python src/pre_market_scanner.py")
    print()
    scan_tickers = all_tickers()
    loaded, missing = load_available_universe(scan_tickers)
    scan_universe = align_universe(loaded)

    if missing:
        print("Error: incomplete universe — re-run download_data.py before backtesting.")
        return

    if not scan_universe:
        print("Error: no ETF data found. Run download_data.py first.")
        return

    print_sector_matrix(scan_universe)

    if len(scan_universe) < 2:
        print("Error: need at least 2 tickers for rotation backtest.")
        return

    result = backtest_rotation(scan_universe)
    print_rotation_backtest(result)


if __name__ == "__main__":
    main()
