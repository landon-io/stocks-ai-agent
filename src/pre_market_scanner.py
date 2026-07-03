"""
Pre-Market Alpha Scanner — forward-looking daily decision tool.

Evening workflow (after the close):
  python src/pre_market_scanner.py

Auto-downloads missing/stale CSV data before scanning (use --no-download to skip).
Writes signals to output/latest_signals.json every run.
For historical backtesting, run analyzer.py or swing_backtest_sandbox.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import pandas as pd
from pandas.tseries.offsets import BDay

from download_data import ensure_fresh_data, expected_completed_session, is_market_data_stale
from analyzer import (
    MAX_POSITIONS,
    POSITION_ALLOC_PCT,
    RSI_OVEREXTENDED,
    SLIPPAGE_PCT,
    align_universe,
    find_missing_tickers,
    load_available_universe,
    load_raw_prices,
    print_missing_data_report,
)
from ticker_config import TICKER_CONFIG, all_tickers, execution_ticker, theme_for_ticker

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
SIGNALS_JSON_PATH = OUTPUT_DIR / "latest_signals.json"

# Hypothetical capital for sizing display only (not persisted).
SUGGESTED_CAPITAL = 100_000.0


@dataclass
class TradeAction:
    action: str
    signal_ticker: str
    exec_ticker: str
    theme: str
    reason: str
    headline: str
    est_fill_price: float
    shares: float = 0.0
    notional: float = 0.0


def _apply_entry_slippage(price: float) -> float:
    return price * (1 + SLIPPAGE_PCT)


def _apply_exit_slippage(price: float) -> float:
    return price * (1 - SLIPPAGE_PCT)


def _est_open_price(exec_ticker: str, price_data: dict[str, pd.DataFrame]) -> float:
    """Use the latest close as a pre-market proxy for today's open."""
    if exec_ticker not in price_data:
        raise ValueError(f"No price data for execution ticker {exec_ticker}")
    return float(price_data[exec_ticker]["Close"].iloc[-1])


def _sentiment_text(row: pd.Series) -> str:
    if bool(row["deep_buy"]):
        return f"deeply oversold (RSI {float(row['RSI_7']):.1f})"
    if bool(row["buy_pullback"]):
        return f"oversold pullback (RSI {float(row['RSI_7']):.1f})"
    if bool(row["overextended"]):
        return f"overextended (RSI {float(row['RSI_7']):.1f})"
    return f"neutral (RSI {float(row['RSI_7']):.1f})"


def build_universe_clean() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.Timestamp]:
    """Load aligned signal data (T-1) and execution price references."""
    expected = all_tickers()
    missing = find_missing_tickers(expected)
    if missing:
        print_missing_data_report(missing)
        raise FileNotFoundError(
            f"Cannot scan: {len(missing)} ETF file(s) missing from data/. "
            "Run: python src/download_data.py"
        )

    available, _ = load_available_universe(expected, report_missing=False)
    signal_tickers = [t for t in available if t not in {"SOXL", "TQQQ"}]
    signal_universe = align_universe({t: available[t] for t in signal_tickers})
    if not signal_universe:
        raise FileNotFoundError("No ETF data found. Run download_data.py first.")

    price_data = {t: load_raw_prices(t) for t in available}
    signal_date = next(iter(signal_universe.values())).index[-1]
    return signal_universe, price_data, signal_date


def next_trading_day(signal_date: pd.Timestamp) -> date:
    """Next NYSE session after the signal close."""
    return (signal_date + BDay(1)).date()


def generate_signals(
    signal_universe: dict[str, pd.DataFrame],
    price_data: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
    execution_date: date,
) -> list[TradeAction]:
    """Build buy/sell advisories from the signal close for the next session open."""
    actions: list[TradeAction] = []
    idx = next(iter(signal_universe.values())).index.get_loc(signal_date)
    if idx < 1:
        return actions

    per_slot_notional = SUGGESTED_CAPITAL * POSITION_ALLOC_PCT

    # --- Exit watch: RSI take-profit cross or overextended with better rotation ---
    deep_buyers = [
        t for t in signal_universe if bool(signal_universe[t].iloc[idx]["deep_buy"])
    ]
    for ticker, df in signal_universe.items():
        row = df.iloc[idx]
        prev = df.iloc[idx - 1]
        prev_rsi = float(prev["RSI_7"])
        rsi = float(row["RSI_7"])
        theme = theme_for_ticker(ticker) or ticker
        exec_t = execution_ticker(ticker)
        if exec_t not in price_data:
            continue

        exit_reason = ""
        if prev_rsi <= RSI_OVEREXTENDED < rsi:
            exit_reason = "take_profit_rsi"
        elif bool(row["overextended"]) and deep_buyers:
            rot = next((t for t in deep_buyers if theme_for_ticker(t) != theme), None)
            if rot:
                exit_reason = f"rotation_to_{rot}"

        if exit_reason:
            fill = _apply_exit_slippage(_est_open_price(exec_t, price_data))
            actions.append(
                TradeAction(
                    action="SELL",
                    signal_ticker=ticker,
                    exec_ticker=exec_t,
                    theme=theme,
                    reason=exit_reason,
                    headline=f"Consider SELL {exec_t} at {execution_date} open — {exit_reason}",
                    est_fill_price=fill,
                )
            )

    # --- Entry candidates from T-1 pullback signals ---
    themes_in_use: set[str] = set()
    open_slots = MAX_POSITIONS

    pullback_candidates = []
    for ticker, df in signal_universe.items():
        row = df.iloc[idx]
        if not bool(row["buy_pullback"]):
            continue
        theme = theme_for_ticker(ticker) or ticker
        if theme in themes_in_use:
            continue
        exec_t = execution_ticker(ticker)
        if exec_t not in price_data:
            continue
        pullback_candidates.append((float(row["rs_score"]), ticker, theme, row))

    pullback_candidates.sort(reverse=True, key=lambda x: x[0])

    overextended = [
        (t, signal_universe[t].iloc[idx])
        for t in signal_universe
        if bool(signal_universe[t].iloc[idx]["overextended"])
    ]
    strongest = max(overextended, key=lambda item: float(item[1]["RSI_7"])) if overextended else None

    for _, ticker, theme, row in pullback_candidates[:open_slots]:
        exec_t = execution_ticker(ticker)
        fill = _apply_entry_slippage(_est_open_price(exec_t, price_data))
        shares = per_slot_notional / fill

        rotation_note = ""
        if strongest and strongest[0] != ticker:
            rot_t, rot_row = strongest
            rotation_note = (
                f"{ticker} is {_sentiment_text(row)} relative to "
                f"{rot_t} ({_sentiment_text(rot_row)}). "
            )

        leverage_note = f" via {exec_t}" if exec_t != ticker else ""
        actions.append(
            TradeAction(
                action="BUY",
                signal_ticker=ticker,
                exec_ticker=exec_t,
                theme=theme,
                reason="buy_pullback",
                headline=(
                    f"BUY{leverage_note} at {execution_date} market open — "
                    f"{rotation_note}"
                    f"RS score {float(row['rs_score']):.1f}, above SMA 200"
                ),
                est_fill_price=fill,
                shares=shares,
                notional=per_slot_notional,
            )
        )
        themes_in_use.add(theme)

    return actions


def print_watchlist(
    signal_universe: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
    trade_date: date,
    actions: list[TradeAction],
) -> None:
    print()
    print("=" * 88)
    print("PRE-MARKET ALPHA SCANNER")
    print("=" * 88)
    print(f"Signal session (T)   : {signal_date.date()}  (last closed trading day)")
    print(f"Execution session    : {trade_date.isoformat()}  (next market open)")
    print(f"Universe scanned     : {len(signal_universe)} ETFs across {len(TICKER_CONFIG)} themes")
    print(f"Data through         : {signal_date.date()}  (expected ≤ {expected_completed_session()})")
    print(f"Sizing reference     : ${SUGGESTED_CAPITAL:,.0f} @ {POSITION_ALLOC_PCT:.0%} per slot (display only)")
    print()

    print("-" * 88)
    print("RELATIVE STRENGTH SNAPSHOT (signal close)")
    print("-" * 88)
    ranked = sorted(
        signal_universe.items(),
        key=lambda item: float(item[1].iloc[-1]["rs_score"]),
        reverse=True,
    )
    for ticker, df in ranked[:5]:
        row = df.iloc[-1]
        theme = theme_for_ticker(ticker) or "?"
        print(
            f"  {ticker:5s} [{theme:14s}]  RS={float(row['rs_score']):5.1f}  "
            f"RSI={float(row['RSI_7']):5.1f}  {_sentiment_text(row)}"
        )
    print()

    print("=" * 88)
    print("PRE-MARKET TRADE WATCHLIST")
    print("=" * 88)

    if not actions:
        print("\nNo actionable signals for today's open.\n")
    else:
        for i, act in enumerate(actions, 1):
            icon = "🚨" if act.action == "BUY" else "🔻"
            print()
            print(f"{icon} [SIGNAL DETECTED] #{i} — {act.action}")
            print(f"   Theme        : {act.theme}")
            print(f"   Signal ETF   : {act.signal_ticker}")
            print(f"   Execute ETF  : {act.exec_ticker}")
            print(f"   Reason       : {act.reason}")
            print(f"   PRE-MARKET ACTION: {act.headline}")
            if act.shares:
                print(
                    f"   Est. fill    : ${act.est_fill_price:,.2f}  "
                    f"| Shares: {act.shares:,.4f}  | Notional: ${act.notional:,.2f}"
                )
            else:
                print(f"   Est. fill    : ${act.est_fill_price:,.2f}")

    print()
    print("=" * 88)


def save_signals_json(
    signal_date: pd.Timestamp,
    execution_date: date,
    actions: list[TradeAction],
    path: Path = SIGNALS_JSON_PATH,
) -> Path:
    """Write signals for next-morning Robinhood MCP / automation handoff."""
    payload = {
        "signal_date": signal_date.date().isoformat(),
        "execution_date": execution_date.isoformat(),
        "suggested_capital": SUGGESTED_CAPITAL,
        "position_alloc_pct": POSITION_ALLOC_PCT,
        "actions": [asdict(a) for a in actions],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-market ETF signal scanner")
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip auto-download even when local CSV data is missing or stale",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Always re-download all tickers before scanning",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.no_download:
        try:
            if ensure_fresh_data(force=args.force_download):
                print("Market data refreshed.\n")
        except RuntimeError as exc:
            print(f"Error: {exc}")
            sys.exit(1)
    else:
        stale, latest, expected = is_market_data_stale()
        if stale:
            print(
                f"Warning: local data ends {latest}; expected through {expected}. "
                "Re-run without --no-download to refresh.\n"
            )
    try:
        signal_universe, price_data, signal_date = build_universe_clean()
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    execution_date = next_trading_day(signal_date)
    actions = generate_signals(signal_universe, price_data, signal_date, execution_date)
    print_watchlist(signal_universe, signal_date, execution_date, actions)

    out = save_signals_json(signal_date, execution_date, actions)
    print(f"Signals saved → {out}")


if __name__ == "__main__":
    main()
