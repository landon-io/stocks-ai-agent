"""
Pre-Market Alpha Scanner — QQQ park + swing sector rotation.

Evening workflow (California / PT, after ~1 PM market close):
  1. Ask Agent to sync Robinhood → output/positions.json (see POSITIONS_SYNC_PROMPT)
  2. python src/pre_market_scanner.py

Auto-downloads stale CSVs. Writes output/latest_signals.json.
SELL signals only appear for swing symbols in positions.json when that file exists.
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
from analyzer import find_missing_tickers, load_raw_prices, print_missing_data_report
from swing_backtest_sandbox import (
    DEFAULT_CONFIG,
    build_rsi_leaderboard,
    bottom_n_oversold_sectors,
    load_and_prepare,
)
from ticker_config import (
    MAXIMUM_SWING_POSITIONS,
    PARK_TICKER,
    SLIPPAGE_PCT,
    STOP_LOSS_PCT,
    SWING_ALLOCATION,
    SWING_SECTORS,
    TAKE_PROFIT_RSI,
    execution_ticker,
    is_park_symbol,
    scanner_tickers,
    swing_sector_for_ticker,
    swing_tickers,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
SIGNALS_JSON_PATH = OUTPUT_DIR / "latest_signals.json"
POSITIONS_JSON_PATH = OUTPUT_DIR / "positions.json"

POSITIONS_SYNC_PROMPT = """\
Sync my Robinhood Agentic account holdings to output/positions.json, then I will run pre_market_scanner.

Steps:
1. get_accounts → use the account with agentic_allowed=true
2. get_equity_positions for that account
3. Write output/positions.json as:
   {
     "account_number": "<full account number>",
     "synced_at": "<ISO-8601 UTC>",
     "positions": [
       {
         "symbol": "QQQ",
         "quantity": 10.5,
         "shares_available_for_sells": 10.5,
         "average_buy_price": 480.25
       }
     ]
   }
Only include positions with shares_available_for_sells > 0.
"""

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
    park_extract_pct: float = 0.0
    repark_to_qqq: bool = False


def _apply_entry_slippage(price: float) -> float:
    return price * (1 + SLIPPAGE_PCT)


def _apply_exit_slippage(price: float) -> float:
    return price * (1 - SLIPPAGE_PCT)


def _est_open_price(exec_ticker: str, price_data: dict[str, pd.DataFrame]) -> float:
    """Use the latest close as a pre-market proxy for today's open."""
    if exec_ticker not in price_data:
        raise ValueError(f"No price data for execution ticker {exec_ticker}")
    return float(price_data[exec_ticker]["Close"].iloc[-1])


def build_swing_universe() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.Timestamp]:
    """Load aligned swing data (RSI + SMA) and raw price references."""
    required = scanner_tickers()
    missing = find_missing_tickers(required)
    if missing:
        print_missing_data_report(missing)
        raise FileNotFoundError(
            f"Cannot scan: {len(missing)} ETF file(s) missing from data/. "
            "Run: python src/download_data.py"
        )

    _, aligned = load_and_prepare(swing_tickers(), DEFAULT_CONFIG)
    price_data = {t: load_raw_prices(t) for t in required}
    signal_date = next(iter(aligned.values())).index[-1]
    return aligned, price_data, signal_date


def next_trading_day(signal_date: pd.Timestamp) -> date:
    """Next NYSE session after the signal close."""
    return (signal_date + BDay(1)).date()


def load_positions_snapshot(path: Path = POSITIONS_JSON_PATH) -> dict | None:
    """Load Robinhood positions snapshot written by Agent (via MCP)."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _position_qty(position: dict) -> float:
    for key in ("shares_available_for_sells", "quantity"):
        raw = position.get(key)
        if raw is None:
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return 0.0


def _position_avg_price(position: dict) -> float | None:
    raw = position.get("average_buy_price")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def held_exec_symbols(snapshot: dict | None) -> set[str]:
    """Execution tickers currently held (sellable qty > 0)."""
    if not snapshot:
        return set()
    held: set[str] = set()
    for pos in snapshot.get("positions", []):
        symbol = pos.get("symbol")
        if symbol and _position_qty(pos) > 0:
            held.add(str(symbol))
    return held


def held_swing_sectors(snapshot: dict | None) -> set[str]:
    """Swing sectors occupied by open Robinhood positions (excludes QQQ park)."""
    sectors: set[str] = set()
    for symbol in held_exec_symbols(snapshot):
        if is_park_symbol(symbol):
            continue
        sector = swing_sector_for_ticker(symbol)
        if sector:
            sectors.add(sector)
    return sectors


def _position_for_symbol(snapshot: dict | None, symbol: str) -> dict | None:
    if not snapshot:
        return None
    for pos in snapshot.get("positions", []):
        if pos.get("symbol") == symbol:
            return pos
    return None


def generate_signals(
    aligned: dict[str, pd.DataFrame],
    price_data: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
    execution_date: date,
    positions: dict | None = None,
) -> list[TradeAction]:
    """Build buy/sell advisories from the signal close for the next session open."""
    actions: list[TradeAction] = []
    idx = next(iter(aligned.values())).index.get_loc(signal_date)
    held_symbols = held_exec_symbols(positions)
    held_sectors = held_swing_sectors(positions)

    # --- Swing exits: take profit or stop loss on held swing positions ---
    for sector, signal_ticker in SWING_SECTORS.items():
        exec_t = execution_ticker(signal_ticker)
        if positions is None or exec_t not in held_symbols:
            continue

        row = aligned[signal_ticker].iloc[idx]
        rsi = float(row["RSI"])
        close = float(row["Adj Close"])
        pos = _position_for_symbol(positions, exec_t)
        held_qty = _position_qty(pos) if pos else 0.0
        avg_buy = _position_avg_price(pos) if pos else None

        exit_reason = ""
        if rsi >= TAKE_PROFIT_RSI:
            exit_reason = "take_profit_rsi"
        elif avg_buy is not None and close <= avg_buy * (1 - STOP_LOSS_PCT):
            exit_reason = "stop_loss"

        if not exit_reason:
            continue

        fill = _apply_exit_slippage(_est_open_price(exec_t, price_data))
        headline = (
            f"SELL {held_qty:.4f} sh {exec_t} at {execution_date} open — {exit_reason}; "
            f"repark proceeds to {PARK_TICKER}"
        )
        actions.append(
            TradeAction(
                action="SELL",
                signal_ticker=signal_ticker,
                exec_ticker=exec_t,
                theme=sector,
                reason=exit_reason,
                headline=headline,
                est_fill_price=fill,
                shares=held_qty,
                notional=held_qty * fill if held_qty > 0 else 0.0,
                repark_to_qqq=True,
            )
        )

    # --- Swing entries: lowest RSI oversold sectors above SMA 200 ---
    held_swing_count = len(held_sectors)
    open_slots = max(0, MAXIMUM_SWING_POSITIONS - held_swing_count)
    if open_slots <= 0:
        return actions

    leaderboard = build_rsi_leaderboard(aligned, idx)
    candidates = bottom_n_oversold_sectors(
        leaderboard, open_slots, aligned, idx, DEFAULT_CONFIG
    )
    candidates = [
        (sector, ticker, rsi)
        for sector, ticker, rsi in candidates
        if sector not in held_sectors
    ]
    if not candidates:
        return actions

    park_extract_pct = SWING_ALLOCATION / len(candidates)
    per_slot_notional = SUGGESTED_CAPITAL * park_extract_pct

    for sector, signal_ticker, rsi in candidates:
        exec_t = execution_ticker(signal_ticker)
        if exec_t in held_symbols:
            continue
        fill = _apply_entry_slippage(_est_open_price(exec_t, price_data))
        shares = per_slot_notional / fill
        leverage_note = f" via {exec_t}" if exec_t != signal_ticker else ""
        actions.append(
            TradeAction(
                action="BUY",
                signal_ticker=signal_ticker,
                exec_ticker=exec_t,
                theme=sector,
                reason="swing_oversold",
                headline=(
                    f"BUY{leverage_note} at {execution_date} open — "
                    f"RSI {rsi:.1f} < {DEFAULT_CONFIG.rsi_oversold_threshold:.0f}, above SMA 200; "
                    f"trim {PARK_TICKER} park ~{park_extract_pct:.0%} to fund"
                ),
                est_fill_price=fill,
                shares=shares,
                notional=per_slot_notional,
                park_extract_pct=park_extract_pct,
            )
        )

    return actions


def print_watchlist(
    aligned: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
    trade_date: date,
    actions: list[TradeAction],
    positions: dict | None = None,
) -> None:
    idx = next(iter(aligned.values())).index.get_loc(signal_date)
    leaderboard = build_rsi_leaderboard(aligned, idx)

    print()
    print("=" * 88)
    print("PRE-MARKET SCANNER — QQQ PARK + SWING ROTATION")
    print("=" * 88)
    print(f"Signal session (T)   : {signal_date.date()}  (last closed trading day)")
    print(f"Execution session    : {trade_date.isoformat()}  (next market open)")
    print(f"Park asset           : {PARK_TICKER}  (core; trim to fund swings)")
    swing_list = ", ".join(SWING_SECTORS.values())
    print(f"Swing universe       : {swing_list}")
    print(f"Entry rule           : RSI < {DEFAULT_CONFIG.rsi_oversold_threshold:.0f} AND Close > SMA_{DEFAULT_CONFIG.sma_trend}")
    print(f"Exit rule            : RSI >= {TAKE_PROFIT_RSI:.0f} or {STOP_LOSS_PCT:.1%} stop")
    print(f"Data through         : {signal_date.date()}  (expected ≤ {expected_completed_session()})")
    print(
        f"Sizing reference     : ${SUGGESTED_CAPITAL:,.0f} @ up to "
        f"{SWING_ALLOCATION:.0%} swing pool / {MAXIMUM_SWING_POSITIONS} slots (display only)"
    )
    if positions:
        synced = positions.get("synced_at", "?")
        held = held_exec_symbols(positions)
        park = sorted(s for s in held if is_park_symbol(s))
        swings = sorted(s for s in held if swing_sector_for_ticker(s))
        print(f"Robinhood park       : {', '.join(park) or 'none'} — synced {synced}")
        print(f"Robinhood swings     : {', '.join(swings) or 'none'}")
    else:
        print("Robinhood positions  : not synced (SELL signals disabled; run Agent sync first)")
    print()

    print("-" * 88)
    print("SWING RSI LEADERBOARD (lowest = most oversold)")
    print("-" * 88)
    for sector, ticker, rsi in leaderboard:
        row = aligned[ticker].iloc[idx]
        sma = row["SMA_200"]
        above = float(row["Adj Close"]) > float(sma) if pd.notna(sma) else False
        flag = "ENTRY OK" if rsi < DEFAULT_CONFIG.rsi_oversold_threshold and above else "watch"
        print(f"  {ticker:5s} [{sector:14s}]  RSI={rsi:5.1f}  {flag}")
    print()

    print("=" * 88)
    print("PRE-MARKET TRADE WATCHLIST")
    print("=" * 88)

    if not actions:
        print("\nNo actionable swing signals for today's open (QQQ park unchanged).\n")
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
            if act.park_extract_pct:
                print(f"   Park trim    : ~{act.park_extract_pct:.0%} of {PARK_TICKER}")
            if act.repark_to_qqq:
                print(f"   Repark       : proceeds → {PARK_TICKER}")
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
    positions: dict | None = None,
    path: Path = SIGNALS_JSON_PATH,
) -> Path:
    """Write signals for next-morning Robinhood MCP / automation handoff."""
    payload = {
        "strategy": "qqq_park_swing",
        "park_ticker": PARK_TICKER,
        "swing_allocation": SWING_ALLOCATION,
        "maximum_swing_positions": MAXIMUM_SWING_POSITIONS,
        "signal_date": signal_date.date().isoformat(),
        "execution_date": execution_date.isoformat(),
        "suggested_capital": SUGGESTED_CAPITAL,
        "positions_synced": positions is not None,
        "held_symbols": sorted(held_exec_symbols(positions)),
        "swing_sectors": sorted(SWING_SECTORS.keys()),
        "actions": [asdict(a) for a in actions],
    }
    if positions:
        payload["positions"] = positions
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
        aligned, price_data, signal_date = build_swing_universe()
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    execution_date = next_trading_day(signal_date)
    positions = load_positions_snapshot()
    actions = generate_signals(
        aligned, price_data, signal_date, execution_date, positions
    )
    print_watchlist(aligned, signal_date, execution_date, actions, positions)

    out = save_signals_json(signal_date, execution_date, actions, positions)
    print(f"Signals saved → {out}")
    if positions is None:
        print()
        print("Tip: sync Robinhood holdings first so SELL signals match your account.")
        print("Copy POSITIONS_SYNC_PROMPT from pre_market_scanner.py into Agent.")


if __name__ == "__main__":
    main()
