"""
Pre-Market Alpha Scanner — forward-looking daily decision tool.

Evening workflow (California / PT, after ~1 PM market close):
  1. Ask Agent to sync Robinhood → output/positions.json (see POSITIONS_SYNC_PROMPT)
  2. python src/pre_market_scanner.py

Auto-downloads stale CSVs. Writes output/latest_signals.json.
SELL signals only appear for symbols in positions.json when that file exists.
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
from ticker_config import LEVERAGED_EXEC, TICKER_CONFIG, all_tickers, execution_ticker, theme_for_ticker

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
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
         "symbol": "IXC",
         "quantity": 1.015,
         "shares_available_for_sells": 1.015,
         "average_buy_price": 49.25
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


def held_themes(snapshot: dict | None) -> set[str]:
    """Themes already occupied by open Robinhood positions."""
    themes: set[str] = set()
    for symbol in held_exec_symbols(snapshot):
        theme = theme_for_ticker(symbol)
        if theme:
            themes.add(theme)
        for signal_ticker, exec_ticker in LEVERAGED_EXEC.items():
            if exec_ticker == symbol:
                signal_theme = theme_for_ticker(signal_ticker)
                if signal_theme:
                    themes.add(signal_theme)
    return themes


def generate_signals(
    signal_universe: dict[str, pd.DataFrame],
    price_data: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
    execution_date: date,
    positions: dict | None = None,
) -> list[TradeAction]:
    """Build buy/sell advisories from the signal close for the next session open."""
    actions: list[TradeAction] = []
    idx = next(iter(signal_universe.values())).index.get_loc(signal_date)
    if idx < 1:
        return actions

    per_slot_notional = SUGGESTED_CAPITAL * POSITION_ALLOC_PCT
    held_symbols = held_exec_symbols(positions)
    themes_in_use = held_themes(positions)

    # --- Exits: only for symbols you actually hold (when positions.json present) ---
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
            if positions is None or exec_t not in held_symbols:
                continue
            held_qty = 0.0
            if positions:
                for pos in positions.get("positions", []):
                    if pos.get("symbol") == exec_t:
                        held_qty = _position_qty(pos)
                        break
            fill = _apply_exit_slippage(_est_open_price(exec_t, price_data))
            headline = f"SELL {exec_t} at {execution_date} open — {exit_reason}"
            if held_qty > 0:
                headline = (
                    f"SELL {held_qty:.4f} sh {exec_t} at {execution_date} open — {exit_reason}"
                )
            actions.append(
                TradeAction(
                    action="SELL",
                    signal_ticker=ticker,
                    exec_ticker=exec_t,
                    theme=theme,
                    reason=exit_reason,
                    headline=headline,
                    est_fill_price=fill,
                    shares=held_qty,
                    notional=held_qty * fill if held_qty > 0 else 0.0,
                )
            )

    # --- Entries: skip themes/symbols already held ---
    held_count = (
        sum(1 for p in positions.get("positions", []) if _position_qty(p) > 0)
        if positions
        else 0
    )
    open_slots = max(0, MAX_POSITIONS - held_count)

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
        if exec_t in held_symbols:
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
    positions: dict | None = None,
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
    if positions:
        synced = positions.get("synced_at", "?")
        held = held_exec_symbols(positions)
        print(f"Robinhood positions  : {len(held)} held ({', '.join(sorted(held)) or 'none'}) — synced {synced}")
    else:
        print("Robinhood positions  : not synced (SELL signals disabled; run Agent sync first)")
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
    positions: dict | None = None,
    path: Path = SIGNALS_JSON_PATH,
) -> Path:
    """Write signals for next-morning Robinhood MCP / automation handoff."""
    payload = {
        "signal_date": signal_date.date().isoformat(),
        "execution_date": execution_date.isoformat(),
        "suggested_capital": SUGGESTED_CAPITAL,
        "position_alloc_pct": POSITION_ALLOC_PCT,
        "positions_synced": positions is not None,
        "held_symbols": sorted(held_exec_symbols(positions)),
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
        signal_universe, price_data, signal_date = build_universe_clean()
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    execution_date = next_trading_day(signal_date)
    positions = load_positions_snapshot()
    actions = generate_signals(
        signal_universe, price_data, signal_date, execution_date, positions
    )
    print_watchlist(signal_universe, signal_date, execution_date, actions, positions)

    out = save_signals_json(signal_date, execution_date, actions, positions)
    print(f"Signals saved → {out}")
    if positions is None:
        print()
        print("Tip: sync Robinhood holdings first so SELL signals match your account.")
        print("Copy POSITIONS_SYNC_PROMPT from pre_market_scanner.py into Agent.")


if __name__ == "__main__":
    main()
