"""
Grid Search Optimizer
---------------------
Systematic sensitivity analysis over swing-backtest hyperparameters.

Runs every combination in the parameter matrix, records metrics in a DataFrame,
and prints a Markdown table sorted by Sharpe ratio.

Usage:
    python src/grid_search_optimizer.py
    python src/grid_search_optimizer.py --workers 8
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

# Allow imports when run as script from project root or src/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from swing_backtest_sandbox import (  # noqa: E402
    TICKERS,
    StrategyConfig,
    load_and_prepare,
    run_simulation,
)

# =============================================================================
# Parameter matrix
# =============================================================================
RSI_PERIOD_GRID = [5, 7, 10, 14]
RSI_OVERSOLD_THRESHOLD_GRID = [25, 30, 35, 40]
STOP_LOSS_PCT_GRID = [0.01, 0.015, 0.02, 0.03]
TAKE_PROFIT_RSI_GRID = [65, 70, 75]

_ALIGNED_CACHE: dict[int, dict] = {}


def _get_aligned(rsi_period: int) -> dict:
    """Cache aligned data per RSI period to avoid redundant I/O."""
    if rsi_period not in _ALIGNED_CACHE:
        config = StrategyConfig(rsi_period=rsi_period)
        _, aligned = load_and_prepare(TICKERS, config)
        _ALIGNED_CACHE[rsi_period] = aligned
    return _ALIGNED_CACHE[rsi_period]


def _run_single_combo(params: tuple[int, float, float, float]) -> dict:
    """Worker function: run one backtest and return summary metrics."""
    rsi_period, oversold, stop_loss, take_profit = params
    config = StrategyConfig(
        rsi_period=rsi_period,
        rsi_oversold_threshold=oversold,
        stop_loss_pct=stop_loss,
        take_profit_rsi=take_profit,
    )
    aligned = _get_aligned(rsi_period)
    result = run_simulation(config, aligned=aligned)
    return {
        "rsi_period": rsi_period,
        "rsi_oversold_threshold": oversold,
        "stop_loss_pct": stop_loss,
        "take_profit_rsi": take_profit,
        "ending_equity": round(result.ending_equity, 2),
        "total_return_pct": round(result.total_return_pct, 2),
        "alpha_vs_qqq_pct": round(result.alpha_vs_qqq_pct, 2),
        "sharpe_ratio": round(result.sharpe_ratio, 3),
        "win_rate_pct": round(result.win_rate_pct, 1),
        "num_trades": result.num_trades,
        "swing_pnl": round(result.swing_pnl, 2),
        "beats_qqq": result.alpha_vs_qqq_pct > 0,
    }


def build_parameter_grid() -> list[tuple[int, float, float, float]]:
    return list(
        itertools.product(
            RSI_PERIOD_GRID,
            RSI_OVERSOLD_THRESHOLD_GRID,
            STOP_LOSS_PCT_GRID,
            TAKE_PROFIT_RSI_GRID,
        )
    )


def run_grid_search(workers: int = 1) -> pd.DataFrame:
    """Execute all parameter combinations and return results DataFrame."""
    combos = build_parameter_grid()
    total = len(combos)
    results: list[dict] = []

    if workers <= 1:
        for i, combo in enumerate(combos, start=1):
            results.append(_run_single_combo(combo))
            if i % 32 == 0 or i == total:
                print(f"  Progress: {i}/{total} ({100 * i / total:.0f}%)", flush=True)
    else:
        # Pre-warm cache in parent so workers inherit via fork (Unix)
        for period in RSI_PERIOD_GRID:
            _get_aligned(period)

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_single_combo, c): c for c in combos}
            done = 0
            for future in as_completed(futures):
                results.append(future.result())
                done += 1
                if done % 32 == 0 or done == total:
                    print(f"  Progress: {done}/{total} ({100 * done / total:.0f}%)", flush=True)

    df = pd.DataFrame(results)
    return df.sort_values("sharpe_ratio", ascending=False).reset_index(drop=True)


def _summary_stats(df: pd.DataFrame) -> dict:
    return {
        "total_combos": len(df),
        "positive_alpha": int((df["alpha_vs_qqq_pct"] > 0).sum()),
        "negative_alpha": int((df["alpha_vs_qqq_pct"] <= 0).sum()),
        "pct_positive_alpha": round(100 * (df["alpha_vs_qqq_pct"] > 0).mean(), 1),
        "best_sharpe": df["sharpe_ratio"].max(),
        "best_return": df["total_return_pct"].max(),
        "median_return": round(df["total_return_pct"].median(), 2),
        "median_alpha": round(df["alpha_vs_qqq_pct"].median(), 2),
    }


def to_markdown_table(df: pd.DataFrame, top_n: int = 25) -> str:
    """Format top rows as a Markdown table."""
    display = df.head(top_n).copy()
    display["stop_loss_pct"] = display["stop_loss_pct"].map(lambda x: f"{x:.1%}")
    cols = [
        "rsi_period",
        "rsi_oversold_threshold",
        "stop_loss_pct",
        "take_profit_rsi",
        "total_return_pct",
        "alpha_vs_qqq_pct",
        "sharpe_ratio",
        "num_trades",
        "win_rate_pct",
        "swing_pnl",
    ]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in display[cols].iterrows():
        rows.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join([header, sep, *rows])


def print_report(df: pd.DataFrame) -> None:
    stats = _summary_stats(df)
    w = 72

    print()
    print("=" * w)
    print("GRID SEARCH — RSI SECTOR MEAN-REVERSION SENSITIVITY ANALYSIS")
    print("=" * w)
    print(f"Parameter space: {stats['total_combos']} combinations")
    print(f"  RSI_PERIOD              {RSI_PERIOD_GRID}")
    print(f"  RSI_OVERSOLD_THRESHOLD  {RSI_OVERSOLD_THRESHOLD_GRID}")
    print(f"  STOP_LOSS_PCT           {STOP_LOSS_PCT_GRID}")
    print(f"  TAKE_PROFIT_RSI         {TAKE_PROFIT_RSI_GRID}")
    print()
    print("AGGREGATE FINDINGS")
    print("-" * w)
    print(f"  Combos beating QQQ (α > 0) : {stats['positive_alpha']} / {stats['total_combos']} ({stats['pct_positive_alpha']}%)")
    print(f"  Combos with α ≤ 0          : {stats['negative_alpha']} / {stats['total_combos']}")
    print(f"  Best Sharpe ratio          : {stats['best_sharpe']:.3f}")
    print(f"  Best total return          : {stats['best_return']:+.2f}%")
    print(f"  Median total return        : {stats['median_return']:+.2f}%")
    print(f"  Median alpha vs QQQ        : {stats['median_alpha']:+.2f}%")
    print()

    if stats["pct_positive_alpha"] < 20:
        verdict = (
            "VERDICT: RSI-based sector mean-reversion shows WEAK mathematical edge "
            "in this regime — the vast majority of parameter combos fail to beat QQQ."
        )
    elif stats["pct_positive_alpha"] < 50:
        verdict = (
            "VERDICT: Mixed results — some parameter islands exist, but edge is inconsistent."
        )
    else:
        verdict = (
            "VERDICT: A meaningful subset of parameters beats QQQ — worth deeper validation."
        )
    print(verdict)
    print()
    print("TOP 25 COMBINATIONS (sorted by Sharpe ratio)")
    print("-" * w)
    print(to_markdown_table(df, top_n=25))
    print()
    print("TOP 10 BY TOTAL RETURN")
    print("-" * w)
    by_return = df.sort_values("total_return_pct", ascending=False).head(10)
    print(to_markdown_table(by_return, top_n=10))
    print("=" * w)


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search swing backtest hyperparameters")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes (default: 1; use 4-8 on multi-core machines)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional CSV path to save full results",
    )
    args = parser.parse_args()

    combos = len(build_parameter_grid())
    print(f"Starting grid search: {combos} combinations, workers={args.workers}")
    t0 = time.perf_counter()

    df = run_grid_search(workers=args.workers)

    elapsed = time.perf_counter() - t0
    print(f"Completed in {elapsed:.1f}s ({elapsed / combos:.2f}s per combo)")

    if args.output:
        out_path = Path(args.output)
        df.to_csv(out_path, index=False)
        print(f"Full results saved to {out_path}")

    print_report(df)


if __name__ == "__main__":
    main()
