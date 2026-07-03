# Stocks AI Agent

Cross-sector ETF rotation strategy with pre-market scanning, backtesting, and signal generation.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Download historical data:

```bash
python src/download_data.py
```

Run pre-market scanner (writes signals to `output/`):

```bash
python src/pre_market_scanner.py
```

Run backtest:

```bash
python src/analyzer.py
```

## Project structure

- `src/ticker_config.py` — ETF universe and theme mapping
- `src/download_data.py` — fetch OHLCV via yfinance
- `src/pre_market_scanner.py` — daily signal generation
- `src/analyzer.py` — historical backtester
- `src/swing_backtest_sandbox.py` — sandbox backtests
- `src/grid_search_optimizer.py` — parameter optimization
- `data/` — cached price CSVs
- `output/` — latest trading signals (gitignored)
