#!/usr/bin/env bash
set -euo pipefail

cd /home/jack/stock_rl_project

LOG_DIR=/home/jack/stock_rl_project/logs
mkdir -p "$LOG_DIR"

{
  echo "[$(date --iso-8601=seconds)] weekly market email start"
  if ! PYTHONPATH=src .venv/bin/python -m stock_rl.update_daily_targets \
    --config configs/KRX_E032_liquid48_long_trend_min_exposure.yaml \
    --rule strong_trend_full_else070; then
    echo "[$(date --iso-8601=seconds)] target update failed; continuing with latest available targets"
  fi
  PYTHONPATH=src .venv/bin/python -m stock_rl.weekly_market_report "$@"
  echo "[$(date --iso-8601=seconds)] weekly market email done"
} >> "$LOG_DIR/weekly_market_email.log" 2>&1
