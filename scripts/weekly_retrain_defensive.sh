#!/usr/bin/env bash
set -euo pipefail

cd /home/jack/stock_rl_project

PYTHONPATH=src .venv/bin/python -m stock_rl.weekly_retrain \
  --config configs/KRX_E035_defensive_retrain.yaml \
  --rule strong_trend_full_else070 \
  --refresh-data
