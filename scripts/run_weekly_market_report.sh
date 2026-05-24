#!/usr/bin/env bash
set -euo pipefail

cd /home/jack/stock_rl_project
PYTHONPATH=src .venv/bin/python -m stock_rl.weekly_market_report "$@"
