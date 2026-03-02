#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/jack/trail and error/weekly_market_report"
PYTHON_BIN="/home/jack/trail and error/.venv/bin/python"
LOG_FILE="$PROJECT_DIR/weekly.log"

mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] weekly job start" >> "$LOG_FILE"
if "$PYTHON_BIN" "$PROJECT_DIR/weekly_report.py" >> "$LOG_FILE" 2>&1; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] weekly job end (success)" >> "$LOG_FILE"
else
  rc=$?
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] weekly job end (failed: exit $rc)" >> "$LOG_FILE"
  exit "$rc"
fi
