#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/jack/trail and error/weekly_market_report"
LOG_FILE="$PROJECT_DIR/weekly.log"
PRIMARY_PYTHON_BIN="/home/jack/trail and error/.venv/bin/python"
LEGACY_PYTHON_BIN="/home/jack/venvs/weekly_market_report/bin/python"

mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

if [[ -x "$PRIMARY_PYTHON_BIN" ]]; then
  PYTHON_BIN="$PRIMARY_PYTHON_BIN"
elif [[ -x "$LEGACY_PYTHON_BIN" ]]; then
  PYTHON_BIN="$LEGACY_PYTHON_BIN"
else
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] weekly job end (failed: python not found)" >> "$LOG_FILE"
  exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] weekly job start" >> "$LOG_FILE"
if "$PYTHON_BIN" "$PROJECT_DIR/weekly_report.py" >> "$LOG_FILE" 2>&1; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] weekly job end (success)" >> "$LOG_FILE"
else
  rc=$?
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] weekly job end (failed: exit $rc)" >> "$LOG_FILE"
  exit "$rc"
fi
