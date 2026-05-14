from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any


def load_collection_state(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    state_path = Path(path)
    if not state_path.exists():
        return {}
    return json.loads(state_path.read_text(encoding="utf-8"))


def save_collection_state(path: str | Path | None, state: dict[str, Any]) -> None:
    if path is None:
        return
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def mark_empty_response(state: dict[str, Any], kind: str, market: str, bas_dd: str) -> None:
    empty_responses = state.setdefault("empty_responses", {})
    empty_responses[_key(kind, market, bas_dd)] = _now_utc_iso()


def clear_empty_response(state: dict[str, Any], kind: str, market: str, bas_dd: str) -> None:
    state.get("empty_responses", {}).pop(_key(kind, market, bas_dd), None)


def recently_checked_empty(
    state: dict[str, Any],
    kind: str,
    market: str,
    bas_dd: str,
    ttl_minutes: int,
) -> bool:
    if ttl_minutes <= 0:
        return False
    checked_at = state.get("empty_responses", {}).get(_key(kind, market, bas_dd))
    if not checked_at:
        return False
    try:
        checked = dt.datetime.fromisoformat(str(checked_at))
    except ValueError:
        return False
    if checked.tzinfo is None:
        checked = checked.replace(tzinfo=dt.UTC)
    return _now_utc() - checked <= dt.timedelta(minutes=ttl_minutes)


def _key(kind: str, market: str, bas_dd: str) -> str:
    return f"{kind}:{str(market).upper()}:{bas_dd}"


def _now_utc() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _now_utc_iso() -> str:
    return _now_utc().isoformat()
