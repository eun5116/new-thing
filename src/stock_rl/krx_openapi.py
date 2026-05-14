from __future__ import annotations

import datetime as dt
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from stock_rl.collection_state import (
    clear_empty_response,
    load_collection_state,
    mark_empty_response,
    recently_checked_empty,
    save_collection_state,
)

KOSPI_STOCK_DAILY_URL = "http://data-dbg.krx.co.kr/svc/apis/sto/stk_bydd_trd"
KOSDAQ_STOCK_DAILY_URL = "http://data-dbg.krx.co.kr/svc/apis/sto/ksq_bydd_trd"
KOSPI_INDEX_DAILY_URL = "http://data-dbg.krx.co.kr/svc/apis/idx/kospi_dd_trd"
KOSDAQ_INDEX_DAILY_URL = "http://data-dbg.krx.co.kr/svc/apis/idx/kosdaq_dd_trd"
KOSPI_ISSUE_BASE_URL = "http://data-dbg.krx.co.kr/svc/apis/sto/stk_isu_base_info"
KOSDAQ_ISSUE_BASE_URL = "http://data-dbg.krx.co.kr/svc/apis/sto/ksq_isu_base_info"

STOCK_DAILY_URLS = {
    "KOSPI": KOSPI_STOCK_DAILY_URL,
    "KOSDAQ": KOSDAQ_STOCK_DAILY_URL,
}

INDEX_DAILY_URLS = {
    "KOSPI": KOSPI_INDEX_DAILY_URL,
    "KOSDAQ": KOSDAQ_INDEX_DAILY_URL,
}

ISSUE_BASE_URLS = {
    "KOSPI": KOSPI_ISSUE_BASE_URL,
    "KOSDAQ": KOSDAQ_ISSUE_BASE_URL,
}


@dataclass(frozen=True)
class KrxOpenApiClient:
    auth_key: str
    timeout_seconds: int = 60
    max_retries: int = 3

    @classmethod
    def from_env(cls, env_var: str = "KRX_AUTH_KEY") -> "KrxOpenApiClient":
        _load_local_env(Path.cwd() / ".env")
        auth_key = os.getenv(env_var, "").strip()
        if not auth_key:
            raise RuntimeError(f"{env_var} is not set.")
        return cls(auth_key=auth_key)

    def get_json(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(
                    url,
                    headers={"AUTH_KEY": self.auth_key},
                    params=params or {},
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError(f"Unexpected KRX response type: {type(payload).__name__}")
                return payload
            except requests.RequestException as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(attempt * 2)
        assert last_error is not None
        raise last_error

    def fetch_stock_daily(self, market: str, bas_dd: str) -> pd.DataFrame:
        market_key = _normalize_market(market)
        url = STOCK_DAILY_URLS[market_key]
        payload = self.get_json(url, {"basDd": _format_bas_dd(bas_dd)})
        return pd.json_normalize(payload.get("OutBlock_1", []))

    def fetch_issue_base(self, market: str, bas_dd: str) -> pd.DataFrame:
        market_key = _normalize_market(market)
        url = ISSUE_BASE_URLS[market_key]
        payload = self.get_json(url, {"basDd": _format_bas_dd(bas_dd)})
        return pd.json_normalize(payload.get("OutBlock_1", []))

    def fetch_index_daily(self, market: str, bas_dd: str) -> pd.DataFrame:
        market_key = _normalize_market(market)
        url = INDEX_DAILY_URLS[market_key]
        payload = self.get_json(url, {"basDd": _format_bas_dd(bas_dd)})
        return pd.json_normalize(payload.get("OutBlock_1", []))


def _normalize_market(market: str) -> str:
    market_key = str(market).strip().upper()
    if market_key not in {"KOSPI", "KOSDAQ"}:
        raise ValueError(f"Unsupported KRX market: {market}")
    return market_key


def _load_local_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _format_bas_dd(value: str | dt.date | pd.Timestamp) -> str:
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y%m%d")
    if isinstance(value, dt.date):
        return value.strftime("%Y%m%d")
    text = str(value).strip()
    if "-" in text:
        return pd.to_datetime(text).strftime("%Y%m%d")
    return text


def _to_number(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().replace(",", "")
    if not text or text == "-":
        return None
    return pd.to_numeric(text, errors="coerce")


def normalize_stock_daily(raw: pd.DataFrame, market: str, tickers: list[str] | None = None) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"])

    frame = raw.copy()
    ticker_column = "ISU_SRT_CD" if "ISU_SRT_CD" in frame.columns else "ISU_CD"
    if ticker_column not in frame.columns:
        raise ValueError("KRX stock daily response is missing ISU_SRT_CD/ISU_CD.")

    frame["ticker"] = frame[ticker_column].astype(str).str.zfill(6)
    if tickers:
        wanted = {str(ticker).replace(".KS", "").replace(".KQ", "").zfill(6) for ticker in tickers}
        frame = frame[frame["ticker"].isin(wanted)]

    rename = {
        "BAS_DD": "date",
        "TDD_OPNPRC": "open",
        "TDD_HGPRC": "high",
        "TDD_LWPRC": "low",
        "TDD_CLSPRC": "close",
        "ACC_TRDVOL": "volume",
        "ACC_TRDVAL": "trading_value",
        "MKTCAP": "market_cap",
        "FLUC_RT": "change_pct",
    }
    existing = {source: target for source, target in rename.items() if source in frame.columns}
    frame = frame.rename(columns=existing)
    if "date" not in frame.columns:
        raise ValueError("KRX stock daily response is missing BAS_DD.")

    frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d", errors="coerce").dt.date
    frame["market"] = _normalize_market(market)
    for column in ["open", "high", "low", "close", "volume", "trading_value", "market_cap", "change_pct"]:
        if column in frame.columns:
            frame[column] = frame[column].map(_to_number)
    frame["adj_close"] = frame["close"]

    keep = [
        "date",
        "ticker",
        "market",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "trading_value",
        "market_cap",
        "change_pct",
    ]
    existing_keep = [column for column in keep if column in frame.columns]
    return frame[existing_keep].dropna(subset=["date", "close"]).sort_values(["ticker", "date"])


def normalize_issue_base(raw: pd.DataFrame, market: str) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    frame = raw.copy()
    rename = {
        "ISU_CD": "isin",
        "ISU_SRT_CD": "ticker",
        "ISU_NM": "name",
        "ISU_ABBRV": "abbrv",
        "ISU_ENG_NM": "english_name",
        "LIST_DD": "list_date",
        "MKT_TP_NM": "market",
        "SECUGRP_NM": "security_group",
        "SECT_TP_NM": "section",
        "KIND_STKCERT_TP_NM": "stock_type",
        "PARVAL": "par_value",
        "LIST_SHRS": "listed_shares",
    }
    frame = frame.rename(columns={source: target for source, target in rename.items() if source in frame.columns})
    if "ticker" in frame.columns:
        frame["ticker"] = frame["ticker"].astype(str).str.zfill(6)
    if "list_date" in frame.columns:
        frame["list_date"] = pd.to_datetime(frame["list_date"], format="%Y%m%d", errors="coerce").dt.date
    for column in ["par_value", "listed_shares"]:
        if column in frame.columns:
            frame[column] = frame[column].map(_to_number)
    frame["market"] = _normalize_market(market)
    return frame.sort_values("ticker").reset_index(drop=True)


def normalize_index_daily(raw: pd.DataFrame, market: str) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    frame = raw.copy()
    rename = {
        "BAS_DD": "date",
        "IDX_CLSS": "index_class",
        "IDX_NM": "index_name",
        "CLSPRC_IDX": "close",
        "CMPPREVDD_IDX": "change",
        "FLUC_RT": "change_pct",
        "OPNPRC_IDX": "open",
        "HGPRC_IDX": "high",
        "LWPRC_IDX": "low",
        "ACC_TRDVOL": "volume",
        "ACC_TRDVAL": "trading_value",
        "MKTCAP": "market_cap",
    }
    frame = frame.rename(columns={source: target for source, target in rename.items() if source in frame.columns})
    if "date" not in frame.columns:
        raise ValueError("KRX index daily response is missing BAS_DD.")
    frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d", errors="coerce").dt.date
    frame["market"] = _normalize_market(market)
    for column in ["close", "change", "change_pct", "open", "high", "low", "volume", "trading_value", "market_cap"]:
        if column in frame.columns:
            frame[column] = frame[column].map(_to_number)
    return frame.dropna(subset=["date"]).sort_values(["index_name", "date"]).reset_index(drop=True)


def _read_daily_cache(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
    if "ticker" in frame.columns:
        frame["ticker"] = frame["ticker"].astype(str).str.zfill(6)
    return frame


def _write_daily_cache(frame: pd.DataFrame, path: Path) -> None:
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)


def fetch_stock_prices(
    client: KrxOpenApiClient,
    market: str,
    tickers: list[str],
    start: str,
    end: str | None = None,
    cache_dir: str | Path | None = None,
    progress_every: int = 50,
    refresh_empty_recent_days: int = 7,
    state_path: str | Path | None = None,
    empty_cache_ttl_minutes: int = 60,
) -> pd.DataFrame:
    end_date = pd.Timestamp(end).date() if end else dt.date.today()
    dates = pd.bdate_range(start=start, end=end_date)
    market_key = _normalize_market(market)
    cache_path = Path(cache_dir) if cache_dir else None
    if cache_path:
        cache_path.mkdir(parents=True, exist_ok=True)
    state = load_collection_state(state_path)
    frames: list[pd.DataFrame] = []
    total = len(dates)
    for index, bas_dd in enumerate(dates, start=1):
        bas_dd_text = _format_bas_dd(bas_dd)
        daily_cache = cache_path / f"{market_key}_{bas_dd_text}.parquet" if cache_path else None
        legacy_cache = cache_path / f"{market_key}_{bas_dd_text}.csv" if cache_path else None
        if daily_cache and daily_cache.exists():
            normalized = _read_daily_cache(daily_cache)
            if normalized.empty and _should_refresh_empty_cache(bas_dd, refresh_empty_recent_days):
                normalized = _refresh_stock_cache_if_needed(
                    client,
                    market_key,
                    bas_dd,
                    daily_cache,
                    state,
                    state_path,
                    empty_cache_ttl_minutes,
                )
        elif legacy_cache and legacy_cache.exists():
            normalized = _read_daily_cache(legacy_cache)
            if normalized.empty and _should_refresh_empty_cache(bas_dd, refresh_empty_recent_days):
                normalized = _refresh_stock_cache_if_needed(
                    client,
                    market_key,
                    bas_dd,
                    daily_cache,
                    state,
                    state_path,
                    empty_cache_ttl_minutes,
                )
        else:
            raw = client.fetch_stock_daily(market_key, bas_dd)
            normalized = normalize_stock_daily(raw, market=market_key, tickers=None)
            if daily_cache is not None:
                _write_daily_cache(normalized, daily_cache)
            _record_cache_result(state, state_path, "stock", market_key, bas_dd_text, normalized)
        wanted = {str(ticker).replace(".KS", "").replace(".KQ", "").zfill(6) for ticker in tickers}
        if not normalized.empty and wanted.difference(set(normalized["ticker"].astype(str).str.zfill(6))):
            raw = client.fetch_stock_daily(market_key, bas_dd)
            normalized = normalize_stock_daily(raw, market=market_key, tickers=None)
            if daily_cache is not None:
                _write_daily_cache(normalized, daily_cache)
            _record_cache_result(state, state_path, "stock", market_key, bas_dd_text, normalized)
        if not normalized.empty:
            normalized = normalized[normalized["ticker"].isin(wanted)]
        if not normalized.empty:
            frames.append(normalized)
        if progress_every and (index == 1 or index % progress_every == 0 or index == total):
            print(f"KRX {market_key} {index}/{total} {bas_dd_text} rows={len(normalized)}", flush=True)
    if not frames:
        raise ValueError(f"No KRX price data returned for {market} {tickers}.")
    return pd.concat(frames, ignore_index=True).drop_duplicates(["ticker", "date"], keep="last")


def fetch_index_history(
    client: KrxOpenApiClient,
    market: str,
    start: str,
    end: str | None = None,
    index_names: list[str] | None = None,
    cache_dir: str | Path | None = None,
    progress_every: int = 50,
    refresh_empty_recent_days: int = 7,
    state_path: str | Path | None = None,
    empty_cache_ttl_minutes: int = 60,
) -> pd.DataFrame:
    end_date = pd.Timestamp(end).date() if end else dt.date.today()
    dates = pd.bdate_range(start=start, end=end_date)
    market_key = _normalize_market(market)
    cache_path = Path(cache_dir) if cache_dir else None
    if cache_path:
        cache_path.mkdir(parents=True, exist_ok=True)
    state = load_collection_state(state_path)
    wanted = set(index_names or [])
    frames: list[pd.DataFrame] = []
    total = len(dates)
    for index, bas_dd in enumerate(dates, start=1):
        bas_dd_text = _format_bas_dd(bas_dd)
        daily_cache = cache_path / f"{market_key}_INDEX_{bas_dd_text}.parquet" if cache_path else None
        if daily_cache and daily_cache.exists():
            normalized = _read_daily_cache(daily_cache)
            if normalized.empty and _should_refresh_empty_cache(bas_dd, refresh_empty_recent_days):
                normalized = _refresh_index_cache_if_needed(
                    client,
                    market_key,
                    bas_dd,
                    daily_cache,
                    state,
                    state_path,
                    empty_cache_ttl_minutes,
                )
        else:
            raw = client.fetch_index_daily(market_key, bas_dd)
            normalized = normalize_index_daily(raw, market_key)
            if daily_cache is not None:
                _write_daily_cache(normalized, daily_cache)
            _record_cache_result(state, state_path, "index", market_key, bas_dd_text, normalized)
        if wanted and "index_name" in normalized.columns:
            normalized = normalized[normalized["index_name"].isin(wanted)]
        if not normalized.empty:
            frames.append(normalized)
        if progress_every and (index == 1 or index % progress_every == 0 or index == total):
            print(f"KRX {market_key} index {index}/{total} {bas_dd_text} rows={len(normalized)}", flush=True)
    if not frames:
        raise ValueError(f"No KRX index data returned for {market}.")
    return pd.concat(frames, ignore_index=True).drop_duplicates(["market", "index_name", "date"], keep="last")


def _should_refresh_empty_cache(bas_dd: pd.Timestamp, recent_days: int) -> bool:
    if recent_days <= 0:
        return False
    day = pd.Timestamp(bas_dd).date()
    return day >= dt.date.today() - dt.timedelta(days=recent_days)


def _refresh_stock_cache_if_needed(
    client: KrxOpenApiClient,
    market_key: str,
    bas_dd: pd.Timestamp,
    daily_cache: Path | None,
    state: dict,
    state_path: str | Path | None,
    empty_cache_ttl_minutes: int,
) -> pd.DataFrame:
    bas_dd_text = _format_bas_dd(bas_dd)
    if recently_checked_empty(state, "stock", market_key, bas_dd_text, empty_cache_ttl_minutes):
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close"])
    raw = client.fetch_stock_daily(market_key, bas_dd)
    normalized = normalize_stock_daily(raw, market=market_key, tickers=None)
    if daily_cache is not None:
        _write_daily_cache(normalized, daily_cache)
    _record_cache_result(state, state_path, "stock", market_key, bas_dd_text, normalized)
    return normalized


def _refresh_index_cache_if_needed(
    client: KrxOpenApiClient,
    market_key: str,
    bas_dd: pd.Timestamp,
    daily_cache: Path | None,
    state: dict,
    state_path: str | Path | None,
    empty_cache_ttl_minutes: int,
) -> pd.DataFrame:
    bas_dd_text = _format_bas_dd(bas_dd)
    if recently_checked_empty(state, "index", market_key, bas_dd_text, empty_cache_ttl_minutes):
        return pd.DataFrame()
    raw = client.fetch_index_daily(market_key, bas_dd)
    normalized = normalize_index_daily(raw, market_key)
    if daily_cache is not None:
        _write_daily_cache(normalized, daily_cache)
    _record_cache_result(state, state_path, "index", market_key, bas_dd_text, normalized)
    return normalized


def _record_cache_result(
    state: dict,
    state_path: str | Path | None,
    kind: str,
    market_key: str,
    bas_dd_text: str,
    frame: pd.DataFrame,
) -> None:
    if frame.empty:
        mark_empty_response(state, kind, market_key, bas_dd_text)
    else:
        clear_empty_response(state, kind, market_key, bas_dd_text)
    save_collection_state(state_path, state)
