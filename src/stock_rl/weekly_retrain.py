from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from stock_rl.build_features import build_features
from stock_rl.build_target_change_report import build_target_change_report
from stock_rl.build_trading_sheet import build_trading_sheet
from stock_rl.collect_krx_reference import collect_index_series
from stock_rl.collect_prices import collect_prices
from stock_rl.config import load_config, project_path
from stock_rl.evaluate_regime_exposure_cap import evaluate_split
from stock_rl.generate_current_targets import generate_targets
from stock_rl.krx_openapi import KrxOpenApiClient
from stock_rl.train_rl import train
from stock_rl.update_daily_targets import (
    _config_markets,
    infer_index_collection_starts,
    infer_market_collection_starts,
)


DEFAULT_CONFIG = "configs/KRX_E035_defensive_retrain.yaml"
DEFAULT_RULE = "strong_trend_full_else070"


@dataclass(frozen=True)
class RetrainResult:
    run_id: str
    config: str
    model_path: str | None
    output_dir: str
    refreshed_data: bool
    feature_paths: dict[str, str]
    evaluation_paths: dict[str, dict[str, str]]
    target_path: str | None
    sheet_paths: dict[str, str]
    change_paths: dict[str, str]


def _run_id(now: datetime | None = None) -> str:
    return (now or datetime.now()).strftime("%Y%m%d_%H%M%S")


def _stringify_paths(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _stringify_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stringify_paths(item) for item in value]
    return value


def refresh_krx_features(
    config_path: str,
    start: str | None = None,
    end: str | None = None,
    skip_collect: bool = False,
    skip_indices: bool = False,
) -> dict[str, str]:
    config = load_config(config_path)
    stock_starts = {market: start for market in _config_markets(config)} if start else infer_market_collection_starts(config_path)
    index_starts = {market: start for market in _config_markets(config)} if start else infer_index_collection_starts(config_path)

    if not skip_collect:
        collect_prices(config_path, start=start, end=end, start_by_market=stock_starts)

    if not skip_indices:
        client = KrxOpenApiClient.from_env()
        for market in _config_markets(config):
            collect_index_series(
                client,
                config["project"]["data_dir"],
                market,
                index_starts[market],
                end=end,
                allow_empty=True,
                config_path=config_path,
            )

    return {name: str(path) for name, path in build_features(config_path).items()}


def weekly_retrain(
    config_path: str = DEFAULT_CONFIG,
    rule: str = DEFAULT_RULE,
    splits: tuple[str, ...] = ("valid", "test"),
    refresh_data: bool = False,
    start: str | None = None,
    end: str | None = None,
    skip_collect: bool = False,
    skip_indices: bool = False,
    skip_train: bool = False,
    skip_targets: bool = False,
    out_dir: str | None = None,
) -> RetrainResult:
    run_id = _run_id()
    output_dir = Path(out_dir) if out_dir else project_path(config_path, "reports", "retrain", run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_paths: dict[str, str] = {}
    if refresh_data:
        feature_paths = refresh_krx_features(config_path, start=start, end=end, skip_collect=skip_collect, skip_indices=skip_indices)

    model_path = None if skip_train else train(config_path)

    evaluation_paths: dict[str, dict[str, str]] = {}
    for split in splits:
        paths = evaluate_split(config_path, split=split, out_dir=str(output_dir))
        evaluation_paths[split] = {name: str(path) for name, path in paths.items()}

    target_path = None
    sheet_paths: dict[str, str] = {}
    change_paths: dict[str, str] = {}
    if not skip_targets:
        target_path = str(generate_targets(config_path, rule_name=rule))
        sheet_paths = {name: str(path) for name, path in build_trading_sheet(config_path, rule=rule, target_path=target_path).items()}
        change_paths = {
            name: str(path)
            for name, path in build_target_change_report(config_path, rule=rule, current_target_path=target_path).items()
        }

    result = RetrainResult(
        run_id=run_id,
        config=config_path,
        model_path=model_path,
        output_dir=str(output_dir),
        refreshed_data=refresh_data,
        feature_paths=feature_paths,
        evaluation_paths=evaluation_paths,
        target_path=target_path,
        sheet_paths=sheet_paths,
        change_paths=change_paths,
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(_stringify_paths(asdict(result)), indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--rule", default=DEFAULT_RULE)
    parser.add_argument("--splits", nargs="+", choices=["train", "valid", "test"], default=["valid", "test"])
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument("--skip-indices", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-targets", action="store_true")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    result = weekly_retrain(
        config_path=args.config,
        rule=args.rule,
        splits=tuple(args.splits),
        refresh_data=args.refresh_data,
        start=args.start,
        end=args.end,
        skip_collect=args.skip_collect,
        skip_indices=args.skip_indices,
        skip_train=args.skip_train,
        skip_targets=args.skip_targets,
        out_dir=args.out_dir,
    )
    print(json.dumps(_stringify_paths(asdict(result)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
