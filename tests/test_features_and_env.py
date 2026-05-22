import pandas as pd
import matplotlib.pyplot as plt

from stock_rl.build_features import add_event_features, add_market_features, add_price_features, split_and_write
from stock_rl.build_report_dashboard import build_report_dashboard
from stock_rl.build_portfolio_policy_sheet import build_portfolio_policy_sheet, classify_asset
from stock_rl.build_portfolio_decision_sheet import build_decision_sheet
from stock_rl.build_trading_sheet import build_trading_sheet
from stock_rl.build_rebalance_orders import build_rebalance_orders
from stock_rl.build_rebalance_orders import _load_positions as load_rebalance_positions
from stock_rl.analyze_positions import _load_positions as load_analysis_positions
from stock_rl.build_target_change_report import build_target_change_report
from stock_rl.backtest_portfolio_allocator import simulate_portfolio
from stock_rl.collection_state import load_collection_state, mark_empty_response, recently_checked_empty, save_collection_state
from stock_rl.krx_openapi import normalize_stock_daily
from stock_rl.smoke_env import run_smoke
from stock_rl.trading_env import MultiTickerTradingEnv, StockTradingEnv, TradingEnvConfig
from stock_rl.update_us_portfolio_targets import _feature_columns_for_model, _policy_name_cap
from stock_rl.build_features import FEATURE_COLUMNS
from stock_rl.build_us_portfolio_features import US_FEATURE_COLUMNS
from stock_rl.update_daily_targets import (
    _index_summary,
    infer_incremental_start,
    infer_index_collection_starts,
    infer_market_collection_starts,
)
from stock_rl.weekly_retrain import _run_id


def sample_prices():
    dates = pd.bdate_range("2024-01-01", periods=180)
    close = [100 + i * 0.5 for i in range(len(dates))]
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": "SPY",
            "open": close,
            "high": [x + 1 for x in close],
            "low": [x - 1 for x in close],
            "close": close,
            "adj_close": close,
            "volume": [1_000_000 + i * 1000 for i in range(len(dates))],
            "trading_value": [(1_000_000 + i * 1000) * close[i] for i in range(len(dates))],
            "market_cap": [1_000_000_000 + i * 1_000_000 for i in range(len(dates))],
        }
    )


def sample_multi_ticker_prices():
    first = sample_prices()
    second = first.copy()
    second["ticker"] = "QQQ"
    second["adj_close"] = [120 - i * 0.2 for i in range(len(second))]
    second["close"] = second["adj_close"]
    second["open"] = second["adj_close"]
    second["high"] = second["adj_close"] + 1
    second["low"] = second["adj_close"] - 1
    second["trading_value"] = second["volume"] * second["adj_close"]
    return pd.concat([first, second], ignore_index=True)


def test_position_loaders_recalculate_market_value_from_quantity_and_current_price(tmp_path):
    path = tmp_path / "positions.csv"
    path.write_text(
        "ticker,name,quantity,avg_price,current_price,market_value\n"
        "005930,삼성전자,5,210500,281000,281000\n"
        "AMD,AMD,0.084,612000,636060,636060\n",
        encoding="utf-8",
    )

    analysis_positions = load_analysis_positions(path)
    rebalance_positions = load_rebalance_positions(path)

    assert analysis_positions.loc[analysis_positions["ticker"] == "005930", "market_value"].iloc[0] == 1_405_000
    assert rebalance_positions.loc[rebalance_positions["ticker"] == "005930", "market_value"].iloc[0] == 1_405_000
    assert round(float(analysis_positions.loc[analysis_positions["ticker"] == "AMD", "market_value"].iloc[0]), 2) == 53429.04
    assert round(float(rebalance_positions.loc[rebalance_positions["ticker"] == "AMD", "market_value"].iloc[0]), 2) == 53429.04


def test_portfolio_policy_classifies_qqqm_as_core_etf():
    policy = {
        "ticker_groups": {"QQQM": "etf_core"},
        "groups": {"etf_core": {}, "manual_review": {}},
    }
    row = pd.Series({"ticker": "QQQM", "asset_scope": "us_or_global", "volatility_20d_pct": 10, "current_weight_pct": 4})

    assert classify_asset(row, policy) == "etf_core"


def test_us_policy_name_cap_uses_ticker_group_limit():
    policy = {
        "groups": {
            "us_speculative": {"label": "US speculative", "max_name_weight_pct": 2.0},
            "us_large_cap": {"label": "US large cap", "max_name_weight_pct": 10.0},
        },
        "ticker_groups": {"IONQ": "us_speculative"},
    }

    assert _policy_name_cap("IONQ", policy) == (0.02, "us_speculative", "US speculative")
    assert _policy_name_cap("NVDA", policy) == (0.10, "us_large_cap", "US large cap")


def test_weekly_retrain_run_id_uses_timestamp_format():
    assert _run_id(pd.Timestamp("2026-05-22 15:04:05").to_pydatetime()) == "20260522_150405"


def test_add_price_features_uses_next_day_target():
    features = add_price_features(sample_prices()).dropna()

    first = features.iloc[0]
    expected = features.iloc[1]["adj_close"] / first["adj_close"] - 1.0
    assert first["target_return_1d"] == expected
    assert {
        "return_1d",
        "return_60d",
        "return_120d",
        "ma20_gap",
        "ma60_gap",
        "ma120_gap",
        "volatility_20d",
        "turnover_value_ratio",
        "ma20_60_signal",
        "ma20_60_gap",
        "ma20_60_position",
    }.issubset(features.columns)
    assert set(features["ma20_60_position"].dropna().unique()).issubset({0.0, 1.0})


def test_add_market_features_joins_index_context():
    features = add_price_features(sample_prices()).reset_index(drop=True)
    indices = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=90),
            "market": ["KOSPI"] * 90,
            "index_name": ["코스피"] * 90,
            "close": [2500 + i for i in range(90)],
        }
    )
    features["market"] = "KOSPI"

    merged = add_market_features(features, indices).dropna(subset=["market_return_1d"])

    assert "market_return_1d" in merged.columns
    assert "market_return_60d" in merged.columns
    assert "market_return_120d" in merged.columns
    assert "market_volatility_20d" in merged.columns
    assert "market_ma60_gap" in merged.columns
    assert "market_ma120_gap" in merged.columns
    assert "excess_return_1d" in merged.columns
    assert "drawdown_vs_market_60d" in merged.columns
    assert "market_drop_recent_5d" in merged.columns
    assert "market_trend_regime" in merged.columns
    assert "relative_strength_regime" in merged.columns
    assert merged["market_return_1d"].abs().sum() > 0


def test_split_and_write_removes_infinite_volume_change(tmp_path):
    prices = sample_prices()
    prices.loc[20, "volume"] = 0
    features = add_price_features(prices)
    config = {
        "features": {
            "train_end": "2024-02-29",
            "valid_end": "2024-03-29",
        }
    }

    written = split_and_write(features, config, tmp_path)
    train = pd.read_parquet(written["train"])

    assert train["volume_change"].notna().all()
    assert train["volume_change"].map(lambda value: value != float("inf")).all()


def test_split_and_write_keeps_latest_daily_feature_for_inference(tmp_path):
    features = add_price_features(sample_prices())
    config = {
        "features": {
            "train_end": "2024-02-29",
            "valid_end": "2024-03-29",
        }
    }

    written = split_and_write(features, config, tmp_path)
    daily = pd.read_parquet(written["daily_features"])
    train = pd.read_parquet(written["train"])

    assert daily["date"].max() == features["date"].max()
    assert daily.loc[daily["date"] == daily["date"].max(), "target_return_1d"].isna().all()
    assert train["target_return_1d"].notna().all()


def test_infer_incremental_start_from_latest_features(tmp_path):
    project = tmp_path / "project"
    config_dir = project / "configs"
    processed_dir = project / "data_krx" / "processed"
    config_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    config_path = config_dir / "test.yaml"
    config_path.write_text(
        "project:\n"
        "  data_dir: data_krx\n"
        "market:\n"
        "  start: '2020-01-01'\n",
        encoding="utf-8",
    )
    pd.DataFrame({"date": pd.to_datetime(["2026-05-10", "2026-05-11"])}).to_parquet(
        processed_dir / "daily_features.parquet",
        index=False,
    )

    assert infer_incremental_start(config_path) == "2026-05-12"


def test_infer_incremental_start_falls_back_to_config_start(tmp_path):
    project = tmp_path / "project"
    config_dir = project / "configs"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "test.yaml"
    config_path.write_text(
        "project:\n"
        "  data_dir: data_krx\n"
        "market:\n"
        "  start: '2020-01-01'\n",
        encoding="utf-8",
    )

    assert infer_incremental_start(config_path) == "2020-01-01"


def test_index_summary_reports_latest_date(tmp_path):
    path = tmp_path / "kospi_indices.parquet"
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-11", "2026-05-13"]),
            "market": ["KOSPI", "KOSPI"],
            "index_name": ["코스피", "코스피"],
            "close": [3000.0, 3010.0],
        }
    ).to_parquet(path, index=False)

    assert _index_summary([path]) == {"KOSPI": "2026-05-13"}


def test_infer_market_collection_starts_from_raw_prices(tmp_path):
    project = tmp_path / "project"
    config_dir = project / "configs"
    price_dir = project / "data_krx" / "raw" / "prices"
    config_dir.mkdir(parents=True)
    price_dir.mkdir(parents=True)
    config_path = config_dir / "test.yaml"
    config_path.write_text(
        "project:\n"
        "  data_dir: data_krx\n"
        "market:\n"
        "  start: '2020-01-01'\n"
        "  price_source: krx_openapi\n"
        "  tickers: ['000001', '000002']\n"
        "  ticker_markets:\n"
        "    '000001': KOSPI\n"
        "    '000002': KOSPI\n",
        encoding="utf-8",
    )
    pd.DataFrame({"date": pd.to_datetime(["2026-05-11", "2026-05-13"]), "ticker": ["000001", "000001"]}).to_parquet(
        price_dir / "000001.parquet",
        index=False,
    )
    pd.DataFrame({"date": pd.to_datetime(["2026-05-11"]), "ticker": ["000002"]}).to_parquet(
        price_dir / "000002.parquet",
        index=False,
    )

    assert infer_market_collection_starts(config_path) == {"KOSPI": "2026-05-12"}


def test_infer_index_collection_starts_from_raw_indices(tmp_path):
    project = tmp_path / "project"
    config_dir = project / "configs"
    index_dir = project / "data_krx" / "raw" / "indices"
    config_dir.mkdir(parents=True)
    index_dir.mkdir(parents=True)
    config_path = config_dir / "test.yaml"
    config_path.write_text(
        "project:\n"
        "  data_dir: data_krx\n"
        "market:\n"
        "  start: '2020-01-01'\n"
        "  tickers: ['000001']\n"
        "  ticker_markets:\n"
        "    '000001': KOSPI\n",
        encoding="utf-8",
    )
    pd.DataFrame({"date": pd.to_datetime(["2026-05-13"]), "market": ["KOSPI"]}).to_parquet(
        index_dir / "kospi_indices.parquet",
        index=False,
    )

    assert infer_index_collection_starts(config_path) == {"KOSPI": "2026-05-14"}


def test_build_target_change_report_compares_previous_target(tmp_path):
    project = tmp_path / "project"
    config_dir = project / "configs"
    reports_dir = project / "reports"
    config_dir.mkdir(parents=True)
    reports_dir.mkdir(parents=True)
    config_path = config_dir / "test.yaml"
    config_path.write_text(
        "project:\n"
        "  data_dir: data_krx\n",
        encoding="utf-8",
    )
    base_columns = {
        "feature_date": ["2026-05-11", "2026-05-11"],
        "rule": ["strong_trend_full_else070", "strong_trend_full_else070"],
        "model_name": ["model", "model"],
        "assumed_position_ratio": [0.0, 0.0],
        "action": [5, 5],
        "raw_target_ratio": [1.0, 1.0],
        "cap": [1.0, 0.7],
        "cap_reason": ["strong_trend", "none"],
        "market_return_60d": [0.1, 0.1],
        "market_return_120d": [0.1, 0.1],
        "market_ma60_gap": [0.1, 0.1],
        "market_ma120_gap": [0.1, 0.1],
        "relative_strength_20d": [0.1, -0.1],
        "return_20d": [0.2, -0.1],
        "return_60d": [0.3, 0.0],
        "drawdown_60d": [0.0, -0.2],
    }
    previous = pd.DataFrame(
        {
            "as_of_date": ["2026-05-11", "2026-05-11"],
            "ticker": ["000001", "000002"],
            "target_ratio": [0.88, 1.0],
            **base_columns,
        }
    )
    current = pd.DataFrame(
        {
            "as_of_date": ["2026-05-13", "2026-05-13"],
            "ticker": ["000001", "000002"],
            "target_ratio": [1.0, 0.7],
            **base_columns,
        }
    )
    previous.to_csv(reports_dir / "current_targets_20260511_strong_trend_full_else070.csv", index=False)
    current_path = reports_dir / "current_targets_20260513_strong_trend_full_else070.csv"
    current.to_csv(current_path, index=False)

    result = build_target_change_report(str(config_path), current_target_path=str(current_path))
    assert result is not None
    changes = pd.read_csv(result["csv"], dtype={"ticker": str})

    assert set(changes["rebalance_action"]) == {"increase", "reduce"}
    assert changes.loc[changes["ticker"] == "000001", "target_delta_pct"].iloc[0] == 12.0
    assert changes.loc[changes["ticker"] == "000002", "target_delta_pct"].iloc[0] == -30.0
    assert result["png"].exists()
    assert result["png"].stat().st_size > 0


def test_build_trading_sheet_writes_png_snapshot(tmp_path):
    project = tmp_path / "project"
    config_dir = project / "configs"
    reports_dir = project / "reports"
    processed_dir = project / "data_krx" / "processed"
    reference_dir = project / "data_krx" / "raw" / "reference"
    config_dir.mkdir(parents=True)
    reports_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    reference_dir.mkdir(parents=True)
    config_path = config_dir / "test.yaml"
    config_path.write_text(
        "project:\n"
        "  data_dir: data_krx\n",
        encoding="utf-8",
    )
    features = add_price_features(sample_prices()).dropna().reset_index(drop=True)
    features["market"] = "KOSPI"
    features["change_pct"] = 0.0
    features.to_parquet(processed_dir / "daily_features.parquet", index=False)
    pd.DataFrame({"ticker": ["SPY"], "abbrv": ["SPY"], "market": ["KOSPI"]}).to_parquet(
        reference_dir / "kospi_issue_base.parquet",
        index=False,
    )
    target_path = reports_dir / "current_targets_20260513_strong_trend_full_else070.csv"
    pd.DataFrame(
        {
            "as_of_date": ["2026-05-13"],
            "feature_date": ["2026-05-13"],
            "ticker": ["SPY"],
            "rule": ["strong_trend_full_else070"],
            "model_name": ["model"],
            "assumed_position_ratio": [0.0],
            "action": [5],
            "raw_target_ratio": [1.0],
            "cap": [1.0],
            "cap_reason": ["strong_trend"],
            "target_ratio": [1.0],
            "market_return_60d": [0.1],
            "market_return_120d": [0.1],
            "market_ma60_gap": [0.1],
            "market_ma120_gap": [0.1],
            "relative_strength_20d": [0.1],
            "return_20d": [0.2],
            "return_60d": [0.3],
            "drawdown_60d": [0.0],
        }
    ).to_csv(target_path, index=False)

    result = build_trading_sheet(str(config_path), target_path=str(target_path))

    assert result["png"].exists()
    assert result["png"].stat().st_size > 0


def test_collection_state_tracks_recent_empty_response(tmp_path):
    state_path = tmp_path / "collection_state.json"
    state = {}
    mark_empty_response(state, "stock", "KOSPI", "20260514")
    save_collection_state(state_path, state)
    loaded = load_collection_state(state_path)

    assert recently_checked_empty(loaded, "stock", "KOSPI", "20260514", ttl_minutes=60)
    assert not recently_checked_empty(loaded, "stock", "KOSPI", "20260514", ttl_minutes=0)


def test_simulate_portfolio_respects_gross_cap_and_costs():
    features = add_price_features(sample_multi_ticker_prices()).dropna().reset_index(drop=True)
    features["e032_target_ratio"] = 1.0

    returns, trace, allocations = simulate_portfolio(
        features,
        "e032_target_basket",
        top_n=2,
        gross_cap=0.9,
        max_weight=0.6,
        transaction_cost_pct=0.0015,
        rebalance_frequency="weekly",
    )

    assert len(returns) == trace.shape[0]
    assert trace["gross_exposure"].max() <= 0.91
    assert trace["cost"].sum() > 0
    assert allocations["target_weight"].max() <= 0.6


def test_build_rebalance_orders_marks_out_of_universe_holdings(tmp_path):
    project = tmp_path / "project"
    config_dir = project / "configs"
    reports_dir = project / "reports"
    reference_dir = project / "data_krx" / "raw" / "reference"
    config_dir.mkdir(parents=True)
    reports_dir.mkdir()
    reference_dir.mkdir(parents=True)
    config_path = config_dir / "test.yaml"
    config_path.write_text(
        "project:\n"
        "  data_dir: data_krx\n",
        encoding="utf-8",
    )
    pd.DataFrame({"ticker": ["000001"], "abbrv": ["테스트"], "market": ["KOSPI"]}).to_parquet(
        reference_dir / "kospi_issue_base.parquet",
        index=False,
    )
    target_path = reports_dir / "current_targets_20260513_strong_trend_full_else070.csv"
    pd.DataFrame(
        {
            "as_of_date": ["2026-05-13"],
            "feature_date": ["2026-05-13"],
            "ticker": ["000001"],
            "rule": ["strong_trend_full_else070"],
            "model_name": ["model"],
            "assumed_position_ratio": [0.0],
            "action": [5],
            "raw_target_ratio": [1.0],
            "cap": [1.0],
            "cap_reason": ["strong_trend"],
            "target_ratio": [1.0],
            "market_return_60d": [0.1],
            "market_return_120d": [0.1],
            "market_ma60_gap": [0.1],
            "market_ma120_gap": [0.1],
            "relative_strength_20d": [0.1],
            "return_20d": [0.2],
            "return_60d": [0.3],
            "drawdown_60d": [0.0],
        }
    ).to_csv(target_path, index=False)
    positions_path = tmp_path / "positions.csv"
    pd.DataFrame(
        {
            "ticker": ["000001", "NVDA"],
            "name": ["테스트", "엔비디아"],
            "market_value": [100000.0, 100000.0],
        }
    ).to_csv(positions_path, index=False)

    result = build_rebalance_orders(str(config_path), str(positions_path), target_path=str(target_path), top_n=1)
    orders = pd.read_csv(result["csv"], dtype={"ticker": str})

    assert orders.loc[orders["ticker"] == "000001", "asset_scope"].iloc[0] == "model_universe"
    assert orders.loc[orders["ticker"] == "NVDA", "asset_scope"].iloc[0] == "out_of_universe"
    assert orders.loc[orders["ticker"] == "NVDA", "target_weight_pct"].iloc[0] == 0.0
    assert result["png"].exists()
    assert result["png"].stat().st_size > 0


def test_build_decision_sheet_writes_png_snapshot(tmp_path):
    project = tmp_path / "project"
    config_dir = project / "configs"
    reports_dir = project / "reports"
    config_dir.mkdir(parents=True)
    reports_dir.mkdir(parents=True)
    config_path = config_dir / "test.yaml"
    config_path.write_text(
        "project:\n"
        "  data_dir: data_krx\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "ticker": ["000001", "NVDA"],
            "name": ["테스트", "엔비디아"],
            "asset_scope": ["model_universe", "us_or_global"],
            "current_weight_pct": [11.28, 13.37],
            "pnl_pct": [34.92, 16.63],
            "trend_status": ["uptrend", "uptrend"],
            "return_20d_pct": [41.29, 19.04],
            "drawdown_60d_pct": [-0.53, 0.0],
            "volatility_20d_pct": [61.39, 40.61],
        }
    ).to_csv(reports_dir / "current_position_analysis_20260515.csv", index=False)
    pd.DataFrame(
        {
            "ticker": ["000001", "NVDA"],
            "target_weight_pct": [7.5, 0.0],
            "weight_delta_pct": [-3.78, -13.37],
            "order_amount": [-95101.0, -336720.0],
            "order_action": ["sell", "sell"],
        }
    ).to_csv(reports_dir / "rebalance_orders_20260514_strong_trend_full_else070.csv", index=False)

    result = build_decision_sheet(str(config_path))

    assert result["png"].exists()
    assert result["png"].stat().st_size > 0


def test_portfolio_policy_sheet_flags_group_and_name_caps(tmp_path):
    project = tmp_path / "project"
    config_dir = project / "configs"
    reports_dir = project / "reports"
    config_dir.mkdir(parents=True)
    reports_dir.mkdir(parents=True)
    config_path = config_dir / "test.yaml"
    config_path.write_text(
        "project:\n"
        "  data_dir: data_krx\n",
        encoding="utf-8",
    )
    policy_path = config_dir / "portfolio_policy.yaml"
    policy_path.write_text(
        "groups:\n"
        "  us_speculative:\n"
        "    label: US speculative\n"
        "    max_total_weight_pct: 8.0\n"
        "    max_name_weight_pct: 3.0\n"
        "    action: high_vol_cap\n"
        "  us_large_cap:\n"
        "    label: US large cap\n"
        "    max_total_weight_pct: 25.0\n"
        "    max_name_weight_pct: 10.0\n"
        "    action: trend_risk\n"
        "  manual_review:\n"
        "    label: Manual review\n"
        "    max_total_weight_pct: 5.0\n"
        "    max_name_weight_pct: 3.0\n"
        "    action: manual_review\n"
        "ticker_groups:\n"
        "  QUBT: us_speculative\n"
        "  NVDA: us_large_cap\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "ticker": ["QUBT", "IONQ", "NVDA"],
            "name": ["퀀텀 컴퓨팅", "아이온큐", "엔비디아"],
            "asset_scope": ["us_or_global", "us_or_global", "us_or_global"],
            "current_weight_pct": [6.0, 4.0, 9.0],
            "pnl_pct": [-10.0, -5.0, 12.0],
            "trend_status": ["uptrend", "uptrend", "uptrend"],
            "return_20d_pct": [5.0, 4.0, 10.0],
            "drawdown_60d_pct": [-10.0, -9.0, -4.0],
            "volatility_20d_pct": [100.0, 95.0, 40.0],
        }
    ).to_csv(reports_dir / "current_position_analysis_20260518.csv", index=False)

    result = build_portfolio_policy_sheet(str(config_path), policy_path=str(policy_path))
    sheet = pd.read_csv(result["csv"], dtype={"ticker": str})

    assert result["markdown"].exists()
    assert classify_asset(sheet.loc[sheet["ticker"] == "QUBT"].iloc[0], {"ticker_groups": {"QUBT": "us_speculative"}}) == "us_speculative"
    assert sheet.loc[sheet["ticker"] == "QUBT", "policy_decision"].iloc[0] == "trim_to_name_cap"
    assert sheet.loc[sheet["ticker"] == "IONQ", "policy_group"].iloc[0] == "us_speculative"
    assert sheet.loc[sheet["ticker"] == "IONQ", "group_excess_pct"].iloc[0] == 2.0


def test_build_report_dashboard_writes_png_snapshot(tmp_path):
    reports_dir = tmp_path / "project" / "reports"
    reports_dir.mkdir(parents=True)
    config_path = tmp_path / "project" / "configs" / "test.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("project:\n  data_dir: data_krx\n", encoding="utf-8")

    names = [
        "trading_sheet_20260514_strong_trend_full_else070.png",
        "target_changes_20260514_strong_trend_full_else070.png",
        "rebalance_orders_20260514_strong_trend_full_else070.png",
        "current_position_analysis_20260515.png",
        "portfolio_decision_sheet_20260515.png",
    ]
    for name in names:
        fig = plt.figure(figsize=(1, 1))
        plt.plot([0, 1], [0, 1])
        fig.savefig(reports_dir / name, dpi=60)
        plt.close(fig)

    result = build_report_dashboard(str(config_path))

    assert result is not None
    assert result["png"].exists()
    assert result["png"].stat().st_size > 0


def test_add_event_features_supports_all_market_events():
    features = add_price_features(sample_prices()).dropna().reset_index(drop=True)
    event_date = features.iloc[0]["date"]
    events = pd.DataFrame(
        {
            "effective_date": [event_date],
            "ticker": ["ALL"],
            "event_type": ["policy"],
            "event_score": [1.0],
        }
    )

    merged = add_event_features(features, events)

    assert "event_policy" in merged.columns
    assert merged.loc[merged["date"] == event_date, "event_any"].iloc[0] == 1.0
    assert "event_recent_5d" in merged.columns
    later = merged[merged["date"] > event_date].head(1)
    assert later["event_recent_5d"].iloc[0] == 1.0


def test_trading_env_steps_with_discrete_actions():
    features = add_price_features(sample_prices()).dropna().reset_index(drop=True)
    env = StockTradingEnv(features, ticker="SPY")

    obs, info = env.reset()
    assert obs.shape[0] == len(env.feature_columns) + 2

    next_obs, reward, terminated, truncated, info = env.step(2)
    assert next_obs.shape == obs.shape
    assert isinstance(reward, float)
    assert terminated is False
    assert "portfolio_value" in info


def test_trading_env_supports_target_position_actions_and_excess_reward():
    features = add_price_features(sample_prices()).dropna().reset_index(drop=True)
    env = StockTradingEnv(
        features,
        ticker="SPY",
        config=TradingEnvConfig(action_mode="target_position", reward_mode="excess_return"),
    )

    obs, info = env.reset()
    next_obs, reward, terminated, truncated, info = env.step(4)

    assert env.action_space.n == 5
    assert next_obs.shape == obs.shape
    assert isinstance(reward, float)
    assert terminated is False
    assert "benchmark_return" in info


def test_target_position_actions_support_minimum_exposure():
    features = add_price_features(sample_prices()).dropna().reset_index(drop=True)
    env = StockTradingEnv(
        features,
        ticker="SPY",
        config=TradingEnvConfig(
            action_mode="target_position",
            reward_mode="excess_return",
            target_position_bins=6,
            min_target_position_ratio=0.4,
        ),
    )

    env.reset()
    _, _, _, _, info = env.step(0)

    assert env.action_space.n == 6
    assert info["target_ratio"] == 0.4


def test_trading_env_supports_risk_adjusted_reward():
    features = add_price_features(sample_prices()).dropna().reset_index(drop=True)
    env = StockTradingEnv(
        features,
        ticker="SPY",
        config=TradingEnvConfig(
            action_mode="target_position",
            reward_mode="risk_adjusted",
            drawdown_penalty=0.2,
            turnover_penalty=0.01,
        ),
    )

    obs, info = env.reset()
    next_obs, reward, terminated, truncated, info = env.step(4)

    assert next_obs.shape == obs.shape
    assert isinstance(reward, float)
    assert terminated is False
    assert "drawdown" in info
    assert "turnover" in info


def test_trading_env_supports_drawdown_budget_reward():
    features = add_price_features(sample_prices()).dropna().reset_index(drop=True)
    env = StockTradingEnv(
        features,
        ticker="SPY",
        config=TradingEnvConfig(
            action_mode="target_position",
            reward_mode="drawdown_budget",
            drawdown_soft_limit=0.15,
            drawdown_hard_limit=0.20,
        ),
    )

    obs, info = env.reset()
    next_obs, reward, terminated, truncated, info = env.step(5 if env.action_space.n > 5 else 4)

    assert next_obs.shape == obs.shape
    assert isinstance(reward, float)
    assert terminated is False
    assert "drawdown" in info
    assert "turnover" in info


def test_trading_env_supports_ma20_60_relative_reward():
    features = add_price_features(sample_prices()).dropna().reset_index(drop=True)
    env = StockTradingEnv(
        features,
        ticker="SPY",
        config=TradingEnvConfig(
            action_mode="target_position",
            reward_mode="ma20_60_relative",
            turnover_penalty=0.001,
            ma_underperformance_penalty=1.0,
        ),
    )

    obs, info = env.reset()
    next_obs, reward, terminated, truncated, info = env.step(4)

    assert next_obs.shape == obs.shape
    assert isinstance(reward, float)
    assert terminated is False
    assert "ma20_60_return" in info
    assert "target_ratio" in info
    assert "overlay" in info


def test_trading_env_supports_ma20_60_overlay_actions():
    features = add_price_features(sample_prices()).dropna().reset_index(drop=True)
    env = StockTradingEnv(
        features,
        ticker="SPY",
        config=TradingEnvConfig(
            action_mode="ma20_60_overlay",
            reward_mode="ma20_60_relative",
            target_position_bins=5,
            overlay_step_size=0.25,
        ),
    )

    obs, info = env.reset()
    next_obs, reward, terminated, truncated, info = env.step(4)

    assert env.action_space.n == 5
    assert next_obs.shape == obs.shape
    assert isinstance(reward, float)
    assert terminated is False
    assert "ma20_60_return" in info


def test_trading_env_supports_ma20_60_drawdown_hybrid_reward():
    features = add_price_features(sample_prices()).dropna().reset_index(drop=True)
    env = StockTradingEnv(
        features,
        ticker="SPY",
        config=TradingEnvConfig(
            action_mode="ma20_60_overlay",
            reward_mode="ma20_60_drawdown_hybrid",
            target_position_bins=5,
            overlay_step_size=0.25,
            drawdown_penalty=0.5,
        ),
    )

    obs, info = env.reset()
    next_obs, reward, terminated, truncated, info = env.step(2)

    assert next_obs.shape == obs.shape
    assert isinstance(reward, float)
    assert terminated is False
    assert "drawdown" in info


def test_multi_ticker_env_samples_and_reports_ticker():
    features = add_price_features(sample_multi_ticker_prices()).dropna().reset_index(drop=True)
    env = MultiTickerTradingEnv(
        features,
        tickers=["SPY", "QQQ"],
        config=TradingEnvConfig(action_mode="target_position", reward_mode="ma20_60_relative"),
    )

    obs, info = env.reset(seed=7, options={"ticker": "QQQ"})
    next_obs, reward, terminated, truncated, step_info = env.step(2)

    assert info["ticker"] == "QQQ"
    assert step_info["ticker"] == "QQQ"
    assert next_obs.shape == obs.shape
    assert isinstance(reward, float)


def test_smoke_runner_uses_existing_feature_file(tmp_path):
    features = add_price_features(sample_prices()).dropna().reset_index(drop=True)
    path = tmp_path / "features.parquet"
    features.to_parquet(path, index=False)

    result = run_smoke(str(path), "SPY", max_steps=5)

    assert result["ticker"] == "SPY"
    assert result["steps"] == 5
    assert result["last_observation_size"] > 0


def test_us_target_feature_columns_match_model_observation_size():
    features = pd.DataFrame(columns=[*US_FEATURE_COLUMNS, "event_fed_cut"])
    legacy_columns = [*FEATURE_COLUMNS, "event_fed_cut", "event_recent_20d", "event_recent_5d"]

    assert _feature_columns_for_model(features, len(US_FEATURE_COLUMNS) + 2) == US_FEATURE_COLUMNS
    assert _feature_columns_for_model(features, len(US_FEATURE_COLUMNS) + 3) == [*US_FEATURE_COLUMNS, "event_fed_cut"]
    assert _feature_columns_for_model(features, len(FEATURE_COLUMNS) + 2) == FEATURE_COLUMNS
    assert _feature_columns_for_model(features, len(legacy_columns) + 2) == legacy_columns


def test_us_target_feature_columns_reject_unknown_observation_size():
    features = pd.DataFrame(columns=US_FEATURE_COLUMNS)

    try:
        _feature_columns_for_model(features, 999)
    except ValueError as exc:
        assert "no US feature column set matches model observation size 999" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_krx_stock_daily_normalization_filters_tickers():
    raw = pd.DataFrame(
        {
            "BAS_DD": ["20260511", "20260511"],
            "ISU_SRT_CD": ["005930", "000660"],
            "TDD_OPNPRC": ["280,000", "450,000"],
            "TDD_HGPRC": ["290,000", "460,000"],
            "TDD_LWPRC": ["279,000", "449,000"],
            "TDD_CLSPRC": ["285,500", "455,000"],
            "ACC_TRDVOL": ["12,345,678", "1,234,567"],
            "ACC_TRDVAL": ["3,456,789,000", "567,890,000"],
            "MKTCAP": ["100,000,000,000", "50,000,000,000"],
            "FLUC_RT": ["1.25", "-0.50"],
        }
    )

    prices = normalize_stock_daily(raw, market="KOSPI", tickers=["005930"])

    assert list(prices["ticker"]) == ["005930"]
    assert prices.iloc[0]["date"] == pd.Timestamp("2026-05-11").date()
    assert prices.iloc[0]["close"] == 285500
    assert prices.iloc[0]["adj_close"] == 285500
    assert prices.iloc[0]["volume"] == 12345678
