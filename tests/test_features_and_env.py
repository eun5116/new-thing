import pandas as pd

from stock_rl.build_features import add_event_features, add_market_features, add_price_features, split_and_write
from stock_rl.krx_openapi import normalize_stock_daily
from stock_rl.smoke_env import run_smoke
from stock_rl.trading_env import MultiTickerTradingEnv, StockTradingEnv, TradingEnvConfig


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
