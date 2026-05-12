import pandas as pd

from stock_rl.build_features import add_event_features, add_market_features, add_price_features, split_and_write
from stock_rl.krx_openapi import normalize_stock_daily
from stock_rl.smoke_env import run_smoke
from stock_rl.trading_env import StockTradingEnv, TradingEnvConfig


def sample_prices():
    dates = pd.bdate_range("2024-01-01", periods=90)
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


def test_add_price_features_uses_next_day_target():
    features = add_price_features(sample_prices()).dropna()

    first = features.iloc[0]
    expected = features.iloc[1]["adj_close"] / first["adj_close"] - 1.0
    assert first["target_return_1d"] == expected
    assert {
        "return_1d",
        "ma20_gap",
        "volatility_20d",
        "turnover_value_ratio",
        "ma20_60_signal",
        "ma20_60_gap",
    }.issubset(features.columns)


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
    assert "market_volatility_20d" in merged.columns
    assert "excess_return_1d" in merged.columns
    assert "drawdown_vs_market_60d" in merged.columns
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
