from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Performance:
    cumulative_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    max_drawdown: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def performance_from_returns(returns: pd.Series, periods_per_year: int = 252) -> Performance:
    returns = returns.dropna().astype(float)
    if returns.empty:
        raise ValueError("returns series is empty")
    equity = (1.0 + returns).cumprod()
    cumulative_return = float(equity.iloc[-1] - 1.0)
    annualized_return = float(equity.iloc[-1] ** (periods_per_year / len(returns)) - 1.0)
    annualized_vol = float(returns.std(ddof=0) * np.sqrt(periods_per_year))
    sharpe = float(annualized_return / annualized_vol) if annualized_vol > 0 else 0.0
    drawdown = equity / equity.cummax() - 1.0
    return Performance(cumulative_return, annualized_return, annualized_vol, sharpe, float(drawdown.min()))


def buy_and_hold_metrics(features: pd.DataFrame, ticker: str) -> Performance:
    frame = features[features["ticker"] == ticker].sort_values("date")
    if frame.empty:
        raise ValueError(f"ticker not found: {ticker}")
    return performance_from_returns(frame["target_return_1d"])


def cash_metrics(features: pd.DataFrame, ticker: str) -> Performance:
    frame = features[features["ticker"] == ticker].sort_values("date")
    if frame.empty:
        raise ValueError(f"ticker not found: {ticker}")
    return performance_from_returns(pd.Series(np.zeros(len(frame)), index=frame.index))


def moving_average_metrics(features: pd.DataFrame, ticker: str, fast: int = 20, slow: int = 60) -> Performance:
    frame = features[features["ticker"] == ticker].sort_values("date").copy()
    if frame.empty:
        raise ValueError(f"ticker not found: {ticker}")
    fast_ma = frame["adj_close"].rolling(fast).mean()
    slow_ma = frame["adj_close"].rolling(slow).mean()
    position = (fast_ma > slow_ma).astype(float).shift(1).fillna(0.0)
    returns = position * frame["target_return_1d"]
    return performance_from_returns(returns)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/processed/test.parquet")
    parser.add_argument("--ticker", default="SPY")
    args = parser.parse_args()
    features = pd.read_parquet(args.features)
    print(buy_and_hold_metrics(features, args.ticker).to_dict())


if __name__ == "__main__":
    main()
