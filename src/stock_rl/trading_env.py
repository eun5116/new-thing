from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from stock_rl.build_features import FEATURE_COLUMNS


def normalize_ticker(ticker: str) -> str:
    value = str(ticker)
    return value.zfill(6) if value.isdigit() else value


@dataclass(frozen=True)
class TradingEnvConfig:
    initial_cash: float = 1_000_000.0
    transaction_cost_pct: float = 0.001
    volatility_penalty: float = 0.05
    max_position_ratio: float = 1.0
    action_mode: str = "trade"
    reward_mode: str = "absolute"
    target_position_bins: int = 5
    drawdown_penalty: float = 0.2
    turnover_penalty: float = 0.01
    drawdown_soft_limit: float = 0.15
    drawdown_hard_limit: float = 0.20
    drawdown_soft_penalty: float = 0.5
    drawdown_hard_penalty: float = 2.0
    hard_drawdown_terminate: bool = False
    ma_underperformance_penalty: float = 1.0
    ma_outperformance_bonus: float = 0.0
    overlay_step_size: float = 0.25


class StockTradingEnv(gym.Env):
    """Single-asset daily trading environment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: pd.DataFrame,
        ticker: str,
        feature_columns: list[str] | None = None,
        config: TradingEnvConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or TradingEnvConfig()
        self.data = features[features["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        if len(self.data) < 3:
            raise ValueError(f"not enough rows for ticker {ticker}")
        if feature_columns is None:
            event_columns = sorted(column for column in self.data.columns if column.startswith("event_"))
            self.feature_columns = FEATURE_COLUMNS + event_columns
        else:
            self.feature_columns = feature_columns
        missing = set(self.feature_columns + ["adj_close", "target_return_1d"]).difference(self.data.columns)
        if missing:
            raise ValueError(f"features missing columns: {sorted(missing)}")
        required_values = self.data[self.feature_columns + ["adj_close", "target_return_1d"]]
        if not np.isfinite(required_values.to_numpy(dtype=np.float64)).all():
            raise ValueError("features contain non-finite values")

        obs_size = len(self.feature_columns) + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        if self.config.action_mode not in {"trade", "target_position", "ma20_60_overlay"}:
            raise ValueError("action_mode must be 'trade', 'target_position', or 'ma20_60_overlay'")
        if self.config.reward_mode not in {
            "absolute",
            "excess_return",
            "risk_adjusted",
            "drawdown_budget",
            "ma20_60_relative",
            "ma20_60_drawdown_hybrid",
        }:
            raise ValueError(
                "reward_mode must be 'absolute', 'excess_return', 'risk_adjusted', "
                "'drawdown_budget', 'ma20_60_relative', or 'ma20_60_drawdown_hybrid'"
            )
        if self.config.target_position_bins < 2:
            raise ValueError("target_position_bins must be at least 2")
        if not 0.0 <= self.config.drawdown_soft_limit <= self.config.drawdown_hard_limit:
            raise ValueError("drawdown limits must satisfy 0 <= soft_limit <= hard_limit")

        action_count = 3 if self.config.action_mode == "trade" else self.config.target_position_bins
        self.action_space = spaces.Discrete(action_count)
        self._step_index = 0
        self._cash = self.config.initial_cash
        self._shares = 0.0
        self._portfolio_value = self.config.initial_cash
        self._peak_value = self.config.initial_cash
        self._last_target_ratio = 0.0
        self._last_overlay = 0.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._step_index = 0
        self._cash = self.config.initial_cash
        self._shares = 0.0
        self._portfolio_value = self.config.initial_cash
        self._peak_value = self.config.initial_cash
        self._last_target_ratio = 0.0
        self._last_overlay = 0.0
        return self._observation(), {}

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"action must be in [0, {self.action_space.n - 1}]")

        row = self.data.iloc[self._step_index]
        price = float(row["adj_close"])
        prev_value = self._portfolio_value
        traded_value = self._rebalance(action, price, row)

        next_return = float(row["target_return_1d"])
        next_price = price * (1.0 + next_return)
        self._portfolio_value = self._cash + self._shares * next_price
        self._peak_value = max(self._peak_value, self._portfolio_value)

        daily_return = self._portfolio_value / prev_value - 1.0
        cost_penalty = self.config.transaction_cost_pct * traded_value / max(prev_value, 1e-12)
        vol_penalty = self.config.volatility_penalty * float(row.get("volatility_20d", 0.0)) / 252.0
        turnover = traded_value / max(prev_value, 1e-12)
        drawdown = self._portfolio_value / max(self._peak_value, 1e-12) - 1.0
        ma20_60_return = float(row.get("ma20_60_position", 0.0)) * next_return
        benchmark_return = next_return if self.config.reward_mode == "excess_return" else 0.0
        reward = float(daily_return - benchmark_return - cost_penalty - vol_penalty)
        if self.config.reward_mode == "risk_adjusted":
            drawdown_penalty = self.config.drawdown_penalty * abs(min(drawdown, 0.0)) / 252.0
            turnover_penalty = self.config.turnover_penalty * turnover
            reward = float(daily_return - cost_penalty - drawdown_penalty - turnover_penalty)
        elif self.config.reward_mode == "drawdown_budget":
            drawdown_depth = abs(min(drawdown, 0.0))
            soft_excess = max(drawdown_depth - self.config.drawdown_soft_limit, 0.0)
            hard_excess = max(drawdown_depth - self.config.drawdown_hard_limit, 0.0)
            soft_penalty = self.config.drawdown_soft_penalty * soft_excess / 252.0
            hard_penalty = self.config.drawdown_hard_penalty * hard_excess
            turnover_penalty = self.config.turnover_penalty * turnover
            reward = float(daily_return - cost_penalty - soft_penalty - hard_penalty - turnover_penalty)
        elif self.config.reward_mode in {"ma20_60_relative", "ma20_60_drawdown_hybrid"}:
            underperformance = max(ma20_60_return - daily_return, 0.0)
            outperformance = max(daily_return - ma20_60_return, 0.0)
            turnover_penalty = self.config.turnover_penalty * turnover
            drawdown_penalty = 0.0
            if self.config.reward_mode == "ma20_60_drawdown_hybrid":
                drawdown_penalty = self.config.drawdown_penalty * abs(min(drawdown, 0.0)) / 252.0
            reward = float(
                daily_return
                - cost_penalty
                - turnover_penalty
                - drawdown_penalty
                - self.config.ma_underperformance_penalty * underperformance
                + self.config.ma_outperformance_bonus * outperformance
            )

        self._step_index += 1
        hard_limit_breached = abs(min(drawdown, 0.0)) > self.config.drawdown_hard_limit
        terminated = self._portfolio_value <= 0 or (
            self.config.reward_mode == "drawdown_budget"
            and self.config.hard_drawdown_terminate
            and hard_limit_breached
        )
        truncated = self._step_index >= len(self.data) - 1
        return self._observation(), reward, terminated, truncated, {
            "portfolio_value": self._portfolio_value,
            "daily_return": daily_return,
            "benchmark_return": benchmark_return,
            "ma20_60_return": ma20_60_return,
            "drawdown": drawdown,
            "turnover": turnover,
            "traded_value": traded_value,
            "target_ratio": self._last_target_ratio,
            "overlay": self._last_overlay,
        }

    def _rebalance(self, action: int, price: float, row: pd.Series) -> float:
        current_stock_value = self._shares * price
        current_value = self._cash + current_stock_value
        if self.config.action_mode == "target_position":
            target_ratio = action / (self.config.target_position_bins - 1)
            self._last_overlay = 0.0
            target_stock_value = current_value * self.config.max_position_ratio * target_ratio
        elif self.config.action_mode == "ma20_60_overlay":
            center = (self.config.target_position_bins - 1) / 2.0
            overlay = (action - center) * self.config.overlay_step_size
            base_position = float(row.get("ma20_60_position", 0.0))
            target_ratio = float(np.clip(base_position + overlay, 0.0, 1.0))
            self._last_overlay = float(overlay)
            target_stock_value = current_value * self.config.max_position_ratio * target_ratio
        elif action == 0:
            target_ratio = 0.0
            self._last_overlay = 0.0
            target_stock_value = 0.0
        elif action == 1:
            target_ratio = current_stock_value / max(current_value, 1e-12)
            self._last_overlay = 0.0
            target_stock_value = current_stock_value
        else:
            target_ratio = self.config.max_position_ratio
            self._last_overlay = 0.0
            target_stock_value = current_value * self.config.max_position_ratio

        self._last_target_ratio = float(target_ratio)
        delta = target_stock_value - current_stock_value
        traded_value = abs(delta)
        cost = traded_value * self.config.transaction_cost_pct
        if delta > 0:
            spend = min(delta + cost, self._cash)
            buy_value = spend / (1.0 + self.config.transaction_cost_pct)
            self._shares += buy_value / price
            self._cash -= spend
        elif delta < 0:
            sell_value = min(-delta, current_stock_value)
            self._shares -= sell_value / price
            self._cash += sell_value * (1.0 - self.config.transaction_cost_pct)
        return traded_value

    def _observation(self) -> np.ndarray:
        row = self.data.iloc[min(self._step_index, len(self.data) - 1)]
        price = float(row["adj_close"])
        value = self._cash + self._shares * price
        cash_ratio = self._cash / max(value, 1e-12)
        position_ratio = self._shares * price / max(value, 1e-12)
        values = [float(row[col]) for col in self.feature_columns] + [cash_ratio, position_ratio]
        return np.asarray(values, dtype=np.float32)


class MultiTickerTradingEnv(gym.Env):
    """Episode-level sampler over several single-asset trading environments."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: pd.DataFrame,
        tickers: list[str],
        feature_columns: list[str] | None = None,
        config: TradingEnvConfig | None = None,
    ) -> None:
        super().__init__()
        if not tickers:
            raise ValueError("tickers must not be empty")
        self.features = features
        self.tickers = [normalize_ticker(ticker) for ticker in tickers]
        self.feature_columns = feature_columns
        self.config = config or TradingEnvConfig()
        self._rng = np.random.default_rng()
        self._current_ticker = self.tickers[0]
        self._env = self._make_env(self._current_ticker)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    @property
    def current_ticker(self) -> str:
        return self._current_ticker

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        requested_ticker = options.get("ticker") if options else None
        if requested_ticker is not None:
            ticker = normalize_ticker(requested_ticker)
            if ticker not in self.tickers:
                raise ValueError(f"ticker not configured: {ticker}")
        else:
            ticker = str(self._rng.choice(self.tickers))
        self._current_ticker = ticker
        self._env = self._make_env(ticker)
        obs, info = self._env.reset(seed=seed)
        info["ticker"] = ticker
        return obs, info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self._env.step(action)
        info["ticker"] = self._current_ticker
        return obs, reward, terminated, truncated, info

    def _make_env(self, ticker: str) -> StockTradingEnv:
        return StockTradingEnv(
            self.features,
            ticker=ticker,
            feature_columns=self.feature_columns,
            config=self.config,
        )
