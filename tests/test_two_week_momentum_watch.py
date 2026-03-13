import importlib.util
from pathlib import Path
import unittest
from unittest import mock

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "weekly_market_report" / "weekly_report.py"
SP500_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "dataset" / "us_equities_alerts.yaml"
SPEC = importlib.util.spec_from_file_location("weekly_report", MODULE_PATH)
weekly_report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(weekly_report)


class TwoWeekMomentumWatchTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "alert_name": "two_week_momentum_watch",
            "window_trading_days": 10,
            "min_total_return": 0.10,
            "trend_policy": "tolerant",
            "min_up_days": 8,
            "max_consecutive_down_days": 2,
        }

    def test_two_week_momentum_watch_triggers_for_tolerant_uptrend(self):
        series = pd.Series(
            [100, 101, 103, 104, 103, 105, 107, 110, 112, 115],
            index=pd.date_range("2026-03-02", periods=10, freq="B"),
        )

        result = weekly_report.evaluate_two_week_momentum(series, self.config)

        self.assertTrue(result["triggered"])
        self.assertEqual(result["up_days"], 8)
        self.assertEqual(result["max_consecutive_down_days"], 1)
        self.assertAlmostEqual(result["total_return"], 0.15, places=6)

    def test_two_week_momentum_watch_rejects_three_day_down_streak(self):
        series = pd.Series(
            [100, 103, 106, 109, 108, 107, 106, 111, 114, 116],
            index=pd.date_range("2026-03-02", periods=10, freq="B"),
        )

        result = weekly_report.evaluate_two_week_momentum(series, self.config)

        self.assertFalse(result["triggered"])
        self.assertEqual(result["max_consecutive_down_days"], 3)

    def test_load_alert_config_reads_dataset_sot(self):
        config = weekly_report.load_alert_config(SP500_CONFIG_PATH)

        self.assertEqual(config["alert_name"], "two_week_momentum_watch")
        self.assertEqual(config["window_trading_days"], 10)
        self.assertEqual(config["min_total_return"], 0.10)
        self.assertEqual(config["trend_policy"], "tolerant")
        self.assertEqual(config["market_scope"], "sp500")

    def test_resolve_alert_universe_uses_market_scope(self):
        config = weekly_report.load_alert_config(SP500_CONFIG_PATH)

        with mock.patch.object(weekly_report, "get_sp500_tickers", return_value=["AAPL", "MSFT"]):
            symbols, metadata = weekly_report._resolve_alert_universe(config)

        self.assertEqual(symbols, ["AAPL", "MSFT"])
        self.assertEqual(metadata, {})


if __name__ == "__main__":
    unittest.main()
