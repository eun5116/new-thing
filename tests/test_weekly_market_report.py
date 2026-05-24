import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


from stock_rl import weekly_market_report


class KOSPIAlertFallbackTests(unittest.TestCase):
    def test_resolve_alert_universe_uses_cached_kospi_symbols_when_live_fetch_fails(self):
        config = {"market_scope": "kospi", "universe": []}
        with patch.object(weekly_market_report, "_is_offline_mode", return_value=False), patch.object(
            weekly_market_report, "_get_latest_kospi_date", side_effect=RuntimeError("krx unavailable")
        ), patch.object(weekly_market_report, "_get_cached_kospi_symbols", return_value=["005930", "000660"]):
            symbols, metadata = weekly_market_report._resolve_alert_universe(config)

        self.assertEqual(symbols, ["005930", "000660"])
        self.assertEqual(metadata, {"005930": "005930", "000660": "000660"})

    def test_resolve_alert_universe_uses_default_kospi20_when_cache_is_empty(self):
        config = {"market_scope": "kospi", "universe": []}
        with patch.object(weekly_market_report, "_is_offline_mode", return_value=False), patch.object(
            weekly_market_report, "_get_latest_kospi_date", side_effect=RuntimeError("krx unavailable")
        ), patch.object(weekly_market_report, "_get_cached_kospi_symbols", return_value=[]):
            symbols, metadata = weekly_market_report._resolve_alert_universe(config)

        self.assertEqual(symbols, [code for _, code in weekly_market_report.DEFAULT_KOSPI20])
        self.assertEqual(metadata, {code: name for name, code in weekly_market_report.DEFAULT_KOSPI20})

    def test_render_report_shows_nearest_momentum_candidates_when_none_triggered(self):
        config_path = Path(__file__).resolve().parents[1] / "configs" / "weekly_market" / "kospi_alerts.yaml"
        alerts_df = pd.DataFrame(
            [
                {
                    "market": "KOSPI",
                    "name": "Samsung Electronics",
                    "alert_name": "two_week_momentum_watch",
                    "symbol": "005930",
                    "as_of_date": "2026-05-18",
                    "window_trading_days": 10,
                    "total_return": 0.2086,
                    "up_days": 6,
                    "max_consecutive_down_days": 1,
                    "signal": "",
                    "triggered": False,
                }
            ],
            columns=weekly_market_report.ALERT_COLUMNS,
        )

        _, text, html = weekly_market_report.render_report(
            pd.DataFrame(columns=weekly_market_report.KOSPI_COLUMNS),
            "",
            pd.DataFrame(columns=weekly_market_report.SP500_COLUMNS),
            [(config_path, alerts_df)],
            [],
            report_date="2026-05-18",
        )

        self.assertIn("Closest candidates are shown below", text)
        self.assertIn("005930", text)
        self.assertIn("Closest candidates are shown below", html)
        self.assertIn("005930", html)

    def test_two_week_momentum_triggers_short_term_overheat_signal(self):
        config = weekly_market_report.load_alert_config(
            Path(__file__).resolve().parents[1] / "configs" / "weekly_market" / "kospi_alerts.yaml"
        )
        series = pd.Series(
            [100, 101, 99, 103, 108, 110, 109, 112, 114, 116],
            index=pd.date_range("2026-05-05", periods=10, freq="B"),
        )

        result = weekly_market_report.evaluate_two_week_momentum(series, config)

        self.assertGreaterEqual(result["total_return"], 0.15)
        self.assertGreaterEqual(result["up_days"], 4)
        self.assertLess(result["up_days"], config["min_up_days"])
        self.assertTrue(result["triggered"])
        self.assertEqual(result["signal"], "short_term_overheat")

    def test_sp500_top20_uses_static_fallback_when_live_and_cache_are_unavailable(self):
        with patch.object(weekly_market_report, "_is_offline_mode", return_value=False), patch.object(
            weekly_market_report, "get_sp500_tickers", side_effect=RuntimeError("sp500 unavailable")
        ), patch.object(weekly_market_report, "SP500_CACHE_PATH", Path("/tmp/stock_rl_missing_sp500_cache.json")):
            tickers = weekly_market_report.get_sp500_top20()

        self.assertEqual(tickers, weekly_market_report.DEFAULT_SP50020)


if __name__ == "__main__":
    unittest.main()
