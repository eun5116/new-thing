import importlib.util
import json
from pathlib import Path
import tempfile
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

    def test_normalize_sp500_symbol_replaces_dot_with_dash(self):
        self.assertEqual(weekly_report._normalize_sp500_symbol("BRK.B"), "BRK-B")
        self.assertEqual(weekly_report._normalize_sp500_symbol("bf.b"), "BF-B")

    def test_build_report_includes_summary_and_handles_empty_sections(self):
        alert_df = pd.DataFrame(
            [
                {
                    "market": "SP500",
                    "name": "Apple",
                    "alert_name": "two_week_momentum_watch",
                    "symbol": "AAPL",
                    "as_of_date": "2026-03-20",
                    "window_trading_days": 10,
                    "total_return": 0.1234,
                    "up_days": 8,
                    "max_consecutive_down_days": 1,
                    "triggered": True,
                }
            ],
            columns=weekly_report.ALERT_COLUMNS,
        )
        kospi_df = pd.DataFrame(
            [("삼성전자", "005930", "2026-03-16", "2026-03-20", 100.0, 110.0, 10.0)],
            columns=weekly_report.KOSPI_COLUMNS,
        )

        with mock.patch.object(weekly_report, "get_kospi_top10_and_change", return_value=(kospi_df, "Used fallback.")):
            with mock.patch.object(weekly_report, "get_sp500_top20", return_value=[]):
                with mock.patch.object(
                    weekly_report,
                    "get_sp500_weekly_change",
                    return_value=pd.DataFrame(columns=weekly_report.SP500_COLUMNS),
                ):
                    with mock.patch.object(
                        weekly_report,
                        "get_market_momentum_alerts",
                        return_value=(
                            [(SP500_CONFIG_PATH, alert_df)],
                            [(SP500_CONFIG_PATH, "Some symbols skipped.")],
                        ),
                    ):
                        subject, text, html = weekly_report.build_report()

        self.assertIn("Weekly Market Report - ", subject)
        self.assertIn("Weekly Summary:", text)
        self.assertIn("KOSPI status: Used fallback.", text)
        self.assertIn("S&P 500: data unavailable", text)
        self.assertIn("sp500_alerts_v1: 1 triggered, top AAPL (12.34%) [Some symbols skipped.]", text)
        self.assertIn("<h3>Weekly Summary</h3>", html)
        self.assertIn("No S&P 500 rows available.", html)

    def test_get_sp500_top20_uses_stale_cache_when_live_fetch_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "sp500_top20.json"
            cache_path.write_text(
                json.dumps(
                    {
                        "date": "2026-03-01T00:00:00",
                        "tickers": ["AAPL", "MSFT"],
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.object(weekly_report, "SP500_CACHE_PATH", cache_path):
                with mock.patch.object(weekly_report, "get_sp500_tickers", side_effect=RuntimeError("network down")):
                    tickers = weekly_report.get_sp500_top20()

        self.assertEqual(tickers, ["AAPL", "MSFT"])

    def test_save_report_artifacts_writes_text_html_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = weekly_report.save_report_artifacts(
                "Weekly Market Report - 2026-03-20",
                "plain body",
                "<html>body</html>",
                output_dir=Path(tmpdir),
                metadata={"delivery_status": "dry_run"},
            )

            self.assertTrue(artifacts["text"].exists())
            self.assertTrue(artifacts["html"].exists())
            self.assertTrue(artifacts["meta"].exists())
            self.assertEqual(artifacts["text"].read_text(encoding="utf-8"), "plain body")
            self.assertEqual(artifacts["html"].read_text(encoding="utf-8"), "<html>body</html>")
            payload = json.loads(artifacts["meta"].read_text(encoding="utf-8"))
            self.assertEqual(payload["delivery_status"], "dry_run")

    def test_save_market_history_appends_market_and_alert_rows(self):
        kospi_df = pd.DataFrame(
            [("삼성전자", "005930", "2026-03-16", "2026-03-20", 100.0, 110.0, 10.0)],
            columns=weekly_report.KOSPI_COLUMNS,
        )
        sp_df = pd.DataFrame(
            [("AAPL", "2026-03-16", "2026-03-20", 200.0, 210.0, 5.0)],
            columns=weekly_report.SP500_COLUMNS,
        )
        alerts_df = pd.DataFrame(
            [
                {
                    "market": "SP500",
                    "name": "Apple",
                    "alert_name": "two_week_momentum_watch",
                    "symbol": "AAPL",
                    "as_of_date": "2026-03-20",
                    "window_trading_days": 10,
                    "total_return": 0.1234,
                    "up_days": 8,
                    "max_consecutive_down_days": 1,
                    "triggered": True,
                }
            ],
            columns=weekly_report.ALERT_COLUMNS,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = weekly_report.save_market_history(
                "2026-03-20",
                kospi_df,
                sp_df,
                [(SP500_CONFIG_PATH, alerts_df)],
                history_dir=Path(tmpdir),
            )

            kospi_history = pd.read_csv(paths["kospi"], dtype={"symbol": str})
            sp_history = pd.read_csv(paths["sp500"])
            alert_history = pd.read_csv(paths["alerts"])

        self.assertEqual(kospi_history.iloc[0]["symbol"], "005930")
        self.assertEqual(sp_history.iloc[0]["symbol"], "AAPL")
        self.assertEqual(alert_history.iloc[0]["symbol"], "AAPL")

    def test_main_dry_run_skips_email_and_saves_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(
                weekly_report,
                "collect_report_inputs",
                return_value=(
                    pd.DataFrame(columns=weekly_report.KOSPI_COLUMNS),
                    "",
                    pd.DataFrame(columns=weekly_report.SP500_COLUMNS),
                    [],
                    [],
                ),
            ):
                with mock.patch.object(weekly_report, "send_email") as send_email_mock:
                    weekly_report.main(["--dry-run", "--output-dir", tmpdir])

            send_email_mock.assert_not_called()
            generated = list(Path(tmpdir).glob("*.html"))
            self.assertEqual(len(generated), 1)
            payload = json.loads(list(Path(tmpdir).glob("*.json"))[0].read_text(encoding="utf-8"))
            self.assertIn("history_kospi_path", payload)

    def test_main_saves_artifacts_when_email_send_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(
                weekly_report,
                "collect_report_inputs",
                return_value=(
                    pd.DataFrame(columns=weekly_report.KOSPI_COLUMNS),
                    "",
                    pd.DataFrame(columns=weekly_report.SP500_COLUMNS),
                    [],
                    [],
                ),
            ):
                with mock.patch.object(
                    weekly_report,
                    "load_env",
                    return_value={
                        "smtp_host": "smtp.gmail.com",
                        "smtp_port": 587,
                        "smtp_user": "user",
                        "smtp_pass": "abcdefghijklmnop",
                        "email_to": "to@example.com",
                        "email_from": "from@example.com",
                        "tz": "Asia/Seoul",
                    },
                ):
                    with mock.patch.object(
                        weekly_report,
                        "send_email",
                        side_effect=OSError("Temporary failure in name resolution"),
                    ):
                        with self.assertRaises(OSError):
                            weekly_report.main(["--output-dir", tmpdir])

            meta_files = list(Path(tmpdir).glob("*.json"))
            self.assertEqual(len(meta_files), 1)
            payload = json.loads(meta_files[0].read_text(encoding="utf-8"))
            self.assertEqual(payload["delivery_status"], "send_failed")
            self.assertIn("Temporary failure in name resolution", payload["delivery_error"])
            self.assertIn("history_sp500_path", payload)

    def test_offline_mode_uses_cached_sp500_weekly_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "sp500_weekly_rows.json"
            cache_path.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "Ticker": "AAPL",
                                "StartDate": "2026-03-16",
                                "EndDate": "2026-03-20",
                                "Start": 100.0,
                                "End": 110.0,
                                "ChangePct": 10.0,
                            }
                        ],
                        "note": "Cached S&P 500 weekly rows.",
                    }
                ),
                encoding="utf-8",
            )
            weekly_report.set_offline_mode(True)
            try:
                with mock.patch.object(weekly_report, "SP500_REPORT_CACHE_PATH", cache_path):
                    df = weekly_report.get_sp500_weekly_change(["AAPL"])
            finally:
                weekly_report.set_offline_mode(False)

        self.assertEqual(df["Ticker"].tolist(), ["AAPL"])
        self.assertEqual(df.iloc[0]["ChangePct"], 10.0)


if __name__ == "__main__":
    unittest.main()
