import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from database import Database, FeeSchedule, IST, metrics


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.path = Path(self.folder.name) / "trades.db"
        self.db = Database(self.path)

    def tearDown(self):
        self.folder.cleanup()

    def test_atomic_daily_limit_per_mode_survives_restart(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            attempts = list(pool.map(lambda _: self.db.reserve("2026-10-29", "PAPER"), range(20)))
        self.assertEqual(sum(attempts), 1)
        self.assertFalse(Database(self.path).reserve("2026-10-29", "PAPER"))
        self.assertTrue(Database(self.path).reserve("2026-10-29", "LIVE"), "live has its own slot")
        self.assertFalse(self.db.reserve("2026-10-29", "LIVE"))
        self.assertTrue(self.db.reserve("2026-10-30", "PAPER"))

    def test_v1_database_is_migrated_in_place(self):
        import sqlite3
        legacy = Path(self.folder.name) / "legacy.db"
        with sqlite3.connect(legacy) as connection:
            connection.executescript("""
                CREATE TABLE sessions (day TEXT PRIMARY KEY, state TEXT NOT NULL, mode TEXT NOT NULL,
                                       detail TEXT NOT NULL DEFAULT '');
                CREATE TABLE trades (id INTEGER PRIMARY KEY AUTOINCREMENT, day TEXT NOT NULL UNIQUE,
                                     mode TEXT NOT NULL, status TEXT NOT NULL, payload TEXT NOT NULL,
                                     entry_time TEXT NOT NULL, exit_time TEXT, gross REAL DEFAULT 0,
                                     taxes REAL DEFAULT 0, slippage REAL DEFAULT 0, net REAL DEFAULT 0);
                INSERT INTO sessions VALUES ('2026-10-28', 'EXHAUSTED', 'PAPER', '');
                INSERT INTO trades(day, mode, status, payload, entry_time, net)
                    VALUES ('2026-10-28', 'PAPER', 'CLOSED', '{"symbol": "X"}', '2026-10-28T10:00', 5);
            """)
        connection.close()
        db = Database(legacy)
        self.assertEqual(db.trades()[0]["symbol"], "X")
        self.assertEqual(db.session("2026-10-28", "PAPER")["state"], "EXHAUSTED")
        self.assertTrue(db.reserve("2026-10-28", "LIVE"))
        self.assertFalse(db.reserve("2026-10-28", "PAPER"))
        Database(legacy)  # second open must be a no-op

    def test_explicit_fee_components(self):
        fees = FeeSchedule().calculate(100, 120, 50)
        self.assertEqual(fees["brokerage"], 40)
        self.assertEqual(fees["stt"], 7.5)
        self.assertEqual(fees["stamp"], 0.15)
        self.assertEqual(fees["exchange"], 3.85)
        self.assertEqual(fees["gst"], 7.9)
        self.assertAlmostEqual(fees["total"], 59.41)

    def test_trade_recovery_and_net_pnl(self):
        trade = dict(day="2026-10-29", mode="PAPER", entry_premium=100,
                     entry_time=datetime(2026, 10, 29, 10, tzinfo=IST).isoformat(), quantity=50)
        self.db.reserve(trade["day"], trade["mode"])
        self.db.open_trade(trade)
        self.assertEqual(Database(self.path).active_trade(), trade)
        self.db.close_trade(trade, 120, "TARGET", FeeSchedule(),
                            datetime(2026, 10, 29, 11, tzinfo=IST))
        self.assertIsNone(self.db.active_trade())
        result = metrics(self.db.trades())
        self.assertAlmostEqual(result["net"], 940.59)
        self.assertEqual(result["avg_duration"], 60)
        self.assertIsNone(result["sharpe"])


if __name__ == "__main__":
    unittest.main()