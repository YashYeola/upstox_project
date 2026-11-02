"""SIL scenarios: real TradingEngine + simulated market/broker, incl. fault injection."""

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import time as dtime

import pandas as pd

from database import Database
from sil_harness import (BT_END, BT_START, LOT, SimBroker, drive, history_window, make_minutes,
                         new_engine, run_pair, signature, temp_db)
from strategy import Settings, TradingEngine
from upstox_broker import SPOT_KEYS

SEED, INDEX = 0, "NIFTY"


def setup(mode="PAPER", seed=SEED, latency=0.0, sync=True, **settings):
    minutes = history_window(make_minutes(seed, INDEX))
    broker = SimBroker(minutes, INDEX, latency=latency)
    db = temp_db("sil.db")
    engine = new_engine(db, broker, Settings(index=INDEX, interval=5, **settings), mode, sync)
    return engine, broker, db


def events(db, *levels):
    return [e["message"] for e in db.recent_events() if e["level"] in levels]


def first_day_until_entry(engine, broker):
    return drive(engine, broker, BT_START, BT_START, stop=lambda: engine.state.trade is not None)


class PaperWeek(unittest.TestCase):
    def test_week_invariants(self):
        engine, broker, db = setup()
        drive(engine, broker)
        trades = db.trades()
        self.assertTrue(trades)
        days = [t["day"] for t in trades]
        self.assertEqual(len(days), len(set(days)), "more than one trade on a day")
        for t in trades:
            entry, exit_ = pd.Timestamp(t["entry_time"]), pd.Timestamp(t["exit_time"])
            self.assertEqual(t["status"], "CLOSED")
            self.assertGreaterEqual(entry.time(), dtime(9, 35))
            self.assertLess(entry.time(), dtime(15, 0, 4))
            self.assertLessEqual(exit_.time(), dtime(15, 15, 59))
            self.assertLess(entry, exit_)
            self.assertEqual(t["strike"] % 50, 0)
            self.assertLessEqual(abs(t["strike"] - t["entry_spot"]), 25)
            self.assertAlmostEqual(t["net"], t["gross"] - t["taxes"], places=6)
            self.assertAlmostEqual(t["taxes"], t["fees"]["total"], places=6)
            self.assertEqual(t["fees"]["brokerage"], 40)
        self.assertEqual(events(db, "ERROR", "CRITICAL"), [])
        self.assertIsNone(engine.state.trade)
        self.assertEqual(engine.state.banner, "Daily Trade Limit Exhausted - Blocked")


class LiveSIL(unittest.TestCase):
    def test_live_happy_path_orders_and_flat_book(self):
        engine, broker, db = setup("LIVE")
        drive(engine, broker)
        trades = db.trades(mode="LIVE")
        self.assertTrue(trades)
        self.assertEqual(broker.calls["order_BUY"], len(trades))
        self.assertEqual(broker.calls["order_SELL"], len(trades))
        self.assertTrue(all(q == 0 for q in broker.position.values()))
        self.assertTrue(all(t["fees"]["brokerage"] == 40 for t in trades))
        self.assertEqual(events(db, "ERROR", "CRITICAL"), [])

    def test_entry_rejected_blocks_and_halts_without_retry(self):
        engine, broker, db = setup("LIVE")
        broker.order_faults = ["reject"]
        drive(engine, broker)
        self.assertEqual(broker.calls["order_BUY"], 1)
        self.assertEqual(db.trades(), [])
        self.assertIn("HALTED", engine.state.banner)
        self.assertEqual(db.session(str(BT_START), "LIVE")["state"], "BLOCKED")
        self.assertTrue(all(q == 0 for q in broker.position.values()))

    def test_entry_network_error_blocks_and_halts(self):
        engine, broker, db = setup("LIVE")
        broker.order_faults = ["raise"]
        drive(engine, broker)
        self.assertEqual(broker.calls["order_BUY"], 1)
        self.assertEqual(db.trades(), [])
        self.assertTrue(engine.state.halted)

    def test_entry_partial_fill_trades_filled_quantity(self):
        engine, broker, db = setup("LIVE", sizing_value=2)
        broker.order_faults = ["partial"]
        drive(engine, broker, BT_START, BT_START)
        trade = db.trades()[0]
        self.assertEqual(trade["quantity"], LOT[INDEX])
        self.assertEqual(trade["status"], "CLOSED")
        self.assertTrue(all(q == 0 for q in broker.position.values()))

    def test_entry_never_fills_is_cancelled(self):
        engine, broker, db = setup("LIVE")
        engine._await_fill = lambda oid: TradingEngine._await_fill(engine, oid, timeout=0.3)
        broker.order_faults = ["pending"]
        drive(engine, broker, BT_START, BT_START)
        self.assertEqual(list(broker.orders.values())[0]["status"], "cancelled")
        self.assertEqual(db.trades(), [])
        self.assertTrue(engine.state.halted)

    def test_exit_rejected_once_then_filled(self):
        engine, broker, db = setup("LIVE")
        first_day_until_entry(engine, broker)
        broker.order_faults = ["reject"]
        drive(engine, broker, BT_START, BT_START, begin_after=broker.now.floor("min"))
        trade = db.trades()[0]
        self.assertEqual(trade["status"], "CLOSED")
        self.assertEqual(broker.calls["order_SELL"], 2)
        self.assertEqual(trade["fees"]["brokerage"], 40, "rejected orders are not charged")
        self.assertFalse(engine.state.halted)
        self.assertTrue(all(q == 0 for q in broker.position.values()))

    def test_exit_keeps_failing_halts_and_stops_sending_orders(self):
        engine, broker, db = setup("LIVE")
        first_day_until_entry(engine, broker)
        broker.order_faults = ["reject"] * 50
        drive(engine, broker, begin_after=broker.now.floor("min"))
        self.assertEqual(broker.calls["order_SELL"], 3, "orders kept firing after HALT")
        self.assertIsNotNone(db.active_trade())
        self.assertEqual(db.trades()[0]["status"], "OPEN")
        self.assertIn("EXIT FAILED", engine.state.halted)
        self.assertTrue(events(db, "CRITICAL"))
        self.assertEqual(broker.calls["order_BUY"], 1, "new entries after HALT")


class Recovery(unittest.TestCase):
    def test_restart_mid_trade_matches_uninterrupted_run(self):
        reference, _, _ = run_pair(SEED, INDEX, 5)
        engine, broker, db = setup()
        last = first_day_until_entry(engine, broker)
        engine.shutdown()
        restarted = new_engine(db, broker, engine.settings)
        restarted.recover()
        self.assertIsNotNone(restarted.state.trade)
        drive(restarted, broker, begin_after=last)
        self.assertEqual([signature(t) for t in db.trades()], [signature(t) for t in reference])

    def test_feed_outage_falls_back_to_rest_and_still_exits(self):
        engine, broker, db = setup()
        last = first_day_until_entry(engine, broker)
        end_outage = last + pd.Timedelta(minutes=45)
        broker.outage = True
        drive(engine, broker, BT_START, BT_START, begin_after=last,
              on_minute=lambda m: setattr(broker, "outage", m <= end_outage))
        self.assertGreater(broker.calls["spot_quote"], 0)
        trade = db.trades()[0]
        self.assertEqual(trade["status"], "CLOSED")
        self.assertEqual(events(db, "ERROR", "CRITICAL"), [])


class Concurrency(unittest.TestCase):
    def test_simultaneous_exit_triggers_send_one_sell(self):
        engine, broker, db = setup("LIVE", latency=0.01, sync=True)
        first_day_until_entry(engine, broker)
        engine.pool = ThreadPoolExecutor(max_workers=8)
        trade = engine.state.trade
        crossing = trade["levels"]["target"] + 1 if trade["side"] == "CE" else trade["levels"]["target"] - 1
        barrier = threading.Barrier(12)

        def fire():
            barrier.wait()
            engine._tick(SPOT_KEYS[INDEX], crossing, broker.now.to_pydatetime())

        threads = [threading.Thread(target=fire) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        engine.pool.shutdown(wait=True)
        self.assertEqual(broker.calls["order_SELL"], 1)
        self.assertEqual(broker.position[trade["key"]], 0)
        self.assertEqual(db.trades()[0]["status"], "CLOSED")
        self.assertEqual(events(db, "CRITICAL"), [])
        self.assertEqual(len([m for m in events(db, "TRADE") if m.startswith("EXIT")]), 1)

    def test_exit_guard_is_atomic_under_forced_interleaving(self):
        class SlowEvent(threading.Event):
            def is_set(self):
                result = super().is_set()
                threading.Event().wait(0.02)  # widen the check-then-set window
                return result

        engine, broker, db = setup("LIVE", latency=0.01)
        first_day_until_entry(engine, broker)
        engine.pool = ThreadPoolExecutor(max_workers=8)
        engine.exiting = SlowEvent()
        barrier = threading.Barrier(6)

        def fire():
            barrier.wait()
            engine.request_exit("TARGET")

        threads = [threading.Thread(target=fire) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        engine.pool.shutdown(wait=True)
        self.assertEqual(broker.calls["order_SELL"], 1)
        self.assertEqual(events(db, "CRITICAL"), [])


if __name__ == "__main__":
    unittest.main()
