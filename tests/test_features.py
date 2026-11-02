"""Tests for the feature set added after SIL: backstop SL, reconciliation, manual close,
per-mode limits, tick bars, configurable params, trend filter, sweep, TLS, alerts data."""

import json
import ssl
import unittest
from dataclasses import replace
from unittest import mock

import numpy as np
import pandas as pd

from database import FeeSchedule
from sil_harness import (BT_END, BT_START, RefBroker, SimBroker, drive, history_window,
                         make_minutes, new_engine, run_pair, signature, temp_db)
from strategy import (DEFAULT_PARAMS, Params, Settings, TickBars, indicators, run_backtest,
                      run_sweep, sweep_grid, trend_ok)
from upstox_broker import SPOT_KEYS, OrderStream

INDEX = "NIFTY"


def setup(mode="LIVE", seed=0, **params):
    minutes = history_window(make_minutes(seed, INDEX))
    broker = SimBroker(minutes, INDEX)
    db = temp_db("f.db")
    p = replace(DEFAULT_PARAMS, **params).validate()
    engine = new_engine(db, broker, Settings(index=INDEX, interval=5, params=p), mode)
    return engine, broker, db


def until_entry(engine, broker):
    return drive(engine, broker, BT_START, BT_START, stop=lambda: engine.state.trade is not None)


def events(db, *levels):
    return [e["message"] for e in db.recent_events() if e["level"] in levels]


class BackstopTests(unittest.TestCase):
    def test_backstop_placed_then_cancelled_on_normal_exit(self):
        engine, broker, db = setup()
        until_entry(engine, broker)
        trade = engine.state.trade
        self.assertEqual(len(trade["backstops"]), 1)
        backstop = broker.orders[trade["backstops"][0]]
        self.assertEqual(backstop["order_type"], "SL")
        self.assertLess(backstop["trigger"], trade["entry_premium"])
        self.assertLess(backstop["limit"], backstop["trigger"])
        backstop_id = trade["backstops"][0]
        drive(engine, broker, BT_START, BT_START, begin_after=broker.now.floor("min"))
        closed = db.trades()[0]
        self.assertEqual(closed["status"], "CLOSED")
        self.assertEqual(broker.orders[backstop_id]["status"], "cancelled")
        self.assertEqual(broker.calls["order_SELL"], 1)
        self.assertEqual(closed["fees"]["brokerage"], 40, "cancelled backstop is not an executed order")
        self.assertTrue(all(q == 0 for q in broker.position.values()))

    def test_exchange_fired_backstop_is_recorded_without_second_sell(self):
        engine, broker, db = setup()
        until_entry(engine, broker)
        trade = engine.state.trade
        broker.trigger_stop(trade["backstops"][0], price=50.0)
        closed = db.trades()[0]
        self.assertEqual(closed["exit_reason"], "BACKSTOP_SL")
        self.assertEqual(closed["exit_premium"], 50.0)
        self.assertEqual(broker.calls["order_SELL"], 0)
        self.assertEqual(broker.position[trade["key"]], 0)
        self.assertIsNone(engine.state.trade)

    def test_stuck_backstop_cancel_refuses_market_sell(self):
        engine, broker, db = setup()
        until_entry(engine, broker)
        broker.stuck_cancel = True
        engine.request_exit("TARGET")
        self.assertEqual(broker.calls["order_SELL"], 0, "would risk a double SELL")
        self.assertIn("EXIT FAILED", engine.state.halted)
        self.assertEqual(db.trades()[0]["status"], "OPEN")

    def test_backstop_can_be_disabled(self):
        engine, broker, _ = setup(backstop=False)
        until_entry(engine, broker)
        self.assertIsNone(engine.state.trade.get("backstops"))
        self.assertEqual(broker.calls["order_SL"], 0)


class ReconcileAndManualClose(unittest.TestCase):
    def test_matching_book_passes(self):
        engine, broker, db = setup()
        until_entry(engine, broker)
        self.assertEqual(engine.reconcile(), [])
        self.assertFalse(engine.state.halted)

    def test_app_open_but_broker_flat_halts(self):
        engine, broker, db = setup()
        until_entry(engine, broker)
        broker.position[engine.state.trade["key"]] = 0
        self.assertTrue(engine.reconcile())
        self.assertIn("RECONCILIATION", engine.state.halted)
        self.assertTrue(events(db, "CRITICAL"))

    def test_quantity_mismatch_halts(self):
        engine, broker, _ = setup()
        until_entry(engine, broker)
        broker.position[engine.state.trade["key"]] += 75
        self.assertTrue(any("mismatch" in issue for issue in engine.reconcile()))

    def test_untracked_broker_position_from_app_order_halts(self):
        engine, broker, db = setup()
        until_entry(engine, broker)
        with db.connect() as connection:
            connection.execute("DELETE FROM trades")
        engine.state.trade = None
        issues = engine.reconcile()
        self.assertTrue(any("Untracked" in issue for issue in issues))
        self.assertTrue(engine.state.halted)

    def test_manual_close_after_exit_failure_halt(self):
        engine, broker, db = setup()
        until_entry(engine, broker)
        trade = engine.state.trade
        broker.order_faults = ["reject"] * 3
        engine.request_exit("TARGET")
        self.assertIn("EXIT FAILED", engine.state.halted)
        with self.assertRaises(Exception):
            engine.manual_close(120.0)  # broker still long -> refused
        broker.position[trade["key"]] = 0  # operator flattens in the broker terminal
        closed = engine.manual_close(120.0, "closed in Upstox app")
        self.assertEqual(closed["exit_reason"], "MANUAL_CLOSE")
        self.assertEqual(closed["exit_premium"], 120.0)
        self.assertEqual(db.trades()[0]["status"], "CLOSED")
        self.assertEqual(engine.state.halted, "")
        self.assertIsNone(engine.state.trade)

    def test_manual_close_validates_price_and_paper_needs_no_broker_check(self):
        engine, broker, db = setup(mode="PAPER")
        until_entry(engine, broker)
        for bad in (0, -5, float("nan")):
            with self.assertRaises(ValueError):
                engine.manual_close(bad)
        self.assertEqual(engine.manual_close(99.5)["exit_reason"], "MANUAL_CLOSE")


class PerModeLimit(unittest.TestCase):
    def test_paper_trade_does_not_consume_live_slot(self):
        engine, broker, db = setup(mode="PAPER")
        drive(engine, broker, BT_START, BT_START)
        self.assertEqual(len(db.trades(mode="PAPER")), 1)
        self.assertEqual(engine.state.banner, "Daily Trade Limit Exhausted - Blocked")
        engine.set_mode("LIVE", armed=True)
        self.assertFalse(engine.state.blocked, "live slot must still be free")
        self.assertTrue(db.reserve(str(BT_START), "LIVE"))


class TickBarTests(unittest.TestCase):
    def test_tickbars_require_full_minute_coverage(self):
        bars = TickBars()
        start = pd.Timestamp("2026-10-29 10:00", tz="Asia/Kolkata")
        for minute in range(5):
            for second, price in ((0, 100 + minute), (20, 103 + minute), (40, 98 + minute)):
                bars.add(float(price), (start + pd.Timedelta(minutes=minute, seconds=second)).to_pydatetime())
        bar = bars.bar(start, 5)
        self.assertEqual((bar["open"], bar["high"], bar["low"], bar["close"]), (100, 107, 98, 102))
        late = TickBars()
        late.add(100.0, (start + pd.Timedelta(seconds=45)).to_pydatetime())
        self.assertIsNone(late.bar(start, 1), "first tick 45 s late -> open unknown")
        self.assertIsNone(bars.bar(start, 10), "missing minutes -> no bar")

    def test_engine_uses_ticks_and_matches_rest(self):
        ticks, _, _ = run_pair(1, INDEX, 5, params=replace(DEFAULT_PARAMS, bar_source="Ticks"))
        rest, _, _ = run_pair(1, INDEX, 5, params=replace(DEFAULT_PARAMS, bar_source="REST"))
        self.assertEqual([signature(t) for t in ticks], [signature(t) for t in rest])

    def test_ticks_cut_rest_calls_and_fall_back_on_outage(self):
        tick_engine, tick_broker, _ = setup(mode="PAPER", bar_source="Ticks")
        drive(tick_engine, tick_broker, BT_START, BT_START)
        rest_engine, rest_broker, _ = setup(mode="PAPER", bar_source="REST")
        drive(rest_engine, rest_broker, BT_START, BT_START)
        self.assertLess(tick_broker.calls["candles"], rest_broker.calls["candles"] / 5)
        outage_engine, outage_broker, _ = setup(mode="PAPER", bar_source="Ticks")
        stamp = pd.Timestamp(f"{BT_START} 11:00", tz="Asia/Kolkata")
        drive(outage_engine, outage_broker, BT_START, BT_START,
              on_minute=lambda m: setattr(outage_broker, "outage", stamp <= m < stamp + pd.Timedelta(minutes=12)))
        self.assertGreater(outage_broker.calls["candles"], tick_broker.calls["candles"])


class ParamsAndFilters(unittest.TestCase):
    def test_validation_and_warmup(self):
        self.assertEqual(DEFAULT_PARAMS.warmup, 121)
        self.assertEqual(replace(DEFAULT_PARAMS, trend_filter="EMA+ADX", trend_ema=200).warmup, 300)
        bad = [dict(ema_fast=21, ema_slow=9), dict(signal_start="09:00"), dict(square_off="15:45"),
               dict(entry_cutoff="15:20"), dict(sl_atr=0), dict(trend_filter="EMA", trend_ema=10),
               dict(slippage_pct=9), dict(backstop_pct=95)]
        for change in bad:
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(DEFAULT_PARAMS, **change).validate()

    def test_trend_ok_rules(self):
        p = replace(DEFAULT_PARAMS, trend_filter="EMA+ADX", adx_min=25)
        bar = pd.Series({"close": 100.0, "trend_ema": 95.0, "adx": 30.0})
        self.assertTrue(trend_ok(bar, "CE", p))
        self.assertFalse(trend_ok(bar, "PE", p))
        self.assertFalse(trend_ok(bar.copy().replace(30.0, 10.0), "CE", p))
        self.assertFalse(trend_ok(pd.Series({"close": 100.0, "trend_ema": np.nan, "adx": 30.0}), "CE", p))

    def test_short_history_gives_nan_not_crash(self):
        idx = pd.date_range("2026-10-29 09:15", periods=5, freq="5min", tz="Asia/Kolkata")
        frame = pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 0.0}, index=idx)
        out = indicators(frame, replace(DEFAULT_PARAMS, trend_filter="EMA+ADX"))
        self.assertTrue(out[["ema_fast", "ema_slow", "atr", "trend_ema", "adx"]].isna().all().all())

    def test_b2b_with_custom_params_and_filters(self):
        variants = [
            replace(DEFAULT_PARAMS, ema_fast=5, ema_slow=13, sl_atr=1.0, tp_atr=2.0,
                    signal_start="09:45", entry_cutoff="14:30", square_off="15:00", slippage_pct=1.0),
            replace(DEFAULT_PARAMS, trend_filter="EMA", trend_ema=50),
            replace(DEFAULT_PARAMS, trend_filter="ADX", adx_min=20),
            replace(DEFAULT_PARAMS, trend_filter="EMA+ADX", trend_ema=50, adx_min=18),
        ]
        for p in variants:
            with self.subTest(params=p):
                engine, backtest, db = run_pair(2, INDEX, 5, params=p)
                self.assertEqual([signature(t) for t in engine], [signature(t) for t in backtest])
                self.assertEqual(events(db, "ERROR", "CRITICAL"), [])
        custom, _, _ = run_pair(2, INDEX, 5, params=variants[0])
        for t in custom:
            self.assertGreaterEqual(pd.Timestamp(t["entry_time"]).time(), variants[0].start_time)
            self.assertLessEqual(pd.Timestamp(t["exit_time"]).strftime("%H:%M"), "15:00")


class SweepTests(unittest.TestCase):
    def test_grid_validation_and_limit(self):
        grid = sweep_grid([5, 9, 30], [21], [1.5], [3.0], ["Off"], DEFAULT_PARAMS)
        self.assertEqual([p.ema_fast for p in grid], [5, 9])
        with self.assertRaises(ValueError):
            sweep_grid(range(2, 20), range(21, 40), [1, 2], [2, 3], ["Off"], DEFAULT_PARAMS)

    def test_sweep_matches_individual_backtests_and_ranks(self):
        minutes = make_minutes(4, INDEX)
        broker = RefBroker(minutes, INDEX)
        calls = mock.Mock(wraps=broker.backtest_option)
        broker.backtest_option = calls
        grid = sweep_grid([5, 9], [21], [1.5], [2.0, 3.0], ["Off", "ADX"], DEFAULT_PARAMS)
        results = run_sweep(broker, Settings(index=INDEX), BT_START, BT_END, grid)
        self.assertEqual(len(results), len(grid))
        sharpes = [r["sharpe"] if r["sharpe"] is not None else -np.inf for r in results]
        self.assertEqual(sharpes, sorted(sharpes, reverse=True))
        self.assertLessEqual(calls.call_count, 2 * 2 * 5, "option data is cached per day/side/strike")
        best = results[0]
        db = temp_db("sweep.db")
        run_backtest(RefBroker(minutes, INDEX), db, Settings(index=INDEX, params=best["params"]),
                     BT_START, BT_END)
        self.assertEqual(len(db.trades()), best["trades"])
        self.assertAlmostEqual(sum(t["net"] for t in db.trades()), best["net"], places=6)


class TransportAndStreams(unittest.TestCase):
    def test_websocket_tls_is_forced_on(self):
        import websocket
        captured = {}
        base = websocket.WebSocketApp.__mro__[1]
        with mock.patch.object(base, "run_forever", lambda self, **kw: captured.update(kw)):
            websocket.WebSocketApp("wss://example.invalid").run_forever(
                sslopt={"cert_reqs": ssl.CERT_NONE, "check_hostname": False})
        self.assertEqual(captured["sslopt"]["cert_reqs"], ssl.CERT_REQUIRED)
        self.assertTrue(captured["sslopt"]["check_hostname"])
        context = captured["sslopt"]["context"]
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)
        self.assertEqual(type(context).__module__.split(".")[0], "truststore")

    def test_order_stream_parses_only_order_updates(self):
        seen = []
        stream = OrderStream(None, seen.append, lambda *a: None)
        stream._message(json.dumps({"update_type": "order", "order_id": "1", "status": "complete"}))
        stream._message(json.dumps({"update_type": "position", "order_id": "2"}))
        stream._message("not json")
        stream._message({"update_type": "order", "order_id": "3", "status": "open"})
        self.assertEqual([u["order_id"] for u in seen], ["1", "3"])
        self.assertIsNone(OrderStream.watchdog)


class FeeAwarePnL(unittest.TestCase):
    def test_unrealized_net_subtracts_round_trip_charges(self):
        engine, broker, _ = setup(mode="PAPER")
        until_entry(engine, broker)
        trade = engine.state.trade
        engine.state.option_ltp = trade["entry_premium"] + 10
        net, charges = engine.unrealized_net()
        expected = FeeSchedule().calculate(trade["entry_premium"], engine.state.option_ltp,
                                           trade["quantity"])["total"]
        self.assertAlmostEqual(charges, expected)
        self.assertAlmostEqual(net, 10 * trade["quantity"] - expected)


if __name__ == "__main__":
    unittest.main()
