"""Tests for risk/exit, market-rule, operations and analytics features (feature release 2)."""

import gzip
import json
import logging
import ssl
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import notify
from database import (Database, FeeSchedule, HistoryStore, IST, daily_pnl, financial_year,
                      fy_report, slippage_report)
from sil_harness import (BT_END, BT_START, LOT, RefBroker, SimBroker, drive, history_window,
                         make_minutes, new_engine, run_pair, signature, temp_db)
from strategy import (DEFAULT_PARAMS, BSE_EXCHANGE_RATE, ExitTracker, Settings, blended_exit,
                      breaker_reason, monte_carlo, partial_quantity, replay, risk_levels, run_sweep,
                      sweep_grid)
from upstox_broker import InstrumentMaster, SPOT_KEYS, UpstoxBroker, pick_expiry, round_strike

INDEX = "NIFTY"


def setup(mode="PAPER", seed=0, latency=0.0, sync=True, index=INDEX, interval=5, sizing_value=1, **params):
    minutes = history_window(make_minutes(seed, index))
    broker = SimBroker(minutes, index, latency=latency)
    db = temp_db("f2.db")
    p = replace(DEFAULT_PARAMS, **params).validate()
    engine = new_engine(db, broker, Settings(index=index, interval=interval, sizing_value=sizing_value,
                                             params=p), mode, sync)
    return engine, broker, db


def until_entry(engine, broker, day=BT_START):
    return drive(engine, broker, day, day, stop=lambda: engine.state.trade is not None)


def events(db, *levels):
    return [e["message"] for e in db.recent_events() if e["level"] in levels]


def trade_fixture(side="CE"):
    return dict(side=side, entry_spot=1000.0, atr=10.0, levels=risk_levels(side, 1000.0, 10.0))


class ExitTrackerTests(unittest.TestCase):
    def test_defaults_match_plain_stop_target(self):
        for side, prices, expected in (("CE", [1001, 984.9], "STOP_LOSS"), ("CE", [1010, 1030], "TARGET"),
                                       ("PE", [999, 1015.1], "STOP_LOSS"), ("PE", [990, 970], "TARGET")):
            tracker = ExitTracker(trade_fixture(side), DEFAULT_PARAMS)
            results = [tracker.step(p)[0] for p in prices]
            self.assertEqual(results[-1], [expected], (side, prices))

    def test_breakeven_then_trail_ratchets_only_forward(self):
        p = replace(DEFAULT_PARAMS, breakeven_atr=1.0, trail_start_atr=1.5, trail_atr=1.0)
        trade = trade_fixture("CE")
        tracker = ExitTracker(trade, p)
        self.assertEqual(tracker.step(1011)[0], [])
        self.assertEqual(trade["levels"]["stop"], 1000.0)
        self.assertEqual(trade["track"]["kind"], "BREAKEVEN_STOP")
        tracker.step(1020)
        self.assertEqual(trade["levels"]["stop"], 1010.0)
        tracker.step(1015)
        self.assertEqual(trade["levels"]["stop"], 1010.0, "trail never moves backwards")
        self.assertEqual(tracker.step(1010)[0], ["TRAIL_STOP"])
        self.assertEqual(trade["track"]["mfe"], 20.0)

    def test_stop_checked_against_level_before_ratchet(self):
        p = replace(DEFAULT_PARAMS, breakeven_atr=0.5)
        trade = trade_fixture("PE")
        tracker = ExitTracker(trade, p)
        self.assertEqual(tracker.step(1016)[0], ["STOP_LOSS"])
        self.assertEqual(trade["track"]["mae"], 16.0)

    def test_partial_fires_once_and_quantity_rules(self):
        p = replace(DEFAULT_PARAMS, partial_atr=1.0)
        tracker = ExitTracker(trade_fixture("CE"), p)
        self.assertEqual(tracker.step(1010)[0], ["PARTIAL"])
        self.assertEqual(tracker.step(1012)[0], [])
        self.assertEqual(partial_quantity(130, 65, 50), 65)
        self.assertEqual(partial_quantity(65, 65, 50), 0, "single lot cannot be split")
        trade = dict(quantity=130, legs=[dict(qty=65, price=110.0)])
        self.assertEqual(blended_exit(trade, 90.0), 100.0)

    def test_param_validation(self):
        for bad in (dict(partial_atr=3.0), dict(partial_pct=95), dict(max_spread_pct=60),
                    dict(expiry_mode="Weekly"), dict(expiry_cutoff="25:00"), dict(event_days="2026-13-01"),
                    dict(max_week_loss_pct=150), dict(max_consec_losses=99)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                replace(DEFAULT_PARAMS, **bad).validate()


class RiskParity(unittest.TestCase):
    """Back-to-back: live engine (SIL) vs backtest with the new exit and day rules."""

    def test_b2b_breakeven_trail_partial(self):
        variants = [
            dict(breakeven_atr=1.0),
            dict(trail_start_atr=1.0, trail_atr=1.2),
            dict(partial_atr=1.0, partial_pct=50, sizing_value=2),
            dict(breakeven_atr=0.8, trail_start_atr=1.5, trail_atr=1.0, partial_atr=1.2, sizing_value=3),
        ]
        seen = set()
        for change in variants:
            sizing = change.pop("sizing_value", 1)
            for seed in (0, 1, 2):
                with self.subTest(change=change, seed=seed):
                    engine, backtest, db = run_pair(seed, INDEX, 5, sizing_value=sizing,
                                                    params=replace(DEFAULT_PARAMS, **change))
                    self.assertEqual([signature(t) for t in engine], [signature(t) for t in backtest])
                    self.assertEqual(events(db, "ERROR", "CRITICAL"), [])
                    for t in engine:
                        seen.add(t["exit_reason"])
                        seen |= {"PARTIAL"} if t.get("legs") else set()
        self.assertTrue({"BREAKEVEN_STOP", "TRAIL_STOP", "PARTIAL"} <= seen, seen)

    def test_b2b_expiry_modes_and_events(self):
        tuesday = date(2026, 10, 20)
        for change in (dict(expiry_mode="Cutoff", expiry_cutoff="10:00"), dict(expiry_mode="Next expiry"),
                       dict(event_days=f"{tuesday}")):
            with self.subTest(change=change):
                engine, backtest, _ = run_pair(1, INDEX, 5, params=replace(DEFAULT_PARAMS, **change))
                self.assertEqual([signature(t) for t in engine], [signature(t) for t in backtest])
        _, next_exp, _ = run_pair(1, INDEX, 5, params=replace(DEFAULT_PARAMS, expiry_mode="Next expiry"))
        for t in next_exp:
            self.assertNotEqual(t["expiry"], t["day"], "next-expiry mode never trades same-day expiry")
        _, events_bt, _ = run_pair(1, INDEX, 5, params=replace(DEFAULT_PARAMS, event_days=f"{tuesday}"))
        self.assertNotIn(str(tuesday), [t["day"] for t in events_bt])

    def test_b2b_circuit_breaker(self):
        p = replace(DEFAULT_PARAMS, max_consec_losses=1)
        engine, backtest, db = run_pair(0, INDEX, 5, params=p)
        self.assertEqual([signature(t) for t in engine], [signature(t) for t in backtest])
        first_loss = next((i for i, t in enumerate(engine) if t["net"] < 0), None)
        if first_loss is not None:
            self.assertEqual(len(engine), first_loss + 1, "no trades after the breaker trips")

    def test_b2b_min_premium_skips_without_consuming_day(self):
        p = replace(DEFAULT_PARAMS, min_premium=10_000)
        engine, backtest, db = run_pair(0, INDEX, 5, params=p)
        self.assertEqual(engine, [])
        self.assertEqual(backtest, [])
        self.assertIsNone(db.session(str(BT_START), "PAPER"), "rejected signal must not use the day slot")


class BreakerAndDayBlocks(unittest.TestCase):
    def closed(self, day, net):
        return dict(status="CLOSED", day=day, net=net, exit_time=f"{day}T11:00:00+05:30")

    def test_breaker_rules(self):
        today = date(2026, 10, 23)
        p = replace(DEFAULT_PARAMS, max_consec_losses=2, max_week_loss_pct=2, max_month_loss_pct=3)
        self.assertIsNone(breaker_reason([self.closed("2026-10-21", -100)], today, p, 1e5))
        two = [self.closed("2026-10-21", -100), self.closed("2026-10-22", -100)]
        self.assertIn("consecutive", breaker_reason(two, today, p, 1e5))
        self.assertIsNone(breaker_reason(two + [self.closed("2026-10-23", 50)], today, p, 1e5))
        week = [self.closed("2026-10-20", -2500), self.closed("2026-10-21", 400)]
        self.assertIn("week", breaker_reason(week, today, p, 1e5))
        month = [self.closed("2026-10-06", -2900), self.closed("2026-10-14", 400), self.closed("2026-10-21", -600)]
        self.assertIn("month", breaker_reason(month, today, p, 1e5))
        self.assertIsNone(breaker_reason(two, today, p, 1e5, since="2026-10-22T12:00"))

    def test_engine_breaker_blocks_and_reset_rearms(self):
        engine, broker, db = setup(max_consec_losses=1)
        db.reserve("2026-10-16", "PAPER")
        db.open_trade(dict(day="2026-10-16", mode="PAPER", entry_premium=100, quantity=65,
                           entry_time="2026-10-16T10:00:00+05:30", symbol="X"))
        db.close_trade(dict(day="2026-10-16", mode="PAPER", entry_premium=100, quantity=65,
                            entry_time="2026-10-16T10:00:00+05:30", symbol="X"), 50, "STOP_LOSS",
                       FeeSchedule(), datetime(2026, 10, 16, 11, tzinfo=IST))
        drive(engine, broker, BT_START, BT_START)
        self.assertEqual(len(db.trades()), 1)
        self.assertIn("circuit breaker", engine.state.banner)
        engine.reset_breaker()
        drive(engine, broker, BT_START + timedelta(days=1), BT_START + timedelta(days=1))
        self.assertEqual(len(db.trades()), 2)

    def test_holiday_and_event_day_block(self):
        engine, broker, db = setup(event_days=str(BT_START))
        drive(engine, broker, BT_START, BT_START)
        self.assertEqual(db.trades(), [])
        engine2, broker2, db2 = setup()
        broker2.holiday_calendar = {BT_START: ["NSE", "BSE"]}
        drive(engine2, broker2, BT_START, BT_START)
        self.assertEqual(db2.trades(), [])
        self.assertIn("holiday", engine2.state.banner)

    def test_spread_rejection_then_later_signal_can_trade(self):
        engine, broker, db = setup(max_spread_pct=2.0)
        broker.spread_pct = 10.0
        drive(engine, broker, BT_START, BT_START)
        self.assertEqual(db.trades(), [])
        self.assertTrue(any("spread" in m for m in events(db, "WARN")))
        broker.spread_pct = 0.5
        drive(engine, broker, BT_START + timedelta(days=1), BT_START + timedelta(days=1))
        self.assertEqual(len(db.trades()), 1)


class LiveExecution(unittest.TestCase):
    def test_freeze_slicing_entry_backstops_and_exit(self):
        engine, broker, db = setup(mode="LIVE", sizing_value=3)
        broker.freeze_lots = 1
        until_entry(engine, broker)
        trade = engine.state.trade
        self.assertEqual(trade["quantity"], 3 * LOT[INDEX])
        self.assertEqual(broker.calls["order_BUY"], 3)
        self.assertEqual(len(trade["backstops"]), 3)
        self.assertTrue(all(broker.orders[i]["quantity"] == LOT[INDEX] for i in trade["backstops"]))
        drive(engine, broker, BT_START, BT_START, begin_after=broker.now.floor("min"))
        closed = db.trades()[0]
        self.assertEqual(broker.calls["order_SELL"], 3)
        self.assertEqual(closed["fees"]["brokerage"], 6 * 20, "brokerage per executed slice")
        self.assertTrue(all(q == 0 for q in broker.position.values()))

    def test_live_partial_resizes_backstop_and_flattens(self):
        engine, broker, db = setup(mode="LIVE", sizing_value=2, partial_atr=0.6)
        drive(engine, broker)
        partials = [t for t in db.trades() if t.get("legs")]
        self.assertTrue(partials, "a partial exit should occur in the week")
        t = partials[0]
        self.assertEqual(t["legs"][0]["qty"], LOT[INDEX])
        self.assertEqual(t["status"], "CLOSED")
        sl_orders = [o for o in broker.orders.values() if o["order_type"] == "SL"]
        self.assertTrue(any(o["quantity"] == LOT[INDEX] for o in sl_orders), "backstop re-placed for remainder")
        self.assertTrue(all(o["status"] in ("cancelled", "complete") for o in sl_orders))
        self.assertTrue(all(q == 0 for q in broker.position.values()))
        self.assertEqual(events(db, "CRITICAL"), [])

    def test_exit_requested_during_partial_is_not_lost(self):
        engine, broker, db = setup(mode="LIVE", latency=0.02, sizing_value=2, partial_atr=0.6)
        until_entry(engine, broker)
        engine.pool = ThreadPoolExecutor(max_workers=4)
        engine.request_partial()
        engine.request_exit("TARGET")
        deadline = datetime.now() + timedelta(seconds=10)
        while engine.state.trade is not None and datetime.now() < deadline:
            threading.Event().wait(0.05)
        engine.pool.shutdown(wait=True)
        self.assertIsNone(engine.state.trade)
        closed = db.trades()[0]
        self.assertEqual(closed["exit_reason"], "TARGET")
        self.assertLess(sum(leg["qty"] for leg in closed.get("legs", [])), closed["quantity"])
        self.assertTrue(all(q == 0 for q in broker.position.values()))


class AnalyticsTests(unittest.TestCase):
    def test_monte_carlo(self):
        self.assertIsNone(monte_carlo([1, 2, 3], 1e5))
        mc = monte_carlo([500, -300, 800, -200, 400, -100, 650], 1e5, runs=3000, seed=1)
        self.assertLessEqual(mc["net_p5"], mc["net_p50"])
        self.assertLessEqual(mc["net_p50"], mc["net_p95"])
        self.assertLessEqual(mc["dd_p50"], mc["dd_p95"])
        self.assertEqual(monte_carlo([10] * 6, 1e5)["loss_probability"], 0.0)
        self.assertEqual(mc, monte_carlo([500, -300, 800, -200, 400, -100, 650], 1e5, runs=3000, seed=1))

    def test_sweep_out_of_sample_split(self):
        minutes = make_minutes(4, INDEX)
        grid = sweep_grid([5, 9], [21], [1.5], [3.0], ["Off"], DEFAULT_PARAMS)
        split = BT_START + timedelta(days=2)
        results = run_sweep(RefBroker(minutes, INDEX), Settings(index=INDEX), BT_START, BT_END, grid, split=split)
        for r in results:
            self.assertEqual(r["trades"] + r["oos_trades"], len(r["all_trades"]))
            self.assertIn("oos_sharpe", r)
        with self.assertRaises(ValueError):
            run_sweep(RefBroker(minutes, INDEX), Settings(index=INDEX), BT_START, BT_END, grid, split=BT_END)

    def test_reports(self):
        trades = [dict(mode="LIVE", status="CLOSED", day="2026-03-31", entry_ltp=100, entry_premium=101,
                       exit_ltp=120, exit_premium=119.4, quantity=65, gross=1196, taxes=60, net=1136,
                       fees=dict(brokerage=40, stt=9.7)),
                  dict(mode="LIVE", status="CLOSED", day="2026-04-01", entry_ltp=200, entry_premium=203,
                       exit_ltp=150, exit_premium=148.5, quantity=65, gross=-3542.5, taxes=62, net=-3604.5,
                       fees=dict(brokerage=40, stt=12))]
        report = slippage_report(trades)["LIVE"]
        self.assertAlmostEqual(report["entry_pct"], (1.0 + 1.5) / 2)
        self.assertAlmostEqual(report["exit_pct"], (0.5 + 1.0) / 2)
        self.assertEqual(financial_year("2026-03-31"), "FY2025-26")
        self.assertEqual(financial_year("2026-04-01"), "FY2026-27")
        fy = fy_report(trades, "FY2026-27")
        self.assertEqual((fy["trades"], fy["losses"]), (1, 1))
        self.assertAlmostEqual(fy["abs_pnl_turnover"], 3542.5)
        self.assertEqual(fy["charges"]["brokerage"], 40)
        self.assertEqual(daily_pnl(trades)[date(2026, 4, 1)], -3604.5)

    def test_journal(self):
        db = temp_db("journal.db")
        db.reserve("2026-10-20", "PAPER")
        trade_id = db.open_trade(dict(day="2026-10-20", mode="PAPER", entry_time="2026-10-20T10:00", symbol="X"))
        db.set_journal(trade_id, "  chased the move  ", "fomo, , late entry")
        row = db.trades()[0]
        self.assertEqual((row["note"], row["tags"]), ("chased the move", "fomo,late entry"))
        with self.assertRaises(KeyError):
            db.set_journal(999, "x", "")


class HistoryStoreTests(unittest.TestCase):
    def test_store_round_trip_and_incremental_download(self):
        folder = Path(tempfile.mkdtemp())
        store = HistoryStore(folder / "history.db")
        minutes = make_minutes(0, INDEX, first="2026-10-12", last="2026-10-23")
        broker = UpstoxBroker.__new__(UpstoxBroker)
        broker.history_store = store
        calls = []

        def download(index, start, end):
            calls.append((start, end))
            dates = minutes.index.date
            return minutes[(dates >= start) & (dates <= end)]

        broker._download_minutes = download
        with mock.patch("upstox_broker.now_ist", return_value=datetime(2026, 10, 26, 9, tzinfo=IST)):
            first = broker.spot_minutes(INDEX, date(2026, 10, 19), date(2026, 10, 23))
            second = broker.spot_minutes(INDEX, date(2026, 10, 12), date(2026, 10, 23))
            third = broker.spot_minutes(INDEX, date(2026, 10, 12), date(2026, 10, 23))
        self.assertEqual(calls, [(date(2026, 10, 19), date(2026, 10, 23)), (date(2026, 10, 12), date(2026, 10, 16))])
        self.assertEqual(len(first), 5 * 375)
        self.assertEqual(len(second), len(third))
        pd.testing.assert_frame_equal(third, minutes, check_freq=False, check_names=False)
        self.assertEqual(store.coverage()[0][3], 10)


class OpsTests(unittest.TestCase):
    def test_notifier_queue_and_failures(self):
        sent = []
        n = notify.Notifier(sender=lambda c, t, m: sent.append((t, m)))
        n.notify("x", "ignored while unconfigured")
        n.configure(dict(telegram_token="t", telegram_chat="1"))
        n.notify("TRADE", "ENTRY")
        n.wait_idle()
        self.assertEqual(sent, [("TRADE", "ENTRY")])
        bad = notify.Notifier(sender=mock.Mock(side_effect=RuntimeError("offline")))
        bad.configure(dict(smtp_host="h", email_to="a@b"))
        bad.notify("T", "M")
        bad.wait_idle()
        threading.Event().wait(0.1)
        self.assertEqual(bad.failed, 1)

    def test_telegram_and_email_payloads(self):
        with mock.patch("notify.requests.post") as post, mock.patch("notify.smtplib.SMTP") as smtp:
            notify.Notifier._send(dict(telegram_token="TOK", telegram_chat="42", smtp_host="smtp.x",
                                       smtp_port=587, email_to="me@x", smtp_user="u", smtp_password="p"),
                                  "Title", "Body")
        self.assertIn("botTOK/sendMessage", post.call_args.args[0])
        self.assertEqual(post.call_args.kwargs["json"], {"chat_id": "42", "text": "Title\nBody"})
        session = smtp.return_value.__enter__.return_value
        session.starttls.assert_called_once()
        session.login.assert_called_once_with("u", "p")

    def test_schedule_command(self):
        command = notify.schedule_command("08:55")
        self.assertEqual(command[:2], ["schtasks", "/Create"])
        self.assertIn("08:55", command)
        self.assertIn("MON,TUE,WED,THU,FRI", command)
        self.assertTrue(command[-1].endswith("main.py") or command[-1].endswith("main.py\""))
        self.assertEqual(notify.unschedule_command()[:2], ["schtasks", "/Delete"])

    def test_engine_events_reach_log_file_logger(self):
        engine, _, _ = setup()
        with self.assertLogs("strat", level="INFO") as captured:
            engine.emit("CRITICAL", "boom")
        self.assertTrue(any("boom" in line for line in captured.output))


class MarketsAndTls(unittest.TestCase):
    def test_sensex_strikes_contract_and_fees(self):
        self.assertEqual(round_strike(80549.9, "SENSEX"), 80500)
        self.assertEqual(round_strike(80550.0, "SENSEX"), 80600)
        ms = lambda d: int(datetime(2026, 11, d, 15, 30, tzinfo=IST).timestamp() * 1000)
        base = dict(underlying_key=SPOT_KEYS["SENSEX"], instrument_type="CE", strike_price=80500.0,
                    lot_size=20, freeze_quantity=1000)
        master = InstrumentMaster(Path("."))
        master.records = [base | dict(segment="BSE_FO", expiry=ms(5), instrument_key="BSE_FO|1", trading_symbol="S1"),
                          base | dict(segment="BSE_FO", expiry=ms(12), instrument_key="BSE_FO|2", trading_symbol="S2")]
        today = date(2026, 11, 5)
        self.assertEqual(master.option("SENSEX", 80510, "CE", today)["key"], "BSE_FO|1")
        nxt = master.option("SENSEX", 80510, "CE", today, skip_today=True)
        self.assertEqual((nxt["key"], nxt["exchange"]), ("BSE_FO|2", "BSE"))
        self.assertEqual(pick_expiry([today], today, skip_today=True), None)
        engine, _, _ = setup()
        self.assertEqual(engine.fees_for("BSE").exchange_rate, BSE_EXCHANGE_RATE)
        self.assertNotEqual(engine.fees_for("NSE").exchange_rate, BSE_EXCHANGE_RATE)

    def test_missing_strike_in_nearest_expiry_blocks_instead_of_drifting(self):
        ms = lambda d: int(datetime(2026, 11, d, 15, 30, tzinfo=IST).timestamp() * 1000)
        base = dict(underlying_key=SPOT_KEYS["NIFTY"], instrument_type="CE", lot_size=65, segment="NSE_FO")
        master = InstrumentMaster(Path("."))
        master.records = [base | dict(strike_price=24000.0, expiry=ms(3), instrument_key="a", trading_symbol="a"),
                          base | dict(strike_price=24500.0, expiry=ms(10), instrument_key="b", trading_symbol="b")]
        with self.assertRaises(Exception):
            master.option("NIFTY", 24510, "CE", date(2026, 11, 1))

    def test_rest_tls_uses_os_trust_store_with_verification(self):
        context = ssl.create_default_context()
        self.assertEqual(type(context).__module__.split(".")[0], "truststore")
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)


if __name__ == "__main__":
    unittest.main()
