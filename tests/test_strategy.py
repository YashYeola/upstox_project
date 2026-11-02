import tempfile
import unittest
from datetime import datetime, time as dtime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from database import Database, FeeSchedule, IST
from strategy import (Settings, TradingEngine, crossover, exit_reason, indicators, lots_for,
                      risk_levels, simulate_day)
from upstox_broker import SPOT_KEYS, InstrumentMaster, aggregate_candles, round_strike


def ist(h, m, s=0, day=29):
    return datetime(2026, 10, day, h, m, s, tzinfo=IST)


def frame_ending(close_values, last_start):
    index = pd.date_range(end=pd.Timestamp(last_start), periods=len(close_values), freq="5min")
    close = pd.Series(close_values, index=index, dtype=float)
    return indicators(pd.DataFrame({"open": close, "high": close + 5, "low": close - 5,
                                    "close": close, "volume": 0}))


def cross_series(up=True):
    base = np.r_[np.linspace(100, 60, 140), [200.0]] if up else np.r_[np.linspace(60, 100, 140), [-100.0]]
    return base


class FakeMaster:
    loaded_day = None

    def refresh(self):
        self.loaded_day = ist(9, 0).date()

    def nearest_expiry(self, index, today=None, skip_today=False):
        return ist(10, 0, day=3).date().replace(month=11)

    def option(self, index, spot, side, today, skip_today=False):
        return dict(key=f"NSE_FO|{side}", symbol=f"NIFTY {round_strike(spot, index)} {side}",
                    lot_size=75, strike=round_strike(spot, index), side=side,
                    expiry="2026-11-03", freeze=1800)


class FakeBroker:
    def __init__(self):
        self.master, self.channels, self.orders = FakeMaster(), {}, []

    def quote(self, key):
        return 100.0, ist(10, 0)

    def stream(self, *args):
        pass

    def close(self):
        pass


class RuleTests(unittest.TestCase):
    def test_strike_rounding(self):
        self.assertEqual(round_strike(24524.9, "NIFTY"), 24500)
        self.assertEqual(round_strike(24525.0, "NIFTY"), 24550)
        self.assertEqual(round_strike(51249.9, "BANKNIFTY"), 51200)
        self.assertEqual(round_strike(51250.0, "BANKNIFTY"), 51300)

    def test_risk_levels_and_exits(self):
        ce = risk_levels("CE", 24500, 20)
        self.assertEqual(ce, {"stop": 24470, "target": 24560})
        self.assertEqual(exit_reason("CE", 24470, ce), "STOP_LOSS")
        self.assertEqual(exit_reason("CE", 24560, ce), "TARGET")
        self.assertIsNone(exit_reason("CE", 24500, ce))
        pe = risk_levels("PE", 24500, 20)
        self.assertEqual(pe, {"stop": 24530, "target": 24440})
        self.assertEqual(exit_reason("PE", 24530, pe), "STOP_LOSS")
        self.assertEqual(exit_reason("PE", 24440, pe), "TARGET")
        with self.assertRaises(ValueError):
            risk_levels("CE", 24500, float("nan"))

    def test_crossover_direction_and_morning_filter(self):
        up = frame_ending(cross_series(True), "2026-10-29 10:00:00+05:30")
        self.assertEqual(crossover(up, 5, ist(10, 5, 3)), "CE")
        down = frame_ending(cross_series(False), "2026-10-29 10:00:00+05:30")
        self.assertEqual(crossover(down, 5, ist(10, 5, 3)), "PE")
        early = frame_ending(cross_series(True), "2026-10-29 09:25:00+05:30")
        self.assertIsNone(crossover(early, 5, ist(9, 30, 3)))
        self.assertIsNone(crossover(up, 5, ist(10, 15)))  # stale bar
        self.assertIsNone(crossover(up, 5, ist(10, 4)))   # bar not yet closed

    def test_sizing(self):
        self.assertEqual(lots_for("Fixed Lots", 3, 1e5, 100, 75, 1800), 3)
        self.assertEqual(lots_for("Max Risk %", 10, 1e5, 100, 75, 1800), 1)
        self.assertEqual(lots_for("Max Risk %", 1, 1e5, 100, 75, 1800), 0)
        self.assertEqual(lots_for("Fixed Lots", 50, 1e5, 100, 75, 1800), 24)

    def test_resample_aligns_to_0915(self):
        start = pd.Timestamp("2026-10-29 09:15:00+05:30")
        rows = [[(start + pd.Timedelta(minutes=i)).isoformat(), 1, 2, 0.5, 1 + i, 10, 0]
                for i in range(30)]
        bars = aggregate_candles(rows, 15)
        self.assertEqual([t.strftime("%H:%M") for t in bars.index], ["09:15", "09:30"])
        self.assertEqual(bars["close"].iloc[0], 15)
        self.assertEqual(bars["volume"].iloc[0], 150)

    def test_instrument_master_picks_nearest_listed_expiry(self):
        master = InstrumentMaster(Path("."))
        ms = lambda d: int(datetime(2026, 11, d, 15, 30, tzinfo=IST).timestamp() * 1000)
        record = dict(segment="NSE_FO", instrument_type="CE", underlying_key=SPOT_KEYS["NIFTY"],
                      strike_price=24500.0, lot_size=75, trading_symbol="X")
        master.records = [record | dict(expiry=ms(10), instrument_key="far"),
                          record | dict(expiry=ms(3), instrument_key="near"),
                          record | dict(expiry=ms(3), instrument_key="put", instrument_type="PE")]
        self.assertEqual(master.option("NIFTY", 24510, "CE", ist(10, 0).date())["key"], "near")


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.folder.name) / "t.db")
        self.now = ist(10, 0)
        self.engine = TradingEngine(self.db, clock=lambda: self.now)
        self.engine.broker = FakeBroker()
        self.engine.state.spot, self.engine.state.spot_time = 24510.0, self.now

    def tearDown(self):
        self.engine.shutdown()
        self.folder.cleanup()

    def test_paper_trade_lifecycle_and_daily_limit(self):
        self.engine.enter("CE", 20)
        trade = self.engine.state.trade
        self.assertEqual(trade["strike"], 24500)
        self.assertEqual(trade["entry_premium"], 100.5)
        self.assertEqual(trade["levels"], {"stop": 24480, "target": 24570})
        self.assertTrue(self.engine.state.blocked)
        self.engine.enter("PE", 20)  # second signal same day
        self.assertEqual(self.engine.state.trade["side"], "CE")

        self.engine.state.option_ltp, self.engine.state.option_time = 120.0, self.now
        self.engine._tick(SPOT_KEYS["NIFTY"], 24570.0, self.now)
        self.engine.pool.shutdown(wait=True)
        self.assertIsNone(self.engine.state.trade)
        closed = self.db.trades()[0]
        self.assertEqual(closed["exit_reason"], "TARGET")
        self.assertEqual(closed["exit_premium"], 119.4)
        self.assertEqual(self.engine.state.banner, "Daily Trade Limit Exhausted - Blocked")

        self.engine.enter("PE", 20)
        self.assertEqual(len(self.db.trades()), 1)

    def test_live_requires_arming_and_stale_spot_blocks(self):
        self.engine.set_mode("LIVE", armed=False)
        self.engine.enter("CE", 20)
        self.assertIsNone(self.db.session("2026-10-29", "LIVE"))
        self.engine.set_mode("PAPER")
        self.engine.state.spot_time = self.now - timedelta(seconds=30)
        self.engine.enter("CE", 20)
        self.assertIsNone(self.db.session("2026-10-29", "PAPER"))

    def test_square_off_at_1515(self):
        self.engine.enter("PE", 20)
        self.engine.state.option_ltp, self.engine.state.option_time = 90.0, self.now
        self.now = ist(15, 15, 0)
        self.engine.state.option_time = self.now
        self.engine._tick_clock()
        self.engine.pool.shutdown(wait=True)
        self.assertEqual(self.db.trades()[0]["exit_reason"], "SQUARE_OFF_1515")

    def test_restart_recovers_open_position(self):
        self.engine.enter("CE", 20)
        engine2 = TradingEngine(self.db, clock=lambda: self.now)
        engine2.recover()
        self.assertEqual(engine2.state.trade["symbol"], "NIFTY 24500 CE")
        self.assertTrue(engine2.state.blocked)
        engine2.shutdown()


class BacktestTests(unittest.TestCase):
    def minutes(self, last_day_path):
        days = pd.bdate_range("2026-10-01", "2026-10-23")
        frames, price = [], 24500.0
        for d in days:
            idx = pd.date_range(f"{d.date()} 09:15", f"{d.date()} 15:29", freq="1min", tz=IST)
            if d.date() == days[-1].date():
                close = last_day_path(len(idx), price)
            else:
                close = price - np.arange(len(idx)) * 0.05 + np.sin(np.arange(len(idx)) / 7) * 3
            price = float(close[-1])
            frames.append(pd.DataFrame({"open": close, "high": close + 2, "low": close - 2,
                                        "close": close, "volume": 0.0}, index=idx))
        return pd.concat(frames)

    def loader(self, day, side, strike, skip_today=False):
        idx = pd.date_range(f"{day} 09:15", f"{day} 15:29", freq="1min", tz=IST)
        info = dict(key="k", symbol=f"NIFTY {strike} {side}", lot_size=75, strike=strike,
                    side=side, expiry=str(day), freeze=1800)
        return info, pd.DataFrame({"open": 100.0, "close": 100.0}, index=idx)

    def test_bullish_day_targets(self):
        def path(n, start):
            close = start - np.arange(n) * 0.2
            rally = np.arange(n) >= 60
            close[rally] = close[60] + np.arange(rally.sum()) * 4
            return close
        minutes = self.minutes(path)
        settings = Settings(interval=5)
        trade, note = simulate_day(pd.Timestamp("2026-10-23").date(), minutes, settings, 1e5,
                                   self.loader, FeeSchedule())
        self.assertIsNotNone(trade, note)
        self.assertEqual(trade["side"], "CE")
        self.assertEqual(trade["exit_reason"], "TARGET")
        self.assertGreaterEqual(pd.Timestamp(trade["entry_time"]).time(), dtime(9, 35))
        self.assertEqual(trade["entry_premium"], 100.5)
        self.assertEqual(trade["exit_premium"], 99.5)

    def test_same_minute_sl_and_tp_counts_as_stop(self):
        def path(n, start):
            close = start - np.arange(n) * 0.2
            rally = np.arange(n) >= 60
            close[rally] = close[60] + np.arange(rally.sum()) * 4
            return close
        minutes = self.minutes(path)
        day = minutes.index[-1].date()
        first, _ = simulate_day(day, minutes, Settings(interval=5), 1e5, self.loader, FeeSchedule())
        mask = minutes.index >= pd.Timestamp(first["entry_time"])
        minutes.loc[mask, "high"] += 10_000
        minutes.loc[mask, "low"] -= 10_000
        trade, _ = simulate_day(day, minutes, Settings(interval=5), 1e5, self.loader, FeeSchedule())
        self.assertEqual(trade["exit_reason"], "STOP_LOSS")
        self.assertEqual(trade["exit_time"], first["entry_time"])


if __name__ == "__main__":
    unittest.main()
