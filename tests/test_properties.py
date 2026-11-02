"""Property-based tests (Hypothesis) for the pure trading rules and analytics."""

import math
import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, assume, given, settings, strategies as st

from database import IST, FeeSchedule, metrics
from strategy import (DEFAULT_PARAMS, crossover, exit_reason, indicators, lots_for, risk_levels)
from upstox_broker import aggregate_candles, round_strike

SIGNAL_START, ENTRY_CUTOFF = DEFAULT_PARAMS.start_time, DEFAULT_PARAMS.cutoff_time

FAST = settings(max_examples=400, deadline=None)
SLOW = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
prices = st.floats(0.05, 5000, allow_nan=False)


class FeeProperties(unittest.TestCase):
    @FAST
    @given(prices, prices, st.integers(1, 1800), st.integers(1, 6))
    def test_components_and_formulae(self, buy, sell, qty, orders):
        fee = FeeSchedule().calculate(buy, sell, qty, orders)
        parts = ["brokerage", "exchange", "sebi", "stt", "stamp", "gst"]
        self.assertAlmostEqual(fee["total"], sum(fee[p] for p in parts), places=6)
        self.assertTrue(all(fee[p] >= 0 for p in parts))
        self.assertEqual(fee["brokerage"], 20 * orders)
        self.assertAlmostEqual(fee["stt"], sell * qty * 0.00125, delta=0.005 + 1e-9)
        self.assertAlmostEqual(fee["stamp"], buy * qty * 0.00003, delta=0.005 + 1e-9)
        gst_base = 20 * orders + (buy + sell) * qty * (0.0003503 + 0.000001)
        self.assertAlmostEqual(fee["gst"], 0.18 * gst_base, delta=0.01)

    @FAST
    @given(prices, prices, st.integers(1, 1800), st.floats(0.05, 100))
    def test_monotonic_in_sell_value(self, buy, sell, qty, bump):
        low = FeeSchedule().calculate(buy, sell, qty)["total"]
        high = FeeSchedule().calculate(buy, sell + bump, qty)["total"]
        self.assertGreaterEqual(high, low - 0.02)


class StrikeAndRiskProperties(unittest.TestCase):
    @FAST
    @given(st.floats(1000, 200000), st.sampled_from(["NIFTY", "BANKNIFTY"]))
    def test_strike_is_nearest_step_multiple(self, price, index):
        step = 50 if index == "NIFTY" else 100
        strike = round_strike(price, index)
        self.assertEqual(strike % step, 0)
        self.assertLessEqual(abs(strike - price), step / 2 + 1e-9)

    @FAST
    @given(st.sampled_from(["CE", "PE"]), st.floats(1000, 100000), st.floats(0.01, 2000))
    def test_levels_geometry_and_triggers(self, side, spot, atr):
        levels = risk_levels(side, spot, atr)
        sign = 1 if side == "CE" else -1
        self.assertGreater(sign * (levels["target"] - spot), 0)
        self.assertGreater(sign * (spot - levels["stop"]), 0)
        self.assertAlmostEqual(abs(levels["target"] - spot), 2 * abs(spot - levels["stop"]),
                               delta=1e-6 * spot)
        self.assertIsNone(exit_reason(side, spot, levels))
        self.assertEqual(exit_reason(side, levels["stop"], levels), "STOP_LOSS")
        self.assertEqual(exit_reason(side, levels["target"], levels), "TARGET")
        self.assertEqual(exit_reason(side, levels["stop"] - sign * atr, levels), "STOP_LOSS")
        self.assertEqual(exit_reason(side, levels["target"] + sign * atr, levels), "TARGET")

    @FAST
    @given(st.sampled_from([0.0, -1.0, math.nan, math.inf]))
    def test_invalid_atr_rejected(self, atr):
        with self.assertRaises(ValueError):
            risk_levels("CE", 24500, atr)


class SizingProperties(unittest.TestCase):
    @FAST
    @given(st.floats(1000, 1e8), st.floats(0.1, 100), st.floats(0.05, 2000),
           st.sampled_from([15, 25, 35, 50, 75]), st.integers(0, 30))
    def test_risk_percent_never_exceeds_budget_or_freeze(self, capital, pct, premium, lot, freeze_lots):
        freeze = freeze_lots * lot
        lots = lots_for("Max Risk %", pct, capital, premium, lot, freeze)
        self.assertGreaterEqual(lots, 0)
        self.assertLessEqual(lots * premium * lot, capital * pct / 100 + 1e-6)
        if freeze:
            self.assertLessEqual(lots * lot, freeze)
            uncapped = math.floor(capital * pct / 100 / (premium * lot))
            self.assertEqual(lots, min(uncapped, freeze_lots))

    @FAST
    @given(st.integers(1, 100), st.sampled_from([25, 75]), st.integers(1, 40))
    def test_fixed_lots_capped_by_freeze(self, lots, lot, freeze_lots):
        self.assertEqual(lots_for("Fixed Lots", lots, 1e5, 100, lot, freeze_lots * lot),
                         min(lots, freeze_lots))


def one_day(seed):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-10-29 09:15", "2026-10-29 15:29", freq="1min", tz=IST)
    close = 24500 + np.cumsum(rng.normal(0, 6, len(idx)))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + rng.uniform(0, 4, len(idx))
    low = np.minimum(open_, close) - rng.uniform(0, 4, len(idx))
    vol = rng.integers(0, 1000, len(idx)).astype(float)
    rows = [[t.isoformat(), o, h, l, c, v, 0] for t, o, h, l, c, v in zip(idx, open_, high, low, close, vol)]
    return rows, vol.sum()


class ResampleProperties(unittest.TestCase):
    @SLOW
    @given(st.integers(0, 10_000), st.sampled_from([1, 5, 15, 30]), st.integers(0, 40))
    def test_bars_are_consistent_aligned_and_lossless(self, seed, interval, drop):
        rows, volume = one_day(seed)
        rng = np.random.default_rng(seed + 1)
        gaps = set(rng.choice(len(rows), size=drop, replace=False).tolist())
        kept = [r for i, r in enumerate(rows) if i not in gaps]
        bars = aggregate_candles(kept + kept[:5], interval)  # duplicates must be ignored
        self.assertTrue((bars["high"] >= bars[["open", "close"]].max(axis=1) - 1e-9).all())
        self.assertTrue((bars["low"] <= bars[["open", "close"]].min(axis=1) + 1e-9).all())
        offsets = (bars.index - bars.index.normalize() - pd.Timedelta("9h15min")) / pd.Timedelta(minutes=1)
        self.assertTrue((offsets % interval == 0).all())
        self.assertAlmostEqual(bars["volume"].sum(), sum(r[5] for r in kept), places=3)
        self.assertTrue(bars.index.is_monotonic_increasing)


class CrossoverProperties(unittest.TestCase):
    @SLOW
    @given(st.integers(0, 10_000), st.integers(0, 74), st.sampled_from([1, 5]),
           st.integers(0, 20))
    def test_signal_only_on_fresh_sign_change_inside_window(self, seed, bar_no, interval, delay):
        rng = np.random.default_rng(seed)
        last = pd.Timestamp("2026-10-29 09:15", tz=IST) + pd.Timedelta(minutes=interval * bar_no)
        assume(last.time() < datetime(2026, 1, 1, 15, 30).time())
        idx = pd.date_range(end=last, periods=160, freq=f"{interval}min")
        close = pd.Series(24500 + np.cumsum(rng.normal(0, 15, 160)), index=idx)
        frame = indicators(pd.DataFrame({"open": close, "high": close + 5, "low": close - 5,
                                         "close": close, "volume": 0.0}))
        now = (last + pd.Timedelta(minutes=interval, seconds=delay)).to_pydatetime()
        signal = crossover(frame, interval, now)
        prev = frame["ema_fast"].iloc[-2] - frame["ema_slow"].iloc[-2]
        cur = frame["ema_fast"].iloc[-1] - frame["ema_slow"].iloc[-1]
        if signal == "CE":
            self.assertTrue(prev <= 0 < cur)
        if signal == "PE":
            self.assertTrue(prev >= 0 > cur)
        if signal:
            self.assertGreaterEqual(last.time(), SIGNAL_START)
            self.assertLess(now.time(), ENTRY_CUTOFF)
        in_window = SIGNAL_START <= last.time() and now.time() < ENTRY_CUTOFF and last.date() == now.date()
        if in_window and (prev <= 0 < cur or prev >= 0 > cur):
            self.assertIsNotNone(signal)


class MetricsProperties(unittest.TestCase):
    @FAST
    @given(st.lists(st.floats(-20000, 20000), min_size=0, max_size=40), st.floats(10000, 1e7))
    def test_equity_drawdown_and_ratios(self, nets, capital):
        start = datetime(2026, 9, 1, 10, tzinfo=IST)
        trades = [dict(status="CLOSED", day=(start + timedelta(days=i)).date().isoformat(), net=n,
                       taxes=1.0, slippage=0.5, entry_time=(start + timedelta(days=i)).isoformat(),
                       exit_time=(start + timedelta(days=i, minutes=30)).isoformat())
                  for i, n in enumerate(nets)]
        result = metrics(trades, capital)
        self.assertAlmostEqual(result["equity"], capital + sum(nets), delta=1e-6 * capital)
        self.assertGreaterEqual(result["max_dd"], 0)
        self.assertEqual(len(result["curve"]), len(nets) + 1)
        wins, losses = [n for n in nets if n > 0], [-n for n in nets if n < 0]
        if losses:
            self.assertAlmostEqual(result["profit_factor"], sum(wins) / sum(losses), places=6)
        if nets:
            self.assertAlmostEqual(result["avg_duration"], 30.0)
        peak, worst, equity = capital, 0.0, capital
        for n in nets:
            equity += n
            peak = max(peak, equity)
            worst = max(worst, (peak - equity) / peak * 100 if peak > 0 else 0)
        self.assertAlmostEqual(result["max_dd"], worst, places=6)


if __name__ == "__main__":
    unittest.main()
