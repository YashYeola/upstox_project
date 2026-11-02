"""Back-to-back: live TradingEngine (SIL) vs backtest simulator on identical market data."""

import os
import unittest
from collections import Counter

from sil_harness import run_pair, signature

SEEDS = range(int(os.environ.get("B2B_SEEDS", "4")))
MATRIX = [(seed, index, interval) for seed in SEEDS
          for index in ("NIFTY", "BANKNIFTY") for interval in (1, 5, 15, 30)]


class BackToBack(unittest.TestCase):
    def test_engine_matches_backtest_trade_for_trade(self):
        coverage = Counter()
        for seed, index, interval in MATRIX:
            with self.subTest(seed=seed, index=index, interval=interval):
                engine, backtest, db = run_pair(seed, index, interval)
                self.assertEqual([signature(t) for t in engine], [signature(t) for t in backtest])
                errors = [e["message"] for e in db.recent_events() if e["level"] in ("ERROR", "CRITICAL")]
                self.assertEqual(errors, [])
                for trade in engine:
                    coverage[trade["side"]] += 1
                    coverage[trade["exit_reason"]] += 1
                coverage["trades"] += len(engine)
        print(f"\nB2B coverage over {len(MATRIX)} runs: {dict(coverage)}")
        self.assertGreaterEqual(coverage["trades"], len(MATRIX) * 3)
        for key in ("CE", "PE", "TARGET", "STOP_LOSS", "SQUARE_OFF_1515"):
            self.assertGreater(coverage[key], 0, f"matrix never exercised {key}")

    def test_sizing_by_risk_percent_matches(self):
        engine, backtest, _ = run_pair(7, "NIFTY", 5, sizing="Max Risk %", sizing_value=40,
                                       capital=250000)
        self.assertTrue(engine)
        self.assertEqual([signature(t) for t in engine], [signature(t) for t in backtest])
        self.assertTrue(any(t["quantity"] > 75 for t in engine))

    def test_short_history_padding_rule_matches(self):
        # 30-min bars with only ~9 prior sessions: the 121-bar warm-up boundary falls inside the week.
        engine, backtest, _ = run_pair(3, "NIFTY", 30, history_first="2026-10-06")
        self.assertEqual([signature(t) for t in engine], [signature(t) for t in backtest])


if __name__ == "__main__":
    unittest.main()
