"""Headless-ish smoke test: build the GUI with synthetic data, render charts, export dossier."""

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from database import IST, Database, FeeSchedule, build_pdf, export_csv, metrics
from gui import App
from strategy import TradingEngine, indicators, risk_levels

folder = Path(tempfile.mkdtemp())
db = Database(folder / "t.db")
for day, exit_price in [("2026-10-26", 130), ("2026-10-27", 85), ("2026-10-28", 118)]:
    trade = dict(day=day, mode="PAPER", side="CE", symbol=f"NIFTY 24500 CE {day}", key="k",
                 quantity=75, entry_premium=100.5, slippage=75,
                 entry_time=datetime.fromisoformat(day + "T10:05:00+05:30").isoformat())
    db.reserve(day, "PAPER")
    db.open_trade(trade)
    db.close_trade(trade, exit_price, "TARGET", FeeSchedule(),
                   datetime.fromisoformat(day + "T11:20:00+05:30"))

engine = TradingEngine(db)
index = pd.date_range(end=pd.Timestamp("2026-10-29 10:00+05:30"), periods=150, freq="5min")
close = pd.Series(24500 + np.cumsum(np.random.default_rng(1).normal(0, 8, 150)), index=index)
engine.state.frame = indicators(pd.DataFrame({"open": close.shift(1).fillna(close), "high": close + 6,
                                              "low": close - 6, "close": close, "volume": 0}))
engine.state.spot = float(close.iloc[-1])
engine.state.trade = dict(side="CE", symbol="NIFTY 24500 CE", key="k", quantity=75,
                          entry_premium=101.2, entry_spot=engine.state.spot - 10, atr=20,
                          levels=risk_levels("CE", engine.state.spot - 10, 20), expiry="2026-11-03")
engine.state.option_ltp = 110.0

app = App(db, engine)
app.update()
app.update_dashboard(engine.snapshot())
app.draw_charts(engine.snapshot())
app.refresh_reports()
app.outcome.set("Loss Only")
app.refresh_reports()
app.update()
summary = metrics(db.trades())
build_pdf(folder / "dossier.pdf", db.trades(), summary)
export_csv(folder / "dossier.csv", db.trades())

from strategy import DEFAULT_PARAMS  # noqa: E402
from unittest import mock  # noqa: E402

app.param_entries["ema_fast"].delete(0, "end")
app.param_entries["ema_fast"].insert(0, "40")
with mock.patch("gui.messagebox.showerror") as shown:
    app.apply_settings()
assert shown.called and engine.settings.params.ema_fast == 9, "invalid params must be rejected"
app.reset_params()
app.trend_menu.set("EMA+ADX")
app.apply_settings(quiet=True)
assert engine.settings.params.trend_filter == "EMA+ADX"
app.alert("TRADE", "ENTRY PAPER BUY 75 NIFTY 24500 CE @ 101.20")
app.alert("TRADE", "ENTRY PAPER BUY 75 NIFTY 24500 CE @ 101.20")
app.update()
from gui import Toast  # noqa: E402
assert len(Toast.stack) == 1, "duplicate alert must be throttled"
app.open_sweep()
app.update()
app._sweep_done([dict(params=DEFAULT_PARAMS, trades=3, errors=0, net=1500.0, sharpe=1.2,
                      profit_factor=2.0, max_dd=1.0, win_loss=2.0)], None)
app.update()
assert app.apply_best_btn.cget("state") == "normal"
print("new-feature widgets OK")
print("cards:", {k: c.value.cget("text") for k, c in app.cards.items()})
print("banner:", app.banner.cget("text"))
print("metrics:", {k: v.cget("text") for k, v in app.metric_labels.items()})
print("pdf bytes:", (folder / "dossier.pdf").stat().st_size, "csv rows:",
      len((folder / "dossier.csv").read_text(encoding="utf-8-sig").splitlines()) - 1)
engine.shutdown()
app.destroy()
print("SMOKE OK")
