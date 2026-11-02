"""Software-in-the-loop harness: the real TradingEngine driven by a simulated market and broker."""

from __future__ import annotations

import math
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from database import IST, Database
from strategy import Settings, TradingEngine, run_backtest
from upstox_broker import HISTORY_BARS, SPOT_KEYS, BrokerError, resample_bars, round_strike

LOT = {"NIFTY": 65, "BANKNIFTY": 30, "SENSEX": 20}
START_PRICE = {"NIFTY": 24500.0, "BANKNIFTY": 51500.0, "SENSEX": 80500.0}
BT_START, BT_END = date(2026, 10, 19), date(2026, 10, 23)
# Seconds within each minute at which the clock is advanced; ticks are sent at 0/15/30/45.
SECONDS = (0, 3, 15, 30, 45)


def make_minutes(seed: int, index: str = "NIFTY", first: str = "2026-09-21",
                 last: str = "2026-10-23", vol: float = 0.0004) -> pd.DataFrame:
    """Seeded 1-minute OHLC (~1% daily range) with drifting regimes so EMA crossovers occur."""
    rng = np.random.default_rng(seed)
    price, frames = START_PRICE[index], []
    for day in pd.bdate_range(first, last):
        idx = pd.date_range(f"{day.date()} 09:15", f"{day.date()} 15:29", freq="1min", tz=IST)
        n = len(idx)
        drift = np.repeat(rng.normal(0, vol * 0.25, n // 45 + 1), 45)[:n]
        close = price * np.exp(np.cumsum(drift + rng.normal(0, vol, n)))
        open_ = np.r_[price * (1 + rng.normal(0, vol / 3)), close[:-1]]
        wick = np.abs(rng.normal(0, vol * price * 0.6, (2, n)))
        frames.append(pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + wick[0],
                                    "low": np.minimum(open_, close) - wick[1], "close": close,
                                    "volume": 0.0}, index=idx))
        price = float(close[-1])
    return pd.concat(frames)


def premium(spot: float, strike: int, side: str) -> float:
    """Deterministic option price: intrinsic + bell-shaped time value, on a 0.05 tick."""
    intrinsic = max(spot - strike, 0) if side == "CE" else max(strike - spot, 0)
    value = intrinsic + 0.0045 * spot * math.exp(-((spot - strike) / (0.006 * spot)) ** 2 / 2) + 2
    return round(value * 20) / 20


def sim_expiry(day, skip_today: bool = False):
    """Weekly Tuesday expiries (NSE-style) for the simulated market."""
    ahead = (1 - day.weekday()) % 7
    expiry = day + timedelta(days=ahead)
    if skip_today and expiry == day:
        expiry += timedelta(days=7)
    return expiry


def contract(index: str, side: str, strike: int, day, skip_today: bool = False) -> dict:
    expiry = sim_expiry(day, skip_today) if day is not None else None
    return dict(key=f"SIM|{index}|{strike}|{side}|{expiry}", symbol=f"{index} {strike} {side} {expiry}",
                lot_size=LOT[index], strike=strike, side=side, expiry=str(expiry),
                freeze=LOT[index] * 24, exchange="BSE" if index == "SENSEX" else "NSE")


def option_frame(minutes: pd.DataFrame, day, side: str, strike: int) -> pd.DataFrame:
    rows = minutes[minutes.index.date == day]
    return pd.DataFrame({"open": [premium(p, strike, side) for p in rows["open"]],
                         "close": [premium(p, strike, side) for p in rows["close"]]},
                        index=rows.index)


def history_window(minutes: pd.DataFrame, start=BT_START, end=BT_END) -> pd.DataFrame:
    """Same lookback slice that run_backtest downloads, so both sides see identical data."""
    fetch_from = max(start - timedelta(days=20), end - timedelta(days=29))
    dates = minutes.index.date
    return minutes[(dates >= fetch_from) & (dates <= end)]


class SyncPool:
    """Runs engine work inline for deterministic SIL stepping."""

    def submit(self, fn, *args, **kwargs):
        fn(*args, **kwargs)

    def shutdown(self, **_kwargs):
        pass


class SimMaster:
    def __init__(self, broker) -> None:
        self.broker, self.loaded_day, self.records = broker, None, []

    def refresh(self, force: bool = False) -> None:
        self.loaded_day = self.broker.now.date()

    def nearest_expiry(self, index, today=None, skip_today=False):
        return sim_expiry(today or self.broker.now.date(), skip_today)

    def option(self, index, spot, side, today=None, skip_today=False) -> dict:
        info = contract(index, side, round_strike(spot, index), today or self.broker.now.date(), skip_today)
        return dict(info, freeze=LOT[index] * self.broker.freeze_lots)


class SimBroker:
    """Implements the UpstoxBroker surface used by TradingEngine, with fault injection."""

    def __init__(self, minutes: pd.DataFrame, index: str, latency: float = 0.0) -> None:
        self.minutes, self.index, self.latency = minutes, index, latency
        self.master, self.channels = SimMaster(self), {}
        self.now, self.row = minutes.index[0], None
        self.last_spot = None
        self.outage = False
        self.bars: dict[int, pd.DataFrame] = {}
        self.orders: dict[str, dict] = {}
        self.position: dict[str, int] = defaultdict(int)
        self.order_faults: list[str | None] = []
        self.calls = Counter()
        self.listener = None
        self.stuck_cancel = False
        self.tag_day = None
        self.spread_pct = 1.0
        self.freeze_lots = 24
        self.holiday_calendar: dict = {}

    def holidays(self):
        return self.holiday_calendar

    def depth(self, key):
        price, _ = self.quote(key)
        half = price * self.spread_pct / 200
        return dict(bid=round(price - half, 2), ask=round(price + half, 2), ltp=price)

    # --- market data
    def candles(self, index, interval):
        self.calls["candles"] += 1
        if interval not in self.bars:
            self.bars[interval] = resample_bars(self.minutes, interval)
        bars = self.bars[interval]
        done = bars[bars.index + pd.Timedelta(minutes=interval) <= self.now]
        return done.tail(HISTORY_BARS)

    def _intra_minute(self) -> float:
        return self.row[0] if self.now.second < 10 else self.row[3]

    def quote(self, key):
        if key == SPOT_KEYS[self.index]:
            self.calls["spot_quote"] += 1
            price = self._intra_minute() if self.outage else self.last_spot
            return float(price), self.now.to_pydatetime()
        _, _, strike, side, _expiry = key.split("|")
        self.calls["option_quote"] += 1
        return premium(self._intra_minute(), int(strike), side), self.now.to_pydatetime()

    # --- orders
    def market_order(self, key, quantity, side, tag):
        if self.latency:
            time.sleep(self.latency)
        self.calls["order_" + side] += 1
        fault = self.order_faults.pop(0) if self.order_faults else None
        if fault == "raise":
            raise BrokerError("simulated network failure on place_order")
        order_id = f"SIM{len(self.orders) + 1}"
        price, _ = self.quote(key)
        lot = LOT[self.index]
        status = dict(order_id=order_id, status="complete", average_price=price,
                      filled_quantity=quantity, status_message="", side=side, key=key,
                      transaction_type=side, instrument_token=key, tag=tag, order_type="MARKET")
        if fault == "reject":
            status.update(status="rejected", average_price=0, filled_quantity=0,
                          status_message="RMS: simulated rejection")
        elif fault == "partial":
            status.update(status="cancelled", filled_quantity=max(quantity // 2 // lot, 1) * lot)
        elif fault == "pending":
            status.update(status="open", average_price=0, filled_quantity=0)
        self.orders[order_id] = status
        self.position[key] += status["filled_quantity"] * (1 if side == "BUY" else -1)
        return order_id

    def order_status(self, order_id):
        return dict(self.orders[order_id])

    def cancel(self, order_id):
        self.calls["cancel"] += 1
        if self.stuck_cancel:
            return
        if self.orders[order_id]["status"] in ("open", "trigger pending"):
            self.orders[order_id]["status"] = "cancelled"

    def stop_loss_order(self, key, quantity, trigger, limit, tag):
        self.calls["order_SL"] += 1
        order_id = f"SIM{len(self.orders) + 1}"
        self.orders[order_id] = dict(order_id=order_id, status="trigger pending", average_price=0,
                                     filled_quantity=0, status_message="", side="SELL", key=key,
                                     transaction_type="SELL", instrument_token=key, tag=tag,
                                     order_type="SL", quantity=quantity, trigger=trigger, limit=limit)
        return order_id

    def trigger_stop(self, order_id, price=None):
        """Exchange fires the resting SL: fill, move the position, push an order update."""
        order = self.orders[order_id]
        order.update(status="complete", filled_quantity=order["quantity"],
                     average_price=price if price is not None else order["trigger"])
        self.position[order["key"]] -= order["quantity"]
        if self.listener:
            self.listener(dict(order, update_type="order"))

    def subscribe_orders(self, listener, _on_status):
        self.listener = listener

    def order_book(self):
        return [dict(o) for o in self.orders.values()]

    def positions(self):
        if self.latency:
            time.sleep(self.latency)
        return [dict(instrument_token=k, quantity=q) for k, q in self.position.items()]

    def stream(self, *_args):
        pass

    def close(self):
        pass


def new_engine(db: Database, broker: SimBroker, settings: Settings, mode: str = "PAPER",
               sync: bool = True) -> TradingEngine:
    engine = TradingEngine(db, clock=lambda: broker.now.to_pydatetime())
    if sync:
        engine.pool.shutdown(wait=False)
        engine.pool = SyncPool()
    engine.settings = replace(settings)
    engine.broker = broker
    broker.subscribe_orders(engine._on_order_update, engine._feed)
    if mode == "LIVE":
        engine.set_mode("LIVE", armed=True)
    return engine


def drive(engine: TradingEngine, broker: SimBroker, start=BT_START, end=BT_END,
          stop=None, begin_after=None, on_minute=None):
    """Step the clock minute by minute; returns the last minute processed."""
    key = SPOT_KEYS[broker.index]
    data = broker.minutes
    dates = data.index.date
    sel = data[(dates >= start) & (dates <= end)]
    if begin_after is not None:
        sel = sel[sel.index > begin_after]
    last = None
    for minute, o, h, l, c in zip(sel.index, sel["open"], sel["high"], sel["low"], sel["close"]):
        broker.row = (o, h, l, c)
        if on_minute:
            on_minute(minute)
        pair = (l, h)
        for second in SECONDS:
            broker.now = minute + pd.Timedelta(seconds=second)
            engine._tick_clock()
            if second == 3:
                continue
            if second == 15:
                trade = engine.state.trade
                adverse_first = (trade["side"] == "CE") if trade else (c >= o)
                pair = (l, h) if adverse_first else (h, l)
            price = {0: o, 15: pair[0], 30: pair[1], 45: c}[second]
            if broker.outage:
                continue
            broker.last_spot = price
            engine._tick(key, float(price), broker.now.to_pydatetime())
        last = minute
        if stop and stop():
            break
    return last


def temp_db(name: str) -> Database:
    folder = Path(tempfile.mkdtemp(prefix="sil_"))
    return Database(folder / name)


class RefBroker:
    """Adapter so run_backtest reads the same synthetic data."""

    def __init__(self, minutes, index):
        self.minutes, self.index = minutes, index

    def spot_minutes(self, index, start, end):
        dates = self.minutes.index.date
        return self.minutes[(dates >= start) & (dates <= end)]

    def backtest_expiry(self, index, day, skip_today=False):
        return sim_expiry(day, skip_today)

    def backtest_option(self, index, day, side, strike, skip_today=False):
        return (contract(index, side, strike, day, skip_today),
                option_frame(self.minutes, day, side, strike))


def run_pair(seed: int, index: str, interval: int, history_first: str = "2026-09-21",
             **settings_kwargs):
    """Run the SIL engine and the backtest on identical data; return (engine, backtest) trades."""
    minutes = make_minutes(seed, index, first=history_first)
    settings = Settings(index=index, interval=interval, **settings_kwargs)
    window = history_window(minutes)
    broker = SimBroker(window, index)
    engine_db = temp_db("engine.db")
    engine = new_engine(engine_db, broker, settings)
    drive(engine, broker)
    engine.shutdown()
    bt_db = temp_db("backtest.db")
    run_backtest(RefBroker(minutes, index), bt_db, settings, BT_START, BT_END)
    closed = lambda db: [t for t in db.trades() if t["status"] == "CLOSED"]
    return closed(engine_db), closed(bt_db), engine_db


def signature(trade: dict) -> tuple:
    minute = lambda text: pd.Timestamp(text).floor("min").isoformat()
    legs = tuple((leg["qty"], leg["price"], minute(leg["time"])) for leg in trade.get("legs", []))
    return (trade["day"], trade["side"], trade["strike"], trade["quantity"], trade["expiry"],
            minute(trade["entry_time"]), round(trade["entry_spot"], 4),
            round(trade.get("initial_stop", trade["levels"]["stop"]), 4), round(trade["levels"]["stop"], 4),
            round(trade["levels"]["target"], 4), trade["entry_premium"], minute(trade["exit_time"]),
            trade["exit_reason"], round(trade["exit_premium"], 4), legs, trade.get("mfe_pts"),
            trade.get("mae_pts"), round(trade["taxes"], 2), round(trade["net"], 2))
