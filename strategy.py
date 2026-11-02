"""Spot-driven EMA crossover engine with ATR spot exits and one trade per day per mode."""

from __future__ import annotations

import itertools
import logging
import math
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time as dtime, timedelta

import pandas as pd
import pandas_ta as ta
import numpy as np

from database import Database, FeeSchedule, IST, metrics, now_ist
from upstox_broker import (EXCHANGE, HISTORY_BARS, SPOT_KEYS, TERMINAL, BrokerError, resample_bars)

SESSION_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)
INTERVALS = {"1 min": 1, "5 min": 5, "15 min": 15, "30 min": 30}
TREND_FILTERS = ("Off", "EMA", "ADX", "EMA+ADX")
BAR_SOURCES = ("Ticks", "REST")
EXPIRY_MODES = ("Normal", "Cutoff", "Next expiry")
TICK_GAP_SECONDS = 30
SWEEP_LIMIT = 300
# ASSUMPTION - verify on bseindia.com: BSE index-option transaction charge per premium turnover.
BSE_EXCHANGE_RATE = "0.000325"
LOG = logging.getLogger("strat")
LOG.addHandler(logging.NullHandler())


def _hhmm(text: str) -> dtime:
    return datetime.strptime(text, "%H:%M").time()


@dataclass(frozen=True)
class Params:
    ema_fast: int = 9
    ema_slow: int = 21
    atr_length: int = 14
    sl_atr: float = 1.5
    tp_atr: float = 3.0
    signal_start: str = "09:30"
    entry_cutoff: str = "15:00"
    square_off: str = "15:15"
    slippage_pct: float = 0.5
    trend_filter: str = "Off"
    trend_ema: int = 50
    adx_length: int = 14
    adx_min: float = 20.0
    bar_source: str = "Ticks"
    backstop: bool = True
    backstop_pct: float = 40.0
    breakeven_atr: float = 0.0
    trail_start_atr: float = 0.0
    trail_atr: float = 0.0
    partial_atr: float = 0.0
    partial_pct: float = 50.0
    min_premium: float = 0.0
    max_spread_pct: float = 0.0
    expiry_mode: str = "Normal"
    expiry_cutoff: str = "13:00"
    max_week_loss_pct: float = 0.0
    max_month_loss_pct: float = 0.0
    max_consec_losses: int = 0
    event_days: str = ""

    def validate(self) -> "Params":
        times = [_hhmm(t) for t in (self.signal_start, self.entry_cutoff, self.square_off)]
        try:
            _hhmm(self.expiry_cutoff)
            events = self.events
        except ValueError:
            raise ValueError("Expiry cutoff must be HH:MM; event days YYYY-MM-DD, comma separated.")
        checks = [
            (2 <= self.ema_fast < self.ema_slow <= 200, "Need 2 <= EMA fast < EMA slow <= 200."),
            (2 <= self.atr_length <= 100, "ATR length must be 2-100."),
            (0 < self.sl_atr <= 20 and 0 < self.tp_atr <= 40, "SL/TP ATR multiples must be positive."),
            (SESSION_OPEN <= times[0] < times[1] <= times[2] < MARKET_CLOSE,
             "Need 09:15 <= signal start < entry cutoff <= square-off < 15:30."),
            (0 <= self.slippage_pct <= 5, "Slippage must be 0-5 %."),
            (self.trend_filter in TREND_FILTERS, "Unknown trend filter."),
            ("EMA" not in self.trend_filter or self.ema_slow < self.trend_ema <= 400,
             "Trend EMA must be longer than EMA slow (max 400)."),
            (2 <= self.adx_length <= 100 and 0 <= self.adx_min <= 100, "ADX length 2-100, min 0-100."),
            (self.bar_source in BAR_SOURCES, "Unknown bar source."),
            (5 <= self.backstop_pct <= 90, "Backstop must be 5-90 % below the entry premium."),
            (all(0 <= v <= 40 for v in (self.breakeven_atr, self.trail_start_atr, self.trail_atr,
                                       self.partial_atr)), "Breakeven/trail/partial ATR values must be 0-40."),
            (self.trail_atr == 0 or self.trail_atr > 0 and self.trail_start_atr >= 0,
             "Trail distance needs a start level."),
            (self.partial_atr == 0 or self.partial_atr < self.tp_atr,
             "Partial exit level must be below the target."),
            (10 <= self.partial_pct <= 90, "Partial exit size must be 10-90 %."),
            (self.min_premium >= 0 and 0 <= self.max_spread_pct <= 50, "Premium/spread limits invalid."),
            (self.expiry_mode in EXPIRY_MODES, "Unknown expiry mode."),
            (all(0 <= v <= 100 for v in (self.max_week_loss_pct, self.max_month_loss_pct)),
             "Loss limits must be 0-100 %."),
            (0 <= self.max_consec_losses <= 50, "Consecutive-loss limit must be 0-50."),
        ]
        for ok, message in checks:
            if not ok:
                raise ValueError(message)
        return self

    @property
    def events(self) -> frozenset:
        return frozenset(datetime.strptime(d.strip(), "%Y-%m-%d").date()
                         for d in self.event_days.split(",") if d.strip())

    @property
    def expiry_time(self) -> dtime:
        return _hhmm(self.expiry_cutoff)

    @property
    def slippage(self) -> float:
        return self.slippage_pct / 100

    @property
    def start_time(self) -> dtime:
        return _hhmm(self.signal_start)

    @property
    def cutoff_time(self) -> dtime:
        return _hhmm(self.entry_cutoff)

    @property
    def square_time(self) -> dtime:
        return _hhmm(self.square_off)

    @property
    def warmup(self) -> int:
        """Longest indicator lookback + 100 padding bars."""
        longest = max(self.ema_slow, self.atr_length)
        if "EMA" in self.trend_filter:
            longest = max(longest, self.trend_ema)
        if "ADX" in self.trend_filter:
            longest = max(longest, 2 * self.adx_length)
        return longest + 100


DEFAULT_PARAMS = Params()


def _or_nan(series, index) -> pd.Series:
    return pd.Series(math.nan, index=index) if series is None else series


def indicators(frame: pd.DataFrame, p: Params = DEFAULT_PARAMS) -> pd.DataFrame:
    frame = frame.copy()
    close = frame["close"]
    frame["ema_fast"] = _or_nan(ta.ema(close, length=p.ema_fast), frame.index)
    frame["ema_slow"] = _or_nan(ta.ema(close, length=p.ema_slow), frame.index)
    frame["atr"] = _or_nan(ta.atr(frame["high"], frame["low"], close, length=p.atr_length), frame.index)
    if "EMA" in p.trend_filter:
        frame["trend_ema"] = _or_nan(ta.ema(close, length=p.trend_ema), frame.index)
    if "ADX" in p.trend_filter:
        adx = ta.adx(frame["high"], frame["low"], close, length=p.adx_length)
        frame["adx"] = adx[f"ADX_{p.adx_length}"] if adx is not None else math.nan
    return frame


def trend_ok(bar: pd.Series, side: str, p: Params) -> bool:
    if "EMA" in p.trend_filter:
        trend = bar.get("trend_ema", math.nan)
        if not math.isfinite(trend):
            return False
        if (side == "CE" and not bar["close"] > trend) or (side == "PE" and not bar["close"] < trend):
            return False
    if "ADX" in p.trend_filter:
        adx = bar.get("adx", math.nan)
        if not (math.isfinite(adx) and adx >= p.adx_min):
            return False
    return True


def crossover(frame: pd.DataFrame, interval: int, now: datetime,
              p: Params = DEFAULT_PARAMS) -> str | None:
    """CE/PE only for a fresh, filtered crossover on the latest settled bar inside the window."""
    if (len(frame) < max(2, p.warmup)
            or frame[["ema_fast", "ema_slow", "atr"]].iloc[-2:].isna().any().any()):
        return None
    bar_start = frame.index[-1]
    bar_close = bar_start + pd.Timedelta(minutes=interval)
    if (bar_start.date() != now.date() or bar_start.time() < p.start_time
            or not bar_close <= now < bar_close + pd.Timedelta(minutes=interval)
            or now.time() >= p.cutoff_time):
        return None
    previous = frame["ema_fast"].iloc[-2] - frame["ema_slow"].iloc[-2]
    current = frame["ema_fast"].iloc[-1] - frame["ema_slow"].iloc[-1]
    side = "CE" if previous <= 0 < current else "PE" if previous >= 0 > current else None
    if side and not trend_ok(frame.iloc[-1], side, p):
        return None
    return side


def risk_levels(side: str, spot: float, atr: float, p: Params = DEFAULT_PARAMS) -> dict[str, float]:
    if not (math.isfinite(atr) and atr > 0):
        raise ValueError("ATR must be positive and finite.")
    direction = 1 if side == "CE" else -1
    return {"stop": spot - direction * p.sl_atr * atr, "target": spot + direction * p.tp_atr * atr}


def exit_reason(side: str, spot: float, levels: dict[str, float]) -> str | None:
    if side == "CE":
        return "STOP_LOSS" if spot <= levels["stop"] else "TARGET" if spot >= levels["target"] else None
    return "STOP_LOSS" if spot >= levels["stop"] else "TARGET" if spot <= levels["target"] else None


class ExitTracker:
    """Spot exit state machine shared by the live engine and the backtest (same price path -> same result).

    step(price) checks stop then target then partial against the current levels, then ratchets the stop
    (breakeven / trailing) for subsequent prices, and records MFE/MAE in index points.
    """

    def __init__(self, trade: dict, p: Params) -> None:
        self.trade, self.p = trade, p
        self.sign = 1 if trade["side"] == "CE" else -1
        entry = trade["entry_spot"]
        self.state = trade.setdefault("track", dict(best=entry, worst=entry, kind="STOP_LOSS",
                                                    partial_done=False, mfe=0.0, mae=0.0))
        trade.setdefault("initial_stop", trade["levels"]["stop"])

    def step(self, price: float) -> tuple[list[str], bool]:
        """Return (events, stop_moved). Events: PARTIAL and/or one exit reason."""
        s, t, sign = self.state, self.trade, self.sign
        entry, atr, levels = t["entry_spot"], t["atr"], t["levels"]
        if sign * (price - s["best"]) > 0:
            s["best"] = price
        if sign * (s["worst"] - price) > 0:
            s["worst"] = price
        s["mfe"] = max(s["mfe"], sign * (price - entry))
        s["mae"] = max(s["mae"], sign * (entry - price))
        if sign * (price - levels["stop"]) <= 0:
            return [s["kind"]], False
        if sign * (price - levels["target"]) >= 0:
            return ["TARGET"], False
        events = []
        move = sign * (s["best"] - entry)
        p = self.p
        if p.partial_atr and not s["partial_done"] and sign * (price - entry) >= p.partial_atr * atr:
            s["partial_done"] = True
            events.append("PARTIAL")
        stop, kind = levels["stop"], s["kind"]
        if p.breakeven_atr and move >= p.breakeven_atr * atr and sign * (entry - stop) > 0:
            stop, kind = entry, "BREAKEVEN_STOP"
        if p.trail_atr and move >= p.trail_start_atr * atr:
            trail = s["best"] - sign * p.trail_atr * atr
            if sign * (trail - stop) > 0:
                stop, kind = trail, "TRAIL_STOP"
        moved = stop != levels["stop"]
        if moved:
            levels["stop"], s["kind"] = stop, kind
        return events, moved


def partial_quantity(quantity: int, lot: int, pct: float) -> int:
    part = math.floor(quantity * pct / 100 / lot) * lot
    return part if lot <= part < quantity else 0


MAX_SLICES = 10


def params_of(trade: dict, fallback: Params) -> Params:
    """Params a trade was opened with (exit rules must not change mid-trade)."""
    saved = trade.get("params")
    if not saved:
        return fallback
    known = {f for f in Params.__dataclass_fields__}
    return replace(fallback, **{k: v for k, v in saved.items() if k in known})


def backstop_ids(trade: dict) -> list[str]:
    ids = list(trade.get("backstops") or [])
    if trade.get("backstop"):
        ids.append(trade["backstop"])
    return ids


def blended_exit(trade: dict, final_fill: float) -> float:
    """Quantity-weighted exit price over partial legs plus the final fill."""
    legs = trade.get("legs", [])
    if not legs:
        return round(final_fill, 2)
    sold = sum(leg["qty"] for leg in legs)
    value = sum(leg["qty"] * leg["price"] for leg in legs) + (trade["quantity"] - sold) * final_fill
    return round(value / trade["quantity"], 4)


def breaker_reason(trades: list[dict], today, p: Params, capital: float,
                   since: str | None = None) -> str | None:
    """Circuit breaker on closed trades (optionally only those exited after `since`)."""
    closed = sorted((t for t in trades if t.get("status") == "CLOSED"
                     and (not since or t["exit_time"] >= since)), key=lambda t: t["exit_time"])
    if p.max_consec_losses:
        streak = 0
        for trade in reversed(closed):
            if trade["net"] >= 0:
                break
            streak += 1
        if streak >= p.max_consec_losses:
            return f"{streak} consecutive losses (limit {p.max_consec_losses})"
    for pct, label, same in ((p.max_week_loss_pct, "week", lambda d: d.isocalendar()[:2] == today.isocalendar()[:2]),
                             (p.max_month_loss_pct, "month", lambda d: (d.year, d.month) == (today.year, today.month))):
        if pct:
            loss = sum(t["net"] for t in closed if same(datetime.fromisoformat(t["day"]).date()))
            if loss <= -capital * pct / 100:
                return f"{label} loss ₹{-loss:,.0f} reached {pct:g}% of capital"
    return None


def lots_for(sizing: str, value: float, capital: float, premium: float,
             lot_size: int, freeze: int) -> int:
    """Fixed lots, or Max Risk % treating the full premium as worst-case loss."""
    if sizing == "Fixed Lots":
        lots = int(value)
    else:
        lots = math.floor(capital * value / 100 / (premium * lot_size))
    if freeze:
        lots = min(lots, freeze // lot_size)
    return max(lots, 0)


@dataclass
class Settings:
    index: str = "NIFTY"
    interval: int = 5
    sizing: str = "Fixed Lots"
    sizing_value: float = 1
    capital: float = 100000
    params: Params = DEFAULT_PARAMS


class TickBars:
    """1-minute OHLC built from spot ticks; a bar is usable only if every minute is fully covered."""

    def __init__(self) -> None:
        self.minutes: dict[pd.Timestamp, list] = {}

    def add(self, price: float, when: datetime) -> None:
        minute = pd.Timestamp(when).floor("min")
        row = self.minutes.get(minute)
        if row is None:
            self.minutes[minute] = [price, price, price, price, pd.Timestamp(when)]
            if len(self.minutes) > 3000:
                for old in sorted(self.minutes)[:1000]:
                    del self.minutes[old]
        else:
            row[1], row[2], row[3] = max(row[1], price), min(row[2], price), price

    def bar(self, start: pd.Timestamp, interval: int) -> dict | None:
        rows = []
        for i in range(interval):
            minute = start + pd.Timedelta(minutes=i)
            row = self.minutes.get(minute)
            if row is None or (i == 0 and (row[4] - minute).total_seconds() > TICK_GAP_SECONDS):
                return None
            rows.append(row)
        return dict(open=rows[0][0], high=max(r[1] for r in rows), low=min(r[2] for r in rows),
                    close=rows[-1][3], volume=0.0)


OHLC = ["open", "high", "low", "close", "volume"]


@dataclass
class State:
    mode: str = "PAPER"
    live_armed: bool = False
    spot: float | None = None
    spot_time: datetime | None = None
    option_ltp: float | None = None
    option_time: datetime | None = None
    feeds: dict = field(default_factory=lambda: {"A-SPOT": "OFFLINE", "B-OPTION": "IDLE"})
    trade: dict | None = None
    frame: pd.DataFrame | None = None
    banner: str = "System Armed & Ready"
    blocked: bool = False
    halted: str = ""


class TradingEngine:
    def __init__(self, db: Database, fees: FeeSchedule | None = None, clock=now_ist) -> None:
        self.db, self.fees, self.clock = db, fees or FeeSchedule(), clock
        self.bse_fees = replace(self.fees, exchange_rate=BSE_EXCHANGE_RATE)
        self.settings = Settings()
        self.state = State()
        self.broker = None
        self.events: queue.Queue = queue.Queue()
        self.lock = threading.RLock()
        self.exiting = threading.Event()
        self.stop_event = threading.Event()
        self.pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="engine")
        self.loop: threading.Thread | None = None
        self.last_bar: pd.Timestamp | None = None
        self._last_refresh = -math.inf
        self.tick_bars = TickBars()
        self.bar_source_used = ""
        self._pending_exit: str | None = None
        self._tracker: ExitTracker | None = None

    # ---------- wiring ----------
    def emit(self, level: str, message: str) -> None:
        self.db.event(level, message)
        LOG.log({"CRITICAL": logging.CRITICAL, "ERROR": logging.ERROR, "WARN": logging.WARNING}
                .get(level, logging.INFO), "%s %s", level, message)
        self.events.put((level, message))

    def snapshot(self) -> dict:
        with self.lock:
            state = self.state
            trade = dict(state.trade) if state.trade else None
            return dict(mode=state.mode, live_armed=state.live_armed, spot=state.spot,
                        option_ltp=state.option_ltp, feeds=dict(state.feeds), trade=trade,
                        frame=state.frame, banner=state.banner, blocked=state.blocked,
                        halted=state.halted, settings=self.settings)

    def attach(self, broker) -> None:
        self.broker = broker
        self.recover()
        self.restart_feeds()
        if hasattr(broker, "subscribe_orders"):
            broker.subscribe_orders(self._on_order_update, self._feed)
        if hasattr(broker, "holidays"):
            self.pool.submit(lambda: self.entry_block(self.clock()) and self._update_banner())
        if self._live_relevant():
            self.pool.submit(self.reconcile)
        if not self.loop or not self.loop.is_alive():
            self.loop = threading.Thread(target=self._run, name="engine-loop", daemon=True)
            self.loop.start()

    def restart_feeds(self) -> None:
        self.broker.stream("A-SPOT", [SPOT_KEYS[self.settings.index]], self._tick, self._feed)
        if self.state.trade:
            self.broker.stream("B-OPTION", [self.state.trade["key"]], self._tick, self._feed)

    def configure(self, **changes) -> None:
        if "params" in changes:
            changes["params"] = changes["params"].validate()
        with self.lock:
            if self.state.trade and "index" in changes and changes["index"] != self.settings.index:
                raise BrokerError("Cannot change index while a position is open.")
            for key, value in changes.items():
                setattr(self.settings, key, value)
            self.last_bar = None
        if self.broker:
            if "index" in changes:
                self.restart_feeds()
            self.pool.submit(self.refresh_bars)

    def set_mode(self, mode: str, armed: bool = False) -> None:
        with self.lock:
            if self.state.trade:
                raise BrokerError("Cannot switch mode while a position is open.")
            self.state.mode, self.state.live_armed = mode, armed and mode == "LIVE"
        self.emit("WARN" if mode == "LIVE" else "INFO",
                  f"Mode set to {mode}{' (ARMED)' if self.state.live_armed else ''}.")
        self._update_banner()
        if mode == "LIVE" and self.broker and hasattr(self.broker, "positions"):
            self.pool.submit(self.reconcile)

    def clear_halt(self) -> None:
        with self.lock:
            self.state.halted = ""
        self._update_banner()

    # ---------- feeds ----------
    def _feed(self, name: str, status: str) -> None:
        with self.lock:
            self.state.feeds[name] = status
        if status != "CONNECTED":
            self.emit("WARN", f"Channel {name}: {status}")

    def _tick(self, key: str, price: float, when: datetime) -> None:
        events, moved, trade = [], False, None
        with self.lock:
            trade = self.state.trade
            if key == SPOT_KEYS[self.settings.index]:
                self.state.spot, self.state.spot_time = price, when
                self.tick_bars.add(price, when)
                if trade and not self.exiting.is_set():
                    events, moved = self._tracker_for(trade).step(price)
            elif trade and key == trade["key"]:
                self.state.option_ltp, self.state.option_time = price, when
        if moved:
            self.db.update_trade(trade)
            kind = trade["track"]["kind"].replace("_", " ").lower()
            if kind != getattr(self, "_announced_kind", None):
                self._announced_kind = kind
                self.emit("INFO", f"Spot stop moved ({kind}) to {trade['levels']['stop']:.2f}")
        for event in events:
            if event == "PARTIAL":
                self.request_partial()
            else:
                self.request_exit(event)

    def _tracker_for(self, trade: dict) -> ExitTracker:
        tracker = getattr(self, "_tracker", None)
        if tracker is None or tracker.trade is not trade:
            tracker = self._tracker = ExitTracker(trade, params_of(trade, self.settings.params))
        return tracker

    def _on_order_update(self, update: dict) -> None:
        with self.lock:
            trade = self.state.trade
        if (trade and str(update.get("order_id")) in backstop_ids(trade)
                and str(update.get("status", "")).lower() == "complete"):
            self.emit("WARN", "Broker backstop stop-loss executed; recording exit.")
            self.request_exit("BACKSTOP_SL")

    # ---------- main loop ----------
    def _run(self) -> None:
        while not self.stop_event.wait(1):
            try:
                self._tick_clock()
            except Exception as error:
                self.emit("ERROR", f"Engine loop: {error}")

    def _tick_clock(self) -> None:
        now = self.clock()
        if now.date() != getattr(self, "_day", None):
            self._day = now.date()
            self._update_banner()
        with self.lock:
            trade = self.state.trade
            spot_stale = (self.state.spot_time is None
                          or (now - self.state.spot_time).total_seconds() > 5)
        if trade:
            square = self.settings.params.square_time
            if now.time() >= square:
                self.request_exit(f"SQUARE_OFF_{square:%H%M}")
            elif spot_stale and SESSION_OPEN <= now.time() < MARKET_CLOSE:
                self._poll_spot()
        interval = self.settings.interval
        minutes = now.hour * 60 + now.minute - (9 * 60 + 15)
        if 0 <= minutes and now.time() < MARKET_CLOSE and minutes % interval == 0 and now.second >= 3:
            bar = pd.Timestamp(now.replace(second=0, microsecond=0))
            if bar != self.last_bar:
                self.last_bar = bar
                self.pool.submit(self.on_bar_close)
        elif self.state.frame is None and time.monotonic() - self._last_refresh > 30:
            self._last_refresh = time.monotonic()
            self.pool.submit(self.refresh_bars)

    def _poll_spot(self) -> None:
        try:
            price, when = self.broker.quote(SPOT_KEYS[self.settings.index])
            self._tick(SPOT_KEYS[self.settings.index], price, when)
        except Exception as error:
            self.emit("ERROR", f"Spot feed stale and REST fallback failed: {error}")

    def refresh_bars(self) -> pd.DataFrame | None:
        if not self.broker:
            return None
        try:
            frame = indicators(self.broker.candles(self.settings.index, self.settings.interval),
                               self.settings.params)
        except Exception as error:
            self.emit("ERROR", f"Candle refresh failed: {error}")
            return None
        with self.lock:
            self.state.frame = frame
        self.bar_source_used = "REST"
        return frame

    def _tick_frame(self, bar_start: pd.Timestamp) -> pd.DataFrame | None:
        """Append the just-closed bar built from ticks; None if ticks or history are incomplete."""
        interval = self.settings.interval
        with self.lock:
            frame = self.state.frame
        if frame is None or bar_start.time() < SESSION_OPEN:
            return None
        bar = self.tick_bars.bar(bar_start, interval)
        if bar is None:
            return None
        base = frame.loc[frame.index < bar_start, OHLC]
        if not len(base):
            return None
        previous = base.index[-1]
        if bar_start.time() == SESSION_OPEN:
            if previous.date() >= bar_start.date():
                return None
        elif previous != bar_start - pd.Timedelta(minutes=interval):
            return None
        rows = pd.concat([base, pd.DataFrame([bar], index=pd.DatetimeIndex([bar_start]))])
        return indicators(rows.tail(HISTORY_BARS), self.settings.params)

    def on_bar_close(self) -> None:
        now = self.clock()
        interval = self.settings.interval
        bar_start = pd.Timestamp(now.replace(second=0, microsecond=0)) - pd.Timedelta(minutes=interval)
        frame = None
        if self.settings.params.bar_source == "Ticks":
            frame = self._tick_frame(bar_start)
            if frame is not None:
                with self.lock:
                    self.state.frame = frame
                self.bar_source_used = "Ticks"
        if frame is None:
            frame = self.refresh_bars()
        if frame is None:
            return
        side = crossover(frame, interval, now, self.settings.params)
        if side:
            self.emit("INFO", f"{'Bullish' if side == 'CE' else 'Bearish'} EMA crossover at "
                              f"{frame.index[-1]:%H:%M} ({self.bar_source_used} bars).")
            self.enter(side, float(frame["atr"].iloc[-1]))

    # ---------- execution ----------
    def entry_block(self, now: datetime, fetch: bool = True) -> str | None:
        """Day-level reasons not to open a trade: event day, market holiday, circuit breaker.
        fetch=False uses only the cached holiday calendar (safe from the UI thread)."""
        p, mode, today = self.settings.params, self.state.mode, now.date()
        if today in p.events:
            return "event day (no trading configured)"
        holidays = getattr(self.broker, "holidays", None) if fetch else None
        calendar = getattr(self, "_holiday_calendar", None)
        try:
            if holidays:
                calendar = self._holiday_calendar = holidays()
        except Exception as error:
            if not getattr(self, "_holiday_warned", False):
                self._holiday_warned = True
                self.emit("WARN", f"Holiday calendar unavailable: {error}")
        if calendar and EXCHANGE[self.settings.index] in calendar.get(today, []):
            return "exchange holiday"
        reason = breaker_reason(self.db.trades(mode=mode), today, p, self.settings.capital,
                                self.db.get_setting(f"breaker_reset_{mode}") or None)
        return f"circuit breaker - {reason}" if reason else None

    def reset_breaker(self) -> None:
        mode = self.state.mode
        self.db.set_setting(f"breaker_reset_{mode}", self.clock().isoformat())
        self.emit("WARN", f"Circuit breaker reset for {mode} by operator.")
        self._update_banner()

    def fees_for(self, exchange: str) -> FeeSchedule:
        return self.bse_fees if exchange == "BSE" else self.fees

    def _update_banner(self) -> None:
        now = self.clock()
        session = self.db.session(now.date().isoformat(), self.state.mode)
        block = None if (self.state.halted or self.state.trade or session) else self.entry_block(now, fetch=False)
        with self.lock:
            if self.state.halted:
                self.state.banner, self.state.blocked = f"HALTED - {self.state.halted}", True
            elif self.state.trade:
                self.state.banner, self.state.blocked = "Position Active - Monitoring Spot Exits", True
            elif session:
                self.state.banner, self.state.blocked = "Daily Trade Limit Exhausted - Blocked", True
            elif block:
                self.state.banner, self.state.blocked = f"Blocked - {block}", True
            else:
                self.state.banner, self.state.blocked = "System Armed & Ready", False

    def _liquidity_reject(self, contract: dict, premium: float, p: Params) -> str | None:
        if p.min_premium and premium < p.min_premium:
            return f"premium {premium:.2f} below minimum {p.min_premium:g}"
        if p.max_spread_pct and hasattr(self.broker, "depth"):
            quote = self.broker.depth(contract["key"])
            bid, ask = quote["bid"], quote["ask"]
            if bid <= 0 or ask <= 0 or ask < bid:
                return "no two-sided quote"
            spread = (ask - bid) / ((ask + bid) / 2) * 100
            if spread > p.max_spread_pct:
                return f"bid-ask spread {spread:.2f}% above {p.max_spread_pct:g}%"
        return None

    def enter(self, side: str, atr: float) -> None:
        now = self.clock()
        day = now.date().isoformat()
        p, index = self.settings.params, self.settings.index
        with self.lock:
            mode, spot = self.state.mode, self.state.spot
            if self.state.halted or self.state.trade:
                return
            if mode == "LIVE" and not self.state.live_armed:
                self.emit("WARN", "Live signal ignored: live trading is not armed.")
                return
            if spot is None or (now - self.state.spot_time).total_seconds() > 5:
                self.emit("WARN", "Signal ignored: spot price is stale.")
                return
        if self.db.session(day, mode):
            return
        block = self.entry_block(now)
        if block:
            self.emit("INFO", f"Signal ignored: {block}.")
            self._update_banner()
            return
        try:
            master = self.broker.master
            if master.loaded_day != now.date():
                master.refresh()
            expiry_today = master.nearest_expiry(index, now.date()) == now.date()
            if expiry_today and p.expiry_mode == "Cutoff" and now.time() >= p.expiry_time:
                self.emit("INFO", f"Signal ignored: expiry day after {p.expiry_cutoff}.")
                return
            contract = master.option(index, spot, side, now.date(),
                                     skip_today=expiry_today and p.expiry_mode == "Next expiry")
            premium, _ = self.broker.quote(contract["key"])
            reject = self._liquidity_reject(contract, premium, p)
            if reject:
                self.emit("WARN", f"Signal skipped ({contract['symbol']}): {reject}.")
                return
            lots = lots_for(self.settings.sizing, self.settings.sizing_value, self.account_value(),
                            premium, contract["lot_size"], contract["freeze"] * MAX_SLICES)
            if lots < 1:
                self.emit("WARN", "Signal skipped: sizing produced zero lots.")
                return
        except Exception as error:
            self.emit("WARN", f"Signal skipped, contract/quote unavailable: {error}")
            return
        if not self.db.reserve(day, mode):
            self._update_banner()
            return
        self._update_banner()
        try:
            quantity = lots * contract["lot_size"]
            tag = f"ST{now:%y%m%d}"
            if mode == "PAPER":
                fill, filled, order_id, executed = premium * (1 + p.slippage), quantity, "PAPER", 1
                slippage = premium * p.slippage * quantity
            else:
                self.db.session_state(day, mode, "ENTRY_SUBMITTING", tag)
                value, filled, executed = self._sliced(contract["key"], quantity, "BUY", tag,
                                                       contract["lot_size"], contract["freeze"])
                if filled < 1:
                    raise BrokerError("Entry order was not filled.")
                fill, order_id = value / filled, tag
                slippage = max(fill - premium, 0) * filled
                quantity = filled
            with self.lock:
                entry_spot = self.state.spot
            trade = dict(day=day, mode=mode, side=side, symbol=contract["symbol"],
                         key=contract["key"], strike=contract["strike"], expiry=contract["expiry"],
                         exchange=contract.get("exchange", EXCHANGE[index]), freeze=contract["freeze"],
                         quantity=quantity, lot_size=contract["lot_size"], entry_time=now.isoformat(),
                         entry_spot=entry_spot, atr=atr, levels=risk_levels(side, entry_spot, atr, p),
                         entry_premium=round(fill, 2), entry_ltp=premium, slippage=slippage,
                         entry_order=order_id, tag=tag, params=p.__dict__.copy(),
                         exec_orders=executed, expiry_day=expiry_today)
            if mode == "LIVE" and p.backstop:
                trade["backstops"] = self._place_backstops(trade, quantity, p)
            self.db.open_trade(trade)
            with self.lock:
                self.state.trade, self.state.option_ltp = trade, premium
            self.broker.stream("B-OPTION", [trade["key"]], self._tick, self._feed)
            self.emit("TRADE", f"ENTRY {mode} BUY {quantity} {trade['symbol']} @ {fill:.2f} | "
                               f"Spot {entry_spot:.2f} SL {trade['levels']['stop']:.2f} "
                               f"TP {trade['levels']['target']:.2f}")
        except Exception as error:
            self.db.session_state(day, mode, "BLOCKED", str(error))
            if mode == "LIVE":
                self.state.halted = "Entry failed/uncertain - verify broker order book"
            self.emit("ERROR", f"Entry aborted (no retry today): {error}")
        self._update_banner()

    def _sliced(self, key: str, quantity: int, side: str, tag: str, lot: int,
                freeze: int) -> tuple[float, int, int]:
        """Market order split at the exchange freeze limit. Returns (value, filled, executed_orders)."""
        cap = (freeze // lot) * lot if freeze and freeze >= lot else quantity
        value, filled, executed, remaining = 0.0, 0, 0, quantity
        while remaining > 0:
            part = min(remaining, cap)
            order_id = self.broker.market_order(key, part, side, tag)
            try:
                price, got = self._await_fill(order_id)
            except BrokerError as error:
                if not filled:
                    raise
                self.emit("ERROR", f"{side} slice failed after {filled} filled: {error}")
                break
            value, filled, executed = value + price * got, filled + got, executed + (1 if got else 0)
            if got < part:
                break
            remaining -= part
        return value, filled, executed

    def _place_backstops(self, trade: dict, quantity: int, p: Params) -> list[str]:
        """Broker-held SELL SL-limit orders (split at freeze) far below entry; protect if the app dies."""
        tick = 0.05
        trigger = math.floor(trade["entry_premium"] * (1 - p.backstop_pct / 100) / tick) * tick
        limit = math.floor(trigger * 0.9 / tick) * tick
        if limit < tick:
            self.emit("WARN", "Backstop skipped: premium too small for a protective SL.")
            return []
        lot, freeze = trade["lot_size"], trade.get("freeze") or 0
        cap = (freeze // lot) * lot if freeze >= lot else quantity
        ids, remaining = [], quantity
        try:
            while remaining > 0:
                part = min(remaining, cap)
                ids.append(self.broker.stop_loss_order(trade["key"], part, round(trigger, 2),
                                                       round(limit, 2), trade["tag"] + "B"))
                remaining -= part
            self.emit("INFO", f"Backstop SL placed at broker ({len(ids)} order(s)): "
                              f"trigger {trigger:.2f} limit {limit:.2f}")
        except Exception as error:
            self.emit("WARN", f"Backstop SL incomplete ({error}); app-side spot exits still active.")
        return ids

    def _cancel_backstop(self, trade: dict) -> tuple[float, int]:
        """Cancel all backstops; each must be terminal before any market SELL. Returns their fills."""
        value, filled = 0.0, 0
        for order_id in backstop_ids(trade):
            status = self.broker.order_status(order_id)
            if str(status.get("status", "")).lower() not in TERMINAL:
                try:
                    self.broker.cancel(order_id)
                except Exception as error:
                    self.emit("WARN", f"Backstop cancel request failed: {error}")
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    status = self.broker.order_status(order_id)
                    if str(status.get("status", "")).lower() in TERMINAL:
                        break
                    time.sleep(0.2)
                else:
                    raise BrokerError("Backstop SL still working at broker; refusing to send a second SELL.")
            got = int(status.get("filled_quantity") or 0)
            value, filled = value + float(status.get("average_price") or 0) * got, filled + got
            if got:
                trade["backstop_fills"] = trade.get("backstop_fills", 0) + 1
        trade.pop("backstop", None)
        trade["backstops"] = []
        return value, filled

    def _await_fill(self, order_id: str, timeout: float = 20) -> tuple[float, int]:
        deadline = time.monotonic() + timeout
        waiter = getattr(self.broker, "wait_order_update", None)
        status = {}
        while time.monotonic() < deadline:
            try:
                status = self.broker.order_status(order_id)
            except Exception:
                time.sleep(0.5)
                continue
            state = str(status.get("status", "")).lower()
            if state == "complete":
                return float(status["average_price"]), int(status["filled_quantity"])
            if state in {"rejected", "cancelled"}:
                filled = int(status.get("filled_quantity") or 0)
                if filled:
                    return float(status["average_price"]), filled
                raise BrokerError(f"Order {order_id} {state}: {status.get('status_message', '')}")
            if waiter:
                waiter(order_id, 0.4)
            else:
                time.sleep(0.4)
        try:
            self.broker.cancel(order_id)
        except Exception:
            pass
        status = self.broker.order_status(order_id)
        return float(status.get("average_price") or 0), int(status.get("filled_quantity") or 0)

    def request_exit(self, reason: str) -> None:
        # Check-and-set under the lock so concurrent triggers cannot start two exits; a trigger that
        # arrives during a partial exit is queued. While HALTED the operator owns the position.
        with self.lock:
            if self.state.halted:
                return
            if self.exiting.is_set():
                self._pending_exit = self._pending_exit or reason
                return
            self.exiting.set()
        self.pool.submit(self._exit, reason)

    def request_partial(self) -> None:
        with self.lock:
            if self.state.halted or self.exiting.is_set():
                return
            self.exiting.set()
        self.pool.submit(self._partial)

    def _release(self) -> None:
        with self.lock:
            self.exiting.clear()
            pending, self._pending_exit = self._pending_exit, None
            active = self.state.trade is not None
        self._update_banner()
        if pending and active:
            self.request_exit(pending)

    def _partial(self) -> None:
        try:
            with self.lock:
                trade = self.state.trade
            if not trade:
                return
            p = params_of(trade, self.settings.params)
            sold_before = sum(leg["qty"] for leg in trade.get("legs", []))
            qty = partial_quantity(trade["quantity"] - sold_before, trade["lot_size"], p.partial_pct)
            if not qty:
                return
            ltp = self._option_price(trade)
            if trade["mode"] == "PAPER":
                price, got, executed = ltp * (1 - p.slippage), qty, 1
                trade["slippage"] = trade.get("slippage", 0) + ltp * p.slippage * qty
            else:
                value, filled = self._cancel_backstop(trade)
                if filled:
                    raise BrokerError("backstop filled during partial exit")
                value, got, executed = self._sliced(trade["key"], qty, "SELL", trade["tag"] + "P",
                                                    trade["lot_size"], trade.get("freeze") or 0)
                price = value / got if got else 0
                remaining = trade["quantity"] - sold_before - got
                if p.backstop and remaining > 0:
                    trade["backstops"] = self._place_backstops(trade, remaining, p)
            if got:
                trade.setdefault("legs", []).append(dict(qty=got, price=round(price, 2), ltp=ltp,
                                                         time=self.clock().isoformat()))
                trade["exec_orders"] = trade.get("exec_orders", 1) + executed
                self.db.update_trade(trade)
                self.emit("TRADE", f"PARTIAL EXIT {got} {trade['symbol']} @ {price:.2f}")
        except Exception as error:
            with self.lock:
                self.state.halted = "PARTIAL EXIT FAILED - check broker position"
            self.emit("CRITICAL", f"Partial exit failed: {error}")
        finally:
            self._release()

    def _exit(self, reason: str) -> None:
        try:
            with self.lock:
                trade = self.state.trade
            if not trade:
                return
            ltp = self._option_price(trade) if trade["mode"] == "PAPER" else self.state.option_ltp
            trade["exit_ltp"] = ltp
            if trade["mode"] == "PAPER":
                slip = params_of(trade, self.settings.params).slippage
                remaining = trade["quantity"] - sum(leg["qty"] for leg in trade.get("legs", []))
                fill = ltp * (1 - slip)
                trade["slippage"] = trade.get("slippage", 0) + ltp * slip * remaining
                orders = trade.get("exec_orders", 1) + 1
            else:
                fill, orders = self._live_exit(trade)
            self._finish(trade, blended_exit(trade, fill), reason, orders)
        except Exception as error:
            with self.lock:
                self.state.halted = "EXIT FAILED - CLOSE POSITION MANUALLY NOW"
            self.emit("CRITICAL", f"Exit failed for {reason}: {error}")
        finally:
            self._release()

    def _finish(self, trade: dict, fill: float, reason: str, orders: int) -> dict:
        track = trade.get("track", {})
        trade["mfe_pts"], trade["mae_pts"] = round(track.get("mfe", 0.0), 2), round(track.get("mae", 0.0), 2)
        exchange_fees = self.fees_for(trade.get("exchange", "NSE"))
        price = fill if trade.get("legs") else round(fill, 2)
        closed = self.db.close_trade(trade, price, reason, exchange_fees, self.clock(), orders)
        with self.lock:
            self.state.trade = None
            self.state.feeds["B-OPTION"] = "IDLE"
            self._tracker = None
        channel = getattr(self.broker, "channels", {}).get("B-OPTION")
        if channel:
            channel.stop()
        self.emit("TRADE", f"EXIT {reason} {closed['symbol']} @ {fill:.2f} | "
                           f"Net P&L ₹{closed['net']:,.2f} (taxes ₹{closed['taxes']:,.2f})")
        return closed

    def _option_price(self, trade: dict) -> float:
        with self.lock:
            price, when = self.state.option_ltp, self.state.option_time
        if price and when and (self.clock() - when).total_seconds() <= 5:
            return price
        price, _ = self.broker.quote(trade["key"])
        return price

    def _live_exit(self, trade: dict) -> tuple[float, int]:
        """Cancel backstops, then sell what remains (freeze-sliced). Returns (avg price of this
        final leg, total executed orders incl. entry/partials) - brokerage counts executed orders only."""
        value, sold = self._cancel_backstop(trade)
        orders = trade.get("exec_orders", 1) + trade.pop("backstop_fills", 0)
        for attempt in range(3):
            remaining = self._open_quantity(trade["key"])
            if remaining <= 0:
                break
            try:
                got_value, got, executed = self._sliced(trade["key"], remaining, "SELL", trade["tag"] + "X",
                                                        trade["lot_size"], trade.get("freeze") or 0)
                value, sold, orders = value + got_value, sold + got, orders + executed
            except BrokerError as error:
                self.emit("ERROR", f"Exit attempt {attempt + 1}: {error}")
        if self._open_quantity(trade["key"]) > 0:
            raise BrokerError("Position still open at broker after 3 exit attempts.")
        if not sold:
            raise BrokerError("Broker shows flat but no exit fill was confirmed; reconcile manually.")
        return value / sold, orders

    def _open_quantity(self, key: str) -> int:
        positions = self.broker.positions()
        return sum(int(p.get("quantity") or 0) for p in positions if p.get("instrument_token") == key)

    def recover(self) -> None:
        trade = self.db.active_trade()
        day = self.clock().date().isoformat()
        with self.lock:
            self.state.trade = trade
            if trade:
                self.state.mode = trade["mode"]
        if trade:
            self.emit("WARN", f"Recovered open {trade['mode']} position {trade['symbol']}; "
                              "spot exit monitoring resumed.")
        for mode in ("PAPER", "LIVE"):
            session = self.db.session(day, mode)
            if trade and trade["mode"] == mode:
                continue
            if session and session["state"] in {"RESERVED", "ENTRY_SUBMITTING"}:
                self.db.session_state(day, mode, "BLOCKED", "Interrupted entry recovered after restart")
                if mode == "LIVE":
                    with self.lock:
                        self.state.halted = "Interrupted live entry - verify broker order book"
                self.emit("WARN", f"Previous {mode} entry was interrupted; day blocked for safety.")
        self._update_banner()

    def _live_relevant(self) -> bool:
        day = self.clock().date().isoformat()
        trade = self.state.trade
        return (self.state.mode == "LIVE" or bool(trade and trade["mode"] == "LIVE")
                or self.db.session(day, "LIVE") is not None)

    def reconcile(self) -> list[str]:
        """Compare the app's live view with broker positions/orders; HALT on any mismatch."""
        issues: list[str] = []
        try:
            positions = {p.get("instrument_token"): int(p.get("quantity") or 0)
                         for p in self.broker.positions()}
            with self.lock:
                trade = self.state.trade
            live = trade if trade and trade["mode"] == "LIVE" else None
            if live:
                held = positions.get(live["key"], 0)
                if held == 0:
                    issues.append(f"App shows open {live['symbol']} but broker is flat - use Manual Close.")
                elif held != live["quantity"]:
                    issues.append(f"Quantity mismatch on {live['symbol']}: app {live['quantity']}, broker {held}.")
            else:
                tag = f"ST{self.clock():%y%m%d}"
                orders = getattr(self.broker, "order_book", lambda: [])()
                for order in orders:
                    if (order.get("tag") == tag and order.get("transaction_type") == "BUY"
                            and int(order.get("filled_quantity") or 0) > 0
                            and positions.get(order.get("instrument_token"), 0) > 0):
                        issues.append(f"Untracked live position {order.get('instrument_token')} from "
                                      f"today's app order {order.get('order_id')}.")
        except Exception as error:
            issues.append(f"Could not reconcile with broker: {error}")
        if issues:
            with self.lock:
                self.state.halted = "RECONCILIATION MISMATCH - check broker"
            for issue in issues:
                self.emit("CRITICAL", f"Reconcile: {issue}")
        else:
            self.emit("INFO", "Reconcile: broker positions match the app.")
        self._update_banner()
        return issues

    def manual_close(self, exit_price: float, note: str = "") -> dict:
        """Record an operator-closed position. Live: broker must be flat and backstop terminal."""
        if not (math.isfinite(exit_price) and exit_price > 0):
            raise ValueError("Exit price must be a positive number.")
        with self.lock:
            trade = self.state.trade
            if not trade:
                raise BrokerError("No open trade to close.")
            if self.exiting.is_set():
                raise BrokerError("An automatic exit is in progress; wait for it to finish.")
            self.exiting.set()
        try:
            if trade["mode"] == "LIVE":
                self._cancel_backstop(trade)
                held = self._open_quantity(trade["key"])
                if held > 0:
                    raise BrokerError(f"Broker still holds {held} of {trade['symbol']}; close it first.")
            trade = dict(trade, manual_note=note, exit_ltp=exit_price)
            closed = self._finish(trade, blended_exit(trade, exit_price), "MANUAL_CLOSE",
                                  trade.get("exec_orders", 1) + 1)
            with self.lock:
                self.state.halted = ""
            return closed
        finally:
            self.exiting.clear()
            self._update_banner()

    def account_value(self) -> float:
        summary = metrics(self.db.trades(mode=self.state.mode), self.settings.capital)
        return summary["equity"]

    def unrealized(self) -> float:
        with self.lock:
            trade, ltp = self.state.trade, self.state.option_ltp
        if not trade or ltp is None:
            return 0.0
        return (ltp - trade["entry_premium"]) * trade["quantity"]

    def unrealized_net(self) -> tuple[float, float]:
        """(net P&L if closed now incl. partial legs, estimated round-trip charges) for the open trade."""
        with self.lock:
            trade, ltp = self.state.trade, self.state.option_ltp
        if not trade or ltp is None:
            return 0.0, 0.0
        exit_price = blended_exit(trade, ltp)
        orders = trade.get("exec_orders", 1) + len(trade.get("legs", [])) + 1
        charges = self.fees_for(trade.get("exchange", "NSE")).calculate(
            trade["entry_premium"], exit_price, trade["quantity"], orders)["total"]
        return (exit_price - trade["entry_premium"]) * trade["quantity"] - charges, charges

    def shutdown(self) -> None:
        self.stop_event.set()
        if self.broker:
            self.broker.close()
        self.pool.shutdown(wait=False, cancel_futures=True)


# ---------------------------------------------------------------- backtest
def _price_at(frame: pd.DataFrame, when: pd.Timestamp, field_name: str) -> float | None:
    """Option price at minute `when`, else the last close before it."""
    if when in frame.index:
        return float(frame.at[when, field_name])
    earlier = frame[frame.index < when]
    return float(earlier["close"].iloc[-1]) if len(earlier) else None


def fees_for_index(fees: FeeSchedule, index: str) -> FeeSchedule:
    return replace(fees, exchange_rate=BSE_EXCHANGE_RATE) if EXCHANGE[index] == "BSE" else fees


def simulate_day(day, minutes: pd.DataFrame, settings: Settings, capital: float,
                 load_option, fees: FeeSchedule, frame: pd.DataFrame | None = None,
                 expiry_day: bool = False) -> tuple[dict | None, str]:
    """Replay one session with the live rules. Each minute is walked as open -> adverse extreme ->
    favourable extreme -> close (the SIL harness tick order), through the shared ExitTracker."""
    from upstox_broker import round_strike
    p, interval = settings.params, settings.interval
    history = minutes[minutes.index.date <= day]
    if frame is None:
        frame = indicators(resample_bars(history, interval), p)
    frame = frame[frame.index.date <= day]
    today = frame[frame.index.date == day]
    day_minutes = history[history.index.date == day]
    square_reason = f"SQUARE_OFF_{p.square_time:%H%M}"
    skip_today = expiry_day and p.expiry_mode == "Next expiry"
    skipped = []
    for bar_start in today.index:
        # Same warm-up rule as the live engine: enough settled bars at signal time.
        visible = frame.loc[:bar_start].tail(HISTORY_BARS)
        if len(visible) < p.warmup:
            continue
        now = bar_start + pd.Timedelta(minutes=interval, seconds=3)
        side = crossover(visible, interval, now.to_pydatetime(), p)
        if not side:
            continue
        if expiry_day and p.expiry_mode == "Cutoff" and now.time() >= p.expiry_time:
            skipped.append(f"{bar_start:%H:%M} expiry cutoff")
            continue
        entry_minute = bar_start + pd.Timedelta(minutes=interval)
        if entry_minute not in day_minutes.index:
            continue
        entry_spot = float(day_minutes.at[entry_minute, "open"])
        atr = float(frame.at[bar_start, "atr"])
        contract, option = load_option(day, side, round_strike(entry_spot, settings.index), skip_today)
        premium = _price_at(option, entry_minute, "open")
        if not premium:
            skipped.append(f"{bar_start:%H:%M} no option price")
            continue
        if p.min_premium and premium < p.min_premium:
            skipped.append(f"{bar_start:%H:%M} premium {premium:.2f} < {p.min_premium:g}")
            continue
        lots = lots_for(settings.sizing, settings.sizing_value, capital, premium,
                        contract["lot_size"], contract["freeze"] * MAX_SLICES)
        if lots < 1:
            skipped.append(f"{bar_start:%H:%M} sizing 0 lots")
            continue
        quantity = lots * contract["lot_size"]
        trade = dict(day=day.isoformat(), mode="BACKTEST", side=side, symbol=contract["symbol"],
                     key=contract["key"], strike=contract["strike"], expiry=contract["expiry"],
                     exchange=EXCHANGE[settings.index], quantity=quantity, lot_size=contract["lot_size"],
                     entry_time=entry_minute.isoformat(), entry_spot=entry_spot, atr=atr,
                     levels=risk_levels(side, entry_spot, atr, p), entry_premium=round(premium * (1 + p.slippage), 2),
                     entry_ltp=premium, slippage=premium * p.slippage * quantity, status="CLOSED",
                     expiry_day=expiry_day)
        tracker = ExitTracker(trade, p)
        reason, exit_minute, exit_field = square_reason, None, "open"
        square = pd.Timestamp(datetime.combine(day, p.square_time), tz=IST)
        window = day_minutes[day_minutes.index >= entry_minute]
        for minute, o, h, l, c in zip(window.index, window["open"], window["high"], window["low"], window["close"]):
            if minute >= square:
                exit_minute = minute
                break
            path = (o, l, h, c) if side == "CE" else (o, h, l, c)
            for position, price in enumerate(path):
                events, _ = tracker.step(float(price))
                field_name = "open" if position == 0 else "close"
                for event in events:
                    if event == "PARTIAL":
                        part = partial_quantity(quantity, contract["lot_size"], p.partial_pct)
                        if part:
                            ltp = _price_at(option, minute, field_name)
                            trade.setdefault("legs", []).append(
                                dict(qty=part, price=round(ltp * (1 - p.slippage), 2), ltp=ltp, time=minute.isoformat()))
                            trade["slippage"] += ltp * p.slippage * part
                    else:
                        reason, exit_minute, exit_field = event, minute, field_name
                if exit_minute is not None:
                    break
            if exit_minute is not None:
                break
        if exit_minute is None:
            exit_minute = day_minutes.index[-1]
        exit_ltp = _price_at(option, exit_minute, exit_field if reason != square_reason else "open")
        remaining = quantity - sum(leg["qty"] for leg in trade.get("legs", []))
        trade["slippage"] += exit_ltp * p.slippage * remaining
        trade["exit_ltp"] = exit_ltp
        exit_price = blended_exit(trade, exit_ltp * (1 - p.slippage))
        track = trade["track"]
        trade["mfe_pts"], trade["mae_pts"] = round(track["mfe"], 2), round(track["mae"], 2)
        orders = 2 + len(trade.get("legs", []))
        charges = fees_for_index(fees, settings.index).calculate(trade["entry_premium"], exit_price, quantity, orders)
        gross = (exit_price - trade["entry_premium"]) * quantity
        trade.update(exit_premium=exit_price, exit_reason=reason, exit_orders=orders,
                     exit_time=exit_minute.isoformat(), fees=charges, gross=gross,
                     taxes=charges["total"], net=gross - charges["total"])
        return trade, f"{side} {bar_start:%H:%M} -> {reason} net {trade['net']:,.2f}"
    if not len(today):
        return None, "no bars"
    if len(frame.loc[:today.index[-1]].tail(HISTORY_BARS)) < p.warmup:
        return None, "insufficient history padding"
    return None, "no tradable signal" + (f" (skipped: {'; '.join(skipped)})" if skipped else "")


def _fetch_window(start, end):
    return max(start - timedelta(days=20), end - timedelta(days=29))


def _expiry_flag(broker, index: str, day) -> bool:
    try:
        return broker.backtest_expiry(index, day) == day
    except Exception:
        return False


def replay(broker, settings: Settings, minutes: pd.DataFrame, days, fees: FeeSchedule, loader,
           frame: pd.DataFrame | None = None, on_day=lambda day, trade, note: None) -> list[dict]:
    """Day loop shared by backtest and sweep: event days, circuit breaker, expiry flag, sizing equity."""
    p, capital, trades = settings.params, settings.capital, []
    frame = frame if frame is not None else indicators(resample_bars(minutes, settings.interval), p)
    for day in days:
        trade, note = None, ""
        breaker = breaker_reason(trades, day, p, settings.capital)
        if day in p.events:
            note = "event day - skipped"
        elif breaker:
            note = f"circuit breaker - {breaker}"
        else:
            try:
                trade, note = simulate_day(day, minutes, settings, capital, loader, fees, frame,
                                           _expiry_flag(broker, settings.index, day))
            except Exception as error:
                trade, note = None, f"error: {error}"
        if trade:
            trades.append(trade)
            capital += trade["net"]
        on_day(day, trade, note)
    return trades


def run_backtest(broker, db: Database, settings: Settings, start, end,
                 fees: FeeSchedule | None = None, progress=lambda message: None) -> list[str]:
    """Replay [start, end] with real Upstox data into `db` (cleared first)."""
    fees = fees or FeeSchedule()
    fetch_from = _fetch_window(start, end)
    progress(f"Downloading {settings.index} 1-minute spot {fetch_from} -> {end}")
    minutes = broker.spot_minutes(settings.index, fetch_from, end)
    days = sorted({d for d in minutes.index.date if start <= d <= end})
    if not days:
        raise BrokerError("No trading days with data in the selected range.")
    db.reset()
    notes = []
    loader = lambda day, side, strike, skip=False: broker.backtest_option(settings.index, day, side, strike, skip)
    index_fees = fees_for_index(fees, settings.index)

    def record(day, trade, note):
        if trade:
            db.reserve(trade["day"], "BACKTEST")
            db.open_trade(trade)
            db.close_trade(trade, trade["exit_premium"], trade["exit_reason"], index_fees,
                           datetime.fromisoformat(trade["exit_time"]), trade["exit_orders"])
        line = f"{day}: {note}"
        notes.append(line)
        db.event("TRADE" if trade else "INFO", line)
        progress(line)

    replay(broker, settings, minutes, days, fees, loader, on_day=record)
    return notes


def monte_carlo(nets: list[float], capital: float, runs: int = 2000, seed: int = 7) -> dict | None:
    """Reshuffle trade order (bootstrap with replacement) to estimate the drawdown/P&L spread
    a different sequence of the same edge could produce. Needs >= 5 trades."""
    if len(nets) < 5:
        return None
    rng = np.random.default_rng(seed)
    samples = rng.choice(np.asarray(nets, dtype=float), size=(runs, len(nets)), replace=True)
    equity = capital + np.cumsum(samples, axis=1)
    peaks = np.maximum.accumulate(np.concatenate([np.full((runs, 1), capital), equity], axis=1), axis=1)[:, 1:]
    max_dd = ((peaks - equity) / peaks * 100).max(axis=1)
    final = equity[:, -1] - capital
    pct = lambda values, q: float(np.percentile(values, q))
    return dict(runs=runs, trades=len(nets), net_p5=pct(final, 5), net_p50=pct(final, 50),
                net_p95=pct(final, 95), dd_p50=pct(max_dd, 50), dd_p95=pct(max_dd, 95),
                loss_probability=float((final < 0).mean() * 100))


def sweep_grid(ema_fast, ema_slow, sl_atr, tp_atr, trend_filters, base: Params) -> list[Params]:
    grid = []
    for fast, slow, sl, tp, trend in itertools.product(ema_fast, ema_slow, sl_atr, tp_atr, trend_filters):
        try:
            grid.append(replace(base, ema_fast=int(fast), ema_slow=int(slow), sl_atr=float(sl),
                                tp_atr=float(tp), trend_filter=trend).validate())
        except ValueError:
            continue
    if len(grid) > SWEEP_LIMIT:
        raise ValueError(f"{len(grid)} combinations; limit is {SWEEP_LIMIT}.")
    return grid


def run_sweep(broker, settings: Settings, start, end, grid: list[Params],
              fees: FeeSchedule | None = None, progress=lambda message: None,
              cancel: threading.Event | None = None, split=None) -> list[dict]:
    """Backtest every Params in `grid` on one download. With `split` (a date), rank on the
    in-sample days <= split and report out-of-sample (oos_*) metrics for days after it."""
    fees = fees or FeeSchedule()
    fetch_from = _fetch_window(start, end)
    progress(f"Downloading {settings.index} 1-minute spot {fetch_from} -> {end}")
    minutes = broker.spot_minutes(settings.index, fetch_from, end)
    days = sorted({d for d in minutes.index.date if start <= d <= end})
    if not days:
        raise BrokerError("No trading days with data in the selected range.")
    if split is not None and not (days[0] <= split < days[-1]):
        raise ValueError("Out-of-sample split date must fall inside the range (before the last day).")
    cache: dict = {}

    def loader(day, side, strike, skip=False):
        key = (day, side, strike, skip)
        if key not in cache:
            try:
                cache[key] = broker.backtest_option(settings.index, day, side, strike, skip)
            except Exception as error:
                cache[key] = error
        if isinstance(cache[key], Exception):
            raise cache[key]
        return cache[key]

    bars = resample_bars(minutes, settings.interval)
    results = []
    for number, params in enumerate(grid, 1):
        if cancel and cancel.is_set():
            break
        run = replace(settings, params=params)
        errors = []
        trades = replay(broker, run, minutes, days, fees, loader, indicators(bars, params),
                        on_day=lambda day, trade, note: note.startswith("error") and errors.append(note))
        train = [t for t in trades if split is None or date.fromisoformat(t["day"]) <= split]
        test = [t for t in trades if split is not None and date.fromisoformat(t["day"]) > split]
        summary = metrics(train, settings.capital)
        row = dict(params=params, trades=len(train), errors=len(errors), net=summary["net"],
                   sharpe=summary["sharpe"], profit_factor=summary["profit_factor"],
                   max_dd=summary["max_dd"], win_loss=summary["win_loss"], all_trades=trades)
        if split is not None:
            oos = metrics(test, settings.capital)
            row.update(oos_trades=len(test), oos_net=oos["net"], oos_sharpe=oos["sharpe"],
                       oos_profit_factor=oos["profit_factor"], oos_max_dd=oos["max_dd"])
        results.append(row)
        progress(f"Sweep {number}/{len(grid)}: EMA {params.ema_fast}/{params.ema_slow} "
                 f"SL {params.sl_atr} TP {params.tp_atr} {params.trend_filter} -> "
                 f"{len(trades)} trades, net {summary['net']:,.0f}")
    rank = lambda value: -math.inf if value is None else value
    results.sort(key=lambda r: (rank(r["sharpe"]), rank(r["profit_factor"]), r["net"]), reverse=True)
    return results
