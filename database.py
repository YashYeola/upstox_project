"""Durable execution journal, Decimal fee accounting, and report analytics."""

from __future__ import annotations

import csv
import json
import math
import sqlite3
import statistics
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


def now_ist() -> datetime:
    return datetime.now(IST)


@dataclass(frozen=True)
class FeeSchedule:
    brokerage: str = "20"
    exchange_rate: str = "0.0003503"
    sebi_rate: str = "0.000001"
    stt_rate: str = "0.00125"
    stamp_rate: str = "0.00003"
    gst_rate: str = "0.18"

    def calculate(self, buy: float, sell: float, quantity: int,
                  orders: int = 2) -> dict[str, float]:
        buy_value = Decimal(str(buy)) * quantity
        sell_value = Decimal(str(sell)) * quantity
        turnover = buy_value + sell_value
        brokerage = Decimal(self.brokerage) * orders
        exchange = turnover * Decimal(self.exchange_rate)
        sebi = turnover * Decimal(self.sebi_rate)
        values = {
            "brokerage": brokerage, "exchange": exchange, "sebi": sebi,
            "stt": sell_value * Decimal(self.stt_rate),
            "stamp": buy_value * Decimal(self.stamp_rate),
            "gst": (brokerage + exchange + sebi) * Decimal(self.gst_rate),
        }
        rounded = {key: value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                   for key, value in values.items()}
        return {key: float(value) for key, value in
                (rounded | {"total": sum(rounded.values())}).items()}


class Database:
    def __init__(self, path: str | Path = "trades.db") -> None:
        self.path = str(path)
        with self.connect() as connection:
            self._migrate(connection)
            connection.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS sessions (
                    day TEXT NOT NULL, state TEXT NOT NULL,
                    mode TEXT NOT NULL, detail TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (day, mode)
                );
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    day TEXT NOT NULL, mode TEXT NOT NULL,
                    status TEXT NOT NULL, payload TEXT NOT NULL,
                    entry_time TEXT NOT NULL, exit_time TEXT,
                    gross REAL DEFAULT 0, taxes REAL DEFAULT 0,
                    slippage REAL DEFAULT 0, net REAL DEFAULT 0,
                    UNIQUE (day, mode)
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY, time TEXT NOT NULL,
                    level TEXT NOT NULL, message TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL
                );
            """)

    @staticmethod
    def _migrate(connection: sqlite3.Connection) -> None:
        """v1 keyed the daily slot by day only; v2 keys it by (day, mode)."""
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='sessions'").fetchone()
        if not row or "PRIMARY KEY (day, mode)" in row[0]:
            return
        connection.executescript("""
            ALTER TABLE sessions RENAME TO sessions_v1;
            ALTER TABLE trades RENAME TO trades_v1;
            CREATE TABLE sessions (
                day TEXT NOT NULL, state TEXT NOT NULL, mode TEXT NOT NULL,
                detail TEXT NOT NULL DEFAULT '', PRIMARY KEY (day, mode));
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT, day TEXT NOT NULL, mode TEXT NOT NULL,
                status TEXT NOT NULL, payload TEXT NOT NULL, entry_time TEXT NOT NULL,
                exit_time TEXT, gross REAL DEFAULT 0, taxes REAL DEFAULT 0,
                slippage REAL DEFAULT 0, net REAL DEFAULT 0, UNIQUE (day, mode));
            INSERT INTO sessions SELECT day, state, mode, detail FROM sessions_v1;
            INSERT INTO trades SELECT * FROM trades_v1;
            DROP TABLE sessions_v1;
            DROP TABLE trades_v1;
        """)

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def reserve(self, day: str, mode: str) -> bool:
        with self.connect() as connection:
            try:
                connection.execute("INSERT INTO sessions(day,state,mode) VALUES(?,?,?)",
                                   (day, "RESERVED", mode))
                return True
            except sqlite3.IntegrityError:
                return False

    def session(self, day: str, mode: str) -> dict | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM sessions WHERE day=? AND mode=?",
                                     (day, mode)).fetchone()
            return dict(row) if row else None

    def session_state(self, day: str, mode: str, state: str, detail: str = "") -> None:
        with self.connect() as connection:
            connection.execute("UPDATE sessions SET state=?,detail=? WHERE day=? AND mode=?",
                               (state, detail, day, mode))

    def open_trade(self, trade: dict) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                "INSERT INTO trades(day,mode,status,payload,entry_time) VALUES(?,?,?,?,?)",
                (trade["day"], trade["mode"], "OPEN", json.dumps(trade), trade["entry_time"]))
            connection.execute("UPDATE sessions SET state='OPEN' WHERE day=? AND mode=?",
                               (trade["day"], trade["mode"]))
            return int(cursor.lastrowid)

    def update_trade(self, trade: dict) -> None:
        with self.connect() as connection:
            connection.execute("UPDATE trades SET payload=? WHERE day=? AND mode=? AND status='OPEN'",
                               (json.dumps(trade), trade["day"], trade["mode"]))

    def active_trade(self, mode: str | None = None) -> dict | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT payload FROM trades WHERE status='OPEN' AND (? IS NULL OR mode=?) "
                "ORDER BY id LIMIT 1", (mode, mode)).fetchone()
            return json.loads(row["payload"]) if row else None

    def close_trade(self, trade: dict, exit_price: float, reason: str,
                    fees: FeeSchedule, when: datetime | None = None,
                    orders: int = 2) -> dict:
        trade = dict(trade)
        charges = fees.calculate(trade["entry_premium"], exit_price, trade["quantity"], orders)
        gross = (exit_price - trade["entry_premium"]) * trade["quantity"]
        trade.update(exit_premium=exit_price, exit_reason=reason,
                     exit_time=(when or now_ist()).isoformat(), fees=charges,
                     gross=gross, taxes=charges["total"], net=gross - charges["total"])
        with self.connect() as connection:
            connection.execute(
                "UPDATE trades SET status='CLOSED',payload=?,exit_time=?,gross=?,taxes=?,"
                "slippage=?,net=? WHERE day=? AND mode=? AND status='OPEN'",
                (json.dumps(trade), trade["exit_time"], gross, charges["total"],
                 trade.get("slippage", 0), trade["net"], trade["day"], trade["mode"]))
            connection.execute("UPDATE sessions SET state='EXHAUSTED' WHERE day=? AND mode=?",
                               (trade["day"], trade["mode"]))
        return trade

    def trades(self, start: str = "0001-01-01", end: str = "9999-12-31",
               outcome: str = "All", mode: str | None = None) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM trades WHERE day BETWEEN ? AND ? ORDER BY entry_time", (start, end))
            result = [dict(row) | json.loads(row["payload"]) for row in rows]
        return [row for row in result
                if (mode is None or row["mode"] == mode)
                and (outcome != "Win Only" or row["net"] > 0)
                and (outcome != "Loss Only" or row["net"] < 0)]

    def event(self, level: str, message: str) -> None:
        with self.connect() as connection:
            connection.execute("INSERT INTO events(time,level,message) VALUES(?,?,?)",
                               (now_ist().isoformat(), level, message))

    def recent_events(self) -> list[dict]:
        with self.connect() as connection:
            return [dict(row) for row in connection.execute(
                "SELECT * FROM events ORDER BY id DESC LIMIT 100")]

    def reset(self) -> None:
        with self.connect() as connection:
            connection.executescript("DELETE FROM trades; DELETE FROM sessions; DELETE FROM events;")

    def set_journal(self, trade_id: int, note: str, tags: str) -> None:
        """Attach a free-text note and comma-separated tags to a trade (kept in its payload)."""
        with self.connect() as connection:
            row = connection.execute("SELECT payload FROM trades WHERE id=?", (trade_id,)).fetchone()
            if not row:
                raise KeyError(f"trade {trade_id} not found")
            payload = json.loads(row["payload"])
            payload["note"] = note.strip()[:2000]
            payload["tags"] = ",".join(t.strip() for t in tags.split(",") if t.strip())[:200]
            connection.execute("UPDATE trades SET payload=? WHERE id=?", (json.dumps(payload), trade_id))

    def set_setting(self, key: str, value: str) -> None:
        with self.connect() as connection:
            connection.execute("INSERT OR REPLACE INTO settings VALUES(?,?)", (key, value))

    def get_setting(self, key: str, default: str = "") -> str:
        with self.connect() as connection:
            row = connection.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
            return row[0] if row else default


def metrics(trades: list[dict], initial_capital: float = 100000) -> dict:
    closed = [trade for trade in trades if trade.get("status") == "CLOSED"]
    profits = [trade["net"] for trade in closed if trade["net"] > 0]
    losses = [-trade["net"] for trade in closed if trade["net"] < 0]
    equity, peak, max_dd = initial_capital, initial_capital, 0.0
    curve, drawdowns, returns, durations = [equity], [0.0], [], []
    daily: dict[str, float] = {}
    for trade in closed:
        daily[trade["day"]] = daily.get(trade["day"], 0) + trade["net"]
        equity += trade["net"]
        peak = max(peak, equity)
        drawdown = (peak - equity) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, drawdown)
        curve.append(equity)
        drawdowns.append(drawdown)
        durations.append((datetime.fromisoformat(trade["exit_time"]) -
                          datetime.fromisoformat(trade["entry_time"])).total_seconds() / 60)
    prior = initial_capital
    for pnl in daily.values():
        returns.append(pnl / prior if prior > 0 else 0)
        prior += pnl
    deviation = statistics.stdev(returns) if len(returns) > 1 else 0
    return {
        "net": sum(trade["net"] for trade in closed), "equity": equity,
        "profit_factor": sum(profits) / sum(losses) if losses else (math.inf if profits else None),
        "sharpe": statistics.mean(returns) / deviation * math.sqrt(252) if deviation else None,
        "max_dd": max_dd,
        "win_loss": len(profits) / len(losses) if losses else (math.inf if profits else None),
        "avg_duration": statistics.mean(durations) if durations else 0,
        "taxes": sum(trade["taxes"] for trade in closed),
        "slippage": sum(trade.get("slippage", 0) for trade in closed),
        "curve": curve, "drawdowns": drawdowns, "count": len(closed),
    }


def export_csv(path: str | Path, trades: list[dict]) -> None:
    fields = ["id", "day", "mode", "status", "exchange", "symbol", "side", "quantity", "entry_time",
              "exit_time", "entry_spot", "entry_ltp", "entry_premium", "exit_ltp", "exit_premium",
              "exit_reason", "mfe_pts", "mae_pts", "gross", "taxes", "slippage", "net", "tags",
              "note", "legs", "fees"]
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(trades)


class HistoryStore:
    """Local archive of 1-minute spot candles so backtests can go beyond the broker's history window."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        with self._connect() as connection:
            connection.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS candles (
                    idx TEXT NOT NULL, ts TEXT NOT NULL, open REAL, high REAL, low REAL,
                    close REAL, volume REAL, PRIMARY KEY (idx, ts));
            """)

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=15)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def save(self, index: str, frame) -> int:
        rows = [(index, ts.isoformat(), float(r.open), float(r.high), float(r.low), float(r.close),
                 float(r.volume)) for ts, r in frame.iterrows()]
        with self._connect() as connection:
            connection.executemany("INSERT OR REPLACE INTO candles VALUES(?,?,?,?,?,?,?)", rows)
        return len(rows)

    def days(self, index: str, start, end) -> set:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT DISTINCT substr(ts,1,10) FROM candles WHERE idx=? AND ts BETWEEN ? AND ?",
                (index, start.isoformat(), end.isoformat() + "T99")).fetchall()
        return {datetime.fromisoformat(r[0]).date() for r in rows}

    def load(self, index: str, start, end):
        import pandas as pd
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT ts, open, high, low, close, volume FROM candles WHERE idx=? AND ts BETWEEN ? AND ? "
                "ORDER BY ts", (index, start.isoformat(), end.isoformat() + "T99")).fetchall()
        frame = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
        frame["time"] = pd.to_datetime(frame["time"], utc=True).dt.tz_convert(IST)
        return frame.set_index("time").astype(float)

    def coverage(self) -> list[tuple]:
        with self._connect() as connection:
            return connection.execute(
                "SELECT idx, MIN(substr(ts,1,10)), MAX(substr(ts,1,10)), COUNT(DISTINCT substr(ts,1,10)) "
                "FROM candles GROUP BY idx").fetchall()


def slippage_report(trades: list[dict]) -> dict:
    """Average adverse fill vs reference LTP, in % of premium, per mode (entry and exit legs)."""
    report = {}
    for mode in sorted({t["mode"] for t in trades}):
        entry = [(t["entry_premium"] - t["entry_ltp"]) / t["entry_ltp"] * 100 for t in trades
                 if t["mode"] == mode and t.get("entry_ltp")]
        exits = [(t["exit_ltp"] - t["exit_premium"]) / t["exit_ltp"] * 100 for t in trades
                 if t["mode"] == mode and t.get("exit_ltp") and t.get("exit_premium") and not t.get("legs")]
        report[mode] = dict(trades=len([t for t in trades if t["mode"] == mode]),
                            entry_pct=statistics.mean(entry) if entry else None,
                            exit_pct=statistics.mean(exits) if exits else None)
    return report


def financial_year(day: str) -> str:
    d = datetime.fromisoformat(day).date()
    start = d.year if d.month >= 4 else d.year - 1
    return f"FY{start}-{str(start + 1)[2:]}"


def fy_report(trades: list[dict], fy: str) -> dict:
    """Per-financial-year F&O summary to hand to a tax adviser (not tax advice)."""
    rows = [t for t in trades if t.get("status") == "CLOSED" and financial_year(t["day"]) == fy]
    charges = {k: sum((t.get("fees") or {}).get(k, 0) for t in rows)
               for k in ("brokerage", "exchange", "sebi", "stt", "stamp", "gst")}
    return dict(fy=fy, trades=len(rows),
                gross_pnl=sum(t["gross"] for t in rows),
                charges=charges, total_charges=sum(t["taxes"] for t in rows),
                net_pnl=sum(t["net"] for t in rows),
                abs_pnl_turnover=sum(abs(t["gross"]) for t in rows),
                sell_premium_value=sum(t["exit_premium"] * t["quantity"] for t in rows),
                buy_premium_value=sum(t["entry_premium"] * t["quantity"] for t in rows),
                wins=sum(1 for t in rows if t["net"] > 0), losses=sum(1 for t in rows if t["net"] < 0))


def daily_pnl(trades: list[dict]) -> dict:
    result: dict = {}
    for t in trades:
        if t.get("status") == "CLOSED":
            day = datetime.fromisoformat(t["day"]).date()
            result[day] = result.get(day, 0.0) + t["net"]
    return result


def build_pdf(path: str | Path, trades: list[dict], summary: dict) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    def num(value, digits=2, suffix=""):
        if value is None:
            return "-"
        return "inf" if math.isinf(value) else f"{value:,.{digits}f}{suffix}"

    styles = getSampleStyleSheet()
    green, red, ink = colors.HexColor("#10B981"), colors.HexColor("#DC143C"), colors.HexColor("#0D1117")
    days = sorted(trade["day"] for trade in trades)
    span = f"{days[0]} to {days[-1]}" if days else "no trades"
    story = [Paragraph("<b>STRAT TERMINAL</b> - Weekly Analytical Dossier", styles["Title"]),
             Paragraph(f"Period: {span} &nbsp;&nbsp; Generated: {now_ist():%Y-%m-%d %H:%M} IST",
                       styles["Normal"]), Spacer(1, 10)]
    kpis = [["Trades", "Net P&L (Rs)", "Sharpe", "Profit Factor", "Max DD", "Win/Loss",
             "Avg Duration", "Slippage (Rs)", "Taxes (Rs)"],
            [summary["count"], num(summary["net"]), num(summary["sharpe"]),
             num(summary["profit_factor"]), num(summary["max_dd"], suffix="%"),
             num(summary["win_loss"]), num(summary["avg_duration"], 1, " min"),
             num(summary["slippage"]), num(summary["taxes"])]]
    table = Table(kpis)
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), ink),
                               ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                               ("TEXTCOLOR", (1, 1), (1, 1), green if summary["net"] >= 0 else red),
                               ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                               ("FONTSIZE", (0, 0), (-1, -1), 9)]))
    story += [table, Spacer(1, 14)]
    rows = [["Date", "Mode", "Contract", "Qty", "Entry", "Exit", "Reason", "Brokerage",
             "Exch", "SEBI", "STT", "Stamp", "GST", "Net"]]
    for trade in trades:
        fees = trade.get("fees") or {}
        rows.append([trade["day"], trade["mode"], trade.get("symbol", ""), trade.get("quantity"),
                     num(trade.get("entry_premium")), num(trade.get("exit_premium")),
                     trade.get("exit_reason", "")] +
                    [num(fees.get(k)) for k in ("brokerage", "exchange", "sebi", "stt", "stamp", "gst")] +
                    [num(trade.get("net"))])
    ledger = Table(rows, repeatRows=1)
    style = [("BACKGROUND", (0, 0), (-1, 0), ink), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
             ("GRID", (0, 0), (-1, -1), 0.3, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 7.5)]
    for i, trade in enumerate(trades, start=1):
        style.append(("TEXTCOLOR", (-1, i), (-1, i), green if trade.get("net", 0) >= 0 else red))
    ledger.setStyle(TableStyle(style))
    story += [Paragraph("Trade & Regulatory Tax Ledger", styles["Heading2"]), ledger, Spacer(1, 10),
              Paragraph("Charges: Rs 20 brokerage/order, NSE option transaction charge, SEBI fee, "
                        "STT 0.125% on sell premium, stamp 0.003% on buy, GST 18% on brokerage + "
                        "transaction + SEBI. Paper fills include 0.5% adverse slippage.",
                        styles["Italic"])]
    SimpleDocTemplate(str(path), pagesize=landscape(A4), title="Strat Terminal Dossier").build(story)