"""CustomTkinter desktop terminal: dashboard, setup and reports."""

from __future__ import annotations

import json
import math
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, fields, replace
from datetime import datetime, timedelta
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
import keyring
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402

from database import (Database, HistoryStore, build_pdf, daily_pnl, export_csv,  # noqa: E402
                      financial_year, fy_report, metrics, now_ist, slippage_report)
from notify import Notifier, run_schtasks, schedule_command, unschedule_command  # noqa: E402
from strategy import (BAR_SOURCES, DEFAULT_PARAMS, EXPIRY_MODES, INTERVALS, SWEEP_LIMIT,  # noqa: E402
                      TREND_FILTERS, Params, TradingEngine, monte_carlo, run_backtest, run_sweep,
                      sweep_grid)
from upstox_broker import SERVICE, OAuthLogin, TokenVault, UpstoxBroker  # noqa: E402

try:
    import winsound
except ImportError:
    winsound = None

GREEN, RED, AMBER, MUTED = "#10B981", "#DC143C", "#F59E0B", "#9CA3AF"
PANEL, BG = "#151A21", "#0D1117"


def money(value: float) -> str:
    return f"₹{value:,.2f}"


def fmt(value, digits=2, suffix="") -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and math.isinf(value):
        return "∞"
    return f"{value:,.{digits}f}{suffix}"


def load_params(raw: str) -> Params:
    try:
        known = {f.name for f in fields(Params)}
        return Params(**{k: v for k, v in json.loads(raw).items() if k in known}).validate()
    except (ValueError, TypeError):
        return DEFAULT_PARAMS


def parse_list(text: str, cast) -> list:
    values = [cast(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("empty list")
    return values


class Toast(ctk.CTkToplevel):
    """Non-blocking corner notification; stacks upward, auto-closes."""
    stack: list = []

    def __init__(self, master, title: str, message: str, color: str, seconds: int) -> None:
        super().__init__(master)
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.configure(fg_color=PANEL)
        frame = ctk.CTkFrame(self, fg_color=PANEL, border_color=color, border_width=2)
        frame.pack(fill="both", expand=True)
        ctk.CTkLabel(frame, text=title, text_color=color, font=("Segoe UI Black", 14)).pack(
            anchor="w", padx=12, pady=(8, 0))
        ctk.CTkLabel(frame, text=message, wraplength=340, justify="left").pack(
            anchor="w", padx=12, pady=(0, 10))
        frame.bind("<Button-1>", lambda _e: self.close())
        self.update_idletasks()
        Toast.stack.append(self)
        offset = sum(t.winfo_height() + 8 for t in Toast.stack[:-1])
        x = self.winfo_screenwidth() - 380
        y = self.winfo_screenheight() - 120 - self.winfo_height() - offset
        self.geometry(f"360x{self.winfo_height()}+{x}+{max(y, 0)}")
        self.after(seconds * 1000, self.close)

    def close(self) -> None:
        if self in Toast.stack:
            Toast.stack.remove(self)
        if self.winfo_exists():
            self.destroy()


class Card(ctk.CTkFrame):
    def __init__(self, master, title: str) -> None:
        super().__init__(master, fg_color=PANEL, corner_radius=10)
        ctk.CTkLabel(self, text=title, text_color=MUTED, font=("Segoe UI", 12)).pack(
            anchor="w", padx=14, pady=(10, 0))
        self.value = ctk.CTkLabel(self, text="—", font=("Segoe UI Semibold", 22))
        self.value.pack(anchor="w", padx=14, pady=(0, 10))

    def set(self, text: str, color: str = "white") -> None:
        self.value.configure(text=text, text_color=color)


class Gauge(ctk.CTkFrame):
    def __init__(self, master, title: str, color: str) -> None:
        super().__init__(master, fg_color="transparent")
        self.label = ctk.CTkLabel(self, text=title, anchor="w")
        self.label.pack(fill="x")
        self.bar = ctk.CTkProgressBar(self, progress_color=color, height=14)
        self.bar.pack(fill="x", pady=(2, 8))
        self.bar.set(0)
        self.title = title

    def set(self, fraction: float, detail: str) -> None:
        self.bar.set(max(0.0, min(1.0, fraction)))
        self.label.configure(text=f"{self.title}  {detail}")


class App(ctk.CTk):
    def __init__(self, db: Database, engine: TradingEngine) -> None:
        super().__init__()
        ctk.set_appearance_mode("dark")
        self.db, self.engine = db, engine
        self.bt_db = Database(Path(db.path).with_name("backtest.db"))
        self.history = HistoryStore(Path(db.path).with_name("history.db"))
        self.notifier = Notifier()
        self.workers = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ui-io")
        self.chart_stamp = None
        self.sort_key, self.sort_desc = "entry_time", True
        self.alert_times: dict[str, float] = {}
        self.sweep_cancel = threading.Event()
        self.sweep_results: list[dict] = []
        self.title("Strat Terminal — Upstox Index Options")
        self.geometry("1440x900")
        self.minsize(1180, 760)
        self.configure(fg_color=BG)
        tabs = ctk.CTkTabview(self, fg_color=BG, segmented_button_selected_color=GREEN)
        tabs.pack(fill="both", expand=True, padx=12, pady=8)
        self._dashboard(tabs.add("Dashboard"))
        self._setup(tabs.add("Strategy & Setup"))
        self._reports(tabs.add("Reports & Logs"))
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(300, self._try_cached_login)
        self.after(500, self.poll)

    # ================= Dashboard =================
    def _dashboard(self, tab) -> None:
        top = ctk.CTkFrame(tab, fg_color="transparent")
        top.pack(fill="x")
        self.mode_switch = ctk.CTkSwitch(top, text="Live Trading", command=self.toggle_mode,
                                         progress_color=RED, font=("Segoe UI Semibold", 14))
        self.mode_switch.pack(side="left", padx=(4, 12))
        self.paper_bulb = ctk.CTkLabel(top, text="● PAPER", text_color=GREEN,
                                       font=("Segoe UI Black", 14))
        self.paper_bulb.pack(side="left", padx=6)
        self.live_bulb = ctk.CTkLabel(top, text="● LIVE", text_color="#374151",
                                      font=("Segoe UI Black", 14))
        self.live_bulb.pack(side="left", padx=6)
        self.feed_label = ctk.CTkLabel(top, text="", text_color=MUTED)
        self.feed_label.pack(side="right", padx=8)
        self.alerts_switch = ctk.CTkSwitch(top, text="Alerts", command=self._save_alerts)
        if self.db.get_setting("alerts", "1") == "1":
            self.alerts_switch.select()
        self.alerts_switch.pack(side="right", padx=8)
        ctk.CTkButton(top, text="Reconcile", width=100, fg_color="#2563EB",
                      command=self.reconcile).pack(side="right", padx=4)
        self.manual_btn = ctk.CTkButton(top, text="Manual Close…", width=120, fg_color="#7C3AED",
                                        command=self.manual_close)
        self.manual_btn.pack(side="right", padx=4)
        self.clear_btn = ctk.CTkButton(top, text="Acknowledge Halt", fg_color=AMBER,
                                       text_color="black", command=self.clear_halt, width=150)
        self.breaker_btn = ctk.CTkButton(top, text="Reset Circuit Breaker", fg_color=AMBER,
                                         text_color="black", command=self.reset_breaker, width=170)
        self.expiry_label = ctk.CTkLabel(tab, text="Expiry: connect to Upstox", text_color=MUTED, anchor="w")
        self.expiry_label.pack(fill="x", padx=6)

        self.banner = ctk.CTkLabel(tab, text="", height=54, corner_radius=10,
                                   font=("Segoe UI Black", 22))
        self.banner.pack(fill="x", pady=8)

        cards = ctk.CTkFrame(tab, fg_color="transparent")
        cards.pack(fill="x")
        self.cards = {}
        for column, name in enumerate(["Live Net P&L", "Account Value", "Active Strike",
                                       "Entry Premium", "Spot Index"]):
            card = Card(cards, name)
            card.grid(row=0, column=column, sticky="nsew", padx=4)
            cards.grid_columnconfigure(column, weight=1)
            self.cards[name] = card

        body = ctk.CTkFrame(tab, fg_color="transparent")
        body.pack(fill="both", expand=True, pady=8)
        side = ctk.CTkFrame(body, fg_color=PANEL, width=330, corner_radius=10)
        side.pack(side="left", fill="y", padx=(0, 8))
        side.pack_propagate(False)
        ctk.CTkLabel(side, text="Spot Proximity Risk", font=("Segoe UI Semibold", 16)).pack(
            anchor="w", padx=14, pady=(12, 6))
        self.tp_gauge = Gauge(side, "Target", GREEN)
        self.tp_gauge.pack(fill="x", padx=14)
        self.sl_gauge = Gauge(side, "Stop Loss", RED)
        self.sl_gauge.pack(fill="x", padx=14)
        self.levels = ctk.CTkLabel(side, text="No active position", justify="left",
                                   text_color=MUTED, anchor="w")
        self.levels.pack(fill="x", padx=14, pady=4)
        ctk.CTkLabel(side, text="Event Log", font=("Segoe UI Semibold", 16)).pack(
            anchor="w", padx=14, pady=(14, 4))
        self.log = ctk.CTkTextbox(side, font=("Consolas", 11), wrap="word")
        self.log.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        for level, color in {"TRADE": GREEN, "ERROR": RED, "CRITICAL": RED, "WARN": AMBER}.items():
            self.log.tag_config(level, foreground=color)
        for event in reversed(self.db.recent_events()[:40]):
            self._log(event["level"], event["message"], event["time"][11:19])

        chart = ctk.CTkFrame(body, fg_color=PANEL, corner_radius=10)
        chart.pack(side="left", fill="both", expand=True)
        self.figure = Figure(figsize=(8, 6), dpi=100, facecolor=PANEL)
        grid = self.figure.add_gridspec(2, 1, height_ratios=[3, 1.3], hspace=0.12)
        self.ax_price = self.figure.add_subplot(grid[0])
        self.ax_equity = self.figure.add_subplot(grid[1])
        self.ax_dd = self.ax_equity.twinx()
        self.canvas = FigureCanvasTkAgg(self.figure, master=chart)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

    def _style(self, axis) -> None:
        axis.set_facecolor(BG)
        axis.tick_params(colors=MUTED, labelsize=8)
        for spine in axis.spines.values():
            spine.set_color("#2D333B")
        axis.grid(color="#222831", linewidth=0.6)

    def draw_charts(self, snap) -> None:
        frame, trade = snap["frame"], snap["trade"]
        stamp = (None if frame is None else (frame.index[-1], len(frame), tuple(frame.columns)),
                 bool(trade), len(self.db.trades()))
        if stamp == self.chart_stamp:
            return
        self.chart_stamp = stamp
        for axis in (self.ax_price, self.ax_equity, self.ax_dd):
            axis.clear()
            self._style(axis)
        if frame is not None:
            data = frame.tail(90)
            x = range(len(data))
            up = data["close"] >= data["open"]
            colors = [GREEN if flag else RED for flag in up]
            self.ax_price.vlines(x, data["low"], data["high"], color=colors, linewidth=1)
            self.ax_price.bar(x, (data["close"] - data["open"]).abs().clip(lower=0.05),
                              bottom=data[["open", "close"]].min(axis=1), color=colors, width=0.6)
            p = snap["settings"].params
            self.ax_price.plot(x, data["ema_fast"], color="#38BDF8", linewidth=1.3, label=f"EMA {p.ema_fast}")
            self.ax_price.plot(x, data["ema_slow"], color="#F472B6", linewidth=1.3, label=f"EMA {p.ema_slow}")
            if "trend_ema" in data:
                self.ax_price.plot(x, data["trend_ema"], color=AMBER, linewidth=1, linestyle="--",
                                   label=f"Trend EMA {p.trend_ema}")
            if trade:
                levels = trade["levels"]
                self.ax_price.axhline(trade["entry_spot"], color="white", linestyle=":", linewidth=1)
                low, high = sorted([levels["stop"], trade["entry_spot"]])
                self.ax_price.axhspan(low, high, color=RED, alpha=0.12)
                low, high = sorted([levels["target"], trade["entry_spot"]])
                self.ax_price.axhspan(low, high, color=GREEN, alpha=0.12)
                self.ax_price.axhline(levels["stop"], color=RED, linewidth=1.2, label="Spot SL")
                self.ax_price.axhline(levels["target"], color=GREEN, linewidth=1.2, label="Spot TP")
            step = max(1, len(data) // 8)
            self.ax_price.set_xticks(list(x)[::step])
            self.ax_price.set_xticklabels([t.strftime("%d %H:%M") for t in data.index[::step]])
            self.ax_price.legend(loc="upper left", fontsize=8, facecolor=PANEL, labelcolor="white")
            self.ax_price.set_title(f"{snap['settings'].index} Spot · "
                                    f"{snap['settings'].interval} min", color="white", fontsize=10)
        result = metrics(self.db.trades(mode=snap["mode"]), snap["settings"].capital)
        self.ax_equity.plot(result["curve"], color=GREEN, linewidth=1.6, marker="o", markersize=3)
        self.ax_dd.fill_between(range(len(result["drawdowns"])),
                                [-d for d in result["drawdowns"]], color=RED, alpha=0.25)
        self.ax_dd.set_ylabel("DD %", color=MUTED, fontsize=8)
        self.ax_equity.set_ylabel("Equity ₹", color=MUTED, fontsize=8)
        self.ax_equity.set_title(f"Equity Curve · Max DD {result['max_dd']:.2f}%",
                                 color="white", fontsize=9)
        self.canvas.draw_idle()

    # ================= Setup =================
    def _setup(self, tab) -> None:
        box = ctk.CTkFrame(tab, fg_color=PANEL, corner_radius=10)
        box.pack(fill="x", padx=4, pady=6)
        ctk.CTkLabel(box, text="Upstox API Credentials", font=("Segoe UI Semibold", 18)).grid(
            row=0, column=0, columnspan=3, sticky="w", padx=14, pady=(12, 6))
        api_key = self.db.get_setting("api_key")
        self.api_key = self._entry(box, 1, "API Key", api_key)
        secret = keyring.get_password(SERVICE, "secret:" + TokenVault.account(api_key)) if api_key else ""
        self.api_secret = self._entry(box, 2, "API Secret", secret or "", show="•")
        self.redirect = self._entry(box, 3, "Redirect URI",
                                    self.db.get_setting("redirect", "http://127.0.0.1:8765/callback"))
        self.login_btn = ctk.CTkButton(box, text="Authenticate via Upstox Web Login",
                                       fg_color=GREEN, hover_color="#059669", command=self.login)
        self.login_btn.grid(row=4, column=1, sticky="w", padx=8, pady=12)
        self.auth_status = ctk.CTkLabel(box, text="Not connected", text_color=MUTED)
        self.auth_status.grid(row=4, column=2, sticky="w")
        ctk.CTkLabel(box, text="Secret is stored in Windows Credential Manager; the daily access "
                               "token is cached there too and expires at day end.",
                     text_color=MUTED).grid(row=5, column=0, columnspan=3, sticky="w", padx=14,
                                            pady=(0, 12))

        params = ctk.CTkFrame(tab, fg_color=PANEL, corner_radius=10)
        params.pack(fill="x", padx=4, pady=6)
        ctk.CTkLabel(params, text="Strategy Parameters", font=("Segoe UI Semibold", 18)).grid(
            row=0, column=0, columnspan=3, sticky="w", padx=14, pady=(12, 6))
        settings = self.engine.settings
        self.index_menu = self._menu(params, 1, "Target Index", ["NIFTY", "BANKNIFTY", "SENSEX"],
                                     self.db.get_setting("index", settings.index))
        labels = {value: key for key, value in INTERVALS.items()}
        self.tf_menu = self._menu(params, 2, "Strategy Timeframe", list(INTERVALS),
                                  labels[int(self.db.get_setting("interval", "5"))])
        self.sizing_menu = self._menu(params, 3, "Capital Sizing", ["Fixed Lots", "Max Risk %"],
                                      self.db.get_setting("sizing", "Fixed Lots"))
        self.sizing_value = self._entry(params, 4, "Lots / Max Risk %",
                                        self.db.get_setting("sizing_value", "1"))
        self.capital = self._entry(params, 5, "Starting Capital ₹",
                                   self.db.get_setting("capital", "100000"))
        ctk.CTkButton(params, text="Apply Parameters", command=self.apply_settings).grid(
            row=6, column=1, sticky="w", padx=8, pady=12)
        self.rules_label = ctk.CTkLabel(params, text="", text_color=MUTED)
        self.rules_label.grid(row=7, column=0, columnspan=3, sticky="w", padx=14, pady=(0, 12))

        adv = ctk.CTkFrame(tab, fg_color=PANEL, corner_radius=10)
        adv.pack(fill="x", padx=4, pady=6)
        ctk.CTkLabel(adv, text="Strategy Engine Settings", font=("Segoe UI Semibold", 18)).grid(
            row=0, column=0, columnspan=6, sticky="w", padx=14, pady=(12, 6))
        p = load_params(self.db.get_setting("params", "{}"))
        layout = [
            ("ema_fast", "EMA fast"), ("ema_slow", "EMA slow"), ("atr_length", "ATR length"),
            ("sl_atr", "SL × ATR"), ("tp_atr", "TP × ATR"), ("slippage_pct", "Paper slippage %"),
            ("signal_start", "Signals from"), ("entry_cutoff", "No entries after"),
            ("square_off", "Square-off"), ("trend_ema", "Trend EMA"), ("adx_length", "ADX length"),
            ("adx_min", "ADX minimum"), ("backstop_pct", "Backstop SL % below entry"),
            ("breakeven_atr", "Breakeven after ×ATR"), ("trail_start_atr", "Trail starts ×ATR"),
            ("trail_atr", "Trail distance ×ATR"), ("partial_atr", "Partial exit at ×ATR"),
            ("partial_pct", "Partial size %"), ("min_premium", "Min premium ₹"),
            ("max_spread_pct", "Max bid-ask spread %"), ("expiry_cutoff", "Expiry-day cutoff"),
            ("max_consec_losses", "Stop after N losses"), ("max_week_loss_pct", "Max week loss %"),
            ("max_month_loss_pct", "Max month loss %")]
        self.param_entries = {}
        for i, (name, label) in enumerate(layout):
            row, col = 1 + i // 3, (i % 3) * 2
            ctk.CTkLabel(adv, text=label, anchor="w", width=150).grid(row=row, column=col, padx=(14, 4),
                                                                      pady=3, sticky="w")
            entry = ctk.CTkEntry(adv, width=110)
            entry.insert(0, str(getattr(p, name)))
            entry.grid(row=row, column=col + 1, padx=(0, 16), pady=3, sticky="w")
            self.param_entries[name] = entry
        row = 2 + len(layout) // 3
        ctk.CTkLabel(adv, text="Trend filter", anchor="w").grid(row=row, column=0, padx=(14, 4), sticky="w")
        self.trend_menu = ctk.CTkOptionMenu(adv, values=list(TREND_FILTERS), width=110)
        self.trend_menu.set(p.trend_filter)
        self.trend_menu.grid(row=row, column=1, sticky="w")
        ctk.CTkLabel(adv, text="Bar source", anchor="w").grid(row=row, column=2, padx=(14, 4), sticky="w")
        self.bar_menu = ctk.CTkOptionMenu(adv, values=list(BAR_SOURCES), width=110)
        self.bar_menu.set(p.bar_source)
        self.bar_menu.grid(row=row, column=3, sticky="w")
        self.backstop_switch = ctk.CTkSwitch(adv, text="Broker backstop SL (live)")
        if p.backstop:
            self.backstop_switch.select()
        self.backstop_switch.grid(row=row, column=4, columnspan=2, padx=14, sticky="w")
        row += 1
        ctk.CTkLabel(adv, text="Expiry day", anchor="w").grid(row=row, column=0, padx=(14, 4), sticky="w")
        self.expiry_menu = ctk.CTkOptionMenu(adv, values=list(EXPIRY_MODES), width=110)
        self.expiry_menu.set(p.expiry_mode)
        self.expiry_menu.grid(row=row, column=1, sticky="w")
        ctk.CTkLabel(adv, text="No-trade days", anchor="w").grid(row=row, column=2, padx=(14, 4), sticky="w")
        self.events_entry = ctk.CTkEntry(adv, width=420, placeholder_text="YYYY-MM-DD, YYYY-MM-DD (budget, RBI, results)")
        if p.event_days:
            self.events_entry.insert(0, p.event_days)
        self.events_entry.grid(row=row, column=3, columnspan=3, sticky="w")
        ctk.CTkButton(adv, text="Apply Engine Settings", command=self.apply_settings).grid(
            row=row + 1, column=1, columnspan=2, sticky="w", padx=0, pady=12)
        ctk.CTkButton(adv, text="Reset Defaults", fg_color="#374151", command=self.reset_params).grid(
            row=row + 1, column=3, sticky="w", pady=12)
        ctk.CTkLabel(adv, text="Trend filter: EMA = close above/below Trend EMA in the trade direction; "
                               "ADX = trend strength ≥ minimum. (Index spot has no volume, so VWAP is "
                               "not available.) Ticks = bars built live from feed A, REST fallback on gaps. "
                               "0 disables breakeven/trail/partial/limits. Expiry day: Cutoff = no entries "
                               "after the cutoff time; Next expiry = trade next week's contract. Exchange "
                               "holidays come from Upstox; add your own no-trade days above.",
                     text_color=MUTED, wraplength=1100, justify="left").grid(
            row=row + 2, column=0, columnspan=6, sticky="w", padx=14, pady=(0, 12))
        self._notifications_panel(tab)
        self.apply_settings(quiet=True)

    def _notifications_panel(self, tab) -> None:
        box = ctk.CTkFrame(tab, fg_color=PANEL, corner_radius=10)
        box.pack(fill="x", padx=4, pady=6)
        ctk.CTkLabel(box, text="Phone Alerts & Auto-start", font=("Segoe UI Semibold", 18)).grid(
            row=0, column=0, columnspan=6, sticky="w", padx=14, pady=(12, 6))
        try:
            config = Notifier.load_config()
        except Exception:
            config = {}
        self.notifier.configure(config)
        fields = [("telegram_token", "Telegram bot token", True), ("telegram_chat", "Telegram chat id", False),
                  ("smtp_host", "SMTP host", False), ("smtp_port", "SMTP port", False),
                  ("smtp_user", "SMTP user", False), ("smtp_password", "SMTP password", True),
                  ("email_to", "Alert email to", False)]
        self.notify_entries = {}
        for i, (key, label, secret) in enumerate(fields):
            row, col = 1 + i // 3, (i % 3) * 2
            ctk.CTkLabel(box, text=label, anchor="w", width=140).grid(row=row, column=col, padx=(14, 4), pady=3, sticky="w")
            entry = ctk.CTkEntry(box, width=200, show="•" if secret else None)
            entry.insert(0, str(config.get(key, "587" if key == "smtp_port" else "")))
            entry.grid(row=row, column=col + 1, padx=(0, 16), pady=3, sticky="w")
            self.notify_entries[key] = entry
        bar = ctk.CTkFrame(box, fg_color="transparent")
        bar.grid(row=4, column=0, columnspan=6, sticky="w", padx=10, pady=10)
        ctk.CTkButton(bar, text="Save Alerts", width=110, command=self.save_notifications).pack(side="left", padx=4)
        ctk.CTkButton(bar, text="Send Test", width=100, fg_color="#374151",
                      command=lambda: self.notifier.notify("Test", "Strat Terminal alerts are working.")).pack(side="left", padx=4)
        ctk.CTkLabel(bar, text="   Auto-start weekdays at").pack(side="left")
        self.autostart_time = ctk.CTkEntry(bar, width=70)
        self.autostart_time.insert(0, self.db.get_setting("autostart", "09:00"))
        self.autostart_time.pack(side="left", padx=4)
        ctk.CTkButton(bar, text="Schedule", width=90, command=lambda: self.schedule(True)).pack(side="left", padx=4)
        ctk.CTkButton(bar, text="Remove", width=80, fg_color="#374151",
                      command=lambda: self.schedule(False)).pack(side="left", padx=4)
        ctk.CTkLabel(box, text="Secrets are stored in Windows Credential Manager. Alerts are sent for entries, "
                               "exits, CRITICAL events and feed loss. Upstox needs a browser login each day; "
                               "auto-start opens the app so you can log in before 09:15.",
                     text_color=MUTED, wraplength=1100, justify="left").grid(
            row=5, column=0, columnspan=6, sticky="w", padx=14, pady=(0, 12))

    def save_notifications(self) -> None:
        config = {k: e.get().strip() for k, e in self.notify_entries.items() if e.get().strip()}
        try:
            if config.get("smtp_port"):
                int(config["smtp_port"])
            Notifier.save_config(config)
        except ValueError:
            messagebox.showerror("Alerts", "SMTP port must be a number.")
            return
        except Exception as error:
            messagebox.showerror("Alerts", f"Could not store settings securely: {error}")
            return
        self.notifier.configure(config)
        messagebox.showinfo("Alerts", "Saved." if self.notifier.enabled else
                            "Saved, but no complete Telegram or email configuration was entered.")

    def schedule(self, enable: bool) -> None:
        at = self.autostart_time.get().strip()
        try:
            datetime.strptime(at, "%H:%M")
            output = run_schtasks(schedule_command(at) if enable else unschedule_command())
        except ValueError:
            messagebox.showerror("Auto-start", "Use HH:MM, e.g. 09:00")
            return
        except Exception as error:
            messagebox.showerror("Auto-start", str(error))
            return
        self.db.set_setting("autostart", at)
        messagebox.showinfo("Auto-start", output or ("Scheduled." if enable else "Removed."))

    def _read_params(self) -> Params:
        values = {}
        for name, entry in self.param_entries.items():
            raw = entry.get().strip()
            kind = type(getattr(DEFAULT_PARAMS, name))
            values[name] = raw if kind is str else kind(float(raw)) if kind is int else float(raw)
        return Params(**values, trend_filter=self.trend_menu.get(), bar_source=self.bar_menu.get(),
                      backstop=bool(self.backstop_switch.get()), expiry_mode=self.expiry_menu.get(),
                      event_days=self.events_entry.get().strip()).validate()

    def reset_params(self) -> None:
        for name, entry in self.param_entries.items():
            entry.delete(0, "end")
            entry.insert(0, str(getattr(DEFAULT_PARAMS, name)))
        self.trend_menu.set(DEFAULT_PARAMS.trend_filter)
        self.bar_menu.set(DEFAULT_PARAMS.bar_source)
        self.expiry_menu.set(DEFAULT_PARAMS.expiry_mode)
        self.events_entry.delete(0, "end")
        self.backstop_switch.select()

    def _entry(self, master, row, label, value, show=None):
        ctk.CTkLabel(master, text=label, width=170, anchor="w").grid(row=row, column=0,
                                                                     sticky="w", padx=14, pady=4)
        entry = ctk.CTkEntry(master, width=420, show=show)
        entry.insert(0, value)
        entry.grid(row=row, column=1, sticky="w", padx=8, pady=4)
        return entry

    def _menu(self, master, row, label, values, current):
        ctk.CTkLabel(master, text=label, width=170, anchor="w").grid(row=row, column=0,
                                                                     sticky="w", padx=14, pady=4)
        menu = ctk.CTkOptionMenu(master, values=values, width=200)
        menu.set(current)
        menu.grid(row=row, column=1, sticky="w", padx=8, pady=4)
        return menu

    def apply_settings(self, quiet: bool = False) -> None:
        try:
            value, capital = float(self.sizing_value.get()), float(self.capital.get())
            sizing = self.sizing_menu.get()
            if capital <= 0 or value <= 0 or (sizing == "Fixed Lots" and value != int(value)) \
                    or (sizing == "Max Risk %" and value > 100):
                raise ValueError("Lots must be a positive whole number; risk % must be in (0, 100]; "
                                 "capital must be positive.")
            params = self._read_params()
        except ValueError as error:
            messagebox.showerror("Invalid parameters", str(error) or "Check numeric fields and HH:MM times.")
            return
        changes = dict(index=self.index_menu.get(), interval=INTERVALS[self.tf_menu.get()],
                       sizing=sizing, sizing_value=value, capital=capital, params=params)
        try:
            self.engine.configure(**changes)
        except Exception as error:
            messagebox.showerror("Cannot apply", str(error))
            self.index_menu.set(self.engine.settings.index)
            return
        for key, item in changes.items():
            if key == "params":
                self.db.set_setting("params", json.dumps(asdict(item)))
            else:
                self.db.set_setting(key, str(int(item) if key == "interval" else item))
        self.rules_label.configure(text=(
            f"Rules: EMA {params.ema_fast}/{params.ema_slow} on spot · signals from {params.signal_start} · "
            f"1 trade/day per mode · SL {params.sl_atr}×ATR({params.atr_length}), TP {params.tp_atr}×ATR · "
            f"no entries after {params.entry_cutoff} · square-off {params.square_off} · "
            f"filter {params.trend_filter} · warm-up {params.warmup} bars"))
        self.chart_stamp = None
        if not quiet:
            self._log("INFO", f"Parameters applied: {self.index_menu.get()} {self.tf_menu.get()} "
                              f"{sizing} {value}; {asdict(params)}")

    def login(self) -> None:
        key, secret, redirect = self.api_key.get().strip(), self.api_secret.get().strip(), \
            self.redirect.get().strip()
        if not key or not secret:
            messagebox.showerror("Missing credentials", "Enter API key and secret.")
            return
        self.db.set_setting("api_key", key)
        self.db.set_setting("redirect", redirect)
        try:
            keyring.set_password(SERVICE, "secret:" + TokenVault.account(key), secret)
        except Exception as error:
            messagebox.showerror("Credential vault", f"Could not store secret securely: {error}")
            return
        self.login_btn.configure(state="disabled", text="Waiting for browser login…")
        future = self.workers.submit(OAuthLogin().authenticate, key, secret, redirect)
        future.add_done_callback(lambda f: self.after(0, self._login_done, f))

    def _login_done(self, future) -> None:
        self.login_btn.configure(state="normal", text="Authenticate via Upstox Web Login")
        try:
            self._connect(future.result())
        except Exception as error:
            self.auth_status.configure(text=f"Login failed: {error}", text_color=RED)

    def _try_cached_login(self) -> None:
        token = TokenVault().load(self.db.get_setting("api_key"))
        if token:
            self._connect(token)
        elif now_ist().weekday() < 5:
            message = "No Upstox session for today — log in on Strategy & Setup before 09:15."
            self._log("WARN", message)
            self.alert("FEED", message)

    def _connect(self, token: str) -> None:
        self.auth_status.configure(text="Connecting…", text_color=AMBER)

        def work():
            broker = UpstoxBroker(token, Path(self.db.path).with_name(".cache"))
            broker.history_store = self.history
            broker.validate()
            broker.master.refresh()
            return broker

        future = self.workers.submit(work)
        future.add_done_callback(lambda f: self.after(0, self._connected, f))

    def _connected(self, future) -> None:
        try:
            broker = future.result()
        except Exception as error:
            self.auth_status.configure(text=f"Connection failed: {error}", text_color=RED)
            self._log("ERROR", f"Broker connection failed: {error}")
            return
        self.engine.attach(broker)
        self.auth_status.configure(text=f"Connected · instruments {len(broker.master.records):,}",
                                   text_color=GREEN)
        self._log("INFO", "Connected to Upstox; feeds A (spot) and B (option) started.")
        self.workers.submit(self._sync_history, broker, self.engine.settings.index)
        self._show_expiry(broker)

    def _sync_history(self, broker, index: str) -> None:
        """Archive the broker's recent 1-minute history locally so backtests can reach further back."""
        try:
            today = now_ist().date()
            broker.spot_minutes(index, today - timedelta(days=27), today - timedelta(days=1))
            days = {row[0]: row[3] for row in self.history.coverage()}.get(index, 0)
            self.after(0, self._log, "INFO", f"History store: {index} {days} trading days archived.")
        except Exception as error:
            self.after(0, self._log, "WARN", f"History sync failed: {error}")

    def _show_expiry(self, broker) -> None:
        index = self.engine.settings.index
        try:
            expiries = broker.master.expiries(index)
            upcoming = [e for e in expiries if e >= now_ist().date()]
        except Exception:
            upcoming = []
        if not upcoming:
            self.expiry_label.configure(text=f"{index}: no expiry data", text_color=AMBER)
            return
        nearest = upcoming[0]
        gap = (upcoming[1] - nearest).days if len(upcoming) > 1 else 0
        days = (nearest - now_ist().date()).days
        kind = "monthly" if gap > 10 else "weekly"
        note = " — monthly contract: slower decay, wider strikes" if kind == "monthly" else ""
        today = " — EXPIRY TODAY" if days == 0 else ""
        self.expiry_label.configure(text=f"{index} next expiry {nearest:%d %b} ({days}d, {kind}){today}{note}",
                                    text_color=AMBER if (today or kind == "monthly") else MUTED)

    def toggle_mode(self) -> None:
        if self.mode_switch.get():
            dialog = ctk.CTkInputDialog(title="Enable LIVE trading",
                                        text="Real orders will be sent to the exchange.\n"
                                             "Type LIVE to arm live trading:")
            if (dialog.get_input() or "").strip() != "LIVE":
                self.mode_switch.deselect()
                return
            mode = "LIVE"
        else:
            mode = "PAPER"
        try:
            self.engine.set_mode(mode, armed=mode == "LIVE")
        except Exception as error:
            messagebox.showerror("Mode change blocked", str(error))
            self.mode_switch.select() if self.engine.state.mode == "LIVE" else self.mode_switch.deselect()

    def clear_halt(self) -> None:
        if messagebox.askyesno("Acknowledge halt",
                               "Confirm you verified the broker order book/positions and resolved "
                               "any open exposure manually."):
            self.engine.clear_halt()

    def reset_breaker(self) -> None:
        if messagebox.askyesno("Reset circuit breaker",
                               "Trading was paused by your loss limits. Reset and allow new entries?\n\n"
                               "Only losses after this reset will count toward the limits."):
            self.engine.reset_breaker()

    def manual_close(self) -> None:
        trade = self.engine.state.trade
        if not trade:
            messagebox.showinfo("Manual close", "There is no open trade.")
            return
        if not messagebox.askyesno(
                "Manual close",
                f"Record {trade['symbol']} ({trade['quantity']} qty) as closed by you?\n\n"
                "This does NOT send an order. LIVE trades: the broker position must already be "
                "flat; the app verifies this and cancels its backstop SL first."):
            return
        dialog = ctk.CTkInputDialog(title="Manual close", text="Your actual average exit price (₹):")
        raw = (dialog.get_input() or "").strip()
        if not raw:
            return
        try:
            price = float(raw)
        except ValueError:
            messagebox.showerror("Manual close", "Enter a number, e.g. 112.35")
            return
        future = self.workers.submit(self.engine.manual_close, price, "operator manual close")
        future.add_done_callback(lambda f: self.after(0, self._manual_done, f))

    def _manual_done(self, future) -> None:
        try:
            closed = future.result()
            messagebox.showinfo("Manual close", f"Recorded. Net P&L {money(closed['net'])}")
            self.refresh_reports()
        except Exception as error:
            messagebox.showerror("Manual close refused", str(error))

    def reconcile(self) -> None:
        if not self.engine.broker:
            messagebox.showerror("Not connected", "Authenticate with Upstox first.")
            return
        future = self.workers.submit(self.engine.reconcile)
        future.add_done_callback(lambda f: self.after(0, self._reconcile_done, f))

    def _reconcile_done(self, future) -> None:
        try:
            issues = future.result()
        except Exception as error:
            issues = [str(error)]
        if issues:
            messagebox.showwarning("Reconcile", "\n".join(issues))
        else:
            messagebox.showinfo("Reconcile", "Broker positions match the app.")

    def _save_alerts(self) -> None:
        self.db.set_setting("alerts", "1" if self.alerts_switch.get() else "0")

    def alert(self, level: str, message: str) -> None:
        """Toast + sound (+ phone if configured) for TRADE/CRITICAL/feed loss; throttled per message."""
        now = time.monotonic()
        key = message if level != "FEED" else message.split(":")[0]
        if now - self.alert_times.get(key, -math.inf) < 60:
            return
        self.alert_times[key] = now
        styles = {"TRADE": ("Trade", GREEN, 8, "SystemAsterisk"),
                  "CRITICAL": ("CRITICAL", RED, 20, "SystemHand"),
                  "FEED": ("Data feed", AMBER, 8, "SystemExclamation")}
        title, color, seconds, sound = styles[level]
        self.notifier.notify(title, message)
        if not self.alerts_switch.get():
            return
        try:
            Toast(self, title, message, color, seconds)
        except Exception:
            pass
        if winsound:
            winsound.PlaySound(sound, winsound.SND_ALIAS | winsound.SND_ASYNC)
        else:
            self.bell()

    # ================= Reports =================
    def _reports(self, tab) -> None:
        bar = ctk.CTkFrame(tab, fg_color="transparent")
        bar.pack(fill="x", pady=4)
        self.ledger = ctk.CTkSegmentedButton(bar, values=["Paper/Live", "Backtest"],
                                             command=lambda _: self.refresh_reports())
        self.ledger.set("Paper/Live")
        self.ledger.pack(side="left", padx=4)
        self.outcome = ctk.CTkSegmentedButton(bar, values=["All", "Win Only", "Loss Only"],
                                              command=lambda _: self.refresh_reports())
        self.outcome.set("All")
        self.outcome.pack(side="left", padx=4)
        self.date_from = ctk.CTkEntry(bar, placeholder_text="From YYYY-MM-DD", width=140)
        self.date_from.pack(side="left", padx=4)
        self.date_to = ctk.CTkEntry(bar, placeholder_text="To YYYY-MM-DD", width=140)
        self.date_to.pack(side="left", padx=4)
        ctk.CTkButton(bar, text="Apply Filter", width=100, command=self.refresh_reports).pack(
            side="left", padx=4)
        ctk.CTkButton(bar, text="Export PDF/CSV Analytical Dossier", fg_color=GREEN,
                      hover_color="#059669", font=("Segoe UI Semibold", 14),
                      command=self.export).pack(side="right", padx=4)
        ctk.CTkButton(bar, text="FY Tax Report", width=110, fg_color="#374151",
                      command=self.fy_tax_report).pack(side="right", padx=4)
        ctk.CTkButton(bar, text="Slippage", width=90, fg_color="#374151",
                      command=self.show_slippage).pack(side="right", padx=4)
        ctk.CTkButton(bar, text="P&L Calendar", width=110, fg_color="#374151",
                      command=self.show_calendar).pack(side="right", padx=4)

        bt = ctk.CTkFrame(tab, fg_color=PANEL, corner_radius=10)
        bt.pack(fill="x", pady=4)
        today = now_ist().date()
        monday = today - timedelta(days=today.weekday() + 7)
        ctk.CTkLabel(bt, text="Backtest (dummy run on real Upstox history)",
                     font=("Segoe UI Semibold", 14)).pack(side="left", padx=10, pady=8)
        self.bt_from = ctk.CTkEntry(bt, width=110)
        self.bt_from.insert(0, monday.isoformat())
        self.bt_from.pack(side="left", padx=4)
        self.bt_to = ctk.CTkEntry(bt, width=110)
        self.bt_to.insert(0, (monday + timedelta(days=4)).isoformat())
        self.bt_to.pack(side="left", padx=4)
        self.bt_btn = ctk.CTkButton(bt, text="Run Backtest", fg_color="#2563EB", width=130,
                                    command=self.run_backtest)
        self.bt_btn.pack(side="left", padx=8)
        ctk.CTkButton(bt, text="Parameter Sweep…", fg_color="#7C3AED", width=150,
                      command=self.open_sweep).pack(side="left", padx=4)
        self.bt_status = ctk.CTkLabel(bt, text="Uses the Strategy tab's index, timeframe and "
                                               "sizing. Requires Upstox login.", text_color=MUTED)
        self.bt_status.pack(side="left", padx=8)
        self.history_label = ctk.CTkLabel(tab, text=self._history_text(), text_color=MUTED, anchor="w")
        self.history_label.pack(fill="x", padx=8)

        body = ctk.CTkFrame(tab, fg_color="transparent")
        body.pack(fill="both", expand=True)
        self.columns = [("day", "Date", 86), ("mode", "Mode", 60), ("symbol", "Contract", 175),
                        ("quantity", "Qty", 45), ("entry_premium", "Entry", 65),
                        ("exit_premium", "Exit", 65), ("exit_reason", "Reason", 120),
                        ("mfe_pts", "MFE", 55), ("mae_pts", "MAE", 55),
                        ("taxes", "Taxes", 65), ("net", "Net P&L", 90), ("tags", "Tags", 110)]
        table = ctk.CTkFrame(body, fg_color=PANEL, corner_radius=10)
        table.pack(side="left", fill="both", expand=True, padx=(0, 8))
        header = ctk.CTkFrame(table, fg_color="#1F2630")
        header.pack(fill="x", padx=6, pady=(6, 0))
        for column, (key, title, width) in enumerate(self.columns):
            ctk.CTkButton(header, text=title + " ⇅", width=width, height=28, fg_color="transparent",
                          hover_color="#2D3748", anchor="w",
                          command=lambda k=key: self.sort_by(k)).grid(row=0, column=column, padx=1)
        self.rows = ctk.CTkScrollableFrame(table, fg_color=PANEL)
        self.rows.pack(fill="both", expand=True, padx=6, pady=6)

        side = ctk.CTkFrame(body, fg_color=PANEL, width=300, corner_radius=10)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        ctk.CTkLabel(side, text="Quant Metrics", font=("Segoe UI Semibold", 18)).pack(
            anchor="w", padx=14, pady=(12, 6))
        self.metric_labels = {}
        for name in ["Trades", "Net P&L", "Sharpe Ratio", "Profit Factor", "Max Drawdown",
                     "Win/Loss Ratio", "Avg Duration", "Total Slippage Drag", "Total Taxes Paid"]:
            row = ctk.CTkFrame(side, fg_color="transparent")
            row.pack(fill="x", padx=14, pady=3)
            ctk.CTkLabel(row, text=name, text_color=MUTED).pack(side="left")
            self.metric_labels[name] = ctk.CTkLabel(row, text="—", font=("Segoe UI Semibold", 14))
            self.metric_labels[name].pack(side="right")
        ctk.CTkLabel(side, text="Sharpe uses daily returns, √252 annualised;\nshown once ≥2 days.",
                     text_color=MUTED, justify="left").pack(anchor="w", padx=14, pady=10)
        self.refresh_reports()

    def filtered(self) -> list[dict]:
        start = self.date_from.get().strip() or "0001-01-01"
        end = self.date_to.get().strip() or "9999-12-31"
        for value in (start, end):
            datetime.strptime(value, "%Y-%m-%d")
        rows = self.report_db().trades(start, end, self.outcome.get())
        return [row for row in rows if row["status"] == "CLOSED"]

    def report_db(self) -> Database:
        return self.bt_db if self.ledger.get() == "Backtest" else self.db

    def run_backtest(self) -> None:
        if not self.engine.broker:
            messagebox.showerror("Not connected", "Authenticate with Upstox on the Strategy & "
                                                  "Setup tab first; history needs your token.")
            return
        try:
            start = datetime.strptime(self.bt_from.get().strip(), "%Y-%m-%d").date()
            end = datetime.strptime(self.bt_to.get().strip(), "%Y-%m-%d").date()
            if start > end or end >= now_ist().date():
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid range", "Use YYYY-MM-DD, From <= To, and To before today.")
            return
        self.bt_btn.configure(state="disabled", text="Running…")
        report = lambda message: self.after(0, self._bt_progress, message)
        future = self.workers.submit(run_backtest, self.engine.broker, self.bt_db,
                                     replace(self.engine.settings), start, end, None, report)
        future.add_done_callback(lambda f: self.after(0, self._bt_done, f))

    def _bt_range(self):
        start = datetime.strptime(self.bt_from.get().strip(), "%Y-%m-%d").date()
        end = datetime.strptime(self.bt_to.get().strip(), "%Y-%m-%d").date()
        if start > end or end >= now_ist().date():
            raise ValueError("Use YYYY-MM-DD, From <= To, and To before today.")
        return start, end

    def open_sweep(self) -> None:
        if getattr(self, "sweep_win", None) and self.sweep_win.winfo_exists():
            self.sweep_win.focus()
            return
        win = self.sweep_win = ctk.CTkToplevel(self)
        win.title("Backtest Parameter Sweep")
        win.geometry("980x640")
        win.configure(fg_color=BG)
        form = ctk.CTkFrame(win, fg_color=PANEL, corner_radius=10)
        form.pack(fill="x", padx=10, pady=10)
        p = self.engine.settings.params
        defaults = {"EMA fast": f"5, {p.ema_fast}, 12", "EMA slow": f"{p.ema_slow}, 26, 34",
                    "SL × ATR": f"1.0, {p.sl_atr}, 2.0", "TP × ATR": f"2.0, {p.tp_atr}"}
        self.sweep_inputs = {}
        for col, (label, value) in enumerate(defaults.items()):
            ctk.CTkLabel(form, text=label).grid(row=0, column=col, padx=10, pady=(10, 0), sticky="w")
            entry = ctk.CTkEntry(form, width=170)
            entry.insert(0, value)
            entry.grid(row=1, column=col, padx=10, pady=(0, 10))
            self.sweep_inputs[label] = entry
        ctk.CTkLabel(form, text="Trend filters").grid(row=0, column=4, padx=10, pady=(10, 0), sticky="w")
        self.sweep_trends = {}
        box = ctk.CTkFrame(form, fg_color="transparent")
        box.grid(row=1, column=4, padx=10, sticky="w")
        for name in TREND_FILTERS:
            check = ctk.CTkCheckBox(box, text=name, width=60)
            if name in ("Off", p.trend_filter):
                check.select()
            check.pack(side="left")
            self.sweep_trends[name] = check
        controls = ctk.CTkFrame(win, fg_color="transparent")
        controls.pack(fill="x", padx=10)
        self.sweep_btn = ctk.CTkButton(controls, text="Run Sweep", fg_color=GREEN, command=self.run_sweep)
        self.sweep_btn.pack(side="left", padx=4)
        ctk.CTkButton(controls, text="Cancel", fg_color="#374151",
                      command=self.sweep_cancel.set).pack(side="left", padx=4)
        self.apply_best_btn = ctk.CTkButton(controls, text="Apply Best to Strategy", state="disabled",
                                            command=self.apply_best)
        self.apply_best_btn.pack(side="left", padx=4)
        self.sweep_status = ctk.CTkLabel(controls, text=f"Range from Backtest bar; max {SWEEP_LIMIT} "
                                                        "combinations. Same data and costs as Run Backtest.",
                                         text_color=MUTED)
        self.sweep_status.pack(side="left", padx=10)
        self.sweep_rows = ctk.CTkScrollableFrame(win, fg_color=PANEL)
        self.sweep_rows.pack(fill="both", expand=True, padx=10, pady=10)

    def run_sweep(self) -> None:
        if not self.engine.broker:
            messagebox.showerror("Not connected", "Authenticate with Upstox first.", parent=self.sweep_win)
            return
        try:
            start, end = self._bt_range()
            trends = [n for n, c in self.sweep_trends.items() if c.get()] or ["Off"]
            grid = sweep_grid(parse_list(self.sweep_inputs["EMA fast"].get(), int),
                              parse_list(self.sweep_inputs["EMA slow"].get(), int),
                              parse_list(self.sweep_inputs["SL × ATR"].get(), float),
                              parse_list(self.sweep_inputs["TP × ATR"].get(), float),
                              trends, self.engine.settings.params)
            if not grid:
                raise ValueError("No valid combinations (need EMA fast < EMA slow).")
        except ValueError as error:
            messagebox.showerror("Sweep", str(error), parent=self.sweep_win)
            return
        self.sweep_cancel.clear()
        self.sweep_btn.configure(state="disabled", text=f"Running {len(grid)}…")
        self.apply_best_btn.configure(state="disabled")
        settings = replace(self.engine.settings)
        report = lambda message: self.after(0, self.sweep_status.configure,
                                            {"text": message, "text_color": "white"})

        def work():
            try:
                result = run_sweep(self.engine.broker, settings, start, end, grid,
                                   progress=report, cancel=self.sweep_cancel)
                self.after(0, self._sweep_done, result, None)
            except Exception as error:
                self.after(0, self._sweep_done, None, error)

        threading.Thread(target=work, name="sweep", daemon=True).start()

    def _sweep_done(self, results, error) -> None:
        if not self.sweep_win.winfo_exists():
            return
        self.sweep_btn.configure(state="normal", text="Run Sweep")
        if error:
            self.sweep_status.configure(text=f"Sweep failed: {error}", text_color=RED)
            return
        self.sweep_results = results
        for child in self.sweep_rows.winfo_children():
            child.destroy()
        headers = ["#", "EMA", "SL×", "TP×", "Filter", "Trades", "Net ₹", "Sharpe", "PF", "Max DD", "W/L"]
        for col, text in enumerate(headers):
            ctk.CTkLabel(self.sweep_rows, text=text, text_color=MUTED, width=80).grid(row=0, column=col)
        for row, r in enumerate(results[:100], 1):
            p = r["params"]
            values = [row, f"{p.ema_fast}/{p.ema_slow}", p.sl_atr, p.tp_atr, p.trend_filter, r["trades"],
                      fmt(r["net"], 0), fmt(r["sharpe"]), fmt(r["profit_factor"]), fmt(r["max_dd"], suffix="%"),
                      fmt(r["win_loss"])]
            for col, value in enumerate(values):
                color = (GREEN if r["net"] > 0 else RED if r["net"] < 0 else "white") if col == 6 else "white"
                ctk.CTkLabel(self.sweep_rows, text=str(value), width=80, text_color=color).grid(row=row, column=col)
        done = "cancelled" if self.sweep_cancel.is_set() else "done"
        self.sweep_status.configure(text=f"Sweep {done}: {len(results)} combinations ranked by Sharpe, "
                                         "then profit factor. Few trades = unreliable.", text_color=GREEN)
        self.apply_best_btn.configure(state="normal" if results else "disabled")

    def apply_best(self) -> None:
        best = self.sweep_results[0]["params"]
        if not messagebox.askyesno("Apply best", f"Set EMA {best.ema_fast}/{best.ema_slow}, SL {best.sl_atr}×, "
                                                 f"TP {best.tp_atr}×, filter {best.trend_filter}?\n\n"
                                                 "Short-sample optimisation overfits easily; paper-trade "
                                                 "before using these live.", parent=self.sweep_win):
            return
        for name in ("ema_fast", "ema_slow", "sl_atr", "tp_atr"):
            self.param_entries[name].delete(0, "end")
            self.param_entries[name].insert(0, str(getattr(best, name)))
        self.trend_menu.set(best.trend_filter)
        self.apply_settings()

    def _bt_progress(self, message: str) -> None:
        self.bt_status.configure(text=message, text_color="white")
        self._log("INFO", f"[Backtest] {message}")

    def _bt_done(self, future) -> None:
        self.bt_btn.configure(state="normal", text="Run Backtest")
        try:
            notes = future.result()
        except Exception as error:
            self.bt_status.configure(text=f"Backtest failed: {error}", text_color=RED)
            self._log("ERROR", f"[Backtest] {error}")
            return
        trades = len([t for t in self.bt_db.trades() if t["status"] == "CLOSED"])
        self.bt_status.configure(text=f"Done: {len(notes)} days, {trades} trades", text_color=GREEN)
        self.ledger.set("Backtest")
        self.date_from.delete(0, "end")
        self.date_to.delete(0, "end")
        self.refresh_reports()

    def sort_by(self, key: str) -> None:
        self.sort_desc = not self.sort_desc if key == self.sort_key else True
        self.sort_key = key
        self.refresh_reports()

    def refresh_reports(self) -> None:
        try:
            trades = self.filtered()
        except ValueError:
            messagebox.showerror("Invalid date", "Use YYYY-MM-DD.")
            return
        for child in self.rows.winfo_children():
            child.destroy()
        trades.sort(key=lambda t: (t.get(self.sort_key) is None, t.get(self.sort_key) or 0),
                    reverse=self.sort_desc)
        for row_number, trade in enumerate(trades):
            color = GREEN if trade["net"] > 0 else RED if trade["net"] < 0 else "white"
            for column, (key, _, width) in enumerate(self.columns):
                value = trade.get(key)
                text = fmt(value) if isinstance(value, float) else str(value if value not in (None, "") else "—")
                label = ctk.CTkLabel(self.rows, text=text, width=width, anchor="w",
                                     text_color=color if key == "net" else "white", cursor="hand2")
                label.grid(row=row_number, column=column, padx=1, pady=1)
                label.bind("<Button-1>", lambda _e, t=trade: self.edit_journal(t))
        result = metrics(trades, self.engine.settings.capital)
        values = {"Trades": str(result["count"]), "Net P&L": money(result["net"]),
                  "Sharpe Ratio": fmt(result["sharpe"]), "Profit Factor": fmt(result["profit_factor"]),
                  "Max Drawdown": fmt(result["max_dd"], suffix="%"),
                  "Win/Loss Ratio": fmt(result["win_loss"]),
                  "Avg Duration": fmt(result["avg_duration"], 1, " min"),
                  "Total Slippage Drag": money(result["slippage"]),
                  "Total Taxes Paid": money(result["taxes"])}
        for name, text in values.items():
            color = (GREEN if result["net"] > 0 else RED if result["net"] < 0 else "white") \
                if name == "Net P&L" else "white"
            self.metric_labels[name].configure(text=text, text_color=color)

    def _history_text(self) -> str:
        rows = self.history.coverage()
        if not rows:
            return "Local history store: empty (fills automatically after each Upstox login)."
        return "Local history store: " + " · ".join(f"{i} {a} → {b} ({n} days)" for i, a, b, n in rows)

    def edit_journal(self, trade: dict) -> None:
        if not trade.get("id"):
            return
        win = ctk.CTkToplevel(self)
        win.title(f"Journal · {trade.get('symbol', '')} · {trade['day']}")
        win.geometry("520x380")
        win.transient(self)
        ctk.CTkLabel(win, text=f"{trade.get('exit_reason', '')} · net {money(trade.get('net', 0))} · "
                               f"MFE {trade.get('mfe_pts', '—')} / MAE {trade.get('mae_pts', '—')} pts",
                     text_color=MUTED).pack(anchor="w", padx=12, pady=(10, 4))
        tags = ctk.CTkEntry(win, placeholder_text="tags, comma separated (e.g. fomo, news, clean setup)")
        tags.insert(0, trade.get("tags", ""))
        tags.pack(fill="x", padx=12, pady=4)
        note = ctk.CTkTextbox(win, height=220)
        note.insert("1.0", trade.get("note", ""))
        note.pack(fill="both", expand=True, padx=12, pady=4)

        def save():
            target = self.report_db()
            target.set_journal(int(trade["id"]), note.get("1.0", "end"), tags.get())
            win.destroy()
            self.refresh_reports()

        ctk.CTkButton(win, text="Save", fg_color=GREEN, command=save).pack(pady=8)

    def show_calendar(self) -> None:
        try:
            pnl = daily_pnl(self.filtered())
        except ValueError:
            return
        if not pnl:
            messagebox.showinfo("P&L Calendar", "No closed trades in the current filter.")
            return
        months = sorted({(d.year, d.month) for d in pnl})[-6:]
        win = ctk.CTkToplevel(self)
        win.title("Daily P&L Calendar")
        win.geometry(f"{max(420, 330 * min(len(months), 3))}x{280 * ((len(months) + 2) // 3) + 40}")
        figure = Figure(figsize=(3.3 * min(len(months), 3), 2.8 * ((len(months) + 2) // 3)), dpi=100,
                        facecolor=PANEL)
        scale = max(abs(v) for v in pnl.values()) or 1
        for number, (year, month) in enumerate(months):
            axis = figure.add_subplot((len(months) + 2) // 3, min(len(months), 3), number + 1)
            axis.set_facecolor(BG)
            axis.set_title(datetime(year, month, 1).strftime("%b %Y"), color="white", fontsize=9)
            axis.set_xlim(-0.5, 4.5)
            axis.set_ylim(5.5, -0.5)
            axis.set_xticks(range(5), ["M", "T", "W", "T", "F"], color=MUTED, fontsize=7)
            axis.set_yticks([])
            first = datetime(year, month, 1).date()
            for day, value in pnl.items():
                if (day.year, day.month) != (year, month) or day.weekday() > 4:
                    continue
                week = (day.day + first.weekday() - 1) // 7
                shade = min(abs(value) / scale, 1) * 0.85 + 0.15
                axis.add_patch(matplotlib.patches.Rectangle(
                    (day.weekday() - 0.45, week - 0.45), 0.9, 0.9,
                    color=GREEN if value > 0 else RED if value < 0 else MUTED, alpha=shade))
                axis.text(day.weekday(), week, f"{day.day}\n{value / 1000:+.1f}k", ha="center",
                          va="center", fontsize=6, color="white")
        figure.tight_layout()
        FigureCanvasTkAgg(figure, master=win).get_tk_widget().pack(fill="both", expand=True)

    def show_slippage(self) -> None:
        report = slippage_report(self.db.trades())
        assumed = self.engine.settings.params.slippage_pct
        lines = [f"Paper assumption: {assumed:g}% adverse per side.\n"]
        for mode, row in report.items():
            lines.append(f"{mode}: {row['trades']} trades · entry {fmt(row['entry_pct'], 3, '%')} · "
                         f"exit {fmt(row['exit_pct'], 3, '%')} (vs reference LTP)")
        live = report.get("LIVE", {})
        if live.get("entry_pct") is not None and live.get("exit_pct") is not None:
            actual = (live["entry_pct"] + live["exit_pct"]) / 2
            lines.append(f"\nLive average {actual:.3f}% vs paper {assumed:g}% — "
                         + ("paper is conservative." if actual <= assumed else
                            "consider raising the paper slippage setting."))
        else:
            lines.append("\nNo live fills yet; comparison appears after live trades.")
        messagebox.showinfo("Slippage: paper vs live", "\n".join(lines))

    def fy_tax_report(self) -> None:
        trades = self.report_db().trades()
        years = sorted({financial_year(t["day"]) for t in trades if t.get("status") == "CLOSED"})
        if not years:
            messagebox.showinfo("FY Tax Report", "No closed trades yet.")
            return
        report = fy_report(trades, years[-1])
        path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile=f"StratTerminal_{years[-1]}",
                                            filetypes=[("CSV", "*.csv")])
        lines = [f"{report['fy']} · {report['trades']} trades ({report['wins']} wins / {report['losses']} losses)",
                 f"Gross P&L {money(report['gross_pnl'])} · charges {money(report['total_charges'])} · "
                 f"net {money(report['net_pnl'])}",
                 f"Sum of |trade P&L| {money(report['abs_pnl_turnover'])} · sell premium value "
                 f"{money(report['sell_premium_value'])}",
                 "Charges: " + ", ".join(f"{k} {money(v)}" for k, v in report["charges"].items()),
                 "\nFigures for your tax adviser; turnover rules for options are their call."]
        if path:
            import csv as _csv
            with open(path, "w", newline="", encoding="utf-8-sig") as handle:
                writer = _csv.writer(handle)
                for key, value in report.items():
                    if key == "charges":
                        for name, amount in value.items():
                            writer.writerow([f"charge_{name}", round(amount, 2)])
                    else:
                        writer.writerow([key, round(value, 2) if isinstance(value, float) else value])
                writer.writerow([])
                writer.writerow(["day", "symbol", "quantity", "entry_premium", "exit_premium", "gross", "taxes", "net"])
                for t in trades:
                    if t.get("status") == "CLOSED" and financial_year(t["day"]) == report["fy"]:
                        writer.writerow([t["day"], t.get("symbol"), t.get("quantity"), t.get("entry_premium"),
                                         t.get("exit_premium"), round(t["gross"], 2), round(t["taxes"], 2),
                                         round(t["net"], 2)])
            lines.append(f"\nSaved {path}")
        messagebox.showinfo("FY Tax Report", "\n".join(lines))

    def export(self) -> None:
        try:
            trades = self.filtered()
        except ValueError:
            messagebox.showerror("Invalid date", "Use YYYY-MM-DD.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf", initialfile=f"StratDossier_{now_ist():%Y%m%d}",
            filetypes=[("PDF dossier", "*.pdf"), ("CSV ledger", "*.csv")])
        if not path:
            return
        try:
            if path.lower().endswith(".csv"):
                export_csv(path, trades)
            else:
                build_pdf(path, trades, metrics(trades, self.engine.settings.capital))
                export_csv(Path(path).with_suffix(".csv"), trades)
            messagebox.showinfo("Exported", f"Saved {path}")
        except Exception as error:
            messagebox.showerror("Export failed", str(error))

    # ================= Polling =================
    def _log(self, level: str, message: str, stamp: str | None = None) -> None:
        stamp = stamp or now_ist().strftime("%H:%M:%S")
        self.log.insert("end", f"{stamp} {level:<8} {message}\n", level)
        self.log.see("end")

    def poll(self) -> None:
        try:
            trade_event = False
            while True:
                level, message = self.engine.events.get_nowait()
                self._log(level, message)
                trade_event |= level == "TRADE"
                if level in ("TRADE", "CRITICAL"):
                    self.alert(level, message)
                elif level == "WARN" and message.startswith("Channel ") and "RECONNECT" in message:
                    self.alert("FEED", message)
        except queue.Empty:
            pass
        try:
            snap = self.engine.snapshot()
            self.update_dashboard(snap)
            self.draw_charts(snap)
            if trade_event:
                self.refresh_reports()
        except Exception as error:
            self._log("ERROR", f"UI refresh: {error}")
        self.after(500, self.poll)

    def update_dashboard(self, snap) -> None:
        live = snap["mode"] == "LIVE"
        self.paper_bulb.configure(text_color="#374151" if live else GREEN)
        self.live_bulb.configure(text_color=RED if live and snap["live_armed"] else "#374151")
        color = RED if snap["halted"] else AMBER if snap["blocked"] else GREEN
        self.banner.configure(text=snap["banner"], fg_color=color,
                              text_color="black" if color == AMBER else "white")
        if snap["halted"]:
            self.clear_btn.pack(side="right", padx=8)
        else:
            self.clear_btn.pack_forget()
        if "circuit breaker" in snap["banner"]:
            self.breaker_btn.pack(side="right", padx=8)
        else:
            self.breaker_btn.pack_forget()
        self.feed_label.configure(text="   ".join(f"{k}: {v}" for k, v in snap["feeds"].items()))
        trade, spot = snap["trade"], snap["spot"]
        pnl, est_charges = self.engine.unrealized_net()
        realized = metrics(self.db.trades(mode=snap["mode"]), snap["settings"].capital)
        today = [t for t in self.db.trades(now_ist().date().isoformat(), now_ist().date().isoformat(),
                                           mode=snap["mode"]) if t["status"] == "CLOSED"]
        day_net = sum(t["net"] for t in today) + pnl
        self.cards["Live Net P&L"].set(money(day_net), GREEN if day_net > 0 else RED if day_net < 0 else "white")
        self.cards["Account Value"].set(money(realized["equity"] + pnl))
        self.cards["Active Strike"].set(trade["symbol"] if trade else "—")
        self.cards["Entry Premium"].set(money(trade["entry_premium"]) if trade else "—")
        self.cards["Spot Index"].set(fmt(spot) if spot else "—")
        self.manual_btn.configure(state="normal" if trade else "disabled")
        if trade and spot:
            entry, levels = trade["entry_spot"], trade["levels"]
            self.tp_gauge.set((spot - entry) / (levels["target"] - entry),
                              f"{levels['target']:.2f} ({abs(levels['target'] - spot):.1f} pts)")
            self.sl_gauge.set((spot - entry) / (levels["stop"] - entry),
                              f"{levels['stop']:.2f} ({abs(spot - levels['stop']):.1f} pts)")
            backstop = " · broker backstop ON" if trade.get("backstop") else ""
            self.levels.configure(text=f"{trade['side']} · entry spot {entry:.2f}\n"
                                       f"ATR {trade['atr']:.2f} · qty {trade['quantity']} · "
                                       f"exp {trade['expiry']}\n"
                                       f"Open P&L is net of est. charges ₹{est_charges:,.2f}{backstop}",
                                  text_color="white")
        else:
            self.tp_gauge.set(0, "")
            self.sl_gauge.set(0, "")
            self.levels.configure(text="No active position", text_color=MUTED)

    def on_close(self) -> None:
        if self.engine.state.trade and not messagebox.askyesno(
                "Position open", "A position is open. Closing stops spot-exit monitoring until you "
                                 "restart. The 15:15 square-off will NOT run while closed. Exit anyway?"):
            return
        self.sweep_cancel.set()
        self.engine.shutdown()
        self.workers.shutdown(wait=False, cancel_futures=True)
        self.destroy()
