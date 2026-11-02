"""Official Upstox SDK adapter; no GUI calls or automatic order retries."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import secrets
import ssl
import threading
import time
import webbrowser
from datetime import date, datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import keyring
import pandas as pd
import requests
import truststore
import upstox_client as sdk

from database import IST, now_ist

SPOT_KEYS = {"NIFTY": "NSE_INDEX|Nifty 50", "BANKNIFTY": "NSE_INDEX|Nifty Bank",
             "SENSEX": "BSE_INDEX|SENSEX"}
STRIKE_STEP = {"NIFTY": 50, "BANKNIFTY": 100, "SENSEX": 100}
EXCHANGE = {"NIFTY": "NSE", "BANKNIFTY": "NSE", "SENSEX": "BSE"}
FO_SEGMENT = {"NSE": "NSE_FO", "BSE": "BSE_FO"}
SERVICE = "StratTerminal.Upstox"
TERMINAL = {"complete", "rejected", "cancelled"}
HISTORY_BARS = 2000


def trust_context() -> ssl.SSLContext:
    """Verified TLS against the OS trust store (works behind corporate TLS-inspection proxies)."""
    context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    return context


def _enforce_tls() -> None:
    """REST: verify via OS store. Websockets: the SDK passes CERT_NONE with the bearer token; override."""
    import websocket

    truststore.inject_into_ssl()
    base = websocket.WebSocketApp
    if getattr(base, "_tls_enforced", False):
        return

    class VerifiedWebSocketApp(base):
        _tls_enforced = True

        def run_forever(self, *args, **kwargs):
            kwargs["sslopt"] = {"context": trust_context(), "cert_reqs": ssl.CERT_REQUIRED,
                                "check_hostname": True}
            return super().run_forever(*args, **kwargs)

    websocket.WebSocketApp = VerifiedWebSocketApp


_enforce_tls()


class BrokerError(RuntimeError):
    pass


class TokenVault:
    @staticmethod
    def account(api_key: str) -> str:
        return hashlib.sha256(api_key.encode()).hexdigest()[:24]

    def store(self, api_key: str, token: str) -> None:
        if keyring.get_keyring().priority <= 0:
            raise BrokerError("A secure OS credential vault is required for token caching.")
        keyring.set_password(SERVICE, self.account(api_key), json.dumps(
            {"day": now_ist().date().isoformat(), "token": token}))

    def load(self, api_key: str) -> str | None:
        if not api_key:
            return None
        raw = keyring.get_password(SERVICE, self.account(api_key))
        if raw:
            saved = json.loads(raw)
            if saved["day"] == now_ist().date().isoformat():
                return saved["token"]
        return None


class OAuthLogin:
    def authenticate(self, api_key: str, secret: str, redirect: str) -> str:
        parsed = urlparse(redirect)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
            raise BrokerError("Use a registered loopback HTTP redirect, e.g. http://127.0.0.1:8765/callback")
        if not parsed.port:
            raise BrokerError("The registered redirect must include a local port.")
        state = secrets.token_urlsafe(32)
        outcome: dict[str, str] = {}

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                query = parse_qs(urlparse(self.path).query)
                valid = (urlparse(self.path).path == parsed.path
                         and secrets.compare_digest(query.get("state", [""])[0], state))
                if valid:
                    outcome["code"] = query.get("code", [""])[0]
                    outcome["error"] = query.get("error", [""])[0]
                self.send_response(200 if valid else 400)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"Authentication received. Return to Strat Terminal."
                                 if valid else b"Invalid authentication state.")

            def log_message(self, *_args):
                pass

        server = HTTPServer((parsed.hostname, parsed.port), Handler)
        server.timeout = 1
        try:
            webbrowser.open("https://api.upstox.com/v2/login/authorization/dialog?" + urlencode(
                dict(response_type="code", client_id=api_key, redirect_uri=redirect, state=state)))
            deadline = time.monotonic() + 180
            while not outcome and time.monotonic() < deadline:
                server.handle_request()
            if not outcome.get("code"):
                raise BrokerError("Authentication cancelled, denied, or timed out.")
            result = sdk.LoginApi().token(
                "2.0", code=outcome["code"], client_id=api_key, client_secret=secret,
                redirect_uri=redirect, grant_type="authorization_code", _request_timeout=15)
            TokenVault().store(api_key, result.access_token)
            return result.access_token
        finally:
            server.server_close()


def round_strike(price: float, index: str) -> int:
    step = STRIKE_STEP[index]
    return int(math.floor(price / step + 0.5) * step)


def pick_expiry(expiries, today: date, skip_today: bool = False) -> date | None:
    upcoming = sorted(e for e in expiries if e > today or (e == today and not skip_today))
    return upcoming[0] if upcoming else None


class InstrumentMaster:
    def __init__(self, cache: Path) -> None:
        self.cache = cache
        self.records: list[dict] = []
        self.loaded_day: date | None = None

    def _load(self, exchange: str, today: date, force: bool) -> list[dict]:
        path = self.cache / f"{exchange}-{today}.json.gz"
        if force or not path.exists():
            response = requests.get(
                f"https://assets.upstox.com/market-quote/instruments/exchange/{exchange}.json.gz",
                timeout=60)
            response.raise_for_status()
            records = json.loads(gzip.decompress(response.content))
            if not isinstance(records, list) or not records:
                raise BrokerError(f"{exchange} instrument master is empty or malformed.")
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(response.content)
            temporary.replace(path)
            return records
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)

    def refresh(self, force: bool = False) -> None:
        today = now_ist().date()
        self.cache.mkdir(parents=True, exist_ok=True)
        records = []
        for exchange in FO_SEGMENT:
            records += [r for r in self._load(exchange, today, force)
                        if r.get("segment") in FO_SEGMENT.values()
                        and r.get("underlying_key") in SPOT_KEYS.values()]
        self.records = records
        self.loaded_day = today

    def expiries(self, index: str) -> list[date]:
        return sorted({datetime.fromtimestamp(int(r["expiry"]) / 1000, IST).date()
                       for r in self.records if r.get("underlying_key") == SPOT_KEYS[index]
                       and r.get("instrument_type") in ("CE", "PE")})

    def nearest_expiry(self, index: str, today: date | None = None, skip_today: bool = False):
        return pick_expiry(self.expiries(index), today or now_ist().date(), skip_today)

    def option(self, index: str, spot: float, side: str,
               today: date | None = None, skip_today: bool = False) -> dict:
        today = today or now_ist().date()
        strike = round_strike(spot, index)
        expiry = self.nearest_expiry(index, today, skip_today)
        if expiry is None:
            raise BrokerError(f"No upcoming {index} option expiry in the instrument master.")
        segment = FO_SEGMENT[EXCHANGE[index]]
        for record in self.records:
            if (record.get("segment") == segment
                    and record.get("instrument_type") == side
                    and record.get("underlying_key") == SPOT_KEYS[index]
                    and float(record.get("strike_price", 0)) == strike
                    and datetime.fromtimestamp(int(record["expiry"]) / 1000, IST).date() == expiry):
                return dict(key=record["instrument_key"], symbol=record["trading_symbol"],
                            lot_size=int(record["lot_size"]), strike=strike, side=side,
                            expiry=expiry.isoformat(), freeze=int(record.get("freeze_quantity", 0)),
                            exchange=EXCHANGE[index])
        raise BrokerError(f"No listed {index} {strike} {side} for expiry {expiry}. Entry blocked.")


def rows_to_frame(rows: list) -> pd.DataFrame:
    columns = ["time", "open", "high", "low", "close", "volume"]
    if not rows:
        return pd.DataFrame(columns=columns[1:], index=pd.DatetimeIndex([], tz=IST), dtype=float)
    frame = pd.DataFrame([row[:6] for row in rows], columns=columns)
    frame["time"] = pd.to_datetime(frame["time"], utc=True).dt.tz_convert(IST)
    frame = frame.drop_duplicates("time").set_index("time").sort_index()
    return frame.between_time("09:15", "15:29").astype(float)


def resample_bars(frame: pd.DataFrame, interval: int) -> pd.DataFrame:
    bars = frame.resample(f"{interval}min", origin="start_day", offset="15min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return bars.dropna(subset=["close"])


def aggregate_candles(rows: list, interval: int) -> pd.DataFrame:
    return resample_bars(rows_to_frame(rows), interval)


class FeedChannel:
    watchdog: float | None = 35

    def __init__(self, api, keys: list[str], name: str, on_tick, on_status) -> None:
        self.api, self.keys, self.name = api, keys, name
        self.on_tick, self.on_status = on_tick, on_status
        self.stop_event = threading.Event()
        self.failed = threading.Event()
        self.stream = None
        self.connected = False
        self.last_packet = time.monotonic()
        self.thread = threading.Thread(target=self._run, name=f"feed-{name}", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _open(self, *_args) -> None:
        self.connected = True
        self.last_packet = time.monotonic()
        self.on_status(self.name, "CONNECTED")

    def _failure(self, *_args) -> None:
        self.connected = False
        self.failed.set()

    def _message(self, message: dict) -> None:
        self.last_packet = time.monotonic()
        for key, feed in message.get("feeds", {}).items():
            quote = feed.get("ltpc", {})
            price = float(quote.get("ltp", 0))
            timestamp = int(quote.get("ltt", 0))
            if price > 0 and timestamp:
                self.on_tick(key, price, datetime.fromtimestamp(timestamp / 1000, IST))

    def _make_stream(self):
        return sdk.MarketDataStreamerV3(self.api, self.keys, "ltpc")

    def _run(self) -> None:
        attempt = 0
        while not self.stop_event.is_set():
            self.failed.clear()
            opened_at = time.monotonic()
            try:
                self.stream = self._make_stream()
                self.stream.auto_reconnect(False)
                self.stream.on("open", self._open)
                self.stream.on("message", self._message)
                self.stream.on("close", self._failure)
                self.stream.on("error", self._failure)
                self.stream.connect()
                while not self.stop_event.wait(0.5) and not self.failed.is_set():
                    if self.watchdog and time.monotonic() - self.last_packet > self.watchdog:
                        break
            except Exception:
                self.on_status(self.name, "CONNECTION ERROR")
            finally:
                self.connected = False
                if self.stream:
                    try:
                        self.stream.disconnect()
                    except Exception:
                        pass
            if self.stop_event.is_set():
                break
            attempt = 0 if time.monotonic() - opened_at > 60 else attempt + 1
            delay = min(30, 0.5 * 2 ** min(attempt, 6))
            self.on_status(self.name, f"RECONNECT {delay:g}s")
            self.stop_event.wait(delay)

    def stop(self) -> None:
        self.stop_event.set()
        if self.stream:
            self.stream.disconnect()


class OrderStream(FeedChannel):
    """Pushes order updates; the portfolio socket is silent when idle, so no packet watchdog."""
    watchdog = None

    def __init__(self, api, on_update, on_status) -> None:
        super().__init__(api, [], "ORDERS", None, on_status)
        self.on_update = on_update

    def _make_stream(self):
        return sdk.PortfolioDataStreamer(self.api, order_update=True)

    def _message(self, message) -> None:
        self.last_packet = time.monotonic()
        try:
            data = json.loads(message) if isinstance(message, (str, bytes)) else message
        except ValueError:
            return
        if isinstance(data, dict) and data.get("update_type") == "order" and data.get("order_id"):
            self.on_update(data)


class UpstoxBroker:
    def __init__(self, token: str, cache: Path) -> None:
        configuration = sdk.Configuration()
        configuration.access_token = token
        self.api = sdk.ApiClient(configuration)
        self.history = sdk.HistoryApi(self.api)
        self.orders = sdk.OrderApi(self.api)
        self.orders_v3 = sdk.OrderApiV3(self.api)
        self.master = InstrumentMaster(cache)
        self.channels: dict[str, FeedChannel] = {}
        self._history: dict[tuple[str, date], list] = {}
        self.order_updates: dict[str, dict] = {}
        self._order_cv = threading.Condition()
        self.order_listener = None
        self.order_stream: OrderStream | None = None

    def validate(self) -> None:
        sdk.UserApi(self.api).get_profile("2.0", _request_timeout=10)

    # ---------- order updates ----------
    def subscribe_orders(self, listener, on_status) -> None:
        self.order_listener = listener
        if self.order_stream is None:
            self.order_stream = OrderStream(self.api, self._on_order_update, on_status)
            self.order_stream.start()

    def _on_order_update(self, update: dict) -> None:
        with self._order_cv:
            self.order_updates[str(update["order_id"])] = update
            self._order_cv.notify_all()
        if self.order_listener:
            self.order_listener(update)

    def wait_order_update(self, order_id: str, timeout: float) -> None:
        with self._order_cv:
            self._order_cv.wait_for(lambda: str(self.order_updates.get(order_id, {}).get(
                "status", "")).lower() in TERMINAL, timeout=timeout)

    def minute_rows(self, index: str) -> list:
        return list(self.history.get_intra_day_candle_data(
            SPOT_KEYS[index], "1minute", "2.0", _request_timeout=15).data.candles or [])

    def candles(self, index: str, interval: int) -> pd.DataFrame:
        today = now_ist().date()
        if (index, today) not in self._history:
            self._history[(index, today)] = list(self.history.get_historical_candle_data1(
                SPOT_KEYS[index], "1minute", (today - timedelta(days=1)).isoformat(),
                (today - timedelta(days=27)).isoformat(), "2.0",
                _request_timeout=25).data.candles or [])
        intraday = self.minute_rows(index)
        frame = aggregate_candles(self._history[(index, today)] + intraday, interval)
        cutoff = now_ist()
        if intraday:
            last_minute = max(pd.Timestamp(row[0]).tz_convert(IST) for row in intraday)
            cutoff = min(cutoff, last_minute + pd.Timedelta(minutes=1))
        frame = frame[frame.index + pd.Timedelta(minutes=interval) <= cutoff]
        return frame.tail(HISTORY_BARS)

    def quote(self, key: str) -> tuple[float, datetime]:
        response = sdk.MarketQuoteApi(self.api).get_full_market_quote(
            key, "2.0", _request_timeout=6)
        quote = next(iter(response.data.values()))
        price = float(quote.last_price)
        timestamp = str(getattr(quote, "last_trade_time", "") or "")
        if not timestamp:
            raise BrokerError("Quote has no exchange trade timestamp.")
        when = (datetime.fromtimestamp(int(timestamp) / 1000, IST) if timestamp.isdigit()
                else datetime.fromisoformat(timestamp).astimezone(IST))
        if price <= 0 or abs((now_ist() - when).total_seconds()) > 10:
            raise BrokerError("Stale or invalid option quote; paper fill blocked.")
        return price, when

    def depth(self, key: str) -> dict:
        """Best bid/ask/LTP for pre-entry liquidity checks."""
        response = sdk.MarketQuoteApi(self.api).get_full_market_quote(key, "2.0", _request_timeout=6)
        quote = next(iter(response.data.values()))
        best = lambda side: next((d.price for d in (side or []) if d.price and d.quantity), 0.0)
        return dict(bid=float(best(quote.depth.buy if quote.depth else None)),
                    ask=float(best(quote.depth.sell if quote.depth else None)),
                    ltp=float(quote.last_price or 0))

    def holidays(self) -> dict[date, list[str]]:
        """Trading holidays this year -> closed exchanges (Upstox holidays API)."""
        if getattr(self, "_holidays", None) is None:
            rows = sdk.MarketHolidaysAndTimingsApi(self.api).get_holidays(_request_timeout=10).data or []
            self._holidays = {}
            for row in rows:
                if str(row.holiday_type).upper().startswith("TRADING"):
                    day = row._date.date() if hasattr(row._date, "date") else date.fromisoformat(str(row._date)[:10])
                    self._holidays[day] = list(row.closed_exchanges or [])
        return self._holidays

    def market_order(self, key: str, quantity: int, side: str, tag: str) -> str:
        body = sdk.PlaceOrderV3Request(
            quantity=quantity, product="I", validity="DAY", price=0, tag=tag,
            instrument_token=key, order_type="MARKET", transaction_type=side,
            disclosed_quantity=0, trigger_price=0, is_amo=False, slice=False,
            market_protection=-1)
        ids = self.orders_v3.place_order(body, _request_timeout=10).data.order_ids or []
        if len(ids) != 1:
            raise BrokerError(f"Expected one order id, broker returned {ids}.")
        return str(ids[0])

    def stop_loss_order(self, key: str, quantity: int, trigger: float, limit: float, tag: str) -> str:
        """Broker-held SELL stop-loss-limit (SL-M is not allowed on NSE options)."""
        body = sdk.PlaceOrderV3Request(
            quantity=quantity, product="I", validity="DAY", price=limit, tag=tag,
            instrument_token=key, order_type="SL", transaction_type="SELL",
            disclosed_quantity=0, trigger_price=trigger, is_amo=False, slice=False)
        ids = self.orders_v3.place_order(body, _request_timeout=10).data.order_ids or []
        if len(ids) != 1:
            raise BrokerError(f"Expected one order id, broker returned {ids}.")
        return str(ids[0])

    def order_book(self) -> list[dict]:
        return [order.to_dict() for order in
                self.orders.get_order_book("2.0", _request_timeout=8).data]

    def order_status(self, order_id: str) -> dict:
        cached = self.order_updates.get(order_id)
        if cached and str(cached.get("status", "")).lower() in TERMINAL:
            return dict(cached)
        rows = self.orders.get_order_details("2.0", order_id=order_id, _request_timeout=8).data
        if not rows:
            raise BrokerError("Broker has not returned the order yet.")
        return rows[-1].to_dict()

    def cancel(self, order_id: str) -> None:
        self.orders_v3.cancel_order(order_id, _request_timeout=8)

    def positions(self) -> list[dict]:
        return [position.to_dict() for position in sdk.PortfolioApi(self.api)
                .get_positions("2.0", _request_timeout=8).data]

    # ---------- backtest data ----------
    def spot_minutes(self, index: str, start: date, end: date) -> pd.DataFrame:
        """1-minute spot history; uses the local store for days already saved, downloads the rest."""
        store = getattr(self, "history_store", None)
        if store is not None:
            have = store.days(index, start, end)
            wanted = [d.date() for d in pd.bdate_range(start, min(end, now_ist().date() - timedelta(days=1)))]
            missing = [d for d in wanted if d not in have]
            if missing:
                try:
                    store.save(index, self._download_minutes(index, min(missing), max(missing)))
                except Exception:
                    if not have:
                        raise
            return store.load(index, start, end)
        return self._download_minutes(index, start, end)

    def _download_minutes(self, index: str, start: date, end: date) -> pd.DataFrame:
        rows = self.history.get_historical_candle_data1(
            SPOT_KEYS[index], "1minute", end.isoformat(), start.isoformat(), "2.0",
            _request_timeout=30).data.candles or []
        return rows_to_frame(list(rows))

    def _bt_expiries(self, index: str) -> tuple[set, list]:
        cache = self.__dict__.setdefault("_bt", {})
        if ("expired", index) not in cache:
            try:
                expired = sdk.ExpiredInstrumentApi(self.api).get_expiries(
                    SPOT_KEYS[index], _request_timeout=15).data or []
            except Exception as error:
                raise BrokerError(f"Expired-contract data unavailable on this account: {error}")
            cache[("expired", index)] = {date.fromisoformat(str(d)[:10]) for d in expired}
            cache[("active", index)] = sdk.OptionsApi(self.api).get_option_contracts(
                SPOT_KEYS[index], _request_timeout=20).data or []
        return cache[("expired", index)], cache[("active", index)]

    def backtest_expiry(self, index: str, day: date, skip_today: bool = False) -> date:
        expired, active = self._bt_expiries(index)
        expiry = pick_expiry(expired | {c.expiry.astimezone(IST).date() for c in active}, day, skip_today)
        if expiry is None:
            raise BrokerError(f"No {index} option expiry found on/after {day}.")
        return expiry

    def backtest_option(self, index: str, day: date, side: str, strike: int, skip_today: bool = False):
        """Nearest-expiry ATM contract as listed on `day`, plus its 1-minute candles that day."""
        cache = self.__dict__.setdefault("_bt", {})
        expired, active = self._bt_expiries(index)
        expiry = self.backtest_expiry(index, day, skip_today)
        if expiry in expired:
            if ("contracts", index, expiry) not in cache:
                cache[("contracts", index, expiry)] = sdk.ExpiredInstrumentApi(
                    self.api).get_expired_option_contracts(
                    SPOT_KEYS[index], expiry.isoformat(), _request_timeout=20).data or []
            contracts = cache[("contracts", index, expiry)]
        else:
            contracts = [c for c in active if c.expiry.astimezone(IST).date() == expiry]
        match = [c for c in contracts if c.instrument_type == side
                 and float(c.strike_price) == strike]
        if not match:
            raise BrokerError(f"{index} {strike} {side} exp {expiry} not listed.")
        contract = match[0]
        if expiry in expired:
            rows = sdk.ExpiredInstrumentApi(self.api).get_expired_historical_candle_data(
                contract.instrument_key, "1minute", day.isoformat(), day.isoformat(),
                _request_timeout=20).data.candles or []
        else:
            rows = self.history.get_historical_candle_data1(
                contract.instrument_key, "1minute", day.isoformat(), day.isoformat(), "2.0",
                _request_timeout=20).data.candles or []
        info = dict(key=contract.instrument_key, symbol=contract.trading_symbol,
                    lot_size=int(contract.lot_size), strike=strike, side=side,
                    expiry=expiry.isoformat(), freeze=int(contract.freeze_quantity or 0))
        return info, rows_to_frame(list(rows))

    def stream(self, name: str, keys: list[str], on_tick, on_status) -> None:
        if name in self.channels:
            self.channels[name].stop()
        channel = FeedChannel(self.api, keys, name, on_tick, on_status)
        self.channels[name] = channel
        channel.start()

    def close(self) -> None:
        for channel in self.channels.values():
            channel.stop()
        if self.order_stream:
            self.order_stream.stop()