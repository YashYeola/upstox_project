"""Component tests for the broker adapter: feed reconnects, OAuth state check, token vault."""

import json
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
from datetime import timedelta
from unittest import mock

import upstox_broker
from database import now_ist
from upstox_broker import BrokerError, FeedChannel, OAuthLogin, TokenVault


class FakeStreamer:
    instances = []

    def __init__(self, api, keys, mode):
        self.handlers, self.keys, self.mode = {}, keys, mode
        FakeStreamer.instances.append(self)

    def auto_reconnect(self, enabled, *args):
        self.auto = enabled

    def on(self, event, handler):
        self.handlers[event] = handler

    def connect(self):
        self.handlers["open"]()
        stamp = str(int(now_ist().timestamp() * 1000))
        self.handlers["message"]({"feeds": {self.keys[0]: {"ltpc": {"ltp": 24510.5, "ltt": stamp}}}})
        self.handlers["message"]({"feeds": {self.keys[0]: {"ltpc": {"ltp": 0, "ltt": stamp}}}})
        self.handlers["close"]()

    def disconnect(self):
        pass


class FeedChannelTests(unittest.TestCase):
    def test_reconnects_with_exponential_backoff_and_parses_ticks(self):
        FakeStreamer.instances.clear()
        ticks, statuses = [], []
        with mock.patch.object(upstox_broker.sdk, "MarketDataStreamerV3", FakeStreamer):
            channel = FeedChannel(None, ["NSE_INDEX|Nifty 50"], "A-SPOT",
                                  lambda *a: ticks.append(a), lambda n, s: statuses.append(s))
            channel.start()
            deadline = time.monotonic() + 8
            while len(FakeStreamer.instances) < 3 and time.monotonic() < deadline:
                time.sleep(0.05)
            channel.stop()
            channel.thread.join(timeout=3)
        self.assertGreaterEqual(len(FakeStreamer.instances), 3)
        self.assertTrue(all(s.auto is False for s in FakeStreamer.instances))
        self.assertTrue(all(s.mode == "ltpc" for s in FakeStreamer.instances))
        reconnects = [s for s in statuses if s.startswith("RECONNECT")]
        self.assertEqual(reconnects[:2], ["RECONNECT 1s", "RECONNECT 2s"])
        self.assertEqual(len(ticks), len(FakeStreamer.instances), "zero-price ticks must be dropped")
        self.assertEqual(ticks[0][:2], ("NSE_INDEX|Nifty 50", 24510.5))
        self.assertFalse(channel.thread.is_alive())


class OAuthTests(unittest.TestCase):
    def test_rejects_non_loopback_or_portless_redirects(self):
        for uri in ("https://example.com/cb", "http://10.0.0.5:8765/cb", "http://127.0.0.1/cb"):
            with self.assertRaises(BrokerError):
                OAuthLogin().authenticate("key", "secret", uri)

    def test_state_mismatch_is_rejected_and_valid_code_exchanged(self):
        redirect = "http://127.0.0.1:8799/callback"
        results, workers = {}, []

        def browser(url):
            state = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["state"][0]

            def hit():
                time.sleep(0.3)
                try:
                    urllib.request.urlopen(f"{redirect}?code=evil&state=forged", timeout=3)
                except urllib.error.HTTPError as error:
                    results["forged"] = error.code
                results["good"] = urllib.request.urlopen(
                    f"{redirect}?code=real&state={state}", timeout=3).status
            worker = threading.Thread(target=hit, daemon=True)
            workers.append(worker)
            worker.start()
            return True

        token_api = mock.Mock()
        token_api.return_value.token.return_value = mock.Mock(access_token="TOKEN-123")
        with mock.patch.object(upstox_broker.webbrowser, "open", browser), \
                mock.patch.object(upstox_broker.sdk, "LoginApi", token_api), \
                mock.patch.object(TokenVault, "store") as store:
            token = OAuthLogin().authenticate("key", "secret", redirect)
        workers[0].join(timeout=5)
        self.assertEqual(token, "TOKEN-123")
        self.assertEqual(results, {"forged": 400, "good": 200})
        self.assertEqual(token_api.return_value.token.call_args.kwargs["code"], "real")
        store.assert_called_once_with("key", "TOKEN-123")


class TokenVaultTests(unittest.TestCase):
    def test_only_same_day_token_is_reused(self):
        today = now_ist().date()
        store = {}
        with mock.patch.object(upstox_broker.keyring, "get_password",
                               lambda s, a: store.get(a)):
            account = TokenVault.account("key")
            store[account] = json.dumps({"day": today.isoformat(), "token": "T1"})
            self.assertEqual(TokenVault().load("key"), "T1")
            store[account] = json.dumps({"day": (today - timedelta(days=1)).isoformat(), "token": "OLD"})
            self.assertIsNone(TokenVault().load("key"))
            self.assertIsNone(TokenVault().load(""))
        self.assertNotIn("key", account)


if __name__ == "__main__":
    unittest.main()
