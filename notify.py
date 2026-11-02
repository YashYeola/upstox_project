"""Remote alerts (Telegram / email) and Windows auto-start scheduling. Never blocks trading."""

from __future__ import annotations

import json
import logging
import queue
import smtplib
import subprocess
import sys
import threading
import time
from email.message import EmailMessage
from pathlib import Path

import keyring
import requests

SERVICE = "StratTerminal.Notify"
TASK_NAME = "StratTerminal AutoStart"
LOG = logging.getLogger("strat.notify")


class Notifier:
    """Queue-backed sender; drops messages when the queue is full instead of blocking callers."""

    def __init__(self, sender=None) -> None:
        self.config: dict = {}
        self.queue: queue.Queue = queue.Queue(maxsize=200)
        self.sender = sender or self._send
        self.sent = self.failed = 0
        threading.Thread(target=self._worker, name="notify", daemon=True).start()

    @staticmethod
    def load_config() -> dict:
        raw = keyring.get_password(SERVICE, "config")
        return json.loads(raw) if raw else {}

    @staticmethod
    def save_config(config: dict) -> None:
        keyring.set_password(SERVICE, "config", json.dumps(config))

    def configure(self, config: dict) -> None:
        self.config = dict(config)

    @property
    def enabled(self) -> bool:
        c = self.config
        return bool((c.get("telegram_token") and c.get("telegram_chat"))
                    or (c.get("smtp_host") and c.get("email_to")))

    def notify(self, title: str, message: str) -> None:
        if not self.enabled:
            return
        try:
            self.queue.put_nowait((title, message))
        except queue.Full:
            LOG.warning("notification queue full; dropped: %s", title)

    def _worker(self) -> None:
        while True:
            title, message = self.queue.get()
            try:
                self.sender(self.config, title, message)
                self.sent += 1
            except Exception as error:
                self.failed += 1
                LOG.warning("notification failed: %s", error)

    @staticmethod
    def _send(config: dict, title: str, message: str) -> None:
        errors = []
        if config.get("telegram_token") and config.get("telegram_chat"):
            try:
                response = requests.post(
                    f"https://api.telegram.org/bot{config['telegram_token']}/sendMessage",
                    json={"chat_id": config["telegram_chat"], "text": f"{title}\n{message}"}, timeout=10)
                response.raise_for_status()
            except Exception as error:
                errors.append(f"telegram: {type(error).__name__}")
        if config.get("smtp_host") and config.get("email_to"):
            try:
                mail = EmailMessage()
                mail["Subject"] = f"[Strat Terminal] {title}"
                mail["From"] = config.get("smtp_user") or config["email_to"]
                mail["To"] = config["email_to"]
                mail.set_content(message)
                with smtplib.SMTP(config["smtp_host"], int(config.get("smtp_port") or 587), timeout=10) as smtp:
                    smtp.starttls()
                    if config.get("smtp_user"):
                        smtp.login(config["smtp_user"], config.get("smtp_password", ""))
                    smtp.send_message(mail)
            except Exception as error:
                errors.append(f"email: {type(error).__name__}")
        if errors:
            raise RuntimeError("; ".join(errors))

    def wait_idle(self, timeout: float = 5) -> None:
        deadline = time.monotonic() + timeout
        while not self.queue.empty() and time.monotonic() < deadline:
            time.sleep(0.05)


def launch_command() -> list[str]:
    """Command that starts this app: the frozen exe, or the venv Python running main.py."""
    if getattr(sys, "frozen", False):
        return [sys.executable]
    return [sys.executable, str(Path(__file__).resolve().with_name("main.py"))]


def schedule_command(at: str = "09:00") -> list[str]:
    command = subprocess.list2cmdline(launch_command())
    return ["schtasks", "/Create", "/F", "/TN", TASK_NAME, "/SC", "WEEKLY",
            "/D", "MON,TUE,WED,THU,FRI", "/ST", at, "/TR", command]


def unschedule_command() -> list[str]:
    return ["schtasks", "/Delete", "/F", "/TN", TASK_NAME]


def run_schtasks(args: list[str]) -> str:
    result = subprocess.run(args, capture_output=True, text=True, timeout=30,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip() or "schtasks failed")
    return result.stdout.strip()
