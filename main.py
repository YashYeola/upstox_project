"""Strat Terminal entry point."""

import logging
import os
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

from database import Database
from gui import App
from strategy import TradingEngine


def data_dir() -> Path:
    # A frozen .exe runs from a temp/install folder; keep the ledger in a persistent user folder.
    if getattr(sys, "frozen", False):
        base = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "StratTerminal"
    else:
        base = Path(__file__).resolve().parent
    base.mkdir(parents=True, exist_ok=True)
    return base


def setup_logging(base: Path) -> None:
    handler = RotatingFileHandler(base / "strat_terminal.log", maxBytes=2_000_000, backupCount=5,
                                  encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(threadName)s %(name)s %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    sys.excepthook = lambda kind, value, tb: logging.critical("Unhandled exception", exc_info=(kind, value, tb))
    threading.excepthook = lambda args: logging.critical(
        "Unhandled exception in thread %s", args.thread.name if args.thread else "?",
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback))


def main() -> None:
    base = data_dir()
    setup_logging(base)
    logging.info("Strat Terminal starting; data folder %s", base)
    db = Database(base / "trades.db")
    engine = TradingEngine(db)
    app = App(db, engine)
    app.report_callback_exception = lambda kind, value, tb: logging.error(
        "UI callback error", exc_info=(kind, value, tb))
    app.mainloop()
    logging.info("Strat Terminal closed")


if __name__ == "__main__":
    main()
