# Strat Terminal — Upstox Index Options (EMA crossover, spot-driven exits)

Desktop app (CustomTkinter + Matplotlib) that trades one ATM NIFTY/BANKNIFTY option per day
(per mode) from an EMA crossover on the **spot index** (default 9/21, configurable), with exits
driven by **spot** ATR levels.

## Run
```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe main.py
```
Tests (needs `requirements-dev.txt`):
```powershell
$env:PYTHONPATH = "$PWD;$PWD\tests"
.venv\Scripts\python.exe -m unittest discover -s tests
```

## Windows .exe
```powershell
.venv\Scripts\python.exe -m pip install "pyinstaller>=6.10,<7"
.venv\Scripts\python.exe -m PyInstaller strat_terminal.spec --noconfirm
```
Run `dist\StratTerminal\StratTerminal.exe` (copy the whole `dist\StratTerminal` folder).
The .exe keeps `trades.db`, `backtest.db`, logs and the instrument cache in
`%LOCALAPPDATA%\StratTerminal`; running from source keeps them in the project folder.

## First-time setup
1. In the Upstox developer console, register redirect URI `http://127.0.0.1:8765/callback`.
2. *Strategy & Setup* → enter API key/secret → **Authenticate via Upstox Web Login**.
   The secret and the day's access token are stored in Windows Credential Manager, so
   restarts on the same day skip login.
3. Choose index, timeframe (1/5/15/30 min) and sizing, then **Apply Parameters**.

## Modules
| File | Responsibility |
|---|---|
| `main.py` | Entry point |
| `gui.py` | Dashboard, Setup, Reports tabs (UI thread only; polls engine snapshots) |
| `strategy.py` | EMA/ATR (pandas-ta), crossover, risk levels, `TradingEngine` execution |
| `upstox_broker.py` | Official SDK: OAuth, token vault, instrument master, candles, orders, dual WebSockets |
| `database.py` | SQLite `trades.db`, daily-limit journal, fee ledger, metrics, CSV/PDF dossier |

> The broker module is `upstox_broker.py`, not `upstox_client.py`: that name is the official
> SDK's package name and a local file with it shadows the SDK and breaks all imports.

## Rules implemented (defaults; all editable under *Strategy Engine Settings*)
- 1 trade/day **per mode** (PAPER and LIVE have separate slots) via an atomic SQLite
  reservation that survives restarts.
- Signals before 09:30 ignored; no new entries after 15:00; square-off at 15:15 IST.
- ATM strike: spot rounded to 50 (NIFTY) / 100 (BANKNIFTY); nearest listed expiry from the
  daily Upstox NSE instrument master (JSON — Upstox has deprecated the CSV).
- Bars: built live from spot ticks (feed A); if any minute lacks ticks, the bar is rebuilt from
  Upstox 1-minute REST history. Warm-up = longest indicator + 100 bars.
- Optional trend filter: `EMA` (close above/below a trend EMA), `ADX` (strength ≥ minimum) or
  both. Index spot has no traded volume, so VWAP is not available.
- Exits: CE SL `spot ≤ entry − SL·ATR`, TP `spot ≥ entry + TP·ATR`; PE mirrored. MARKET exit.
- Channel A (spot) and B (option) WebSockets, each with independent exponential-backoff
  reconnect (0.5 s → 30 s) and a stale-feed watchdog; if spot ticks stall during a position,
  the engine falls back to REST quotes.
- Paper mode: fills from live option quotes with 0.5 % adverse slippage, no exchange routing.
- Charges: ₹20/order, NSE transaction 0.03503 %, SEBI ₹10/crore, STT 0.125 % sell,
  stamp 0.003 % buy, GST 18 % on brokerage + transaction + SEBI. **Verify rates periodically.**

## Live trading safeguards and limits
- Live mode must be armed by typing `LIVE`; mode/index cannot change with a position open.
- Live MARKET orders use Upstox market protection (required for API market orders).
- **Broker backstop SL** (default on): after a live entry a SELL stop-loss-limit order rests at
  the broker (default 40 % below the entry premium) so the position is protected if the app or
  PC dies. It is cancelled and confirmed cancelled before any app exit; if the cancel cannot be
  confirmed the app halts rather than risk a double SELL. A gap through the limit price can
  leave it unfilled.
- **Startup reconciliation** (LIVE): broker positions/orders are compared with the app; any
  mismatch halts. *Reconcile* re-runs it on demand.
- **Manual Close…** records a position you closed yourself (enter your actual exit price). For
  LIVE it refuses unless the broker shows the position flat.
- Real-time order updates arrive over the Upstox portfolio WebSocket (fills confirmed without
  polling; REST is still used as fallback).
- All SDK WebSockets are forced to verified TLS (the SDK itself disables certificate checks).
- Live entries are never retried. Exits resubmit only after positions confirm exposure remains.
- Any uncertain entry or failed exit **halts** the engine; no automatic orders are sent while
  halted. Verify the broker, then *Acknowledge Halt* or *Manual Close…*.
- Alerts: corner pop-up + sound for entries/exits, CRITICAL events and feed reconnects
  (toggle on the dashboard).
- WebSocket reconnects cannot recover ticks missed during an outage, and MARKET orders can
  slip. App-side exits only run while the app is open (the backstop SL is the safety net).
  Paper-trade first and confirm Upstox algo/static-IP requirements.

## Backtest and parameter sweep
*Reports & Logs*: **Run Backtest** replays a date range with real Upstox data using the exact
live rules. **Parameter Sweep…** backtests a grid of EMA/SL/TP/filter values (max 300) on one
download and ranks them by Sharpe, then profit factor. Short samples overfit easily — paper
trade any "best" settings before going live.
