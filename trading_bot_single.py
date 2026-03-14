from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd
import structlog
import yaml
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv
from pybit.unified_trading import HTTP
from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String
from sqlalchemy.ext.asyncio import AsyncAttrs, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


# ============================== CONFIG ==============================

def _expand_env(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _expand_env(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand_env(v) for v in node]
    if isinstance(node, str) and node.startswith("${") and node.endswith("}"):
        return os.getenv(node[2:-1], "")
    return node


def load_config(path: str = "config.yaml") -> dict[str, Any]:
    load_dotenv()
    with open(path, "r", encoding="utf-8") as f:
        return _expand_env(yaml.safe_load(f))


# ============================== LOGGING ==============================

def setup_logger(level: str = "INFO", log_dir: str = "./logs"):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
    )


# ============================== DATABASE ==============================

class Base(AsyncAttrs, DeclarativeBase):
    pass


class Trade(Base):
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_id: Mapped[str] = mapped_column(String(64), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    symbol: Mapped[str] = mapped_column(String(20))
    side: Mapped[str] = mapped_column(String(10))
    entry_price: Mapped[float] = mapped_column(Float)
    size: Mapped[float] = mapped_column(Float)
    leverage: Mapped[int] = mapped_column(Integer)
    sl_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    tp1_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    tp2_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    pnl_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_reason: Mapped[str | None] = mapped_column(String(50), nullable=True)


class Signal(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    session: Mapped[str] = mapped_column(String(10))
    symbol: Mapped[str] = mapped_column(String(20))
    conditions: Mapped[dict] = mapped_column(JSON)
    signal_valid: Mapped[bool] = mapped_column(Boolean)


async def init_db(database_url: str) -> async_sessionmaker:
    engine = create_async_engine(database_url, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return async_sessionmaker(engine, expire_on_commit=False)


# ============================== BOT STATE ==============================

@dataclass
class BotState:
    paused: bool = False


# ============================== MARKET DATA ==============================

class DataCollector:
    def __init__(self, session: HTTP):
        self.session = session

    def get_prices(self, symbols: list[str]) -> dict[str, float]:
        return {
            s: float(self.session.get_tickers(category="linear", symbol=s)["result"]["list"][0]["lastPrice"])
            for s in symbols
        }

    def get_funding_rates(self, symbols: list[str]) -> dict[str, float]:
        return {
            s: float(self.session.get_tickers(category="linear", symbol=s)["result"]["list"][0]["fundingRate"]) * 100
            for s in symbols
        }

    def get_ohlcv(self, symbol: str, interval: str = "60", limit: int = 200) -> pd.DataFrame:
        resp = self.session.get_kline(category="linear", symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(
            resp["result"]["list"],
            columns=["ts", "open", "high", "low", "close", "volume", "turnover"],
        )
        for col in ["open", "high", "low", "close", "volume", "turnover"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
        return df.sort_values("ts").reset_index(drop=True)

    def get_volume_ratio(self, symbol: str, interval: str, sample_window: int, baseline_window: int) -> float:
        df = self.get_ohlcv(symbol, interval=interval, limit=max(sample_window + baseline_window + 2, 30))
        if len(df) < sample_window + baseline_window:
            return 0.0
        latest = float(df.tail(sample_window)["volume"].mean())
        baseline = float(df.iloc[-(sample_window + baseline_window):-sample_window]["volume"].mean())
        return latest / baseline if baseline > 0 else 0.0

    def get_spread_pct(self, symbol: str) -> float:
        ob = self.session.get_orderbook(category="linear", symbol=symbol, limit=1)["result"]
        bid = float(ob["b"][0][0])
        ask = float(ob["a"][0][0])
        mid = (bid + ask) / 2
        return ((ask - bid) / mid) * 100

    def get_avg_daily_notional_7d(self, symbol: str) -> float:
        df = self.get_ohlcv(symbol, interval="D", limit=8)
        if df.empty:
            return 0.0
        return float(mean(df.tail(7)["turnover"].tolist()))

    def get_btc_dominance(self) -> float:
        return 0.0  # TODO provider

    def get_etf_flows(self) -> dict[str, float]:
        return {"inflow": 0.0}  # TODO provider

    def get_economic_calendar(self) -> list[dict[str, Any]]:
        return []  # TODO provider


# ============================== ANALYTICS ==============================

class AnalyticsEngine:
    def __init__(self, collector: DataCollector, cfg: dict[str, Any]):
        self.collector = collector
        self.cfg = cfg

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, pd.NA)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high, low, close = df["high"], df["low"], df["close"]
        plus_dm = (high.diff()).clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).fillna(0)
        return dx.rolling(period).mean()

    def check_monday(self, symbol: str) -> dict[str, Any]:
        flt = self.cfg["sessions"]["monday"]["filters"]
        funding = self.collector.get_funding_rates([symbol])[symbol]
        vr = self.collector.get_volume_ratio(symbol, "60", 1, 3)
        h1 = self.collector.get_ohlcv(symbol, "60", 4)
        btc_d = self.collector.get_btc_dominance()
        etf = self.collector.get_etf_flows()["inflow"]
        etf_ok = etf <= flt["etf_outflow_threshold"] if flt.get("require_etf_filter") else True
        return {
            "funding_ok": funding > flt["funding_threshold"],
            "volume_ok": vr >= flt["volume_multiplier"],
            "price_ok": float(h1.iloc[-1]["close"]) < float(h1.iloc[-2]["open"]),
            "btc_ok": btc_d >= flt["btc_dominance_min"],
            "etf_ok": etf_ok,
            "raw": {"funding": funding, "volume_ratio": vr, "btc_d": btc_d, "etf": etf},
        }

    def check_tuesday_or_friday(self, symbol: str, session_name: str) -> dict[str, Any]:
        flt = self.cfg["sessions"][session_name]["filters"]
        funding = self.collector.get_funding_rates([symbol])[symbol]
        vr = self.collector.get_volume_ratio(symbol, "60", 1, 24)
        btc_d = self.collector.get_btc_dominance()
        btc_ok = True if not flt.get("use_btc_dominance") else btc_d >= flt["btc_dominance_min"]
        news_ok = not flt.get("block_on_news")
        return {
            "funding_ok": funding > flt["funding_min"],
            "volume_ok": vr >= flt["volume_ratio_min"],
            "btc_ok": btc_ok,
            "news_ok": news_ok,
            "raw": {"funding": funding, "volume_ratio": vr, "btc_d": btc_d},
        }

    def has_high_impact_event(self) -> bool:
        return any(str(e.get("impact", "")).lower() == "high" for e in self.collector.get_economic_calendar())

    def check_wednesday_trend(self, symbol: str) -> dict[str, Any]:
        flt = self.cfg["sessions"]["wednesday"]["filters"]
        m5 = self.collector.get_ohlcv(symbol, "5", 120)
        h1 = self.collector.get_ohlcv(symbol, "60", 100)
        adx = float(self._adx(h1, flt["adx_period"]).iloc[-1])
        direction = "up" if m5["close"].iloc[-1] > m5["close"].iloc[-8] else "down"
        extremes_ok = (
            bool((m5.tail(8)["high"].diff().tail(3) > 0).sum() >= 2)
            if direction == "up"
            else bool((m5.tail(8)["low"].diff().tail(3) < 0).sum() >= 2)
        )
        vr = self.collector.get_volume_ratio(symbol, "5", 1, 24)
        pre = float(m5.iloc[-30]["close"])
        cur = float(m5.iloc[-1]["close"])
        move_pct = abs((cur - pre) / pre) * 100
        valid = all([
            adx >= flt["adx_thresholds"][symbol],
            extremes_ok,
            vr >= flt["volume_multiplier"][symbol],
            move_pct <= flt["max_move_before_entry"][symbol],
        ])
        return {"valid": valid, "direction": direction, "raw": {"adx": adx, "volume_ratio": vr, "move_pct": move_pct}}

    def check_sunday(self, symbol: str) -> dict[str, Any]:
        flt = self.cfg["sessions"]["sunday"]["filters"]
        h1 = self.collector.get_ohlcv(symbol, "60", 12)
        m15 = self.collector.get_ohlcv(symbol, "15", 40)
        level = float(h1.iloc[-flt["breakout_period_hours"]:]["high"].max())
        price = float(m15.iloc[-1]["close"])
        vr = self.collector.get_volume_ratio(symbol, "60", 1, 4)
        rsi = float(self._rsi(h1["close"], flt["rsi_period"]).iloc[-1])
        spread = self.collector.get_spread_pct(symbol)
        max_notional = self.collector.get_avg_daily_notional_7d(symbol) * (flt["max_position_size_pct"] / 100)
        return {
            "breakout_ok": price > level,
            "volume_ok": vr >= flt["volume_multiplier"],
            "rsi_ok": rsi < flt["rsi_threshold"],
            "spread_ok": spread <= flt["max_spread"],
            "max_notional": max_notional,
            "raw": {"level": level, "price": price, "volume_ratio": vr, "rsi": rsi, "spread": spread},
        }


# ============================== EXECUTOR ==============================

class PositionExecutor:
    def __init__(self, session: HTTP, session_factory: async_sessionmaker):
        self.session = session
        self.session_factory = session_factory

    async def _log_trade(self, payload: dict[str, Any]):
        async with self.session_factory() as db:
            db.add(Trade(**payload))
            await db.commit()

    async def _log_signal(self, session_name: str, symbol: str, conditions: dict[str, Any], valid: bool):
        async with self.session_factory() as db:
            db.add(Signal(timestamp=datetime.now(timezone.utc), session=session_name, symbol=symbol, conditions=conditions, signal_valid=valid))
            await db.commit()

    def _slippage_ok(self, symbol: str, side: str, max_slippage_pct: float) -> bool:
        tick = self.session.get_tickers(category="linear", symbol=symbol)["result"]["list"][0]
        last = float(tick["lastPrice"])
        ob = self.session.get_orderbook(category="linear", symbol=symbol, limit=1)["result"]
        bid, ask = float(ob["b"][0][0]), float(ob["a"][0][0])
        ref = ask if side == "long" else bid
        return abs((last - ref) / ref) * 100 <= max_slippage_pct

    async def open_position(self, symbol: str, side: str, leverage: int, capital_share: float, max_slippage_pct: float) -> str | None:
        if not self._slippage_ok(symbol, side, max_slippage_pct):
            return None
        tick = self.session.get_tickers(category="linear", symbol=symbol)["result"]["list"][0]
        price = float(tick["lastPrice"])
        qty = max((capital_share * leverage) / price, 1)
        self.session.set_leverage(category="linear", symbol=symbol, buyLeverage=str(leverage), sellLeverage=str(leverage))
        order = self.session.place_order(
            category="linear",
            symbol=symbol,
            side="Buy" if side == "long" else "Sell",
            orderType="Market",
            qty=f"{qty:.0f}",
            reduceOnly=False,
        )
        oid = order["result"]["orderId"]
        await self._log_trade(
            {
                "order_id": oid,
                "timestamp": datetime.now(timezone.utc),
                "symbol": symbol,
                "side": side,
                "entry_price": price,
                "size": qty,
                "leverage": leverage,
                "sl_price": None,
                "tp1_price": None,
                "tp2_price": None,
                "exit_price": None,
                "pnl_usd": None,
                "exit_reason": None,
            }
        )
        return oid

    def set_sl_tp(self, symbol: str, side: str, sl_pct: float, tp1_pct: float):
        mark = float(self.session.get_tickers(category="linear", symbol=symbol)["result"]["list"][0]["markPrice"])
        long = side == "long"
        sl = mark * (1 - sl_pct / 100) if long else mark * (1 + sl_pct / 100)
        tp = mark * (1 + tp1_pct / 100) if long else mark * (1 - tp1_pct / 100)
        self.session.set_trading_stop(
            category="linear",
            symbol=symbol,
            stopLoss=f"{sl:.6f}",
            takeProfit=f"{tp:.6f}",
            tpslMode="Full",
            positionIdx=0,
        )

    def close_all_positions(self, symbols: list[str]) -> list[str]:
        ids: list[str] = []
        for symbol in symbols:
            for p in self.session.get_positions(category="linear", symbol=symbol)["result"]["list"]:
                size = float(p["size"])
                if size <= 0:
                    continue
                side = "Sell" if p["side"] == "Buy" else "Buy"
                order = self.session.place_order(category="linear", symbol=symbol, side=side, orderType="Market", qty=str(int(size)), reduceOnly=True)
                ids.append(order["result"]["orderId"])
        return ids

    def get_open_positions(self, symbols: list[str]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for symbol in symbols:
            for p in self.session.get_positions(category="linear", symbol=symbol)["result"]["list"]:
                if float(p.get("size", 0)) > 0:
                    out.append(p)
        return out


# ============================== TELEGRAM ==============================

class TelegramNotifier:
    def __init__(self, bot: Bot, chat_id: str):
        self.bot = bot
        self.chat_id = chat_id

    async def send(self, text: str):
        await self.bot.send_message(chat_id=self.chat_id, text=text)


def build_dispatcher(
    bot_state: BotState,
    cfg: dict[str, Any],
    scheduler: AsyncIOScheduler,
    executor: PositionExecutor,
    notifier: TelegramNotifier,
) -> Dispatcher:
    dp = Dispatcher()

    @dp.message(Command("status"))
    async def status(msg: Message):
        symbols = sorted({s for ss in cfg["sessions"].values() for s in ss["symbols"]})
        pos = executor.get_open_positions(symbols)
        jobs = [f"{j.id} @ {j.next_run_time}" for j in scheduler.get_jobs()]
        await notifier.send(f"paused={bot_state.paused}\nopen_positions={len(pos)}\nnext_jobs=\n" + "\n".join(jobs[:15]))
        await msg.answer("ok")

    @dp.message(Command("kill_all"))
    async def kill_all(msg: Message):
        symbols = sorted({s for ss in cfg["sessions"].values() for s in ss["symbols"]})
        closed = executor.close_all_positions(symbols)
        await notifier.send(f"🛑 kill_all: {closed}")
        await msg.answer("closed")

    @dp.message(Command("pause"))
    async def pause(msg: Message):
        bot_state.paused = True
        await msg.answer("paused")

    @dp.message(Command("resume"))
    async def resume(msg: Message):
        bot_state.paused = False
        await msg.answer("resumed")

    @dp.message(Command("config"))
    async def config(msg: Message):
        await msg.answer(json.dumps(cfg, ensure_ascii=False, indent=2))

    @dp.message(Command("logs"))
    async def logs(msg: Message):
        n = 50
        parts = (msg.text or "").split()
        if len(parts) > 1 and parts[1].isdigit():
            n = min(int(parts[1]), 200)
        log_file = Path(cfg["logging"]["dir"]) / "bot.log"
        if not log_file.exists():
            await msg.answer("log file not found")
            return
        lines = log_file.read_text(encoding="utf-8", errors="ignore").splitlines()[-n:]
        await msg.answer("\n".join(lines) if lines else "empty")

    @dp.message(F.text)
    async def unknown(msg: Message):
        await msg.answer("/status /kill_all /pause /resume /config /logs N")

    return dp


# ============================== STRATEGIES ==============================

def _weights(session_cfg: dict[str, Any]) -> dict[str, float]:
    if session_cfg.get("weights"):
        return session_cfg["weights"]
    symbols = session_cfg["symbols"]
    w = 1 / len(symbols)
    return {s: w for s in symbols}


async def run_monday(state: BotState, cfg: dict[str, Any], analytics: AnalyticsEngine, executor: PositionExecutor, notifier: TelegramNotifier):
    if state.paused:
        await notifier.send("⏸ monday paused")
        return
    s = cfg["sessions"]["monday"]
    passed: list[str] = []
    for sym in s["symbols"]:
        checks = analytics.check_monday(sym)
        valid = all([checks["funding_ok"], checks["volume_ok"], checks["price_ok"], checks["btc_ok"], checks["etf_ok"]])
        await executor._log_signal("monday", sym, checks["raw"], valid)
        if valid:
            passed.append(sym)
    if len(passed) < s["filters"]["min_coins"]:
        await notifier.send(f"⏭ monday skipped ({len(passed)} symbols)")
        return
    weights = _weights(s)
    for sym in passed:
        oid = await executor.open_position(sym, "short", s["execution"]["leverage"], cfg["trading"]["capital"] * weights[sym], s["execution"]["max_slippage_pct"])
        if oid is None:
            await notifier.send(f"⚠️ monday {sym} slippage")
            continue
        r = s["sl_tp"][sym]
        executor.set_sl_tp(sym, "short", r["sl"], r["tp1"])
        await notifier.send(f"✅ monday short {sym}: {oid}")


async def run_tuesday_like(name: str, state: BotState, cfg: dict[str, Any], analytics: AnalyticsEngine, executor: PositionExecutor, notifier: TelegramNotifier):
    if state.paused:
        await notifier.send(f"⏸ {name} paused")
        return
    s = cfg["sessions"][name]
    w = _weights(s)
    for sym in s["symbols"]:
        checks = analytics.check_tuesday_or_friday(sym, name)
        valid = all([checks["funding_ok"], checks["volume_ok"], checks["btc_ok"], checks["news_ok"]])
        await executor._log_signal(name, sym, checks["raw"], valid)
        if not valid:
            continue
        oid = await executor.open_position(sym, "short", s["execution"]["leverage"], cfg["trading"]["capital"] * w[sym], s["execution"]["max_slippage_pct"])
        if oid is None:
            await notifier.send(f"⚠️ {name} {sym} slippage")
            continue
        r = s["sl_tp"][sym]
        executor.set_sl_tp(sym, "short", r["sl"], r["tp1"])
        await notifier.send(f"✅ {name} short {sym}: {oid}")


async def run_wednesday(state: BotState, cfg: dict[str, Any], analytics: AnalyticsEngine, executor: PositionExecutor, notifier: TelegramNotifier):
    if state.paused:
        await notifier.send("⏸ wednesday paused")
        return
    s = cfg["sessions"]["wednesday"]
    if s["filters"]["require_high_impact_event"] and not analytics.has_high_impact_event():
        await notifier.send("⏭ wednesday skipped: no high impact event")
        return
    w = _weights(s)
    opened = 0
    for sym in s["symbols"]:
        trend = analytics.check_wednesday_trend(sym)
        await executor._log_signal("wednesday", sym, trend["raw"], trend["valid"])
        if not trend["valid"]:
            continue
        side = "long" if trend["direction"] == "up" else "short"
        oid = await executor.open_position(sym, side, s["execution"]["leverage"], cfg["trading"]["capital"] * w[sym], s["execution"]["max_slippage_pct"])
        if oid is None:
            await notifier.send(f"⚠️ wednesday {sym} slippage")
            continue
        r = s["sl_tp"][sym]
        executor.set_sl_tp(sym, side, r["sl"], r["tp1"])
        await notifier.send(f"✅ wednesday {side} {sym}: {oid}")
        opened += 1
    if opened == 0:
        await notifier.send("⏭ wednesday skipped: no valid trend")


async def run_sunday(state: BotState, cfg: dict[str, Any], analytics: AnalyticsEngine, executor: PositionExecutor, notifier: TelegramNotifier):
    if state.paused:
        await notifier.send("⏸ sunday paused")
        return
    s = cfg["sessions"]["sunday"]
    w = _weights(s)
    opened = 0
    for sym in s["symbols"]:
        checks = analytics.check_sunday(sym)
        valid = all([checks["breakout_ok"], checks["volume_ok"], checks["rsi_ok"], checks["spread_ok"]])
        await executor._log_signal("sunday", sym, checks["raw"], valid)
        if not valid:
            continue
        capital_share = cfg["trading"]["capital"] * w[sym]
        if capital_share > checks["max_notional"]:
            await notifier.send(f"⚠️ sunday {sym}: max size exceeded")
            continue
        oid = await executor.open_position(sym, "long", s["execution"]["leverage"], capital_share, s["execution"]["max_slippage_pct"])
        if oid is None:
            await notifier.send(f"⚠️ sunday {sym} slippage")
            continue
        r = s["sl_tp"][sym]
        executor.set_sl_tp(sym, "long", r["sl"], r["tp"][0])
        await notifier.send(f"✅ sunday long {sym}: {oid}")
        opened += 1
    if opened == 0:
        await notifier.send("⏭ sunday skipped")


# ============================== SCHEDULER ==============================

def _hm(s: str) -> tuple[int, int]:
    h, m = s.split(":")
    return int(h), int(m)


def register_jobs(scheduler: AsyncIOScheduler, cfg: dict[str, Any], state: BotState, analytics: AnalyticsEngine, executor: PositionExecutor, notifier: TelegramNotifier):
    sessions = cfg["sessions"]

    if sessions["monday"]["enabled"]:
        h, m = _hm(sessions["monday"]["entry_time"])
        scheduler.add_job(run_monday, CronTrigger(day_of_week="mon", hour=h, minute=m), args=[state, cfg, analytics, executor, notifier], id="monday_entry")
        eh, em = _hm(sessions["monday"]["exit_time"])
        scheduler.add_job(executor.close_all_positions, CronTrigger(day_of_week="mon", hour=eh, minute=em), args=[sessions["monday"]["symbols"]], id="monday_exit")

    if sessions["tuesday"]["enabled"]:
        h, m = _hm(sessions["tuesday"]["entry_time"])
        scheduler.add_job(run_tuesday_like, CronTrigger(day_of_week="tue", hour=h, minute=m), args=["tuesday", state, cfg, analytics, executor, notifier], id="tuesday_entry")
        eh, em = _hm(sessions["tuesday"]["exit_time"])
        scheduler.add_job(executor.close_all_positions, CronTrigger(day_of_week="tue", hour=eh, minute=em), args=[sessions["tuesday"]["symbols"]], id="tuesday_exit")

    if sessions["wednesday"]["enabled"]:
        h, m = _hm(sessions["wednesday"]["entry_time"])
        scheduler.add_job(run_wednesday, CronTrigger(day_of_week="wed", hour=h, minute=m), args=[state, cfg, analytics, executor, notifier], id="wednesday_entry")
        eh, em = _hm(sessions["wednesday"]["exit_time"])
        scheduler.add_job(executor.close_all_positions, CronTrigger(day_of_week="thu", hour=eh, minute=em), args=[sessions["wednesday"]["symbols"]], id="wednesday_exit")

    if sessions["friday"]["enabled"]:
        h, m = _hm(sessions["friday"]["entry_time"])
        scheduler.add_job(run_tuesday_like, CronTrigger(day_of_week="fri", hour=h, minute=m), args=["friday", state, cfg, analytics, executor, notifier], id="friday_entry")
        eh, em = _hm(sessions["friday"]["exit_time"])
        scheduler.add_job(executor.close_all_positions, CronTrigger(day_of_week="fri", hour=eh, minute=em), args=[sessions["friday"]["symbols"]], id="friday_exit")

    if sessions["sunday"]["enabled"]:
        h, m = _hm(sessions["sunday"]["entry_time"])
        scheduler.add_job(run_sunday, CronTrigger(day_of_week="sun", hour=h, minute=m), args=[state, cfg, analytics, executor, notifier], id="sunday_entry")
        eh, em = _hm(sessions["sunday"]["exit_time"])
        scheduler.add_job(executor.close_all_positions, CronTrigger(day_of_week="mon", hour=eh, minute=em), args=[sessions["sunday"]["symbols"]], id="sunday_exit")


# ============================== MAIN ==============================

async def main() -> None:
    cfg = load_config("config.yaml")
    setup_logger(cfg["logging"]["level"], cfg["logging"]["dir"])

    db_factory = await init_db(cfg["database"]["url"])

    bybit = HTTP(
        testnet=bool(cfg["exchange"].get("testnet", True)),
        api_key=cfg["exchange"]["api_key"],
        api_secret=cfg["exchange"]["api_secret"],
    )

    collector = DataCollector(bybit)
    analytics = AnalyticsEngine(collector, cfg)
    executor = PositionExecutor(bybit, db_factory)
    state = BotState()

    tg = Bot(token=cfg["telegram"]["bot_token"])
    notifier = TelegramNotifier(tg, cfg["telegram"]["chat_id"])

    scheduler = AsyncIOScheduler(timezone="Europe/Moscow")
    register_jobs(scheduler, cfg, state, analytics, executor, notifier)
    scheduler.start()

    dp = build_dispatcher(state, cfg, scheduler, executor, notifier)
    await notifier.send("🤖 5-day bot started (single-file mode)")
    await dp.start_polling(tg)


if __name__ == "__main__":
    asyncio.run(main())
