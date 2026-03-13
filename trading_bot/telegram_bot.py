from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message

from core.executor import PositionExecutor


@dataclass(slots=True)
class TelegramNotifier:
    bot: Bot
    chat_id: str

    async def send_text(self, text: str) -> None:
        await self.bot.send_message(chat_id=self.chat_id, text=text)

    async def send_trade_alert(self, symbol: str, side: str, price: float, sl: float, tp: float) -> None:
        await self.send_text(f"🚀 {symbol} {side}\nentry={price:.5f} sl={sl:.5f} tp={tp:.5f}")

    async def send_status(self, positions: list[dict[str, Any]]) -> None:
        await self.send_text(f"📊 Open positions: {positions}")

    async def send_emergency_close(self, reason: str) -> None:
        await self.send_text(f"🛑 Emergency close executed: {reason}")


def build_dispatcher(executor: PositionExecutor, notifier: TelegramNotifier, cfg: dict[str, Any]) -> Dispatcher:
    dp = Dispatcher()
    paused = {"value": False}

    @dp.message(Command("status"))
    async def status_handler(msg: Message) -> None:
        positions = []
        for symbol in cfg["trading"]["symbols"]:
            raw = executor.session.get_positions(category="linear", symbol=symbol)["result"]["list"]
            for p in raw:
                if float(p.get("size", 0)) > 0:
                    positions.append(p)
        await notifier.send_status(positions)
        await msg.answer("ok")

    @dp.message(Command("kill_all"))
    async def kill_all_handler(msg: Message) -> None:
        order_ids = executor.close_all_positions()
        await notifier.send_emergency_close(f"/kill_all, closed: {order_ids}")
        await msg.answer("all closed")

    @dp.message(Command("pause"))
    async def pause_handler(msg: Message) -> None:
        paused["value"] = True
        await msg.answer("paused")

    @dp.message(Command("resume"))
    async def resume_handler(msg: Message) -> None:
        paused["value"] = False
        await msg.answer("resumed")

    @dp.message(Command("config"))
    async def config_handler(msg: Message) -> None:
        await msg.answer(str(cfg))

    @dp.message(Command("logs"))
    async def logs_handler(msg: Message) -> None:
        await msg.answer("Смотри файл logs/bot.log")

    @dp.message(F.text)
    async def unknown(msg: Message) -> None:
        await msg.answer("commands: /status /kill_all /pause /resume /config /logs")

    return dp
