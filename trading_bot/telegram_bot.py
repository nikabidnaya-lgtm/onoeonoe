from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message

from core.executor import PositionExecutor
from core.state import BotState


@dataclass(slots=True)
class TelegramNotifier:
    bot: Bot
    chat_id: str

    async def send_text(self, text: str) -> None:
        await self.bot.send_message(chat_id=self.chat_id, text=text)

    async def send_status(self, payload: str) -> None:
        await self.send_text(payload)

    async def send_emergency_close(self, reason: str) -> None:
        await self.send_text(f"🛑 Emergency close executed: {reason}")


def build_dispatcher(
    executor: PositionExecutor,
    notifier: TelegramNotifier,
    cfg: dict[str, Any],
    state: BotState,
    scheduler: Any,
) -> Dispatcher:
    dp = Dispatcher()

    @dp.message(Command("status"))
    async def status_handler(msg: Message) -> None:
        symbols = sorted({s for session in cfg["sessions"].values() for s in session["symbols"]})
        positions = executor.get_open_positions(symbols)
        jobs = [f"{j.id} @ {j.next_run_time}" for j in scheduler.scheduler.get_jobs()]
        text = f"paused={state.paused}\nopen_positions={len(positions)}\nnext_jobs:\n" + "\n".join(jobs[:10])
        await notifier.send_status(text)
        await msg.answer("ok")

    @dp.message(Command("kill_all"))
    async def kill_all_handler(msg: Message) -> None:
        symbols = sorted({s for session in cfg["sessions"].values() for s in session["symbols"]})
        order_ids = executor.close_all_positions(symbols)
        await notifier.send_emergency_close(f"/kill_all, closed: {order_ids}")
        await msg.answer("all closed")

    @dp.message(Command("pause"))
    async def pause_handler(msg: Message) -> None:
        state.paused = True
        await msg.answer("paused")

    @dp.message(Command("resume"))
    async def resume_handler(msg: Message) -> None:
        state.paused = False
        await msg.answer("resumed")

    @dp.message(Command("config"))
    async def config_handler(msg: Message) -> None:
        await msg.answer(str(cfg))

    @dp.message(Command("logs"))
    async def logs_handler(msg: Message) -> None:
        parts = (msg.text or "").split()
        n = 50
        if len(parts) > 1 and parts[1].isdigit():
            n = min(int(parts[1]), 200)
        log_file = Path(cfg["logging"]["dir"]) / "bot.log"
        if not log_file.exists():
            await msg.answer("log file not found")
            return
        lines = log_file.read_text(encoding="utf-8", errors="ignore").splitlines()[-n:]
        await msg.answer("\n".join(lines) if lines else "logs empty")

    @dp.message(F.text)
    async def unknown(msg: Message) -> None:
        await msg.answer("commands: /status /kill_all /pause /resume /config /logs N")

    return dp
