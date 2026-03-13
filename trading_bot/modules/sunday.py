from __future__ import annotations

from dataclasses import dataclass

from core.analytics import AnalyticsEngine
from core.executor import PositionExecutor
from telegram_bot import TelegramNotifier


@dataclass(slots=True)
class SundayStrategy:
    analytics: AnalyticsEngine
    executor: PositionExecutor
    notifier: TelegramNotifier
    config: dict

    async def run(self) -> None:
        opened = 0
        for symbol in self.config["trading"]["symbols"]:
            checks = self.analytics.check_sunday_breakout(symbol)
            if not all(checks[k] for k in ["breakout_ok", "volume_ok", "rsi_ok", "spread_ok"]):
                continue
            capital_share = self.config["trading"]["capital"] * self.config["trading"]["weights"][symbol]
            order_id = await self.executor.open_position(symbol, side="long", leverage=2, capital_share=capital_share)
            await self.notifier.send_text(f"✅ Sunday long opened {symbol}, order={order_id}")
            opened += 1

        if opened == 0:
            await self.notifier.send_text("⏭ Sunday skipped: no breakout")
