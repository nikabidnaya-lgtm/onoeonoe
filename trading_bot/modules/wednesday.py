from __future__ import annotations

from dataclasses import dataclass

from core.analytics import AnalyticsEngine
from core.executor import PositionExecutor
from telegram_bot import TelegramNotifier


@dataclass(slots=True)
class WednesdayStrategy:
    analytics: AnalyticsEngine
    executor: PositionExecutor
    notifier: TelegramNotifier
    config: dict

    async def run(self) -> None:
        opened = 0
        for symbol in self.config["trading"]["symbols"]:
            trend = self.analytics.calculate_trend_strength(symbol)
            if not trend["valid"]:
                continue
            side = "long" if trend["direction"] == "up" else "short"
            capital_share = self.config["trading"]["capital"] * self.config["trading"]["weights"][symbol]
            order_id = await self.executor.open_position(symbol, side=side, leverage=3, capital_share=capital_share)
            await self.notifier.send_text(f"✅ Wednesday {side} opened {symbol}, order={order_id}")
            opened += 1

        if opened == 0:
            await self.notifier.send_text("⏭ Wednesday skipped: no confirmed trend")
