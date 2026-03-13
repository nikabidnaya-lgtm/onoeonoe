from __future__ import annotations

from dataclasses import dataclass

from core.analytics import AnalyticsEngine
from core.executor import PositionExecutor
from telegram_bot import TelegramNotifier


@dataclass(slots=True)
class MondayStrategy:
    analytics: AnalyticsEngine
    executor: PositionExecutor
    notifier: TelegramNotifier
    config: dict

    async def run(self) -> None:
        passed: list[str] = []
        for symbol in self.config["trading"]["symbols"]:
            checks = self.analytics.check_monday_conditions(symbol)
            if all(bool(checks[k]) for k in ["funding_ok", "volume_ok", "price_ok", "btc_dominance_ok", "etf_ok"]):
                passed.append(symbol)

        if len(passed) < self.config["monday"]["min_coins"]:
            await self.notifier.send_text(f"⏭ Monday skipped: only {len(passed)} symbols passed")
            return

        for symbol in passed:
            capital_share = self.config["trading"]["capital"] * self.config["trading"]["weights"][symbol]
            order_id = await self.executor.open_position(symbol, side="short", leverage=3, capital_share=capital_share)
            risk = self.config["risk"]["monday"][symbol]
            self.executor.set_sl_tp(symbol, "short", risk["sl_pct"], risk["tp1_pct"])
            await self.notifier.send_text(f"✅ Monday short opened {symbol}, order={order_id}")
