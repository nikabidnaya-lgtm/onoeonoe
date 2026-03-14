from __future__ import annotations

from dataclasses import dataclass

from core.analytics import AnalyticsEngine
from core.executor import PositionExecutor
from core.state import BotState
from modules.common import resolve_weights
from telegram_bot import TelegramNotifier


@dataclass(slots=True)
class WednesdayStrategy:
    analytics: AnalyticsEngine
    executor: PositionExecutor
    notifier: TelegramNotifier
    config: dict
    state: BotState

    async def run(self) -> None:
        if self.state.paused:
            await self.notifier.send_text("⏸ Wednesday skipped: bot paused")
            return

        session_cfg = self.config["sessions"]["wednesday"]
        filters = session_cfg["filters"]
        if filters["require_high_impact_event"] and not self.analytics.has_high_impact_event():
            await self.notifier.send_text("⏭ Wednesday skipped: no high-impact macro event")
            return

        weights = resolve_weights(session_cfg)
        opened = 0
        for symbol in session_cfg["symbols"]:
            trend = self.analytics.check_wednesday_trend(symbol)
            if not trend["valid"]:
                continue
            side = "long" if trend["direction"] == "up" else "short"
            capital_share = self.config["trading"]["capital"] * weights[symbol]
            order_id = await self.executor.open_position(
                symbol,
                side=side,
                leverage=session_cfg["execution"]["leverage"],
                capital_share=capital_share,
                max_slippage_pct=session_cfg["execution"]["max_slippage_pct"],
            )
            if order_id is None:
                await self.notifier.send_text(f"⚠️ Wednesday {symbol}: order cancelled due to slippage")
                continue
            risk = session_cfg["sl_tp"][symbol]
            self.executor.set_sl_tp(symbol, side, risk["sl"], risk["tp1"])
            await self.notifier.send_text(f"✅ Wednesday {side} opened {symbol}, order={order_id}")
            opened += 1

        if opened == 0:
            await self.notifier.send_text("⏭ Wednesday skipped: no confirmed trend")
