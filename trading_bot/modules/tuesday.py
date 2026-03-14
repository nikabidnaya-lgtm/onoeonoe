from __future__ import annotations

from dataclasses import dataclass

from core.analytics import AnalyticsEngine
from core.executor import PositionExecutor
from core.state import BotState
from modules.common import resolve_weights
from telegram_bot import TelegramNotifier


@dataclass(slots=True)
class TuesdayStrategy:
    analytics: AnalyticsEngine
    executor: PositionExecutor
    notifier: TelegramNotifier
    config: dict
    state: BotState

    async def run(self) -> None:
        if self.state.paused:
            await self.notifier.send_text("⏸ Tuesday skipped: bot paused")
            return

        session_cfg = self.config["sessions"]["tuesday"]
        weights = resolve_weights(session_cfg)

        for symbol in session_cfg["symbols"]:
            checks = self.analytics.check_tuesday_conditions(symbol, "tuesday")
            if not all(bool(checks[k]) for k in ["funding_ok", "volume_ok", "btc_ok", "news_ok"]):
                continue
            capital_share = self.config["trading"]["capital"] * weights[symbol]
            order_id = await self.executor.open_position(
                symbol,
                side="short",
                leverage=session_cfg["execution"]["leverage"],
                capital_share=capital_share,
                max_slippage_pct=session_cfg["execution"]["max_slippage_pct"],
            )
            if order_id is None:
                await self.notifier.send_text(f"⚠️ Tuesday {symbol}: order cancelled due to slippage")
                continue
            risk = session_cfg["sl_tp"][symbol]
            self.executor.set_sl_tp(symbol, "short", risk["sl"], risk["tp1"])
            await self.notifier.send_text(f"✅ Tuesday short opened {symbol}, order={order_id}")
