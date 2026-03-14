from __future__ import annotations

from dataclasses import dataclass

from core.analytics import AnalyticsEngine
from core.executor import PositionExecutor
from core.state import BotState
from modules.common import resolve_weights
from telegram_bot import TelegramNotifier


@dataclass(slots=True)
class SundayStrategy:
    analytics: AnalyticsEngine
    executor: PositionExecutor
    notifier: TelegramNotifier
    config: dict
    state: BotState

    async def run(self) -> None:
        if self.state.paused:
            await self.notifier.send_text("⏸ Sunday skipped: bot paused")
            return

        session_cfg = self.config["sessions"]["sunday"]
        weights = resolve_weights(session_cfg)

        opened = 0
        for symbol in session_cfg["symbols"]:
            checks = self.analytics.check_sunday_breakout(symbol)
            if not all(checks[k] for k in ["breakout_ok", "volume_ok", "rsi_ok", "spread_ok"]):
                continue
            capital_share = self.config["trading"]["capital"] * weights[symbol]
            if capital_share > checks["max_pos_notional"]:
                await self.notifier.send_text(f"⚠️ Sunday {symbol}: exceeds max position notional")
                continue

            order_id = await self.executor.open_position(
                symbol,
                side="long",
                leverage=session_cfg["execution"]["leverage"],
                capital_share=capital_share,
                max_slippage_pct=session_cfg["execution"]["max_slippage_pct"],
            )
            if order_id is None:
                await self.notifier.send_text(f"⚠️ Sunday {symbol}: order cancelled due to slippage")
                continue

            risk = session_cfg["sl_tp"][symbol]
            self.executor.set_sl_tp(symbol, "long", risk["sl"], risk["tp"][0])
            await self.notifier.send_text(f"✅ Sunday long opened {symbol}, order={order_id}")
            opened += 1

        if opened == 0:
            await self.notifier.send_text("⏭ Sunday skipped: no breakout")
