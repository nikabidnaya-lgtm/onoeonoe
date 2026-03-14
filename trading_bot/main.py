from __future__ import annotations

import asyncio

from aiogram import Bot
from pybit.unified_trading import HTTP

from core.analytics import AnalyticsEngine
from core.config import load_config
from core.data_collector import DataCollector
from core.executor import PositionExecutor
from core.logger import setup_logger
from core.scheduler import BotScheduler
from core.state import BotState
from database import init_db, make_session_factory
from modules.friday import FridayStrategy
from modules.monday import MondayStrategy
from modules.sunday import SundayStrategy
from modules.tuesday import TuesdayStrategy
from modules.wednesday import WednesdayStrategy
from telegram_bot import TelegramNotifier, build_dispatcher


async def run() -> None:
    cfg = load_config("config.yaml").raw
    setup_logger(cfg["logging"]["level"], cfg["logging"]["dir"])

    await init_db(cfg["database"]["url"])
    db_factory = make_session_factory(cfg["database"]["url"])

    collector = DataCollector.from_config(cfg)
    analytics = AnalyticsEngine(collector=collector, config=cfg)

    session = HTTP(
        testnet=bool(cfg["exchange"].get("testnet", True)),
        api_key=cfg["exchange"]["api_key"],
        api_secret=cfg["exchange"]["api_secret"],
    )
    executor = PositionExecutor(session=session, db_session_factory=db_factory)
    state = BotState()

    tg_bot = Bot(token=cfg["telegram"]["bot_token"])
    notifier = TelegramNotifier(bot=tg_bot, chat_id=cfg["telegram"]["chat_id"])

    monday = MondayStrategy(analytics=analytics, executor=executor, notifier=notifier, config=cfg, state=state)
    tuesday = TuesdayStrategy(analytics=analytics, executor=executor, notifier=notifier, config=cfg, state=state)
    wednesday = WednesdayStrategy(analytics=analytics, executor=executor, notifier=notifier, config=cfg, state=state)
    friday = FridayStrategy(analytics=analytics, executor=executor, notifier=notifier, config=cfg, state=state)
    sunday = SundayStrategy(analytics=analytics, executor=executor, notifier=notifier, config=cfg, state=state)

    scheduler = BotScheduler(
        monday=monday,
        tuesday=tuesday,
        wednesday=wednesday,
        friday=friday,
        sunday=sunday,
        executor=executor,
        config=cfg,
    )
    scheduler.start()

    dp = build_dispatcher(executor=executor, notifier=notifier, cfg=cfg, state=state, scheduler=scheduler)
    await notifier.send_text("🤖 Trading bot started (Bybit 5-day strategy, testnet)")
    await dp.start_polling(tg_bot)


if __name__ == "__main__":
    asyncio.run(run())
