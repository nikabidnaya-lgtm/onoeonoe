from __future__ import annotations

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from core.executor import PositionExecutor
from modules.friday import FridayStrategy
from modules.monday import MondayStrategy
from modules.sunday import SundayStrategy
from modules.tuesday import TuesdayStrategy
from modules.wednesday import WednesdayStrategy


class BotScheduler:
    def __init__(
        self,
        monday: MondayStrategy,
        tuesday: TuesdayStrategy,
        wednesday: WednesdayStrategy,
        friday: FridayStrategy,
        sunday: SundayStrategy,
        executor: PositionExecutor,
        config: dict,
    ) -> None:
        self.scheduler = AsyncIOScheduler(timezone="Europe/Moscow")
        self.monday = monday
        self.tuesday = tuesday
        self.wednesday = wednesday
        self.friday = friday
        self.sunday = sunday
        self.executor = executor
        self.config = config

    async def _close_session_positions(self, session_name: str) -> None:
        symbols = self.config["sessions"][session_name]["symbols"]
        self.executor.close_all_positions(symbols)

    def _add_entry(self, day_of_week: str, time_str: str, func, job_id: str) -> None:
        hour, minute = map(int, time_str.split(":"))
        self.scheduler.add_job(func, CronTrigger(day_of_week=day_of_week, hour=hour, minute=minute), id=job_id)

    def _add_exit(self, day_of_week: str, time_str: str, session_name: str, job_id: str) -> None:
        hour, minute = map(int, time_str.split(":"))
        self.scheduler.add_job(
            self._close_session_positions,
            CronTrigger(day_of_week=day_of_week, hour=hour, minute=minute),
            args=[session_name],
            id=job_id,
        )

    def register_jobs(self) -> None:
        cfg = self.config["sessions"]
        if cfg["monday"]["enabled"]:
            self._add_entry("mon", cfg["monday"]["entry_time"], self.monday.run, "monday_entry")
            self._add_exit("mon", cfg["monday"]["exit_time"], "monday", "monday_exit")
        if cfg["tuesday"]["enabled"]:
            self._add_entry("tue", cfg["tuesday"]["entry_time"], self.tuesday.run, "tuesday_entry")
            self._add_exit("tue", cfg["tuesday"]["exit_time"], "tuesday", "tuesday_exit")
        if cfg["wednesday"]["enabled"]:
            self._add_entry("wed", cfg["wednesday"]["entry_time"], self.wednesday.run, "wednesday_entry")
            self._add_exit("thu", cfg["wednesday"]["exit_time"], "wednesday", "wednesday_exit")
        if cfg["friday"]["enabled"]:
            self._add_entry("fri", cfg["friday"]["entry_time"], self.friday.run, "friday_entry")
            self._add_exit("fri", cfg["friday"]["exit_time"], "friday", "friday_exit")
        if cfg["sunday"]["enabled"]:
            self._add_entry("sun", cfg["sunday"]["entry_time"], self.sunday.run, "sunday_entry")
            self._add_exit("mon", cfg["sunday"]["exit_time"], "sunday", "sunday_exit")

    def start(self) -> None:
        self.register_jobs()
        self.scheduler.start()
