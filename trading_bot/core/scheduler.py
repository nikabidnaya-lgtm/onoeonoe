from __future__ import annotations

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from modules.monday import MondayStrategy
from modules.sunday import SundayStrategy
from modules.wednesday import WednesdayStrategy


class BotScheduler:
    def __init__(self, monday: MondayStrategy, wednesday: WednesdayStrategy, sunday: SundayStrategy) -> None:
        self.scheduler = AsyncIOScheduler(timezone="Europe/Moscow")
        self.monday = monday
        self.wednesday = wednesday
        self.sunday = sunday

    def register_jobs(self) -> None:
        self.scheduler.add_job(self.sunday.run, CronTrigger(day_of_week="sun", hour=20, minute=0), id="sunday_entry")
        self.scheduler.add_job(self.monday.run, CronTrigger(day_of_week="mon", hour=11, minute=0), id="monday_entry")
        self.scheduler.add_job(self.wednesday.run, CronTrigger(day_of_week="wed", hour=15, minute=0), id="wednesday_entry")

    def start(self) -> None:
        self.register_jobs()
        self.scheduler.start()
