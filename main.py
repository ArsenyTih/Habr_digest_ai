from __future__ import annotations

import asyncio
import logging
import signal
import sys

from aiogram import Bot
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from config import settings
from database import init_db
from graph import build_graph, run_digest
from rag import RAGStore
from telegram_bot import create_bot_and_dispatcher

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


async def main() -> None:
    logger.info("Запуск бота дайджестов...")

    await init_db()
    rag = RAGStore()

    bot = Bot(token=settings.telegram_bot_token)
    graph = build_graph(bot, rag)

    async def digest_callback(dry_run: bool, manual: bool) -> str:
        return await run_digest(graph, dry_run=dry_run, manual=manual)

    _, dp = create_bot_and_dispatcher(digest_callback, rag)

    scheduler = AsyncIOScheduler(timezone=settings.timezone)

    async def scheduled_digest() -> None:
        logger.info("Запуск планового дайджеста...")
        result = await run_digest(graph, dry_run=False, manual=False)
        logger.info("Результат: %s", result)

    scheduler.add_job(
        scheduled_digest,
        trigger=CronTrigger(hour=settings.digest_hour_1, minute=0, timezone=settings.timezone),
        id="digest_morning",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=600,
    )
    scheduler.add_job(
        scheduled_digest,
        trigger=CronTrigger(hour=settings.digest_hour_2, minute=0, timezone=settings.timezone),
        id="digest_evening",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=600,
    )
    scheduler.start()
    logger.info(
        "Расписание активно: %02d:00 и %02d:00 (%s)",
        settings.digest_hour_1,
        settings.digest_hour_2,
        settings.timezone,
    )

    stop_event = asyncio.Event()

    def _on_signal() -> None:
        logger.info("Получен сигнал остановки.")
        stop_event.set()

    # Обработка остановки по Ctrl+C
    loop = asyncio.get_running_loop()
    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _on_signal)

    try:
        polling_task = asyncio.create_task(dp.start_polling(bot, handle_signals=(sys.platform != "win32")))
        await stop_event.wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        logger.info("Останавливаем сервисы...")
        scheduler.shutdown(wait=False)
        polling_task.cancel()
        try:
            await polling_task
        except (asyncio.CancelledError, Exception):
            pass
        await bot.session.close()
        logger.info("Бот остановлен.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Работа завершена.")
        sys.exit(0)
