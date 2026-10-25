from __future__ import annotations

import asyncio
import html
import logging
from typing import Callable, Awaitable, Any

from aiogram import Bot, Dispatcher, F, Router
from aiogram.exceptions import TelegramAPIError, TelegramRetryAfter
from aiogram.filters import Command, CommandObject
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from config import settings
from database import (
    add_preference,
    get_last_digest_articles,
    get_last_sent_articles,
    get_preferences,
    remove_preference,
    save_feedback,
    update_preference_weight,
    get_article_topics,
)
from rag import RAGStore

logger = logging.getLogger(__name__)


async def safe_send(message: Message, text: str, **kwargs: Any) -> None:
    try:
        await message.answer(text, **kwargs)
    except TelegramRetryAfter as exc:
        await asyncio.sleep(exc.retry_after)
        await message.answer(text, **kwargs)
    except TelegramAPIError as exc:
        logger.error("Ошибка Telegram API: %s", exc)


async def safe_reply(callback: CallbackQuery, text: str) -> None:
    try:
        await callback.answer(text)
    except TelegramAPIError as exc:
        logger.error("Ошибка Telegram callback: %s", exc)


def create_router(
    run_digest_callback: Callable[[bool, bool], Awaitable[str]],
    rag: RAGStore,
) -> Router:
    router = Router()

    # Фильтруем все сообщения: бот отвечает только владельцу
    router.message.filter(F.from_user.id == settings.telegram_user_id)
    router.callback_query.filter(F.from_user.id == settings.telegram_user_id)

    @router.message(Command("start"))
    async def cmd_start(message: Message) -> None:
        text = (
            "👋 <b>Бот дайджестов Хабра</b>\n\n"
            "Команды:\n"
            "/digest — собрать дайджест сейчас\n"
            "/digest_test — тестовый прогон (без записи в базу)\n"
            "/topics — список моих тем\n"
            "/addtopic &lt;тема&gt; — добавить тему\n"
            "/removetopic &lt;тема&gt; — удалить тему\n"
            "/history — последние статьи\n"
            "/feedback — оценить последний дайджест"
        )
        await safe_send(message, text, parse_mode="HTML")

    @router.message(Command("digest"))
    async def cmd_digest(message: Message) -> None:
        await safe_send(message, "⏳ Собираю и анализирую статьи...")
        result = await run_digest_callback(False, True)
        if "уже формируется" in result:
            await safe_send(message, result)

    @router.message(Command("digest_test"))
    async def cmd_digest_test(message: Message) -> None:
        await safe_send(message, "⏳ Тестовый запуск дайджеста...")
        result = await run_digest_callback(True, True)
        if "уже формируется" in result:
            await safe_send(message, result)

    @router.message(Command("topics"))
    async def cmd_topics(message: Message) -> None:
        prefs = await get_preferences()
        if not prefs:
            await safe_send(message, "Список тем пуст. Добавьте: /addtopic Название")
            return
        lines = ["<b>Мои темы:</b>"]
        for p in prefs:
            lines.append(f"  • {html.escape(p['topic'])} — вес {p['weight']}/10")
        await safe_send(message, "\n".join(lines), parse_mode="HTML")

    @router.message(Command("addtopic"))
    async def cmd_addtopic(message: Message, command: CommandObject) -> None:
        topic = (command.args or "").strip()
        if not topic:
            await safe_send(message, "Пример: /addtopic Python")
            return
        added = await add_preference(topic)
        if added:
            await safe_send(message, f"Тема «{html.escape(topic)}» добавлена.")
        else:
            await safe_send(message, f"Тема «{html.escape(topic)}» уже есть.")

    @router.message(Command("removetopic"))
    async def cmd_removetopic(message: Message, command: CommandObject) -> None:
        topic = (command.args or "").strip()
        if not topic:
            await safe_send(message, "Пример: /removetopic Python")
            return
        removed = await remove_preference(topic)
        if removed:
            await safe_send(message, f"Тема «{html.escape(topic)}» удалена.")
        else:
            await safe_send(message, f"Тема «{html.escape(topic)}» не найдена.")

    @router.message(Command("history"))
    async def cmd_history(message: Message) -> None:
        articles = await get_last_sent_articles(limit=15)
        if not articles:
            await safe_send(message, "История пуста, дайджестов ещё не было.")
            return
        lines = ["<b>Последние отправленные статьи:</b>\n"]
        for art in articles:
            title = html.escape(art["title"])
            url = html.escape(art["url"])
            sent = art["sent_at"][:10] if art.get("sent_at") else "?"
            score = art.get("score") or "?"
            lines.append(f"• <a href=\"{url}\">{title}</a> ({sent}, ⭐{score})")
        await safe_send(message, "\n".join(lines), parse_mode="HTML", disable_web_page_preview=True)

    @router.message(Command("feedback"))
    async def cmd_feedback(message: Message) -> None:
        articles = await get_last_digest_articles()
        if not articles:
            await safe_send(message, "Дайджестов в базе пока нет.")
            return

        lines = ["<b>Статьи последнего дайджеста:</b>\n"]
        for art in articles:
            lines.append(f"• <a href=\"{html.escape(art['url'])}\">{html.escape(art['title'])}</a>")
        lines.append("\nОцените качество подборки:")

        buttons = [
            [
                InlineKeyboardButton(text=str(i), callback_data=f"feedback:{i}")
                for i in range(1, 6)
            ]
        ]
        keyboard = InlineKeyboardMarkup(inline_keyboard=buttons)
        await safe_send(
            message,
            "\n".join(lines),
            parse_mode="HTML",
            disable_web_page_preview=True,
            reply_markup=keyboard,
        )

    @router.callback_query(F.data.startswith("feedback:"))
    async def callback_feedback(callback: CallbackQuery) -> None:
        try:
            rating = int(callback.data.split(":")[1])  # type: ignore[union-attr]
        except (ValueError, IndexError):
            await safe_reply(callback, "Ошибка данных.")
            return

        articles = await get_last_digest_articles()
        if not articles:
            await safe_reply(callback, "Дайджест не найден.")
            return

        for art in articles:
            url = art["url"]
            await save_feedback(url, rating)

            topics = await get_article_topics(url)
            if rating >= 4:
                for topic in topics:
                    await update_preference_weight(topic, +1)
            elif rating <= 2:
                for topic in topics:
                    await update_preference_weight(topic, -1)
                await rag.delete_article(url)

        await safe_reply(callback, f"Оценка {rating}/5 сохранена!")
        try:
            if callback.message:
                await callback.message.edit_text(  # type: ignore[union-attr]
                    f"Спасибо! Оценка <b>{rating}/5</b> учтена для будущих подборок.",
                    parse_mode="HTML",
                )
        except TelegramAPIError:
            pass

    return router


def create_bot_and_dispatcher(
    run_digest_callback: Callable[[bool, bool], Awaitable[str]],
    rag: RAGStore,
) -> tuple[Bot, Dispatcher]:
    bot = Bot(token=settings.telegram_bot_token)
    dp = Dispatcher()
    dp.include_router(create_router(run_digest_callback, rag))
    return bot, dp
