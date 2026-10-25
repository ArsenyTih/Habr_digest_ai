from __future__ import annotations

import asyncio
import html
import logging
from typing import Optional

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from config import settings

logger = logging.getLogger(__name__)

TG_LIMIT = 4096


class DigestItem(BaseModel):
    index: int = Field(description="Номер статьи из входного списка (начиная с 0)")
    russian_title: str = Field(description="Понятный заголовок статьи")
    brief: str = Field(description="Краткая суть в 1-2 предложениях")
    benefit: str = Field(description="Чем полезна статья")


class DigestOutput(BaseModel):
    items: list[DigestItem] = Field(description="Статьи для дайджеста")


def _format_article_block(
    num: int,
    title: str,
    url: str,
    brief: str,
    benefit: str,
    score: int,
) -> str:
    # Экранируем HTML, чтобы Telegram не ругался на некорректные теги
    t = html.escape(title)
    b = html.escape(brief)
    bene = html.escape(benefit)
    safe_url = html.escape(url)
    return (
        f"<b>{num}. <a href=\"{safe_url}\">{t}</a></b>\n"
        f"{b}\n"
        f"💡 <i>{bene}</i>\n"
        f"⭐ Оценка: {score}/10\n"
    )


def _split_into_messages(blocks: list[str], header: str = "") -> list[str]:
    # Разбиваем дайджест на несколько сообщений, если превышен лимит 4096 символов
    messages: list[str] = []
    current = header

    for block in blocks:
        candidate = (current + "\n" + block).strip() if current else block
        if len(candidate) <= TG_LIMIT:
            current = candidate
        else:
            if current:
                messages.append(current)
            if len(block) > TG_LIMIT:
                block = block[: TG_LIMIT - 3] + "..."
            current = block

    if current:
        messages.append(current)
    return messages or ["Нет данных."]


async def create_digest(
    passed_articles: list[dict],
    dry_run: bool = False,
) -> tuple[list[str], list[str], list[tuple[str, str]]]:
    if not passed_articles:
        notice = "Новых интересных публикаций нет."
        if dry_run:
            notice = f"[ТЕСТОВЫЙ ЗАПУСК]\n{notice}"
        return [notice], [], []

    # Сортируем статьи по убыванию оценки и берем топ
    sorted_articles = sorted(passed_articles, key=lambda a: a["score"], reverse=True)
    selected = sorted_articles[: settings.max_digest_items]
    skipped = sorted_articles[settings.max_digest_items :]
    skipped_pairs = [(a["url"], a["title"]) for a in skipped]

    items_text = []
    for i, art in enumerate(selected):
        items_text.append(
            f"[{i}] {art['title']}\n"
            f"Суть: {art.get('summary', '')}\n"
            f"Польза: {art.get('reason', '')}"
        )

    prompt = (
        "Ты редактор дайджеста. Отредактируй статьи:\n"
        "1. Улучши заголовок (если нужно).\n"
        "2. Напиши краткое содержание (1-2 предложения).\n"
        "3. Напиши, чем полезна статья.\n\n"
        "Используй только предоставленный текст, ничего не придумывай.\n\n"
        + "\n\n".join(items_text)
    )

    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,  # type: ignore[arg-type]
        base_url=settings.effective_openai_base_url,
        temperature=0.5,
        max_tokens=1500,
    )
    llm_with_output = llm.with_structured_output(DigestOutput)

    edited_map: dict[int, DigestItem] = {}
    try:
        if settings.llm_delay_seconds > 0:
            await asyncio.sleep(settings.llm_delay_seconds)

        digest_output: DigestOutput = await llm_with_output.ainvoke(  # type: ignore[assignment]
            [{"role": "user", "content": prompt}]
        )
        for item in digest_output.items:
            if 0 <= item.index < len(selected):
                edited_map[item.index] = item
    except Exception as exc:
        logger.error("Ошибка редактора при вызове модели: %s", exc)

    # Собираем блоки (ссылки берем строго из проверенных данных статьи)
    blocks: list[str] = []
    digest_urls: list[str] = []

    for i, art in enumerate(selected):
        url = art["url"]
        digest_urls.append(url)

        if i in edited_map:
            item = edited_map[i]
            title = item.russian_title or art["title"]
            brief = item.brief or art.get("summary", "")
            benefit = item.benefit or art.get("reason", "")
        else:
            title = art["title"]
            brief = art.get("summary", "")
            benefit = art.get("reason", "")

        blocks.append(
            _format_article_block(
                num=i + 1,
                title=title,
                url=url,
                brief=brief,
                benefit=benefit,
                score=art["score"],
            )
        )

    test_mark = "[ТЕСТОВЫЙ ЗАПУСК]\n" if dry_run else ""
    header = f"{test_mark}📰 <b>Дайджест Хабра</b>\n"

    messages = _split_into_messages(blocks, header=header)
    return messages, digest_urls, skipped_pairs
