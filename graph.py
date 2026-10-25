from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from agents.analyst import analyze_articles
from agents.collector import collect_articles
from agents.editor import create_digest
from config import settings
from database import (
    get_seen_urls,
    mark_articles_rejected,
    mark_articles_sent,
    mark_articles_skipped,
    save_article,
)
from rag import RAGStore
from rss_parser import fetch_rss

logger = logging.getLogger(__name__)

# Блокировка, чтобы не запускать несколько генераций дайджеста одновременно
_digest_lock = asyncio.Lock()


class DigestState(TypedDict):
    rss_items: list[dict]
    fetched_articles: list[dict]
    passed_articles: list[dict]
    rejected_articles: list[dict]
    digest_parts: list[str]
    recipient_id: int
    digest_urls: list[str]
    skipped_pairs: list[tuple[str, str]]
    errors: list[str]
    dry_run: bool


def build_graph(bot: Any, rag: RAGStore):
    async def node_collect(state: DigestState) -> dict:
        seen_urls = await get_seen_urls()
        rss_items = state.get("rss_items") or []

        if not rss_items:
            rss_items = await fetch_rss()

        articles, errors = await collect_articles(rss_items, seen_urls)
        return {
            "fetched_articles": articles,
            "errors": state.get("errors", []) + errors,
        }

    async def node_analyze(state: DigestState) -> dict:
        articles = state.get("fetched_articles", [])
        if not articles:
            return {"passed_articles": [], "rejected_articles": []}

        passed, rejected = await analyze_articles(articles, rag)

        # Сохраняем отсеянные статьи в базу (чтобы повторно не скачивать)
        if not state.get("dry_run"):
            rejected_data = [
                (a["url"], a["title"], a.get("score", 0), a.get("topics", []))
                for a in rejected
            ]
            await mark_articles_rejected(rejected_data)

        return {"passed_articles": passed, "rejected_articles": rejected}

    async def node_create_digest(state: DigestState) -> dict:
        passed = state.get("passed_articles", [])
        dry_run = state.get("dry_run", False)
        parts, digest_urls, skipped_pairs = await create_digest(passed, dry_run=dry_run)
        return {
            "digest_parts": parts,
            "digest_urls": digest_urls,
            "skipped_pairs": skipped_pairs,
        }

    async def node_send_digest(state: DigestState) -> dict:
        parts = state.get("digest_parts", [])
        recipient_id = state.get("recipient_id", settings.telegram_user_id)
        dry_run = state.get("dry_run", False)
        digest_urls = state.get("digest_urls", [])
        skipped_pairs = state.get("skipped_pairs", [])
        passed_articles = state.get("passed_articles", [])
        errors = state.get("errors", [])

        if not parts:
            parts = ["Нет данных для отправки."]

        send_success = True
        try:
            for part in parts:
                await bot.send_message(
                    chat_id=recipient_id,
                    text=part,
                    parse_mode="HTML",
                    disable_web_page_preview=True,
                )
        except Exception as exc:
            logger.error("Ошибка отправки в Telegram: %s", exc)
            errors.append(f"Ошибка Telegram: {exc}")
            send_success = False

        # В тестовом режиме базу не меняем
        if dry_run or not send_success:
            return {"errors": errors}

        sent_at = datetime.now(timezone.utc).isoformat()
        if digest_urls:
            await mark_articles_sent(digest_urls, sent_at)

        articles_by_url = {a["url"]: a for a in passed_articles}
        for url in digest_urls:
            art = articles_by_url.get(url)
            if art:
                await save_article(
                    url=art["url"],
                    title=art["title"],
                    published_at=art.get("published_at"),
                    text=art.get("text"),
                    status="sent",
                    score=art.get("score"),
                    topics=art.get("topics"),
                    sent_at=sent_at,
                )
                if art.get("text"):
                    await rag.index_article(url, art["title"], art["text"])

        if skipped_pairs:
            await mark_articles_skipped(skipped_pairs)

        return {"errors": errors}

    builder = StateGraph(DigestState)
    builder.add_node("collect_articles", node_collect)
    builder.add_node("analyze_articles", node_analyze)
    builder.add_node("create_digest", node_create_digest)
    builder.add_node("send_digest", node_send_digest)

    builder.add_edge(START, "collect_articles")
    builder.add_edge("collect_articles", "analyze_articles")
    builder.add_edge("analyze_articles", "create_digest")
    builder.add_edge("create_digest", "send_digest")
    builder.add_edge("send_digest", END)

    return builder.compile()


async def run_digest(
    graph: Any,
    dry_run: bool = False,
    manual: bool = False,
    rss_items: Optional[list[dict]] = None,
) -> str:
    if _digest_lock.locked():
        return "Дайджест уже формируется. Подождите пару минут."

    async with _digest_lock:
        state: DigestState = {
            "rss_items": rss_items or [],
            "fetched_articles": [],
            "passed_articles": [],
            "rejected_articles": [],
            "digest_parts": [],
            "recipient_id": settings.telegram_user_id,
            "digest_urls": [],
            "skipped_pairs": [],
            "errors": [],
            "dry_run": dry_run or settings.dry_run,
        }

        try:
            result = await graph.ainvoke(state)
            errors = result.get("errors", [])
            if errors:
                return f"Дайджест готов. Были ошибки: {'; '.join(errors)}"
            return "Дайджест успешно отправлен."
        except Exception as exc:
            logger.exception("Ошибка при выполнении графа: %s", exc)
            return f"Ошибка формирования дайджеста: {exc}"
