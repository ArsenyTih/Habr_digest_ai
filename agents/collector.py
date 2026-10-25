from __future__ import annotations

import logging

from mcp_client import fetch_url, open_fetch_session
from config import settings

logger = logging.getLogger(__name__)


async def collect_articles(
    rss_items: list[dict],
    seen_urls: set[str],
) -> tuple[list[dict], list[str]]:
    errors: list[str] = []

    # Отсекаем статьи, которые уже были в прошлых дайджестах
    new_items = [item for item in rss_items if item["url"] not in seen_urls]
    new_items = new_items[: settings.max_new_articles]

    if not new_items:
        logger.info("Новых статей нет.")
        return [], errors

    logger.info("Загружаем через MCP новые статьи: %d шт.", len(new_items))

    articles: list[dict] = []
    try:
        async with open_fetch_session() as session:
            for item in new_items:
                url = item["url"]
                try:
                    text = await fetch_url(session, url)
                    if not text or not text.strip():
                        errors.append(f"Пустой текст: {url}")
                        continue

                    articles.append(
                        {
                            "title": item["title"],
                            "url": url,
                            "published_at": item.get("published_at"),
                            "text": text,
                        }
                    )
                except Exception as exc:
                    msg = f"Ошибка загрузки {url}: {exc}"
                    logger.error(msg)
                    errors.append(msg)
    except Exception as exc:
        msg = f"MCP-сервер недоступен: {exc}"
        logger.error(msg)
        errors.append(msg)
        return [], errors

    return articles, errors
