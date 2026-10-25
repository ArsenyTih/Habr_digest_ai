from __future__ import annotations

import html
import logging
import re
import sys
from datetime import datetime, timezone
from typing import Optional

import feedparser
import httpx

from config import settings

logger = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) HabrDigestBot/1.0"
DESCRIPTION_MAX_CHARS = 500
TIMEOUT = 20.0


def _strip_html(text: str) -> str:
    # Очищаем текст от HTML-тегов и декодируем спецсимволы
    clean = re.sub(r"<[^>]+>", " ", text)
    clean = html.unescape(clean)
    return re.sub(r"\s+", " ", clean).strip()


def _parse_date(entry: feedparser.util.FeedParserDict) -> Optional[str]:
    try:
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            return dt.isoformat()
    except Exception:
        pass
    return None


async def fetch_rss(url: Optional[str] = None) -> list[dict]:
    rss_url = url or settings.habr_rss_url

    try:
        async with httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT},
            timeout=TIMEOUT,
            follow_redirects=True,
        ) as client:
            response = await client.get(rss_url)
            response.raise_for_status()
            raw_content = response.content
    except Exception as exc:
        logger.error("Ошибка при получении RSS: %s", exc)
        return []

    feed = feedparser.parse(raw_content)
    articles: list[dict] = []

    for entry in feed.entries:
        try:
            link = getattr(entry, "link", None)
            title = getattr(entry, "title", "").strip()
            if not link or not title:
                continue

            # Очищаем ссылку от UTM-меток, чтобы robots.txt Хабра не блокировал загрузку
            link = link.split("?")[0].rstrip("/") + "/"

            raw_desc = (
                getattr(entry, "summary", "")
                or getattr(entry, "description", "")
                or ""
            )
            description = _strip_html(raw_desc)[:DESCRIPTION_MAX_CHARS]

            articles.append(
                {
                    "title": title,
                    "url": link,
                    "published_at": _parse_date(entry),
                    "description": description,
                }
            )
        except Exception:
            continue

    logger.info("Получено статей из RSS: %d", len(articles))
    return articles


if __name__ == "__main__":
    import asyncio

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    async def _main() -> None:
        url = sys.argv[1] if len(sys.argv) > 1 else None
        items = await fetch_rss(url)
        print(f"Всего статей: {len(items)}")
        for i, item in enumerate(items[:3], 1):
            print(f"{i}. [{item['published_at']}] {item['title']}")
            print(f"   {item['url']}")

    asyncio.run(_main())
