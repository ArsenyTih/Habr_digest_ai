from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

import aiosqlite

from config import settings

logger = logging.getLogger(__name__)

DDL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS articles (
    url          TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    published_at TEXT,
    text         TEXT,
    sent_at      TEXT,
    status       TEXT NOT NULL DEFAULT 'rejected',
    score        INTEGER,
    topics       TEXT
);

CREATE TABLE IF NOT EXISTS user_preferences (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    topic  TEXT NOT NULL UNIQUE,
    weight INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS feedback (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    url        TEXT NOT NULL,
    rating     INTEGER NOT NULL,
    comment    TEXT,
    created_at TEXT NOT NULL
);
"""


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


async def init_db() -> None:
    settings.ensure_data_dirs()
    async with aiosqlite.connect(settings.sqlite_path) as db:
        await db.executescript(DDL)
        await db.commit()

        # Если таблица предпочтений пустая, заполняем темами по умолчанию
        cursor = await db.execute("SELECT COUNT(*) FROM user_preferences")
        row = await cursor.fetchone()
        if row and row[0] == 0:
            for topic in settings.get_default_topics_list():
                await db.execute(
                    "INSERT OR IGNORE INTO user_preferences (topic, weight) VALUES (?, 1)",
                    (topic,),
                )
            await db.commit()


async def get_seen_urls() -> set[str]:
    async with aiosqlite.connect(settings.sqlite_path) as db:
        cursor = await db.execute("SELECT url FROM articles")
        rows = await cursor.fetchall()
        return {row[0] for row in rows}


async def save_article(
    url: str,
    title: str,
    published_at: Optional[str],
    text: Optional[str],
    status: str,
    score: Optional[int] = None,
    topics: Optional[list[str]] = None,
    sent_at: Optional[str] = None,
) -> None:
    topics_json = json.dumps(topics, ensure_ascii=False) if topics else None
    async with aiosqlite.connect(settings.sqlite_path) as db:
        await db.execute(
            """
            INSERT INTO articles (url, title, published_at, text, sent_at, status, score, topics)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
                title        = excluded.title,
                published_at = excluded.published_at,
                text         = excluded.text,
                sent_at      = excluded.sent_at,
                status       = excluded.status,
                score        = excluded.score,
                topics       = excluded.topics
            """,
            (url, title, published_at, text, sent_at, status, score, topics_json),
        )
        await db.commit()


async def mark_articles_sent(urls: list[str], sent_at: str) -> None:
    if not urls:
        return
    async with aiosqlite.connect(settings.sqlite_path) as db:
        await db.executemany(
            "UPDATE articles SET status = 'sent', sent_at = ? WHERE url = ?",
            [(sent_at, url) for url in urls],
        )
        await db.commit()


async def mark_articles_rejected(items: list[tuple[str, str, int, list[str]]]) -> None:
    for url, title, score, topics in items:
        await save_article(
            url=url,
            title=title,
            published_at=None,
            text=None,
            status="rejected",
            score=score,
            topics=topics,
        )


async def mark_articles_skipped(urls_titles: list[tuple[str, str]]) -> None:
    for url, title in urls_titles:
        await save_article(url=url, title=title, published_at=None, text=None, status="skipped")


async def get_preferences() -> list[dict]:
    async with aiosqlite.connect(settings.sqlite_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT topic, weight FROM user_preferences ORDER BY weight DESC, topic")
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


async def add_preference(topic: str) -> bool:
    try:
        async with aiosqlite.connect(settings.sqlite_path) as db:
            await db.execute(
                "INSERT INTO user_preferences (topic, weight) VALUES (?, 1)",
                (topic,),
            )
            await db.commit()
        return True
    except aiosqlite.IntegrityError:
        return False


async def remove_preference(topic: str) -> bool:
    async with aiosqlite.connect(settings.sqlite_path) as db:
        cursor = await db.execute("DELETE FROM user_preferences WHERE topic = ?", (topic,))
        await db.commit()
        return cursor.rowcount > 0


async def update_preference_weight(topic: str, delta: int) -> None:
    # Ограничиваем вес темы в диапазоне от 1 до 10
    async with aiosqlite.connect(settings.sqlite_path) as db:
        await db.execute(
            """
            INSERT INTO user_preferences (topic, weight) VALUES (?, MAX(1, MIN(10, 1 + ?)))
            ON CONFLICT(topic) DO UPDATE SET
                weight = MAX(1, MIN(10, weight + ?))
            """,
            (topic, delta, delta),
        )
        await db.commit()


async def save_feedback(url: str, rating: int, comment: Optional[str] = None) -> None:
    async with aiosqlite.connect(settings.sqlite_path) as db:
        await db.execute(
            "INSERT INTO feedback (url, rating, comment, created_at) VALUES (?, ?, ?, ?)",
            (url, rating, comment, _now_utc()),
        )
        await db.commit()


async def get_last_digest_articles() -> list[dict]:
    async with aiosqlite.connect(settings.sqlite_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """
            SELECT url, title, sent_at, score, topics
            FROM articles
            WHERE status = 'sent' AND sent_at = (
                SELECT MAX(sent_at) FROM articles WHERE status = 'sent'
            )
            ORDER BY score DESC
            """
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


async def get_last_sent_articles(limit: int = 20) -> list[dict]:
    async with aiosqlite.connect(settings.sqlite_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """
            SELECT url, title, sent_at, score
            FROM articles
            WHERE status = 'sent' AND sent_at IS NOT NULL
            ORDER BY sent_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


async def get_article_topics(url: str) -> list[str]:
    async with aiosqlite.connect(settings.sqlite_path) as db:
        cursor = await db.execute("SELECT topics FROM articles WHERE url = ?", (url,))
        row = await cursor.fetchone()
        if not row or not row[0]:
            return []
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return []
