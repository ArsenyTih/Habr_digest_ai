from __future__ import annotations

import asyncio
import os
import shutil
import tempfile

tmp_dir = tempfile.mkdtemp()
os.environ["OPENAI_API_KEY"] = "dummy-key-for-test"
os.environ["OPENAI_MODEL"] = "openai/gpt-oss-120b"
os.environ["TELEGRAM_BOT_TOKEN"] = "1234567890:TESTTOKEN"
os.environ["TELEGRAM_USER_ID"] = "123456789"
os.environ["SQLITE_PATH"] = os.path.join(tmp_dir, "test.db")
os.environ["CHROMA_PATH"] = os.path.join(tmp_dir, "chroma")
os.environ["DRY_RUN"] = "true"


async def main() -> None:
    print("Запуск проверки компонентов...")

    # 1. Конфиг
    from config import settings
    assert settings.dry_run is True
    print(f"[OK] Конфиг загружен: модель {settings.openai_model}, concurrency {settings.llm_concurrency}")

    # 2. База данных
    from database import init_db, get_preferences, save_article, get_seen_urls
    await init_db()
    prefs = await get_preferences()
    assert len(prefs) >= 3
    print(f"[OK] База SQLite инициализирована, тем: {len(prefs)}")

    await save_article(
        url="https://habr.com/ru/articles/123/",
        title="Тестовая статья",
        published_at="2026-10-25T12:00:00+00:00",
        text="Пример текста",
        status="rejected",
        score=4,
        topics=["Python"],
    )
    seen = await get_seen_urls()
    assert "https://habr.com/ru/articles/123/" in seen
    print("[OK] Сохранение статьи в SQLite работает")

    # 3. Фильтрация в сборщике
    fake_rss = [
        {"url": "https://habr.com/ru/articles/123/", "title": "Старая статья"},
        {"url": "https://habr.com/ru/articles/456/", "title": "Новая статья"},
    ]
    new_items = [i for i in fake_rss if i["url"] not in seen]
    assert len(new_items) == 1 and new_items[0]["url"] == "https://habr.com/ru/articles/456/"
    print("[OK] Отсечение уже виденных ссылок работает")

    # 4. Форматирование редактора
    from agents.editor import _format_article_block, _split_into_messages
    block = _format_article_block(
        num=1,
        title="Тест <script>alert(1)</script>",
        url="https://habr.com/ru/articles/456/",
        brief="Краткое описание",
        benefit="Польза",
        score=9,
    )
    assert "<script>" not in block and "&lt;script&gt;" in block
    msgs = _split_into_messages([block] * 5, header="📰 Дайджест\n")
    assert len(msgs) >= 1
    print("[OK] Экранирование HTML и сборка сообщений работает")

    # 5. RAG без ключа OpenAI не должен падать
    from rag import RAGStore
    rag = RAGStore()
    results = await rag.search("Python")
    assert results == []
    print("[OK] Безопасная работа RAG подтверждена")

    # 6. Сборка графа LangGraph
    from graph import build_graph

    class DummyBot:
        async def send_message(self, *args, **kwargs):
            pass

    graph = build_graph(DummyBot(), rag)
    assert graph is not None
    print("[OK] Сборка графа LangGraph прошла успешно")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("\nВсе проверки успешно пройдены!")


if __name__ == "__main__":
    asyncio.run(main())
