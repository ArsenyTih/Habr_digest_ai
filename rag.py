from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Optional

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "habr_articles"
CHUNK_SIZE = 2500
CHUNK_OVERLAP = 250
MAX_RESULTS = 3


def _chunk_id(url: str, idx: int) -> str:
    # Уникальный хеш для фрагмента статьи
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    return f"{url_hash}-{idx}"


class RAGStore:
    def __init__(self) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        self._vectorstore: Optional[Chroma] = None

        # Проверяем, есть ли валидные настройки для эмбеддингов
        api_key = settings.effective_embedding_api_key
        base_url = settings.effective_embedding_base_url

        # Если используется Groq (ключ gsk_...) и нет отдельного ключа для эмбеддингов — отключаем RAG
        if not api_key or (api_key.startswith("gsk_") and not settings.embedding_api_key) or (base_url and "groq.com" in base_url):
            logger.info("Не задан отдельный ключ эмбеддингов. RAG отключен.")
            return

        try:
            embeddings = OpenAIEmbeddings(
                model=settings.embedding_model,
                api_key=api_key,  # type: ignore[arg-type]
                base_url=base_url,
            )
            self._vectorstore = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
                persist_directory=settings.chroma_path,
            )
        except Exception as exc:
            logger.warning("Не удалось инициализировать ChromaDB: %s. Работаем без RAG.", exc)
            self._vectorstore = None

    async def index_article(self, url: str, title: str, text: str) -> None:
        if not self._vectorstore:
            return

        try:
            already = await asyncio.to_thread(self._has_url, url)
            if already:
                return

            full_text = f"{title}\n\n{text}"
            chunks = self._splitter.split_text(full_text)
            if not chunks:
                return

            documents = [
                Document(
                    page_content=chunk,
                    metadata={"url": url, "title": title},
                )
                for chunk in chunks
            ]
            ids = [_chunk_id(url, i) for i in range(len(chunks))]

            await asyncio.to_thread(self._vectorstore.add_documents, documents, ids=ids)
        except Exception as exc:
            logger.warning("Ошибка индексации статьи в RAG %s: %s", url, exc)

    def _has_url(self, url: str) -> bool:
        if not self._vectorstore:
            return False
        try:
            results = self._vectorstore.get(where={"url": url}, limit=1)
            return bool(results and results.get("ids"))
        except Exception:
            return False

    async def search(self, query: str, exclude_url: Optional[str] = None) -> list[dict]:
        if not self._vectorstore or not query.strip():
            return []

        try:
            kwargs: dict = {"k": MAX_RESULTS}
            if exclude_url:
                kwargs["filter"] = {"url": {"$ne": exclude_url}}

            docs = await asyncio.to_thread(self._vectorstore.similarity_search, query, **kwargs)
            return [
                {
                    "title": doc.metadata.get("title", ""),
                    "url": doc.metadata.get("url", ""),
                    "text": doc.page_content,
                }
                for doc in docs
            ]
        except Exception as exc:
            logger.warning("Ошибка поиска в RAG: %s", exc)
            return []

    def build_context_prompt(self, fragments: list[dict]) -> str:
        if not fragments:
            return ""

        lines = ["Ранее прочитанные похожие статьи:"]
        for f in fragments:
            lines.append(f"- «{f['title']}» ({f['url']})")
            lines.append(f"  Фрагмент: {f['text'][:250]}")
        return "\n".join(lines)

    async def delete_article(self, url: str) -> None:
        if not self._vectorstore:
            return
        try:
            await asyncio.to_thread(self._delete_by_url, url)
        except Exception as exc:
            logger.warning("Ошибка удаления из RAG %s: %s", url, exc)

    def _delete_by_url(self, url: str) -> None:
        if not self._vectorstore:
            return
        results = self._vectorstore.get(where={"url": url})
        ids = results.get("ids", [])
        if ids:
            self._vectorstore.delete(ids=ids)
