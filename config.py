from __future__ import annotations

import shlex
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Настройки LLM
    openai_api_key: str = Field(..., repr=False)
    openai_model: str = "openai/gpt-oss-120b"
    openai_base_url: Optional[str] = None

    # Настройки эмбеддингов
    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: Optional[str] = Field(default=None, repr=False)
    embedding_base_url: Optional[str] = None

    # Бот Telegram
    telegram_bot_token: str = Field(..., repr=False)
    telegram_user_id: int

    # Источник статей
    habr_rss_url: str = "https://habr.com/ru/rss/articles/"

    # Время отправки дайджестов
    timezone: str = "Europe/Moscow"
    digest_hour_1: int = 9
    digest_hour_2: int = 18

    # Базы данных
    sqlite_path: str = "data/digest.db"
    chroma_path: str = "data/chroma"

    # MCP-сервер для загрузки страниц
    mcp_server_command: str = "python"
    mcp_server_args: str = "-m mcp_server_fetch"
    mcp_ignore_robots: bool = True
    fetch_max_length: int = 20000
    mcp_timeout_seconds: int = 60

    # Лимиты для модели (подогнаны под 30 RPM, 8K TPM, 1K/день)
    article_max_chars_for_llm: int = 2500
    min_score: int = 6
    max_new_articles: int = 5
    max_digest_items: int = 3
    llm_concurrency: int = 1
    llm_delay_seconds: float = 2.5

    # Общие настройки
    default_topics: str = "Python, ИИ-агенты, LLM"
    send_empty_notice: bool = False
    dry_run: bool = False

    @field_validator("openai_api_key", "telegram_bot_token", mode="before")
    @classmethod
    def _check_not_empty(cls, value: str) -> str:
        if not value or not str(value).strip():
            raise ValueError("Ключ или токен не может быть пустым")
        return str(value).strip()

    def get_default_topics_list(self) -> list[str]:
        return [t.strip() for t in self.default_topics.split(",") if t.strip()]

    def get_mcp_args(self) -> list[str]:
        raw = self.mcp_server_args.strip()
        if raw.startswith("["):
            import json
            args = json.loads(raw)
        else:
            args = shlex.split(raw)

        if self.mcp_ignore_robots and "--ignore-robots-txt" not in args:
            args.append("--ignore-robots-txt")
        return args

    def ensure_data_dirs(self) -> None:
        Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.chroma_path).mkdir(parents=True, exist_ok=True)

    @property
    def effective_openai_base_url(self) -> Optional[str]:
        if self.openai_base_url and self.openai_base_url.strip():
            return self.openai_base_url.strip()
        # Автоматически подставляем Groq, если ключ от Groq или выбрана модель gpt-oss
        if (self.openai_api_key and self.openai_api_key.startswith("gsk_")) or "gpt-oss" in self.openai_model.lower():
            return "https://api.groq.com/openai/v1"
        return None

    @property
    def effective_embedding_api_key(self) -> str:
        return self.embedding_api_key or self.openai_api_key

    @property
    def effective_embedding_base_url(self) -> Optional[str]:
        if self.embedding_base_url and self.embedding_base_url.strip():
            return self.embedding_base_url.strip()
        # У Groq нет сервиса эмбеддингов, не отправляем туда запросы
        base = self.effective_openai_base_url
        if base and "groq.com" in base:
            return None
        return base


settings = Settings()
