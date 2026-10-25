from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

from config import settings

logger = logging.getLogger(__name__)


def _resolve_command(command: str) -> str:
    # Если указан python, используем интерпретатор текущего окружения
    if command.lower() in ("python", "python3", "py"):
        return sys.executable
    return command


@asynccontextmanager
async def open_fetch_session() -> AsyncIterator[ClientSession]:
    # Запуск MCP-сервера в отдельном процессе через stdio
    server_params = StdioServerParameters(
        command=_resolve_command(settings.mcp_server_command),
        args=settings.get_mcp_args(),
        env=None,
    )
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            yield session


async def list_tools(session: ClientSession) -> list[dict]:
    # Получаем список инструментов от MCP-сервера
    result = await session.list_tools()
    tools = []
    for tool in result.tools:
        schema = getattr(tool, "input_schema", getattr(tool, "inputSchema", {})) or {}
        tools.append(
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": schema,
            }
        )
    return tools


async def fetch_url(session: ClientSession, url: str) -> Optional[str]:
    # Ищем подходящий инструмент (обычно fetch)
    tools = await list_tools(session)
    fetch_tool = next((t for t in tools if "fetch" in t["name"].lower()), None)
    if fetch_tool is None:
        names = [t["name"] for t in tools]
        raise RuntimeError(f"Инструмент fetch не найден. Доступно: {names}")

    try:
        result = await asyncio.wait_for(
            session.call_tool(
                fetch_tool["name"],
                arguments={
                    "url": url,
                    "max_length": settings.fetch_max_length,
                },
            ),
            timeout=settings.mcp_timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.error("Таймаут MCP при загрузке %s", url)
        return None
    except Exception as exc:
        logger.error("Ошибка MCP при загрузке %s: %s", url, exc)
        return None

    is_err = getattr(result, "is_error", getattr(result, "isError", False))
    if is_err or not result.content:
        return None

    parts: list[str] = []
    for part in result.content:
        if isinstance(part, TextContent) and part.text:
            parts.append(part.text)

    if not parts:
        return None

    return "\n".join(parts)[: settings.fetch_max_length]


async def _cli_list_tools() -> None:
    print("Подключаюсь к MCP-серверу...")
    try:
        async with open_fetch_session() as session:
            tools = await list_tools(session)
    except Exception as exc:
        print(f"Не удалось подключиться: {exc}")
        sys.exit(1)

    print(f"Доступные инструменты ({len(tools)}):")
    for t in tools:
        print(f"  {t['name']}: {t['description']}")


async def _cli_fetch(url: str) -> None:
    print(f"Загружаю {url} через MCP...")
    try:
        async with open_fetch_session() as session:
            text = await fetch_url(session, url)
    except Exception as exc:
        print(f"Ошибка: {exc}")
        sys.exit(1)

    if not text:
        print("Страница пуста или недоступна.")
        return

    print("Первые 500 символов:")
    print(text[:500])


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING)

    if "--list-tools" in sys.argv:
        asyncio.run(_cli_list_tools())
    elif "--fetch" in sys.argv:
        idx = sys.argv.index("--fetch")
        if idx + 1 >= len(sys.argv):
            print("Укажите ссылку: python mcp_client.py --fetch <url>")
            sys.exit(1)
        asyncio.run(_cli_fetch(sys.argv[idx + 1]))
    else:
        print("Использование:")
        print("  python mcp_client.py --list-tools")
        print("  python mcp_client.py --fetch <url>")
