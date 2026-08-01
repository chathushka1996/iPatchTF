from typing import Any

import redis.asyncio as aioredis
import structlog
from meilisearch_python_sdk import AsyncClient as MeiliAsyncClient

from app.config import settings

logger = structlog.get_logger(__name__)

redis_client: aioredis.Redis | None = None
meili_client: MeiliAsyncClient | None = None


async def init_redis() -> aioredis.Redis:
    global redis_client
    redis_client = aioredis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
    )
    await redis_client.ping()
    logger.info("redis_connected", url=settings.REDIS_URL)
    return redis_client


async def close_redis() -> None:
    global redis_client
    if redis_client is not None:
        await redis_client.aclose()
        redis_client = None
        logger.info("redis_disconnected")


def init_meilisearch() -> MeiliAsyncClient:
    global meili_client
    meili_client = MeiliAsyncClient(
        settings.MEILI_URL,
        settings.MEILI_MASTER_KEY,
    )
    logger.info("meilisearch_initialized", url=settings.MEILI_URL)
    return meili_client


async def close_meilisearch() -> None:
    global meili_client
    if meili_client is not None:
        await meili_client.aclose()
        meili_client = None
        logger.info("meilisearch_disconnected")


def get_redis() -> aioredis.Redis | None:
    return redis_client


def get_meilisearch() -> MeiliAsyncClient | None:
    return meili_client


async def check_redis_health() -> dict[str, Any]:
    if redis_client is None:
        return {"status": "disconnected"}
    try:
        await redis_client.ping()
        return {"status": "ok"}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


async def check_meilisearch_health() -> dict[str, Any]:
    if meili_client is None:
        return {"status": "disconnected"}
    try:
        health = await meili_client.health()
        return {"status": "ok", "detail": health.status}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}
