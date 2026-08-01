from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
from fastapi import FastAPI

from app.api.v1.chat import ws_router
from app.api.v1.router import router as v1_router
from app.config import get_settings, settings
from app.database import close_db, init_db
from app.exceptions import EXCEPTION_HANDLERS
from app.infrastructure import close_meilisearch, close_redis, init_meilisearch, init_redis
from app.middleware.cors import setup_cors
from app.middleware.rate_limit import RateLimitMiddleware
from app.middleware.request_id import RequestIDMiddleware

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(0),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    logger.info("startup", app_name=settings.APP_NAME, env=settings.APP_ENV)

    await init_db()
    try:
        await init_redis()
    except Exception as exc:
        logger.warning("redis_startup_failed", error=str(exc))

    try:
        init_meilisearch()
    except Exception as exc:
        logger.warning("meilisearch_startup_failed", error=str(exc))

    yield

    await close_meilisearch()
    await close_redis()
    await close_db()
    logger.info("shutdown_complete")


def create_app() -> FastAPI:
    app_settings = get_settings()

    app = FastAPI(
        title="GameVault API",
        version="1.0.0",
        description="Interactive Game Database & Community Platform",
        lifespan=lifespan,
    )

    setup_cors(app, app_settings)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(RateLimitMiddleware)

    for exc_class, handler in EXCEPTION_HANDLERS.items():
        app.add_exception_handler(exc_class, handler)

    app.include_router(v1_router, prefix="/api/v1")
    app.include_router(ws_router, prefix="/ws")

    return app


app = create_app()
