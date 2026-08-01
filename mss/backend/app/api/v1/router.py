from typing import Any

from fastapi import APIRouter
from sqlalchemy import text

from app.api.v1.admin import router as admin_router
from app.api.v1.auth import router as auth_router
from app.api.v1.chat import router as chat_router
from app.api.v1.collections import router as collections_router
from app.api.v1.forums import router as forums_router
from app.api.v1.games import router as games_router
from app.api.v1.notifications import router as notifications_router
from app.api.v1.reviews import router as reviews_router
from app.api.v1.search import router as search_router
from app.api.v1.threads import router as threads_router
from app.api.v1.uploads import router as uploads_router
from app.api.v1.users import router as users_router
from app.database import async_session_factory
from app.infrastructure import check_meilisearch_health, check_redis_health

router = APIRouter()

router.include_router(auth_router)
router.include_router(users_router)
router.include_router(games_router)
router.include_router(reviews_router)
router.include_router(forums_router)
router.include_router(threads_router)
router.include_router(chat_router)
router.include_router(notifications_router)
router.include_router(collections_router)
router.include_router(search_router)
router.include_router(admin_router)
router.include_router(uploads_router)


@router.get("/health")
async def health_check() -> dict[str, Any]:
    """Check API and dependency health."""
    checks: dict[str, Any] = {
        "status": "ok",
        "service": "GameVault API",
        "version": "1.0.0",
        "dependencies": {},
    }

    try:
        async with async_session_factory() as session:
            await session.execute(text("SELECT 1"))
        checks["dependencies"]["database"] = {"status": "ok"}
    except Exception as exc:
        checks["dependencies"]["database"] = {"status": "error", "detail": str(exc)}
        checks["status"] = "degraded"

    redis_health = await check_redis_health()
    checks["dependencies"]["redis"] = redis_health
    if redis_health["status"] != "ok":
        checks["status"] = "degraded"

    meili_health = await check_meilisearch_health()
    checks["dependencies"]["meilisearch"] = meili_health
    if meili_health["status"] != "ok":
        checks["status"] = "degraded"

    return checks
