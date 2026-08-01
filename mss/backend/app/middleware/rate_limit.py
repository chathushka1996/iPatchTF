import time
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.exceptions import RateLimitError


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Redis-backed rate limiting middleware (placeholder for Redis integration)."""

    def __init__(self, app, redis_client=None, requests_per_minute: int = 60) -> None:
        super().__init__(app)
        self.redis_client = redis_client
        self.requests_per_minute = requests_per_minute

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if self.redis_client is None:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        key = f"rate_limit:{client_ip}:{int(time.time() // 60)}"

        count = await self.redis_client.incr(key)
        if count == 1:
            await self.redis_client.expire(key, 60)

        if count > self.requests_per_minute:
            raise RateLimitError()

        return await call_next(request)
