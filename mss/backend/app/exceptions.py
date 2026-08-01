from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse


class AppException(Exception):
    """Base application exception."""

    def __init__(
        self,
        detail: str,
        error_code: str = "APP_ERROR",
        status_code: int = 500,
    ) -> None:
        self.detail = detail
        self.error_code = error_code
        self.status_code = status_code
        super().__init__(detail)


class NotFoundError(AppException):
    def __init__(self, detail: str = "Resource not found", error_code: str = "NOT_FOUND") -> None:
        super().__init__(detail=detail, error_code=error_code, status_code=404)


class UnauthorizedError(AppException):
    def __init__(
        self,
        detail: str = "Not authenticated",
        error_code: str = "UNAUTHORIZED",
    ) -> None:
        super().__init__(detail=detail, error_code=error_code, status_code=401)


class ForbiddenError(AppException):
    def __init__(
        self,
        detail: str = "Permission denied",
        error_code: str = "FORBIDDEN",
    ) -> None:
        super().__init__(detail=detail, error_code=error_code, status_code=403)


class ConflictError(AppException):
    def __init__(
        self,
        detail: str = "Resource conflict",
        error_code: str = "CONFLICT",
    ) -> None:
        super().__init__(detail=detail, error_code=error_code, status_code=409)


class ValidationError(AppException):
    def __init__(
        self,
        detail: str = "Validation failed",
        error_code: str = "VALIDATION_ERROR",
    ) -> None:
        super().__init__(detail=detail, error_code=error_code, status_code=422)


class RateLimitError(AppException):
    def __init__(
        self,
        detail: str = "Rate limit exceeded",
        error_code: str = "RATE_LIMIT_EXCEEDED",
    ) -> None:
        super().__init__(detail=detail, error_code=error_code, status_code=429)


def _error_response(exc: AppException) -> dict[str, Any]:
    return {
        "detail": exc.detail,
        "error_code": exc.error_code,
        "status": exc.status_code,
    }


async def app_exception_handler(_request: Request, exc: AppException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_response(exc),
    )


EXCEPTION_HANDLERS = {
    AppException: app_exception_handler,
    NotFoundError: app_exception_handler,
    UnauthorizedError: app_exception_handler,
    ForbiddenError: app_exception_handler,
    ConflictError: app_exception_handler,
    ValidationError: app_exception_handler,
    RateLimitError: app_exception_handler,
}
