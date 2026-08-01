from collections.abc import AsyncGenerator, Callable
from typing import Annotated
from uuid import UUID

from fastapi import Depends, Query
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_async_session
from app.exceptions import ForbiddenError, UnauthorizedError
from app.models.user import User, UserRole

security_scheme = HTTPBearer(auto_error=False)


class PaginationParams(BaseModel):
    page: int = Field(default=1, ge=1)
    per_page: int = Field(default=24, ge=1, le=100)

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.per_page

    @property
    def limit(self) -> int:
        return self.per_page


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async for session in get_async_session():
        yield session


def get_pagination(
    page: Annotated[int, Query(ge=1)] = 1,
    per_page: Annotated[int, Query(ge=1, le=100)] = 24,
) -> PaginationParams:
    return PaginationParams(page=page, per_page=per_page)


def _decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        return payload
    except JWTError as exc:
        raise UnauthorizedError(detail="Invalid or expired token") from exc


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security_scheme)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    if credentials is None:
        raise UnauthorizedError(detail="Authentication required")

    payload = _decode_token(credentials.credentials)
    user_id = payload.get("sub")
    if user_id is None:
        raise UnauthorizedError(detail="Invalid token payload")

    try:
        user_uuid = UUID(str(user_id))
    except ValueError as exc:
        raise UnauthorizedError(detail="Invalid token subject") from exc

    result = await db.execute(select(User).where(User.id == user_uuid))
    user = result.scalar_one_or_none()
    if user is None:
        raise UnauthorizedError(detail="User not found")

    return user


async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    if not current_user.is_active:
        raise ForbiddenError(detail="Inactive user account")
    return current_user


def require_role(*roles: UserRole) -> Callable:
    allowed_roles = set(roles)

    async def role_checker(
        current_user: Annotated[User, Depends(get_current_active_user)],
    ) -> User:
        if current_user.role not in allowed_roles:
            raise ForbiddenError(detail="Insufficient permissions")
        return current_user

    return role_checker
