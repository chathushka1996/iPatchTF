from dataclasses import dataclass
from typing import Any, TypeVar

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

T = TypeVar("T")


@dataclass
class PaginatedResult:
    items: list[Any]
    total: int
    page: int
    per_page: int

    @property
    def pages(self) -> int:
        if self.per_page <= 0:
            return 0
        return max(1, (self.total + self.per_page - 1) // self.per_page) if self.total else 0


async def paginate(
    session: AsyncSession,
    query: Select[tuple[T]],
    page: int = 1,
    per_page: int = 24,
) -> PaginatedResult:
    page = max(1, page)
    per_page = max(1, min(per_page, 100))
    offset = (page - 1) * per_page

    count_query = select(func.count()).select_from(query.subquery())
    total_result = await session.execute(count_query)
    total = total_result.scalar_one()

    result = await session.execute(query.offset(offset).limit(per_page))
    items = list(result.scalars().all())

    return PaginatedResult(items=items, total=total, page=page, per_page=per_page)


async def cursor_paginate(
    session: AsyncSession,
    query: Select[tuple[T]],
    cursor: str | None,
    limit: int = 24,
    cursor_column: Any = None,
) -> tuple[list[T], str | None]:
    limit = max(1, min(limit, 100))
    if cursor and cursor_column is not None:
        query = query.where(cursor_column > cursor)

    result = await session.execute(query.order_by(cursor_column).limit(limit + 1))
    items = list(result.scalars().all())

    next_cursor = None
    if len(items) > limit:
        items = items[:limit]
        if cursor_column is not None and items:
            next_cursor = str(getattr(items[-1], cursor_column.key))

    return items, next_cursor
