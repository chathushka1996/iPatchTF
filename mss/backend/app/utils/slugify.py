from slugify import slugify as _slugify
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


def generate_slug(text: str) -> str:
    return _slugify(text, max_length=200)


async def generate_unique_slug(
    text: str,
    model: type,
    session: AsyncSession,
    slug_field: str = "slug",
) -> str:
    base_slug = generate_slug(text)
    slug = base_slug
    counter = 1

    while True:
        result = await session.execute(
            select(model).where(getattr(model, slug_field) == slug)
        )
        if result.scalar_one_or_none() is None:
            return slug
        slug = f"{base_slug}-{counter}"
        counter += 1
