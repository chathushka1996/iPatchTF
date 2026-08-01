"""CLI to create the first admin user."""

import argparse
import asyncio
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select

from app.database import async_session_factory
from app.models.base import UserRole
from app.models.user import User, UserProfile
from app.utils.security import hash_password
from app.utils.validators import validate_password_strength, validate_username


async def create_superuser(
    username: str,
    email: str,
    password: str,
) -> None:
    validate_username(username)
    validate_password_strength(password)

    async with async_session_factory() as session:
        existing = await session.execute(
            select(User).where(
                (User.username == username) | (User.email == email)
            )
        )
        if existing.scalar_one_or_none():
            print("Error: User with that username or email already exists.")
            sys.exit(1)

        user = User(
            username=username,
            email=email,
            password_hash=hash_password(password),
            display_name=username,
            role=UserRole.ADMIN,
            is_verified=True,
            is_active=True,
        )
        session.add(user)
        await session.flush()
        session.add(UserProfile(user_id=user.id))
        await session.commit()
        print(f"Superuser '{username}' created successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a GameVault admin user")
    parser.add_argument("--username", required=True)
    parser.add_argument("--email", required=True)
    parser.add_argument("--password", default=None)
    args = parser.parse_args()

    password = args.password or getpass.getpass("Password: ")
    asyncio.run(create_superuser(args.username, args.email, password))


if __name__ == "__main__":
    main()
