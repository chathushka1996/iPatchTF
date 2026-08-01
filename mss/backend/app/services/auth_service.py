import base64
import hashlib
import hmac
import secrets
import struct
import time
from typing import Any
from uuid import UUID

import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.exceptions import ConflictError, NotFoundError, UnauthorizedError, ValidationError
from app.models.user import User, UserProfile
from app.repositories.user_repo import UserRepository
from app.schemas.user import TokenResponse, UserCreate
from app.utils.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    generate_verification_token,
    hash_password,
    verify_password,
)
from app.utils.validators import validate_password_strength, validate_username


class AuthService:
    TOKEN_PREFIX = "auth:token:"
    RESET_PREFIX = "auth:reset:"
    VERIFY_PREFIX = "auth:verify:"

    def __init__(
        self,
        session: AsyncSession,
        redis: aioredis.Redis | None = None,
    ) -> None:
        self.session = session
        self.user_repo = UserRepository(session)
        self._redis = redis

    async def _get_redis(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        return self._redis

    async def register(self, data: UserCreate | dict[str, Any]) -> User:
        if isinstance(data, dict):
            data = UserCreate(**data)

        validate_username(data.username)
        validate_password_strength(data.password)

        if await self.user_repo.get_by_email(data.email):
            raise ConflictError(detail="Email already registered")
        if await self.user_repo.get_by_username(data.username):
            raise ConflictError(detail="Username already taken")

        user = await self.user_repo.create(
            {
                "username": data.username,
                "email": data.email,
                "password_hash": hash_password(data.password),
                "display_name": data.username,
            }
        )
        profile = UserProfile(user_id=user.id)
        self.session.add(profile)
        await self.session.flush()

        token = generate_verification_token()
        redis = await self._get_redis()
        await redis.setex(f"{self.VERIFY_PREFIX}{token}", 86400, str(user.id))

        return user

    async def login(self, email: str, password: str) -> TokenResponse:
        user = await self.user_repo.get_by_email(email)
        if not user or not verify_password(password, user.password_hash):
            raise UnauthorizedError(detail="Invalid email or password")
        if not user.is_active:
            raise UnauthorizedError(detail="Account is inactive")

        await self.user_repo.update_last_login(user.id)
        return self._create_tokens(user)

    def _create_tokens(self, user: User) -> TokenResponse:
        payload = {"sub": str(user.id), "role": user.role.value}
        return TokenResponse(
            access_token=create_access_token(payload),
            refresh_token=create_refresh_token(payload),
        )

    async def refresh_token(self, token: str) -> TokenResponse:
        redis = await self._get_redis()
        if await redis.exists(f"{self.TOKEN_PREFIX}{token}"):
            raise UnauthorizedError(detail="Token has been revoked")

        try:
            payload = decode_token(token)
        except ValueError as exc:
            raise UnauthorizedError(detail="Invalid refresh token") from exc

        if payload.get("type") != "refresh":
            raise UnauthorizedError(detail="Invalid token type")

        user = await self.user_repo.get_by_id(UUID(str(payload["sub"])))
        if not user or not user.is_active:
            raise UnauthorizedError(detail="User not found or inactive")

        return self._create_tokens(user)

    async def logout(self, token: str) -> None:
        try:
            payload = decode_token(token)
        except ValueError:
            return

        exp = payload.get("exp", 0)
        ttl = max(int(exp - time.time()), 0)
        if ttl > 0:
            redis = await self._get_redis()
            await redis.setex(f"{self.TOKEN_PREFIX}{token}", ttl, "1")

    async def forgot_password(self, email: str) -> None:
        user = await self.user_repo.get_by_email(email)
        if not user:
            return

        token = generate_verification_token()
        redis = await self._get_redis()
        await redis.setex(f"{self.RESET_PREFIX}{token}", 3600, str(user.id))

    async def reset_password(self, token: str, password: str) -> None:
        validate_password_strength(password)
        redis = await self._get_redis()
        user_id = await redis.get(f"{self.RESET_PREFIX}{token}")
        if not user_id:
            raise ValidationError(detail="Invalid or expired reset token")

        user = await self.user_repo.get_by_id(UUID(user_id))
        if not user:
            raise NotFoundError(detail="User not found")

        user.password_hash = hash_password(password)
        await self.session.flush()
        await redis.delete(f"{self.RESET_PREFIX}{token}")

    async def verify_email(self, token: str) -> User:
        redis = await self._get_redis()
        user_id = await redis.get(f"{self.VERIFY_PREFIX}{token}")
        if not user_id:
            raise ValidationError(detail="Invalid or expired verification token")

        user = await self.user_repo.get_by_id(UUID(user_id))
        if not user:
            raise NotFoundError(detail="User not found")

        user.is_verified = True
        await self.session.flush()
        await redis.delete(f"{self.VERIFY_PREFIX}{token}")
        return user

    async def setup_2fa(self, user_id: UUID) -> dict[str, str]:
        user = await self.user_repo.get_by_id(user_id)
        if not user:
            raise NotFoundError(detail="User not found")

        secret = base64.b32encode(secrets.token_bytes(20)).decode("utf-8").rstrip("=")
        user.two_factor_secret = secret
        user.two_factor_enabled = False
        await self.session.flush()

        return {
            "secret": secret,
            "provisioning_uri": (
                f"otpauth://totp/GameVault:{user.email}?secret={secret}&issuer=GameVault"
            ),
        }

    def _verify_totp(self, secret: str, code: str, window: int = 1) -> bool:
        secret_padded = secret + "=" * (-len(secret) % 8)
        key = base64.b32decode(secret_padded.upper())
        current_counter = int(time.time()) // 30

        for offset in range(-window, window + 1):
            counter = struct.pack(">Q", current_counter + offset)
            digest = hmac.new(key, counter, hashlib.sha1).digest()
            offset_byte = digest[-1] & 0x0F
            truncated = struct.unpack(">I", digest[offset_byte : offset_byte + 4])[0]
            truncated &= 0x7FFFFFFF
            if str(truncated % 1_000_000).zfill(6) == code:
                return True
        return False

    async def verify_2fa(self, user_id: UUID, code: str) -> bool:
        user = await self.user_repo.get_by_id(user_id)
        if not user or not user.two_factor_secret:
            raise ValidationError(detail="2FA not configured")

        if not self._verify_totp(user.two_factor_secret, code):
            raise ValidationError(detail="Invalid 2FA code")

        user.two_factor_enabled = True
        await self.session.flush()
        return True

    async def disable_2fa(self, user_id: UUID) -> None:
        user = await self.user_repo.get_by_id(user_id)
        if not user:
            raise NotFoundError(detail="User not found")

        user.two_factor_enabled = False
        user.two_factor_secret = None
        await self.session.flush()
