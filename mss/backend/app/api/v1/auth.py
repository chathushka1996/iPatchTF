from typing import Annotated

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_current_active_user, get_db
from app.models.user import User
from app.schemas.common import MessageResponse
from app.schemas.user import (
    PasswordResetConfirm,
    PasswordResetRequest,
    TokenResponse,
    UserCreate,
    UserLogin,
)
from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    refresh_token: str | None = None


class TwoFactorSetupResponse(BaseModel):
    secret: str
    provisioning_uri: str


class TwoFactorVerifyRequest(BaseModel):
    code: str = Field(min_length=6, max_length=6)


class TwoFactorDisableRequest(BaseModel):
    code: str = Field(min_length=6, max_length=6)
    password: str


def get_auth_service(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AuthService:
    return AuthService(db)


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new account",
)
async def register(
    data: UserCreate,
    service: Annotated[AuthService, Depends(get_auth_service)],
) -> TokenResponse:
    """Register a new user and return access and refresh tokens."""
    return await service.register(data)


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Authenticate and obtain tokens",
)
async def login(
    data: UserLogin,
    service: Annotated[AuthService, Depends(get_auth_service)],
) -> TokenResponse:
    """Log in with email and password."""
    return await service.login(data)


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh access token",
)
async def refresh_token(
    data: RefreshTokenRequest,
    service: Annotated[AuthService, Depends(get_auth_service)],
) -> TokenResponse:
    """Exchange a valid refresh token for a new token pair."""
    return await service.refresh_tokens(data.refresh_token)


@router.post(
    "/logout",
    response_model=MessageResponse,
    summary="Log out and blacklist tokens",
)
async def logout(
    data: LogoutRequest,
    current_user: Annotated[User, Depends(get_current_active_user)],
    service: Annotated[AuthService, Depends(get_auth_service)],
) -> MessageResponse:
    """Blacklist the current session tokens."""
    await service.logout(current_user, data.refresh_token)
    return MessageResponse(message="Logged out successfully")


@router.post(
    "/forgot-password",
    response_model=MessageResponse,
    summary="Request a password reset email",
)
async def forgot_password(
    data: PasswordResetRequest,
    service: Annotated[AuthService, Depends(get_auth_service)],
) -> MessageResponse:
    """Send a password reset link to the user's email address."""
    await service.forgot_password(data.email)
    return MessageResponse(message="If the email exists, a reset link has been sent")


@router.post(
    "/reset-password",
    response_model=MessageResponse,
    summary="Reset password with token",
)
async def reset_password(
    data: PasswordResetConfirm,
    service: Annotated[AuthService, Depends(get_auth_service)],
) -> MessageResponse:
    """Consume a password reset token and set a new password."""
    await service.reset_password(data.token, data.new_password)
    return MessageResponse(message="Password reset successfully")


@router.post(
    "/verify-email/{token}",
    response_model=MessageResponse,
    summary="Verify email address",
)
async def verify_email(
    token: str,
    service: Annotated[AuthService, Depends(get_auth_service)],
) -> MessageResponse:
    """Verify a user's email address using the token from the verification email."""
    await service.verify_email(token)
    return MessageResponse(message="Email verified successfully")


@router.post(
    "/2fa/setup",
    response_model=TwoFactorSetupResponse,
    summary="Set up two-factor authentication",
)
async def setup_two_factor(
    current_user: Annotated[User, Depends(get_current_active_user)],
    service: Annotated[AuthService, Depends(get_auth_service)],
) -> TwoFactorSetupResponse:
    """Generate a TOTP secret and provisioning URI for 2FA setup."""
    return await service.setup_two_factor(current_user)


@router.post(
    "/2fa/verify",
    response_model=MessageResponse,
    summary="Confirm two-factor authentication setup",
)
async def verify_two_factor(
    data: TwoFactorVerifyRequest,
    current_user: Annotated[User, Depends(get_current_active_user)],
    service: Annotated[AuthService, Depends(get_auth_service)],
) -> MessageResponse:
    """Confirm TOTP setup by verifying a code from the authenticator app."""
    await service.verify_two_factor(current_user, data.code)
    return MessageResponse(message="Two-factor authentication enabled")


@router.post(
    "/2fa/disable",
    response_model=MessageResponse,
    summary="Disable two-factor authentication",
)
async def disable_two_factor(
    data: TwoFactorDisableRequest,
    current_user: Annotated[User, Depends(get_current_active_user)],
    service: Annotated[AuthService, Depends(get_auth_service)],
) -> MessageResponse:
    """Turn off two-factor authentication after verifying credentials."""
    await service.disable_two_factor(current_user, data.code, data.password)
    return MessageResponse(message="Two-factor authentication disabled")
