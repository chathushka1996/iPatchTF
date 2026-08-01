import re
from urllib.parse import urlparse

from app.exceptions import ValidationError

USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_]{3,50}$")
URL_PATTERN = re.compile(
    r"^https?://"
    r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"
    r"localhost|"
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"
    r"(?::\d+)?"
    r"(?:/?|[/?]\S+)$",
    re.IGNORECASE,
)


def validate_password_strength(password: str) -> None:
    if len(password) < 8:
        raise ValidationError(detail="Password must be at least 8 characters")
    if not re.search(r"[A-Za-z]", password):
        raise ValidationError(detail="Password must contain at least one letter")
    if not re.search(r"\d", password):
        raise ValidationError(detail="Password must contain at least one digit")


def validate_username(username: str) -> None:
    if not USERNAME_PATTERN.match(username):
        raise ValidationError(
            detail="Username must be 3-50 characters and contain only letters, numbers, and underscores"
        )


def validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValidationError(detail="Invalid URL format")
    if not URL_PATTERN.match(url):
        raise ValidationError(detail="Invalid URL format")
