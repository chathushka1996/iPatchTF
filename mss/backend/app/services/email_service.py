import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from app.config import settings

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates" / "emails"


class EmailService:
    def __init__(self) -> None:
        self._env = Environment(
            loader=FileSystemLoader(str(TEMPLATE_DIR)),
            autoescape=select_autoescape(["html", "xml"]),
        )

    def _render_template(self, template: str, context: dict[str, Any]) -> str:
        try:
            tmpl = self._env.get_template(template)
            return tmpl.render(**context)
        except Exception:
            return self._fallback_render(template, context)

    def _fallback_render(self, template: str, context: dict[str, Any]) -> str:
        if template == "welcome.html":
            return f"Welcome to GameVault, {context.get('username', 'there')}!"
        if template == "password_reset.html":
            return f"Reset your password: {context.get('reset_url', '')}"
        if template == "email_verification.html":
            return f"Verify your email: {context.get('verify_url', '')}"
        if template == "review_notification.html":
            return (
                f"{context.get('reviewer', 'Someone')} reviewed "
                f"{context.get('game_title', 'your game')}"
            )
        return str(context)

    async def send_email(
        self,
        to: str,
        subject: str,
        template: str,
        context: dict[str, Any],
    ) -> None:
        html_body = self._render_template(template, context)
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = settings.EMAIL_FROM
        msg["To"] = to
        msg.attach(MIMEText(html_body, "html"))

        if settings.SMTP_PASSWORD:
            with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
                server.starttls()
                server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
                server.send_message(msg)

    async def send_welcome_email(self, user: Any) -> None:
        await self.send_email(
            to=user.email,
            subject="Welcome to GameVault",
            template="welcome.html",
            context={"username": user.username},
        )

    async def send_password_reset(self, user: Any, token: str) -> None:
        await self.send_email(
            to=user.email,
            subject="Reset your GameVault password",
            template="password_reset.html",
            context={
                "username": user.username,
                "reset_url": f"/auth/reset-password?token={token}",
            },
        )

    async def send_verification_email(self, user: Any, token: str) -> None:
        await self.send_email(
            to=user.email,
            subject="Verify your GameVault email",
            template="email_verification.html",
            context={
                "username": user.username,
                "verify_url": f"/auth/verify-email/{token}",
            },
        )

    async def send_review_notification(
        self, author: Any, game: Any, reviewer: Any
    ) -> None:
        await self.send_email(
            to=author.email,
            subject=f"New review on {game.title}",
            template="review_notification.html",
            context={
                "author": author.username,
                "game_title": game.title,
                "reviewer": reviewer.username,
                "game_slug": game.slug,
            },
        )
