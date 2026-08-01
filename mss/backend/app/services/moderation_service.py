from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.exceptions import NotFoundError
from app.models.report import AuditLog, Report, ReportReason, ReportStatus
from app.repositories.user_repo import UserRepository


class ModerationService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.user_repo = UserRepository(session)

    async def _log_action(
        self,
        user_id: UUID,
        action: str,
        target_type: str,
        target_id: UUID,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        log = AuditLog(
            user_id=user_id,
            action=action,
            target_type=target_type,
            target_id=target_id,
            metadata_=metadata or {},
        )
        self.session.add(log)
        await self.session.flush()

    async def create_report(self, reporter_id: UUID, data: dict[str, Any]) -> Report:
        target_id = UUID(str(data["target_id"]))
        report = Report(
            reporter_id=reporter_id,
            reason=ReportReason(data["reason"]),
            description=data.get("description"),
            target_type=data["target_type"],
            target_id=target_id,
        )
        self.session.add(report)
        await self.session.flush()
        await self._log_action(
            reporter_id,
            "report.create",
            data["target_type"],
            target_id,
        )
        return report

    async def list_reports(
        self,
        status: str | None = None,
        skip: int = 0,
        limit: int = 24,
    ) -> list[Report]:
        query = select(Report)
        if status:
            query = query.where(Report.status == ReportStatus(status))
        query = query.order_by(desc(Report.created_at)).offset(skip).limit(limit)
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def resolve_report(
        self,
        report_id: UUID,
        moderator_id: UUID,
        status: str,
        note: str | None = None,
    ) -> Report:
        report = await self.session.get(Report, report_id)
        if not report:
            raise NotFoundError(detail="Report not found")

        report.status = ReportStatus(status)
        report.moderator_id = moderator_id
        report.resolution_note = note
        report.resolved_at = datetime.now(UTC)
        await self.session.flush()

        await self._log_action(
            moderator_id,
            "report.resolve",
            "report",
            report_id,
            metadata={"status": status, "note": note},
        )
        return report

    async def ban_user(self, user_id: UUID, moderator_id: UUID) -> None:
        user = await self.user_repo.get_by_id(user_id)
        if not user:
            raise NotFoundError(detail="User not found")

        user.is_active = False
        await self.session.flush()
        await self._log_action(moderator_id, "user.ban", "user", user_id)

    async def unban_user(self, user_id: UUID, moderator_id: UUID) -> None:
        user = await self.user_repo.get_by_id(user_id)
        if not user:
            raise NotFoundError(detail="User not found")

        user.is_active = True
        await self.session.flush()
        await self._log_action(moderator_id, "user.unban", "user", user_id)
