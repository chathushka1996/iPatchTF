from app.schemas.admin import (
    AdminDashboardResponse,
    EngineCreate,
    ReportUpdateRequest,
    RoleChangeRequest,
    TagCreate,
)
from app.schemas.collection import (
    CollectionCreate,
    CollectionDetailResponse,
    CollectionGameAdd,
    CollectionGameResponse,
    CollectionResponse,
    CollectionUpdate,
    ReorderRequest,
)
from app.schemas.common import (
    ErrorResponse,
    HealthResponse,
    MessageResponse,
    PaginatedResponse,
    PaginationParams,
)
from app.schemas.forum import (
    ChatMessageCreate,
    ChatMessageResponse,
    ForumCategoryResponse,
    PostCreate,
    PostResponse,
    PostUpdate,
    ThreadCreate,
    ThreadDetailResponse,
    ThreadResponse,
)
from app.schemas.game import (
    DownloadCreate,
    DownloadResponse,
    EngineResponse,
    GameCreate,
    GameListResponse,
    GameResponse,
    GameSearchParams,
    GameUpdate,
    GameVersionCreate,
    GameVersionResponse,
    ScreenshotResponse,
    TagResponse,
    VersionResponse,
)
from app.schemas.notification import (
    MarkReadRequest,
    NotificationResponse,
    UnreadCountResponse,
)
from app.schemas.review import (
    ReviewCreate,
    ReviewResponse,
    ReviewUpdate,
    ReviewVoteRequest,
)
from app.schemas.upload import PresignRequest, PresignResponse
from app.schemas.user import (
    NotificationPreferencesUpdate,
    PasswordChangeRequest,
    PasswordResetConfirm,
    PasswordResetRequest,
    TokenResponse,
    UserBriefResponse,
    UserCreate,
    UserLogin,
    UserPublicResponse,
    UserResponse,
    UserUpdate,
)

__all__ = [
    # admin
    "AdminDashboardResponse",
    "EngineCreate",
    "ReportUpdateRequest",
    "RoleChangeRequest",
    "TagCreate",
    # collection
    "CollectionCreate",
    "CollectionDetailResponse",
    "CollectionGameAdd",
    "CollectionGameResponse",
    "CollectionResponse",
    "CollectionUpdate",
    "ReorderRequest",
    # common
    "ErrorResponse",
    "HealthResponse",
    "MessageResponse",
    "PaginatedResponse",
    "PaginationParams",
    # forum
    "ChatMessageCreate",
    "ChatMessageResponse",
    "ForumCategoryResponse",
    "PostCreate",
    "PostResponse",
    "PostUpdate",
    "ThreadCreate",
    "ThreadDetailResponse",
    "ThreadResponse",
    # game
    "DownloadCreate",
    "DownloadResponse",
    "EngineResponse",
    "GameCreate",
    "GameListResponse",
    "GameResponse",
    "GameSearchParams",
    "GameUpdate",
    "GameVersionCreate",
    "GameVersionResponse",
    "ScreenshotResponse",
    "TagResponse",
    "VersionResponse",
    # notification
    "MarkReadRequest",
    "NotificationResponse",
    "UnreadCountResponse",
    # review
    "ReviewCreate",
    "ReviewResponse",
    "ReviewUpdate",
    "ReviewVoteRequest",
    # upload
    "PresignRequest",
    "PresignResponse",
    # user
    "NotificationPreferencesUpdate",
    "PasswordChangeRequest",
    "PasswordResetConfirm",
    "PasswordResetRequest",
    "TokenResponse",
    "UserBriefResponse",
    "UserCreate",
    "UserLogin",
    "UserPublicResponse",
    "UserResponse",
    "UserUpdate",
]
