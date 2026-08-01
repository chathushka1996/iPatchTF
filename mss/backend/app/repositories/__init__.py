from app.repositories.base import BaseRepository
from app.repositories.collection_repo import CollectionRepository
from app.repositories.forum_repo import ForumRepository
from app.repositories.game_repo import GameRepository
from app.repositories.notification_repo import NotificationRepository
from app.repositories.review_repo import ReviewRepository
from app.repositories.user_repo import UserRepository

__all__ = [
    "BaseRepository",
    "CollectionRepository",
    "ForumRepository",
    "GameRepository",
    "NotificationRepository",
    "ReviewRepository",
    "UserRepository",
]
