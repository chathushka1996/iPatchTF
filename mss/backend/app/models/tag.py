"""Tag models are defined in game.py — re-exported here for convenience."""

from app.models.game import GameTag, Tag, TagCategory

__all__ = ["Tag", "TagCategory", "GameTag"]
