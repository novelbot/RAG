"""
Common mixins for SQLAlchemy models.
"""

from sqlalchemy import Column, DateTime, func


class TimestampMixin:
    """Mixin to add timestamp fields to models."""
    
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False)