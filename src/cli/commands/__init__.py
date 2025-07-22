"""
CLI command modules for the RAG server.
"""

from .data import data_group
from .user import user_group
from .database import database_group
from .model import model_group
from .config import config_group

__all__ = [
    'data_group',
    'user_group', 
    'database_group',
    'model_group',
    'config_group'
]