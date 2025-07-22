"""
Milvus vector database integration for RAG server.
"""

from .client import MilvusClient, MilvusConnectionPool
from .collection import MilvusCollection, CollectionManager
from .schema import RAGCollectionSchema, SchemaManager, create_default_rag_schema
from .index import IndexManager, IndexType, MetricType, IndexConfig, create_index_manager
from .search import SearchManager, SearchQuery, SearchStrategy, create_search_manager
from .rbac import RBACManager, UserContext, Permission, AccessScope, create_rbac_manager

__all__ = [
    # Client
    "MilvusClient",
    "MilvusConnectionPool",
    
    # Collection
    "MilvusCollection", 
    "CollectionManager",
    
    # Schema
    "RAGCollectionSchema",
    "SchemaManager",
    "create_default_rag_schema",
    
    # Index
    "IndexManager",
    "IndexType",
    "MetricType", 
    "IndexConfig",
    "create_index_manager",
    
    # Search
    "SearchManager",
    "SearchQuery",
    "SearchStrategy",
    "create_search_manager",
    
    # RBAC
    "RBACManager",
    "UserContext",
    "Permission",
    "AccessScope",
    "create_rbac_manager"
]