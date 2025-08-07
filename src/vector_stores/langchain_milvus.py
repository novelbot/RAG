"""
LangChain-based Milvus Vector Store Implementation.

This module provides a Milvus vector store implementation using LangChain's
vector store interfaces, supporting semantic search, metadata filtering,
and hybrid search capabilities.
"""

import time
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
import uuid

# LangChain imports
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Milvus

# Milvus imports
from pymilvus import (
    connections, Collection, FieldSchema, CollectionSchema, DataType,
    utility, MilvusClient
)

from src.core.logging import LoggerMixin
from src.core.exceptions import StorageError, SearchError
from src.milvus.index import MetricType


@dataclass
class LangChainMilvusConfig:
    """Configuration for LangChain Milvus vector store."""
    
    # Connection settings
    host: str = "localhost"
    port: int = 19530
    user: Optional[str] = None
    password: Optional[str] = None
    database: str = "default"
    
    # Collection settings
    collection_name: str = "langchain_documents"
    dimension: int = 768
    index_type: str = "IVF_FLAT"
    metric_type: str = "L2"
    index_params: Optional[Dict[str, Any]] = None
    search_params: Optional[Dict[str, Any]] = None
    
    # Performance settings
    consistency_level: str = "Session"
    batch_size: int = 100
    enable_dynamic_field: bool = True
    
    # Additional features
    partition_key_field: Optional[str] = None
    partition_names: Optional[List[str]] = None
    replica_number: int = 1
    resource_groups: Optional[List[str]] = None


class LangChainMilvusVectorStore(VectorStore, LoggerMixin):
    """
    Milvus vector store implementation using LangChain.
    
    Features:
    - Semantic similarity search
    - Metadata filtering
    - Hybrid search (vector + keyword)
    - Batch operations
    - Async support
    - Collection management
    - Index optimization
    """
    
    def __init__(
        self,
        embedding_function: Embeddings,
        config: LangChainMilvusConfig,
        **kwargs
    ):
        """
        Initialize LangChain Milvus vector store.
        
        Args:
            embedding_function: LangChain embeddings instance
            config: Milvus configuration
            **kwargs: Additional arguments for Milvus
        """
        self.embedding_function = embedding_function
        self.config = config
        
        # Initialize connection
        self._init_connection()
        
        # Initialize or load collection
        self._init_collection()
        
        # Setup LangChain Milvus wrapper
        connection_args = {
            "host": config.host,
            "port": config.port,
            "user": config.user,
            "password": config.password,
            "db_name": config.database,
        }
        
        self.vector_store = Milvus(
            embedding_function=embedding_function,
            collection_name=config.collection_name,
            connection_args=connection_args,
            consistency_level=config.consistency_level,
            index_params=config.index_params,
            search_params=config.search_params,
            **kwargs
        )
        
        self.logger.info(f"Initialized LangChain Milvus vector store: {config.collection_name}")
    
    def _init_connection(self) -> None:
        """Initialize Milvus connection."""
        try:
            alias = f"langchain_{self.config.database}"
            
            # Check if already connected
            if alias not in connections.list_connections():
                connections.connect(
                    alias=alias,
                    host=self.config.host,
                    port=self.config.port,
                    user=self.config.user,
                    password=self.config.password,
                    db_name=self.config.database
                )
                self.logger.info(f"Connected to Milvus: {self.config.host}:{self.config.port}")
            
        except Exception as e:
            raise StorageError(f"Failed to connect to Milvus: {e}")
    
    def _init_collection(self) -> None:
        """Initialize or load Milvus collection."""
        try:
            # Check if collection exists
            if not utility.has_collection(self.config.collection_name):
                self._create_collection()
            else:
                self.logger.info(f"Loaded existing collection: {self.config.collection_name}")
                
        except Exception as e:
            raise StorageError(f"Failed to initialize collection: {e}")
    
    def _create_collection(self) -> None:
        """Create new Milvus collection with schema."""
        try:
            # Define schema
            fields = [
                FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.config.dimension),
            ]
            
            # Add metadata field if dynamic fields are enabled
            if self.config.enable_dynamic_field:
                fields.append(
                    FieldSchema(name="metadata", dtype=DataType.JSON, enable_dynamic_field=True)
                )
            
            schema = CollectionSchema(
                fields=fields,
                description=f"LangChain vector store: {self.config.collection_name}"
            )
            
            # Create collection
            collection = Collection(
                name=self.config.collection_name,
                schema=schema,
                consistency_level=self.config.consistency_level
            )
            
            # Create index
            index_params = self.config.index_params or {
                "index_type": self.config.index_type,
                "metric_type": self.config.metric_type,
                "params": {"nlist": 128}
            }
            
            collection.create_index(
                field_name="vector",
                index_params=index_params
            )
            
            # Load collection
            collection.load()
            
            self.logger.info(f"Created new collection: {self.config.collection_name}")
            
        except Exception as e:
            raise StorageError(f"Failed to create collection: {e}")
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        """
        Add texts to the vector store.
        
        Args:
            texts: List of texts to add
            metadatas: Optional metadata for each text
            ids: Optional IDs for each text
            **kwargs: Additional arguments
            
        Returns:
            List of IDs for added texts
        """
        return self.vector_store.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            **kwargs
        )
    
    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            ids: Optional IDs for each document
            **kwargs: Additional arguments
            
        Returns:
            List of IDs for added documents
        """
        return self.vector_store.add_documents(
            documents=documents,
            ids=ids,
            **kwargs
        )
    
    async def aadd_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        """
        Add texts asynchronously.
        
        Args:
            texts: List of texts to add
            metadatas: Optional metadata for each text
            ids: Optional IDs for each text
            **kwargs: Additional arguments
            
        Returns:
            List of IDs for added texts
        """
        # Run in executor if async not natively supported
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.add_texts,
            texts,
            metadatas,
            ids
        )
    
    async def aadd_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        """
        Add documents asynchronously.
        
        Args:
            documents: List of documents to add
            ids: Optional IDs for each document
            **kwargs: Additional arguments
            
        Returns:
            List of IDs for added documents
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.add_documents,
            documents,
            ids
        )
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs
    ) -> List[Document]:
        """
        Perform similarity search.
        
        Args:
            query: Query text
            k: Number of results to return
            filter: Optional metadata filter
            **kwargs: Additional search parameters
            
        Returns:
            List of similar documents
        """
        return self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter,
            **kwargs
        )
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with scores.
        
        Args:
            query: Query text
            k: Number of results to return
            filter: Optional metadata filter
            **kwargs: Additional search parameters
            
        Returns:
            List of (document, score) tuples
        """
        return self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter,
            **kwargs
        )
    
    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs
    ) -> List[Document]:
        """
        Search by embedding vector.
        
        Args:
            embedding: Query embedding vector
            k: Number of results to return
            filter: Optional metadata filter
            **kwargs: Additional search parameters
            
        Returns:
            List of similar documents
        """
        return self.vector_store.similarity_search_by_vector(
            embedding=embedding,
            k=k,
            filter=filter,
            **kwargs
        )
    
    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs
    ) -> List[Document]:
        """
        Perform similarity search asynchronously.
        
        Args:
            query: Query text
            k: Number of results to return
            filter: Optional metadata filter
            **kwargs: Additional search parameters
            
        Returns:
            List of similar documents
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.similarity_search,
            query,
            k,
            filter
        )
    
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict] = None,
        **kwargs
    ) -> List[Document]:
        """
        Perform MMR search for diverse results.
        
        Args:
            query: Query text
            k: Number of results to return
            fetch_k: Number of candidates to fetch
            lambda_mult: Diversity parameter (0=max diversity, 1=max relevance)
            filter: Optional metadata filter
            **kwargs: Additional parameters
            
        Returns:
            List of diverse relevant documents
        """
        return self.vector_store.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs
        )
    
    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[dict] = None,
        **kwargs
    ) -> Optional[bool]:
        """
        Delete documents from the vector store.
        
        Args:
            ids: IDs of documents to delete
            filter: Filter for documents to delete
            **kwargs: Additional parameters
            
        Returns:
            Success status
        """
        try:
            if ids:
                collection = Collection(self.config.collection_name)
                expr = f"pk in {ids}"
                collection.delete(expr)
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to delete documents: {e}")
            return False
    
    def from_texts(
        self,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs
    ) -> "LangChainMilvusVectorStore":
        """
        Create vector store from texts.
        
        Args:
            texts: List of texts
            embedding: Embedding function
            metadatas: Optional metadata
            ids: Optional IDs
            **kwargs: Additional parameters
            
        Returns:
            New vector store instance
        """
        instance = LangChainMilvusVectorStore(
            embedding_function=embedding,
            config=self.config,
            **kwargs
        )
        instance.add_texts(texts, metadatas, ids)
        return instance
    
    def from_documents(
        self,
        documents: List[Document],
        embedding: Embeddings,
        ids: Optional[List[str]] = None,
        **kwargs
    ) -> "LangChainMilvusVectorStore":
        """
        Create vector store from documents.
        
        Args:
            documents: List of documents
            embedding: Embedding function
            ids: Optional IDs
            **kwargs: Additional parameters
            
        Returns:
            New vector store instance
        """
        instance = LangChainMilvusVectorStore(
            embedding_function=embedding,
            config=self.config,
            **kwargs
        )
        instance.add_documents(documents, ids)
        return instance
    
    def as_retriever(self, **kwargs) -> Any:
        """
        Return retriever interface.
        
        Args:
            **kwargs: Retriever configuration
            
        Returns:
            Retriever instance
        """
        return self.vector_store.as_retriever(**kwargs)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            collection = Collection(self.config.collection_name)
            
            stats = {
                "collection_name": self.config.collection_name,
                "num_entities": collection.num_entities,
                "schema": str(collection.schema),
                "indexes": collection.indexes,
                "loaded": collection.is_loaded,
                "partitions": [p.name for p in collection.partitions]
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {e}")
            return {}
    
    def optimize_index(self) -> bool:
        """Optimize collection index."""
        try:
            collection = Collection(self.config.collection_name)
            
            # Flush data
            collection.flush()
            
            # Compact if needed
            collection.compact()
            
            self.logger.info(f"Optimized index for collection: {self.config.collection_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to optimize index: {e}")
            return False
    
    def create_partition(self, partition_name: str) -> bool:
        """Create a new partition."""
        try:
            collection = Collection(self.config.collection_name)
            collection.create_partition(partition_name)
            self.logger.info(f"Created partition: {partition_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create partition: {e}")
            return False
    
    def drop_collection(self) -> bool:
        """Drop the entire collection."""
        try:
            utility.drop_collection(self.config.collection_name)
            self.logger.info(f"Dropped collection: {self.config.collection_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to drop collection: {e}")
            return False


def create_langchain_milvus(
    embedding_function: Embeddings,
    config: LangChainMilvusConfig,
    **kwargs
) -> LangChainMilvusVectorStore:
    """Factory function to create LangChain Milvus vector store."""
    return LangChainMilvusVectorStore(
        embedding_function=embedding_function,
        config=config,
        **kwargs
    )