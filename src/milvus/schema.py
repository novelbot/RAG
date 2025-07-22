"""
Collection schema management for Milvus with access control metadata.
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json

from pymilvus import FieldSchema, CollectionSchema, DataType
from loguru import logger

from src.core.exceptions import MilvusError, SchemaError
from src.core.logging import LoggerMixin


class MilvusDataType(Enum):
    """Milvus data types enumeration."""
    BOOL = DataType.BOOL
    INT8 = DataType.INT8
    INT16 = DataType.INT16
    INT32 = DataType.INT32
    INT64 = DataType.INT64
    FLOAT = DataType.FLOAT
    DOUBLE = DataType.DOUBLE
    STRING = DataType.STRING
    VARCHAR = DataType.VARCHAR
    JSON = DataType.JSON
    ARRAY = DataType.ARRAY
    FLOAT_VECTOR = DataType.FLOAT_VECTOR
    BINARY_VECTOR = DataType.BINARY_VECTOR
    FLOAT16_VECTOR = DataType.FLOAT16_VECTOR
    BFLOAT16_VECTOR = DataType.BFLOAT16_VECTOR
    SPARSE_FLOAT_VECTOR = DataType.SPARSE_FLOAT_VECTOR


@dataclass
class FieldConfig:
    """Configuration for a Milvus field."""
    name: str
    data_type: MilvusDataType
    description: Optional[str] = None
    is_primary: bool = False
    auto_id: bool = False
    max_length: Optional[int] = None
    dim: Optional[int] = None
    max_capacity: Optional[int] = None
    element_type: Optional[MilvusDataType] = None
    nullable: bool = False
    partition_key: bool = False
    clustering_key: bool = False
    
    def to_field_schema(self) -> FieldSchema:
        """Convert to PyMilvus FieldSchema."""
        kwargs = {
            "name": self.name,
            "dtype": self.data_type.value,
            "description": self.description or "",
            "is_primary": self.is_primary,
            "auto_id": self.auto_id,
            "nullable": self.nullable,
            "partition_key": self.partition_key,
            "clustering_key": self.clustering_key
        }
        
        # Add type-specific parameters
        if self.data_type in [MilvusDataType.VARCHAR, MilvusDataType.STRING]:
            if self.max_length is not None:
                kwargs["max_length"] = self.max_length
            else:
                kwargs["max_length"] = 65535  # Default max length
        
        if self.data_type in [
            MilvusDataType.FLOAT_VECTOR,
            MilvusDataType.BINARY_VECTOR,
            MilvusDataType.FLOAT16_VECTOR,
            MilvusDataType.BFLOAT16_VECTOR
        ]:
            if self.dim is None:
                raise SchemaError(f"Vector field {self.name} requires dim parameter")
            kwargs["dim"] = self.dim
        
        if self.data_type == MilvusDataType.ARRAY:
            if self.element_type is None:
                raise SchemaError(f"Array field {self.name} requires element_type parameter")
            kwargs["element_type"] = self.element_type.value
            if self.max_capacity is not None:
                kwargs["max_capacity"] = self.max_capacity
        
        return FieldSchema(**kwargs)


@dataclass
class CollectionConfig:
    """Configuration for a Milvus collection."""
    name: str
    description: Optional[str] = None
    consistency_level: str = "Bounded"
    partition_key_isolation: bool = False
    enable_dynamic_fields: bool = True
    shard_num: int = 1
    ttl_seconds: Optional[int] = None
    fields: List[FieldConfig] = field(default_factory=list)
    
    def add_field(self, field_config: FieldConfig) -> None:
        """Add a field to the collection."""
        self.fields.append(field_config)
    
    def get_primary_field(self) -> Optional[FieldConfig]:
        """Get the primary field configuration."""
        for field in self.fields:
            if field.is_primary:
                return field
        return None
    
    def get_vector_fields(self) -> List[FieldConfig]:
        """Get all vector fields."""
        vector_types = [
            MilvusDataType.FLOAT_VECTOR,
            MilvusDataType.BINARY_VECTOR,
            MilvusDataType.FLOAT16_VECTOR,
            MilvusDataType.BFLOAT16_VECTOR,
            MilvusDataType.SPARSE_FLOAT_VECTOR
        ]
        return [field for field in self.fields if field.data_type in vector_types]
    
    def validate(self) -> None:
        """Validate collection configuration."""
        if not self.fields:
            raise SchemaError("Collection must have at least one field")
        
        # Check for primary key
        primary_fields = [f for f in self.fields if f.is_primary]
        if len(primary_fields) != 1:
            raise SchemaError("Collection must have exactly one primary field")
        
        # Check for vector fields
        vector_fields = self.get_vector_fields()
        if not vector_fields:
            raise SchemaError("Collection must have at least one vector field")
        
        # Check field name uniqueness
        field_names = [f.name for f in self.fields]
        if len(field_names) != len(set(field_names)):
            raise SchemaError("Field names must be unique")
        
        # Validate field configurations
        for field in self.fields:
            if field.is_primary and field.data_type not in [
                MilvusDataType.INT64, MilvusDataType.VARCHAR
            ]:
                raise SchemaError(f"Primary field {field.name} must be INT64 or VARCHAR")


class RAGCollectionSchema(LoggerMixin):
    """
    Pre-configured schema for RAG collections with access control metadata.
    
    Based on Context7 documentation for Milvus collection schema:
    - Uses FieldSchema for defining field structure
    - Supports VARCHAR primary keys for document IDs
    - Includes metadata fields for access control
    - Provides vector field for embeddings
    """
    
    def __init__(self, 
                 collection_name: str,
                 vector_dim: int = 1536,
                 description: Optional[str] = None):
        """
        Initialize RAG collection schema.
        
        Args:
            collection_name: Name of the collection
            vector_dim: Dimension of vector embeddings
            description: Collection description
        """
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.description = description or f"RAG collection for {collection_name}"
        self.config = self._create_default_config()
    
    def _create_default_config(self) -> CollectionConfig:
        """Create default RAG collection configuration."""
        config = CollectionConfig(
            name=self.collection_name,
            description=self.description,
            consistency_level="Bounded",
            enable_dynamic_fields=True
        )
        
        # Primary key field (document chunk ID)
        config.add_field(FieldConfig(
            name="id",
            data_type=MilvusDataType.VARCHAR,
            description="Unique identifier for document chunk",
            is_primary=True,
            auto_id=False,
            max_length=100
        ))
        
        # Vector field for embeddings
        config.add_field(FieldConfig(
            name="vector",
            data_type=MilvusDataType.FLOAT_VECTOR,
            description="Vector embedding for similarity search",
            dim=self.vector_dim
        ))
        
        # Document metadata fields
        config.add_field(FieldConfig(
            name="document_id",
            data_type=MilvusDataType.VARCHAR,
            description="Original document identifier",
            max_length=100
        ))
        
        config.add_field(FieldConfig(
            name="source_id",
            data_type=MilvusDataType.VARCHAR,
            description="Data source identifier",
            max_length=100
        ))
        
        config.add_field(FieldConfig(
            name="chunk_index",
            data_type=MilvusDataType.INT32,
            description="Index of chunk within document"
        ))
        
        config.add_field(FieldConfig(
            name="content",
            data_type=MilvusDataType.VARCHAR,
            description="Text content of the chunk",
            max_length=32768
        ))
        
        # Access control fields
        config.add_field(FieldConfig(
            name="user_id",
            data_type=MilvusDataType.VARCHAR,
            description="User ID for access control",
            max_length=100
        ))
        
        config.add_field(FieldConfig(
            name="group_ids",
            data_type=MilvusDataType.ARRAY,
            description="Group IDs for access control",
            element_type=MilvusDataType.VARCHAR,
            max_capacity=50
        ))
        
        config.add_field(FieldConfig(
            name="permissions",
            data_type=MilvusDataType.JSON,
            description="JSON permissions metadata"
        ))
        
        # Timestamp fields
        config.add_field(FieldConfig(
            name="created_at",
            data_type=MilvusDataType.INT64,
            description="Creation timestamp (Unix epoch)"
        ))
        
        config.add_field(FieldConfig(
            name="updated_at",
            data_type=MilvusDataType.INT64,
            description="Last update timestamp (Unix epoch)"
        ))
        
        # Metadata field for additional information
        config.add_field(FieldConfig(
            name="metadata",
            data_type=MilvusDataType.JSON,
            description="Additional metadata in JSON format"
        ))
        
        return config
    
    def add_custom_field(self, field_config: FieldConfig) -> None:
        """Add a custom field to the schema."""
        self.config.add_field(field_config)
        self.logger.info(f"Added custom field: {field_config.name}")
    
    def remove_field(self, field_name: str) -> None:
        """Remove a field from the schema."""
        self.config.fields = [f for f in self.config.fields if f.name != field_name]
        self.logger.info(f"Removed field: {field_name}")
    
    def get_field_config(self, field_name: str) -> Optional[FieldConfig]:
        """Get field configuration by name."""
        for field in self.config.fields:
            if field.name == field_name:
                return field
        return None
    
    def create_collection_schema(self) -> CollectionSchema:
        """
        Create PyMilvus CollectionSchema.
        
        Returns:
            CollectionSchema: PyMilvus collection schema
        """
        try:
            # Validate configuration
            self.config.validate()
            
            # Create field schemas
            field_schemas = []
            for field_config in self.config.fields:
                field_schema = field_config.to_field_schema()
                field_schemas.append(field_schema)
            
            # Create collection schema
            schema = CollectionSchema(
                fields=field_schemas,
                description=self.config.description,
                enable_dynamic_field=self.config.enable_dynamic_fields
            )
            
            self.logger.info(f"Created collection schema: {self.collection_name}")
            return schema
            
        except Exception as e:
            self.logger.error(f"Failed to create collection schema: {e}")
            raise SchemaError(f"Schema creation failed: {e}")
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Get schema information."""
        return {
            "collection_name": self.collection_name,
            "description": self.description,
            "vector_dim": self.vector_dim,
            "field_count": len(self.config.fields),
            "fields": [
                {
                    "name": field.name,
                    "type": field.data_type.name,
                    "description": field.description,
                    "is_primary": field.is_primary,
                    "auto_id": field.auto_id,
                    "max_length": field.max_length,
                    "dim": field.dim,
                    "nullable": field.nullable
                }
                for field in self.config.fields
            ],
            "primary_field": self.config.get_primary_field().name if self.config.get_primary_field() else None,
            "vector_fields": [field.name for field in self.config.get_vector_fields()],
            "access_control_fields": ["user_id", "group_ids", "permissions"],
            "consistency_level": self.config.consistency_level,
            "enable_dynamic_fields": self.config.enable_dynamic_fields
        }
    
    def to_json(self) -> str:
        """Export schema configuration as JSON."""
        return json.dumps(self.get_schema_info(), indent=2)
    
    def from_json(self, json_str: str) -> None:
        """Import schema configuration from JSON."""
        try:
            data = json.loads(json_str)
            
            # Update basic configuration
            self.collection_name = data["collection_name"]
            self.description = data.get("description", "")
            self.vector_dim = data.get("vector_dim", 1536)
            
            # Recreate configuration
            self.config = CollectionConfig(
                name=self.collection_name,
                description=self.description,
                consistency_level=data.get("consistency_level", "Bounded"),
                enable_dynamic_fields=data.get("enable_dynamic_fields", True)
            )
            
            # Add fields
            for field_info in data.get("fields", []):
                field_config = FieldConfig(
                    name=field_info["name"],
                    data_type=MilvusDataType[field_info["type"]],
                    description=field_info.get("description"),
                    is_primary=field_info.get("is_primary", False),
                    auto_id=field_info.get("auto_id", False),
                    max_length=field_info.get("max_length"),
                    dim=field_info.get("dim"),
                    nullable=field_info.get("nullable", False)
                )
                self.config.add_field(field_config)
            
            self.logger.info(f"Imported schema from JSON: {self.collection_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to import schema from JSON: {e}")
            raise SchemaError(f"JSON import failed: {e}")


class SchemaManager(LoggerMixin):
    """Manager for multiple collection schemas."""
    
    def __init__(self):
        self._schemas: Dict[str, RAGCollectionSchema] = {}
    
    def create_schema(self, 
                     collection_name: str,
                     vector_dim: int = 1536,
                     description: Optional[str] = None) -> RAGCollectionSchema:
        """
        Create a new RAG collection schema.
        
        Args:
            collection_name: Name of the collection
            vector_dim: Vector dimension
            description: Schema description
            
        Returns:
            RAGCollectionSchema: Created schema
        """
        if collection_name in self._schemas:
            raise SchemaError(f"Schema already exists: {collection_name}")
        
        schema = RAGCollectionSchema(
            collection_name=collection_name,
            vector_dim=vector_dim,
            description=description
        )
        
        self._schemas[collection_name] = schema
        self.logger.info(f"Created schema: {collection_name}")
        
        return schema
    
    def get_schema(self, collection_name: str) -> Optional[RAGCollectionSchema]:
        """Get schema by collection name."""
        return self._schemas.get(collection_name)
    
    def remove_schema(self, collection_name: str) -> None:
        """Remove schema by collection name."""
        if collection_name in self._schemas:
            del self._schemas[collection_name]
            self.logger.info(f"Removed schema: {collection_name}")
    
    def list_schemas(self) -> List[str]:
        """List all schema names."""
        return list(self._schemas.keys())
    
    def validate_all_schemas(self) -> Dict[str, bool]:
        """Validate all schemas."""
        results = {}
        for name, schema in self._schemas.items():
            try:
                schema.config.validate()
                results[name] = True
            except Exception as e:
                results[name] = False
                self.logger.error(f"Schema validation failed for {name}: {e}")
        
        return results
    
    def get_all_schema_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information for all schemas."""
        return {
            name: schema.get_schema_info()
            for name, schema in self._schemas.items()
        }
    
    def export_schemas(self) -> Dict[str, str]:
        """Export all schemas to JSON."""
        return {
            name: schema.to_json()
            for name, schema in self._schemas.items()
        }
    
    def import_schemas(self, schemas_json: Dict[str, str]) -> None:
        """Import schemas from JSON."""
        for name, json_str in schemas_json.items():
            try:
                schema = RAGCollectionSchema("temp", 1536)
                schema.from_json(json_str)
                self._schemas[name] = schema
                self.logger.info(f"Imported schema: {name}")
            except Exception as e:
                self.logger.error(f"Failed to import schema {name}: {e}")
                raise SchemaError(f"Import failed for {name}: {e}")


def create_default_rag_schema(collection_name: str, 
                             vector_dim: int = 1536) -> RAGCollectionSchema:
    """
    Create a default RAG collection schema with access control.
    
    Args:
        collection_name: Name of the collection
        vector_dim: Vector dimension (default: 1536 for OpenAI embeddings)
        
    Returns:
        RAGCollectionSchema: Configured schema
    """
    return RAGCollectionSchema(
        collection_name=collection_name,
        vector_dim=vector_dim,
        description=f"RAG collection for {collection_name} with RBAC support"
    )