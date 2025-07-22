"""
Unit tests for Milvus Schema module.
"""
import pytest
from unittest.mock import Mock, patch
import json

from src.milvus.schema import (
    RAGCollectionSchema, SchemaManager, FieldConfig, CollectionConfig,
    create_default_rag_schema
)
from src.core.exceptions import SchemaError, ValidationError


class TestFieldConfig:
    """Test FieldConfig class."""
    
    def test_field_config_creation(self):
        """Test FieldConfig creation."""
        config = FieldConfig(
            name="vector",
            data_type="FLOAT_VECTOR",
            dimension=768,
            is_primary=False,
            description="Vector field"
        )
        
        assert config.name == "vector"
        assert config.data_type == "FLOAT_VECTOR"
        assert config.dimension == 768
        assert config.is_primary is False
        assert config.description == "Vector field"

    def test_field_config_defaults(self):
        """Test FieldConfig with default values."""
        config = FieldConfig(name="id", data_type="INT64")
        
        assert config.is_primary is False
        assert config.auto_id is False
        assert config.max_length is None
        assert config.dimension is None
        assert config.description == ""

    def test_field_config_validation_success(self):
        """Test field config validation - success."""
        # Vector field
        vector_config = FieldConfig(
            name="vector",
            data_type="FLOAT_VECTOR",
            dimension=768
        )
        vector_config.validate()  # Should not raise
        
        # String field with max_length
        string_config = FieldConfig(
            name="text",
            data_type="VARCHAR",
            max_length=1000
        )
        string_config.validate()  # Should not raise

    def test_field_config_validation_vector_no_dimension(self):
        """Test field config validation - vector without dimension."""
        config = FieldConfig(
            name="vector",
            data_type="FLOAT_VECTOR"
            # Missing dimension
        )
        
        with pytest.raises(ValidationError, match="Vector fields must have dimension"):
            config.validate()

    def test_field_config_validation_string_no_max_length(self):
        """Test field config validation - string without max_length."""
        config = FieldConfig(
            name="text",
            data_type="VARCHAR"
            # Missing max_length
        )
        
        with pytest.raises(ValidationError, match="VARCHAR fields must have max_length"):
            config.validate()

    def test_field_config_validation_invalid_data_type(self):
        """Test field config validation - invalid data type."""
        config = FieldConfig(
            name="field",
            data_type="INVALID_TYPE"
        )
        
        with pytest.raises(ValidationError, match="Invalid data type"):
            config.validate()

    def test_field_config_to_dict(self):
        """Test FieldConfig serialization."""
        config = FieldConfig(
            name="vector",
            data_type="FLOAT_VECTOR",
            dimension=768,
            is_primary=False,
            description="Vector field"
        )
        
        result = config.to_dict()
        
        assert result["name"] == "vector"
        assert result["data_type"] == "FLOAT_VECTOR"
        assert result["dimension"] == 768
        assert result["is_primary"] is False
        assert result["description"] == "Vector field"

    def test_field_config_from_dict(self):
        """Test FieldConfig creation from dictionary."""
        data = {
            "name": "vector",
            "data_type": "FLOAT_VECTOR",
            "dimension": 768,
            "is_primary": False,
            "description": "Vector field"
        }
        
        config = FieldConfig.from_dict(data)
        
        assert config.name == "vector"
        assert config.data_type == "FLOAT_VECTOR"
        assert config.dimension == 768
        assert config.is_primary is False
        assert config.description == "Vector field"


class TestCollectionConfig:
    """Test CollectionConfig class."""
    
    def test_collection_config_creation(self):
        """Test CollectionConfig creation."""
        fields = [
            FieldConfig("id", "INT64", is_primary=True, auto_id=True),
            FieldConfig("vector", "FLOAT_VECTOR", dimension=768)
        ]
        
        config = CollectionConfig(
            name="test_collection",
            description="Test collection",
            fields=fields
        )
        
        assert config.name == "test_collection"
        assert config.description == "Test collection"
        assert len(config.fields) == 2
        assert config.enable_dynamic_field is True

    def test_collection_config_validation_success(self):
        """Test collection config validation - success."""
        fields = [
            FieldConfig("id", "INT64", is_primary=True, auto_id=True),
            FieldConfig("vector", "FLOAT_VECTOR", dimension=768)
        ]
        
        config = CollectionConfig("test_collection", fields=fields)
        config.validate()  # Should not raise

    def test_collection_config_validation_no_primary_key(self):
        """Test collection config validation - no primary key."""
        fields = [
            FieldConfig("vector", "FLOAT_VECTOR", dimension=768)
            # No primary key field
        ]
        
        config = CollectionConfig("test_collection", fields=fields)
        
        with pytest.raises(ValidationError, match="Collection must have exactly one primary key"):
            config.validate()

    def test_collection_config_validation_multiple_primary_keys(self):
        """Test collection config validation - multiple primary keys."""
        fields = [
            FieldConfig("id1", "INT64", is_primary=True),
            FieldConfig("id2", "INT64", is_primary=True),
            FieldConfig("vector", "FLOAT_VECTOR", dimension=768)
        ]
        
        config = CollectionConfig("test_collection", fields=fields)
        
        with pytest.raises(ValidationError, match="Collection must have exactly one primary key"):
            config.validate()

    def test_collection_config_validation_no_vector_field(self):
        """Test collection config validation - no vector field."""
        fields = [
            FieldConfig("id", "INT64", is_primary=True, auto_id=True),
            FieldConfig("text", "VARCHAR", max_length=1000)
            # No vector field
        ]
        
        config = CollectionConfig("test_collection", fields=fields)
        
        with pytest.raises(ValidationError, match="Collection must have at least one vector field"):
            config.validate()

    def test_collection_config_get_primary_field(self):
        """Test getting primary field."""
        fields = [
            FieldConfig("id", "INT64", is_primary=True, auto_id=True),
            FieldConfig("vector", "FLOAT_VECTOR", dimension=768)
        ]
        
        config = CollectionConfig("test_collection", fields=fields)
        primary_field = config.get_primary_field()
        
        assert primary_field.name == "id"
        assert primary_field.is_primary is True

    def test_collection_config_get_vector_fields(self):
        """Test getting vector fields."""
        fields = [
            FieldConfig("id", "INT64", is_primary=True, auto_id=True),
            FieldConfig("vector1", "FLOAT_VECTOR", dimension=768),
            FieldConfig("vector2", "BINARY_VECTOR", dimension=128),
            FieldConfig("text", "VARCHAR", max_length=1000)
        ]
        
        config = CollectionConfig("test_collection", fields=fields)
        vector_fields = config.get_vector_fields()
        
        assert len(vector_fields) == 2
        assert vector_fields[0].name == "vector1"
        assert vector_fields[1].name == "vector2"

    def test_collection_config_to_dict(self):
        """Test CollectionConfig serialization."""
        fields = [
            FieldConfig("id", "INT64", is_primary=True, auto_id=True),
            FieldConfig("vector", "FLOAT_VECTOR", dimension=768)
        ]
        
        config = CollectionConfig(
            name="test_collection",
            description="Test collection",
            fields=fields
        )
        
        result = config.to_dict()
        
        assert result["name"] == "test_collection"
        assert result["description"] == "Test collection"
        assert len(result["fields"]) == 2
        assert result["enable_dynamic_field"] is True

    def test_collection_config_from_dict(self):
        """Test CollectionConfig creation from dictionary."""
        data = {
            "name": "test_collection",
            "description": "Test collection",
            "enable_dynamic_field": True,
            "fields": [
                {
                    "name": "id",
                    "data_type": "INT64",
                    "is_primary": True,
                    "auto_id": True
                },
                {
                    "name": "vector",
                    "data_type": "FLOAT_VECTOR",
                    "dimension": 768
                }
            ]
        }
        
        config = CollectionConfig.from_dict(data)
        
        assert config.name == "test_collection"
        assert config.description == "Test collection"
        assert len(config.fields) == 2
        assert config.enable_dynamic_field is True


class TestRAGCollectionSchema:
    """Test RAGCollectionSchema class."""
    
    @pytest.fixture
    def mock_data_type(self):
        """Mock DataType from pymilvus."""
        with patch('src.milvus.schema.DataType') as mock:
            mock.INT64 = "INT64"
            mock.VARCHAR = "VARCHAR"
            mock.FLOAT_VECTOR = "FLOAT_VECTOR"
            mock.JSON = "JSON"
            yield mock

    @pytest.fixture
    def mock_field_schema(self):
        """Mock FieldSchema from pymilvus."""
        with patch('src.milvus.schema.FieldSchema') as mock:
            yield mock

    @pytest.fixture
    def mock_collection_schema(self):
        """Mock CollectionSchema from pymilvus."""
        with patch('src.milvus.schema.CollectionSchema') as mock:
            yield mock

    def test_rag_schema_creation(self, mock_data_type, mock_field_schema, mock_collection_schema):
        """Test RAGCollectionSchema creation."""
        config = CollectionConfig(
            name="rag_collection",
            fields=[
                FieldConfig("id", "INT64", is_primary=True, auto_id=True),
                FieldConfig("vector", "FLOAT_VECTOR", dimension=768)
            ]
        )
        
        schema = RAGCollectionSchema(config)
        
        assert schema.config == config
        assert schema.collection_name == "rag_collection"
        assert schema.vector_dim == 768

    def test_create_schema_success(self, mock_data_type, mock_field_schema, mock_collection_schema):
        """Test schema creation - success."""
        config = CollectionConfig(
            name="rag_collection",
            fields=[
                FieldConfig("id", "INT64", is_primary=True, auto_id=True),
                FieldConfig("vector", "FLOAT_VECTOR", dimension=768),
                FieldConfig("text", "VARCHAR", max_length=1000)
            ]
        )
        
        schema = RAGCollectionSchema(config)
        result = schema.create_schema()
        
        # Verify FieldSchema was called for each field
        assert mock_field_schema.call_count == 3
        
        # Verify CollectionSchema was called
        mock_collection_schema.assert_called_once()

    def test_create_schema_with_rbac_fields(self, mock_data_type, mock_field_schema, mock_collection_schema):
        """Test schema creation with RBAC fields."""
        config = CollectionConfig(
            name="rag_collection",
            fields=[
                FieldConfig("id", "INT64", is_primary=True, auto_id=True),
                FieldConfig("vector", "FLOAT_VECTOR", dimension=768)
            ]
        )
        
        schema = RAGCollectionSchema(config, enable_rbac=True)
        result = schema.create_schema()
        
        # Should include additional RBAC fields
        assert mock_field_schema.call_count > 2

    def test_validate_schema_success(self):
        """Test schema validation - success."""
        config = CollectionConfig(
            name="rag_collection",
            fields=[
                FieldConfig("id", "INT64", is_primary=True, auto_id=True),
                FieldConfig("vector", "FLOAT_VECTOR", dimension=768)
            ]
        )
        
        schema = RAGCollectionSchema(config)
        schema.validate_schema()  # Should not raise

    def test_validate_schema_invalid_config(self):
        """Test schema validation - invalid config."""
        config = CollectionConfig(
            name="rag_collection",
            fields=[
                # No primary key field
                FieldConfig("vector", "FLOAT_VECTOR", dimension=768)
            ]
        )
        
        schema = RAGCollectionSchema(config)
        
        with pytest.raises(SchemaError):
            schema.validate_schema()

    def test_get_field_by_name(self):
        """Test getting field by name."""
        config = CollectionConfig(
            name="rag_collection",
            fields=[
                FieldConfig("id", "INT64", is_primary=True, auto_id=True),
                FieldConfig("vector", "FLOAT_VECTOR", dimension=768),
                FieldConfig("text", "VARCHAR", max_length=1000)
            ]
        )
        
        schema = RAGCollectionSchema(config)
        
        vector_field = schema.get_field_by_name("vector")
        assert vector_field.name == "vector"
        assert vector_field.data_type == "FLOAT_VECTOR"
        
        missing_field = schema.get_field_by_name("missing")
        assert missing_field is None

    def test_get_rbac_fields(self):
        """Test getting RBAC fields."""
        config = CollectionConfig(
            name="rag_collection",
            fields=[
                FieldConfig("id", "INT64", is_primary=True, auto_id=True),
                FieldConfig("vector", "FLOAT_VECTOR", dimension=768)
            ]
        )
        
        schema = RAGCollectionSchema(config, enable_rbac=True)
        rbac_fields = schema.get_rbac_fields()
        
        assert len(rbac_fields) > 0
        field_names = [field.name for field in rbac_fields]
        assert "user_id" in field_names
        assert "group_ids" in field_names
        assert "permissions" in field_names

    def test_to_dict(self):
        """Test schema serialization."""
        config = CollectionConfig(
            name="rag_collection",
            fields=[
                FieldConfig("id", "INT64", is_primary=True, auto_id=True),
                FieldConfig("vector", "FLOAT_VECTOR", dimension=768)
            ]
        )
        
        schema = RAGCollectionSchema(config)
        result = schema.to_dict()
        
        assert result["collection_name"] == "rag_collection"
        assert result["vector_dim"] == 768
        assert result["enable_rbac"] is False
        assert "config" in result

    def test_from_dict(self):
        """Test schema creation from dictionary."""
        data = {
            "collection_name": "rag_collection",
            "vector_dim": 768,
            "enable_rbac": True,
            "config": {
                "name": "rag_collection",
                "description": "Test collection",
                "enable_dynamic_field": True,
                "fields": [
                    {
                        "name": "id",
                        "data_type": "INT64",
                        "is_primary": True,
                        "auto_id": True
                    },
                    {
                        "name": "vector",
                        "data_type": "FLOAT_VECTOR",
                        "dimension": 768
                    }
                ]
            }
        }
        
        schema = RAGCollectionSchema.from_dict(data)
        
        assert schema.collection_name == "rag_collection"
        assert schema.vector_dim == 768
        assert schema.enable_rbac is True


class TestSchemaManager:
    """Test SchemaManager class."""
    
    @pytest.fixture
    def mock_client(self):
        """Mock Milvus client."""
        client = Mock()
        client.alias = "default"
        return client

    def test_schema_manager_creation(self, mock_client):
        """Test SchemaManager creation."""
        manager = SchemaManager(mock_client)
        
        assert manager.client == mock_client
        assert len(manager._schemas) == 0

    def test_register_schema(self, mock_client):
        """Test schema registration."""
        config = CollectionConfig(
            name="test_collection",
            fields=[
                FieldConfig("id", "INT64", is_primary=True, auto_id=True),
                FieldConfig("vector", "FLOAT_VECTOR", dimension=768)
            ]
        )
        
        manager = SchemaManager(mock_client)
        schema = manager.register_schema(config)
        
        assert isinstance(schema, RAGCollectionSchema)
        assert "test_collection" in manager._schemas
        assert manager._schemas["test_collection"] == schema

    def test_get_schema_exists(self, mock_client):
        """Test getting existing schema."""
        config = CollectionConfig(
            name="test_collection",
            fields=[
                FieldConfig("id", "INT64", is_primary=True, auto_id=True),
                FieldConfig("vector", "FLOAT_VECTOR", dimension=768)
            ]
        )
        
        manager = SchemaManager(mock_client)
        schema = manager.register_schema(config)
        
        retrieved_schema = manager.get_schema("test_collection")
        assert retrieved_schema == schema

    def test_get_schema_not_exists(self, mock_client):
        """Test getting non-existing schema."""
        manager = SchemaManager(mock_client)
        
        schema = manager.get_schema("non_existent")
        assert schema is None

    def test_list_schemas(self, mock_client):
        """Test listing all schemas."""
        config1 = CollectionConfig(
            name="collection1",
            fields=[
                FieldConfig("id", "INT64", is_primary=True, auto_id=True),
                FieldConfig("vector", "FLOAT_VECTOR", dimension=768)
            ]
        )
        
        config2 = CollectionConfig(
            name="collection2",
            fields=[
                FieldConfig("id", "INT64", is_primary=True, auto_id=True),
                FieldConfig("vector", "FLOAT_VECTOR", dimension=512)
            ]
        )
        
        manager = SchemaManager(mock_client)
        manager.register_schema(config1)
        manager.register_schema(config2)
        
        schemas = manager.list_schemas()
        
        assert len(schemas) == 2
        assert "collection1" in schemas
        assert "collection2" in schemas

    def test_remove_schema(self, mock_client):
        """Test schema removal."""
        config = CollectionConfig(
            name="test_collection",
            fields=[
                FieldConfig("id", "INT64", is_primary=True, auto_id=True),
                FieldConfig("vector", "FLOAT_VECTOR", dimension=768)
            ]
        )
        
        manager = SchemaManager(mock_client)
        manager.register_schema(config)
        
        assert "test_collection" in manager._schemas
        
        manager.remove_schema("test_collection")
        
        assert "test_collection" not in manager._schemas

    def test_validate_all_schemas(self, mock_client):
        """Test validating all schemas."""
        valid_config = CollectionConfig(
            name="valid_collection",
            fields=[
                FieldConfig("id", "INT64", is_primary=True, auto_id=True),
                FieldConfig("vector", "FLOAT_VECTOR", dimension=768)
            ]
        )
        
        manager = SchemaManager(mock_client)
        manager.register_schema(valid_config)
        
        # Should not raise
        result = manager.validate_all_schemas()
        
        assert result is True

    def test_export_schemas(self, mock_client):
        """Test schema export."""
        config = CollectionConfig(
            name="test_collection",
            fields=[
                FieldConfig("id", "INT64", is_primary=True, auto_id=True),
                FieldConfig("vector", "FLOAT_VECTOR", dimension=768)
            ]
        )
        
        manager = SchemaManager(mock_client)
        manager.register_schema(config)
        
        exported = manager.export_schemas()
        
        assert "test_collection" in exported
        assert exported["test_collection"]["collection_name"] == "test_collection"

    def test_import_schemas(self, mock_client):
        """Test schema import."""
        schema_data = {
            "test_collection": {
                "collection_name": "test_collection",
                "vector_dim": 768,
                "enable_rbac": False,
                "config": {
                    "name": "test_collection",
                    "description": "",
                    "enable_dynamic_field": True,
                    "fields": [
                        {
                            "name": "id",
                            "data_type": "INT64",
                            "is_primary": True,
                            "auto_id": True
                        },
                        {
                            "name": "vector",
                            "data_type": "FLOAT_VECTOR",
                            "dimension": 768
                        }
                    ]
                }
            }
        }
        
        manager = SchemaManager(mock_client)
        manager.import_schemas(schema_data)
        
        assert "test_collection" in manager._schemas
        schema = manager.get_schema("test_collection")
        assert schema.collection_name == "test_collection"


class TestCreateDefaultRAGSchema:
    """Test create_default_rag_schema function."""
    
    def test_create_default_rag_schema_basic(self):
        """Test creating default RAG schema - basic."""
        schema = create_default_rag_schema(
            collection_name="rag_collection",
            vector_dim=768
        )
        
        assert isinstance(schema, RAGCollectionSchema)
        assert schema.collection_name == "rag_collection"
        assert schema.vector_dim == 768
        assert schema.enable_rbac is False

    def test_create_default_rag_schema_with_rbac(self):
        """Test creating default RAG schema - with RBAC."""
        schema = create_default_rag_schema(
            collection_name="rag_collection",
            vector_dim=768,
            enable_rbac=True
        )
        
        assert schema.enable_rbac is True
        rbac_fields = schema.get_rbac_fields()
        assert len(rbac_fields) > 0

    def test_create_default_rag_schema_custom_description(self):
        """Test creating default RAG schema - custom description."""
        schema = create_default_rag_schema(
            collection_name="rag_collection",
            vector_dim=768,
            description="Custom RAG collection"
        )
        
        assert schema.config.description == "Custom RAG collection"

    def test_create_default_rag_schema_validation(self):
        """Test default schema validation."""
        schema = create_default_rag_schema(
            collection_name="rag_collection",
            vector_dim=768
        )
        
        # Should not raise
        schema.validate_schema()