"""
Unit tests for Database Introspection module.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import json

from src.database.introspection import (
    DatabaseIntrospector, 
    ColumnInfo, 
    ColumnType, 
    TableInfo, 
    SchemaInfo,
    IndexInfo,
    ForeignKeyInfo
)
from src.core.exceptions import DatabaseError


class TestColumnInfo:
    """Test ColumnInfo dataclass."""
    
    def test_column_info_creation(self):
        """Test ColumnInfo creation with basic parameters."""
        column = ColumnInfo(
            name="test_column",
            type=ColumnType.STRING,
            nullable=False,
            primary_key=True
        )
        
        assert column.name == "test_column"
        assert column.type == ColumnType.STRING
        assert column.nullable is False
        assert column.primary_key is True
        assert column.default is None
        assert column.foreign_key is None
        assert column.unique is False
        assert column.indexed is False
        assert column.autoincrement is False

    def test_column_info_with_all_fields(self):
        """Test ColumnInfo with all fields populated."""
        column = ColumnInfo(
            name="id",
            type=ColumnType.INTEGER,
            nullable=False,
            default=None,
            primary_key=True,
            foreign_key=None,
            unique=True,
            indexed=True,
            autoincrement=True,
            comment="Primary key column",
            max_length=None,
            precision=10,
            scale=0
        )
        
        assert column.name == "id"
        assert column.type == ColumnType.INTEGER
        assert column.nullable is False
        assert column.primary_key is True
        assert column.unique is True
        assert column.indexed is True
        assert column.autoincrement is True
        assert column.comment == "Primary key column"
        assert column.precision == 10
        assert column.scale == 0


class TestTableInfo:
    """Test TableInfo dataclass."""
    
    def test_table_info_creation(self):
        """Test TableInfo creation."""
        table = TableInfo(name="test_table", schema="public")
        
        assert table.name == "test_table"
        assert table.schema == "public"
        assert table.columns == []
        assert table.primary_keys == []
        assert table.foreign_keys == []
        assert table.indexes == []
        assert table.unique_constraints == []
        assert table.comment is None
        assert table.row_count is None

    def test_table_info_with_columns(self):
        """Test TableInfo with columns."""
        column1 = ColumnInfo("id", ColumnType.INTEGER, primary_key=True)
        column2 = ColumnInfo("name", ColumnType.STRING)
        
        table = TableInfo(
            name="users",
            columns=[column1, column2],
            primary_keys=["id"]
        )
        
        assert len(table.columns) == 2
        assert table.columns[0].name == "id"
        assert table.columns[1].name == "name"
        assert table.primary_keys == ["id"]


class TestSchemaInfo:
    """Test SchemaInfo dataclass."""
    
    def test_schema_info_creation(self):
        """Test SchemaInfo creation."""
        schema = SchemaInfo(name="public")
        
        assert schema.name == "public"
        assert schema.tables == []
        assert schema.views == []
        assert schema.sequences == []
        assert schema.functions == []
        assert schema.procedures == []


class TestDatabaseIntrospector:
    """Test DatabaseIntrospector class."""
    
    @pytest.fixture
    def mock_pool(self):
        """Mock connection pool."""
        pool = Mock()
        pool.engine = Mock()
        return pool

    @pytest.fixture
    def mock_inspector(self):
        """Mock SQLAlchemy inspector."""
        inspector = Mock()
        return inspector

    @pytest.fixture
    def introspector(self, mock_pool, mock_inspector):
        """Create DatabaseIntrospector with mocked dependencies."""
        with patch('src.database.introspection.inspect', return_value=mock_inspector):
            return DatabaseIntrospector(mock_pool)

    def test_introspector_creation(self, mock_pool, mock_inspector):
        """Test DatabaseIntrospector creation."""
        with patch('src.database.introspection.inspect', return_value=mock_inspector):
            introspector = DatabaseIntrospector(mock_pool)
            
            assert introspector.pool == mock_pool
            assert introspector.engine == mock_pool.engine
            assert introspector.inspector == mock_inspector
            assert isinstance(introspector._type_mapping, dict)

    def test_type_mapping_creation(self, introspector):
        """Test type mapping dictionary creation."""
        type_mapping = introspector._type_mapping
        
        assert type_mapping['VARCHAR'] == ColumnType.STRING
        assert type_mapping['INTEGER'] == ColumnType.INTEGER
        assert type_mapping['FLOAT'] == ColumnType.FLOAT
        assert type_mapping['BOOLEAN'] == ColumnType.BOOLEAN
        assert type_mapping['DATE'] == ColumnType.DATE
        assert type_mapping['JSON'] == ColumnType.JSON

    def test_get_schemas(self, introspector):
        """Test getting schema names."""
        introspector.inspector.get_schema_names.return_value = ["public", "test", "app"]
        
        schemas = introspector.get_schemas()
        
        assert schemas == ["public", "test", "app"]
        introspector.inspector.get_schema_names.assert_called_once()

    def test_get_schemas_error(self, introspector):
        """Test getting schema names with error."""
        introspector.inspector.get_schema_names.side_effect = Exception("Database error")
        
        schemas = introspector.get_schemas()
        
        assert schemas == []

    def test_get_tables(self, introspector):
        """Test getting table names."""
        introspector.inspector.get_table_names.return_value = ["users", "posts", "comments"]
        
        tables = introspector.get_tables("public")
        
        assert tables == ["users", "posts", "comments"]
        introspector.inspector.get_table_names.assert_called_once_with(schema="public")

    def test_get_tables_default_schema(self, introspector):
        """Test getting table names with default schema."""
        introspector.inspector.get_table_names.return_value = ["table1", "table2"]
        
        tables = introspector.get_tables()
        
        assert tables == ["table1", "table2"]
        introspector.inspector.get_table_names.assert_called_once_with(schema=None)

    def test_get_views(self, introspector):
        """Test getting view names."""
        introspector.inspector.get_view_names.return_value = ["user_stats", "post_summary"]
        
        views = introspector.get_views("public")
        
        assert views == ["user_stats", "post_summary"]
        introspector.inspector.get_view_names.assert_called_once_with(schema="public")

    def test_map_column_type_exact_match(self, introspector):
        """Test column type mapping with exact match."""
        mock_type = Mock()
        mock_type.__str__ = Mock(return_value="VARCHAR(255)")
        
        result = introspector._map_column_type(mock_type)
        
        assert result == ColumnType.STRING

    def test_map_column_type_unknown(self, introspector):
        """Test column type mapping with unknown type."""
        mock_type = Mock()
        mock_type.__str__ = Mock(return_value="CUSTOM_TYPE")
        
        result = introspector._map_column_type(mock_type)
        
        assert result == ColumnType.UNKNOWN

    def test_build_column_info(self, introspector):
        """Test building ColumnInfo from SQLAlchemy data."""
        mock_type = Mock()
        mock_type.__str__ = Mock(return_value="VARCHAR")
        mock_type.length = 255
        
        column_data = {
            'name': 'username',
            'type': mock_type,
            'nullable': False,
            'default': None,
            'autoincrement': False,
            'comment': 'User login name'
        }
        
        column_info = introspector._build_column_info(column_data)
        
        assert column_info.name == 'username'
        assert column_info.type == ColumnType.STRING
        assert column_info.nullable is False
        assert column_info.default is None
        assert column_info.autoincrement is False
        assert column_info.comment == 'User login name'
        assert column_info.max_length == 255

    def test_build_foreign_key_info(self, introspector):
        """Test building ForeignKeyInfo from SQLAlchemy data."""
        fk_data = {
            'name': 'fk_user_id',
            'constrained_columns': ['user_id'],
            'referred_table': 'users',
            'referred_columns': ['id'],
            'options': {
                'ondelete': 'CASCADE',
                'onupdate': 'RESTRICT'
            }
        }
        
        fk_info = introspector._build_foreign_key_info(fk_data)
        
        assert fk_info.name == 'fk_user_id'
        assert fk_info.columns == ['user_id']
        assert fk_info.referred_table == 'users'
        assert fk_info.referred_columns == ['id']
        assert fk_info.on_delete == 'CASCADE'
        assert fk_info.on_update == 'RESTRICT'

    def test_build_index_info(self, introspector):
        """Test building IndexInfo from SQLAlchemy data."""
        index_data = {
            'name': 'idx_username',
            'column_names': ['username'],
            'unique': True,
            'type': 'btree'
        }
        
        index_info = introspector._build_index_info(index_data)
        
        assert index_info.name == 'idx_username'
        assert index_info.columns == ['username']
        assert index_info.unique is True
        assert index_info.type == 'btree'

    def test_get_table_comment_success(self, introspector):
        """Test getting table comment successfully."""
        introspector.inspector.get_table_comment.return_value = {'text': 'User data table'}
        
        comment = introspector._get_table_comment('users', 'public')
        
        assert comment == 'User data table'
        introspector.inspector.get_table_comment.assert_called_once_with('users', schema='public')

    def test_get_table_comment_error(self, introspector):
        """Test getting table comment with error."""
        introspector.inspector.get_table_comment.side_effect = Exception("Not supported")
        
        comment = introspector._get_table_comment('users', 'public')
        
        assert comment is None

    @patch('src.database.introspection.text')
    def test_get_table_row_count_success(self, mock_text, introspector):
        """Test getting table row count successfully."""
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.scalar.return_value = 1000
        mock_connection.execute.return_value = mock_result
        
        introspector.pool.get_connection.return_value.__enter__ = Mock(return_value=mock_connection)
        introspector.pool.get_connection.return_value.__exit__ = Mock(return_value=None)
        
        count = introspector._get_table_row_count('users', 'public')
        
        assert count == 1000
        mock_connection.execute.assert_called_once()

    @patch('src.database.introspection.text')
    def test_get_table_row_count_error(self, mock_text, introspector):
        """Test getting table row count with error."""
        mock_connection = Mock()
        mock_connection.execute.side_effect = Exception("Query failed")
        
        introspector.pool.get_connection.return_value.__enter__ = Mock(return_value=mock_connection)
        introspector.pool.get_connection.return_value.__exit__ = Mock(return_value=None)
        
        count = introspector._get_table_row_count('users', 'public')
        
        assert count is None

    def test_introspect_table_basic(self, introspector):
        """Test basic table introspection."""
        # Mock column data
        column_data = [{
            'name': 'id',
            'type': Mock(__str__=Mock(return_value="INTEGER")),
            'nullable': False,
            'default': None,
            'autoincrement': True
        }]
        
        # Mock inspector methods
        introspector.inspector.get_columns.return_value = column_data
        introspector.inspector.get_pk_constraint.return_value = {'constrained_columns': ['id']}
        introspector.inspector.get_foreign_keys.return_value = []
        introspector.inspector.get_indexes.return_value = []
        introspector.inspector.get_unique_constraints.return_value = []
        
        with patch.object(introspector, '_get_table_comment', return_value=None):
            with patch.object(introspector, '_get_table_row_count', return_value=100):
                table_info = introspector.introspect_table('users', 'public')
        
        assert table_info.name == 'users'
        assert table_info.schema == 'public'
        assert len(table_info.columns) == 1
        assert table_info.columns[0].name == 'id'
        assert table_info.columns[0].primary_key is True
        assert table_info.primary_keys == ['id']
        assert table_info.row_count == 100

    def test_introspect_table_error(self, introspector):
        """Test table introspection with error."""
        introspector.inspector.get_columns.side_effect = Exception("Table not found")
        
        with pytest.raises(DatabaseError, match="Table introspection failed"):
            introspector.introspect_table('nonexistent', 'public')

    def test_find_tables_with_column(self, introspector):
        """Test finding tables with specific column."""
        introspector.inspector.get_table_names.return_value = ['users', 'posts', 'comments']
        
        # Mock column data for different tables
        def mock_get_columns(table, schema=None):
            if table == 'users':
                return [{'name': 'id'}, {'name': 'username'}]
            elif table == 'posts':
                return [{'name': 'id'}, {'name': 'user_id'}, {'name': 'title'}]
            elif table == 'comments':
                return [{'name': 'id'}, {'name': 'post_id'}, {'name': 'content'}]
            return []
        
        introspector.inspector.get_columns.side_effect = mock_get_columns
        
        tables = introspector.find_tables_with_column('id', 'public')
        
        assert set(tables) == {'users', 'posts', 'comments'}

    def test_find_related_tables(self, introspector):
        """Test finding related tables through foreign keys."""
        introspector.inspector.get_table_names.return_value = ['users', 'posts', 'comments']
        
        # Mock foreign key data
        def mock_get_foreign_keys(table, schema=None):
            if table == 'posts':
                return [{'referred_table': 'users'}]
            elif table == 'comments':
                return [{'referred_table': 'posts'}]
            return []
        
        introspector.inspector.get_foreign_keys.side_effect = mock_get_foreign_keys
        
        related = introspector.find_related_tables('users', 'public')
        
        # Should find posts table that references users
        table_names = [rel[0] for rel in related]
        assert 'posts' in table_names

    @patch('src.database.introspection.text')
    def test_get_sample_data(self, mock_text, introspector):
        """Test getting sample data from table."""
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.fetchall.return_value = [(1, 'John'), (2, 'Jane')]
        mock_result.keys.return_value = ['id', 'name']
        mock_connection.execute.return_value = mock_result
        
        introspector.pool.get_connection.return_value.__enter__ = Mock(return_value=mock_connection)
        introspector.pool.get_connection.return_value.__exit__ = Mock(return_value=None)
        
        sample_data = introspector.get_sample_data('users', 'public', limit=2)
        
        assert len(sample_data) == 2
        assert sample_data[0] == {'id': 1, 'name': 'John'}
        assert sample_data[1] == {'id': 2, 'name': 'Jane'}

    def test_export_schema_json(self, introspector):
        """Test exporting schema as JSON."""
        # Create mock schema info
        column = ColumnInfo('id', ColumnType.INTEGER, primary_key=True)
        table = TableInfo('users', schema='public', columns=[column], primary_keys=['id'])
        schema_info = SchemaInfo('public', tables=[table])
        
        with patch.object(introspector, 'introspect_schema', return_value=schema_info):
            json_str = introspector.export_schema_json('public')
        
        # Parse JSON to verify structure
        data = json.loads(json_str)
        assert data['name'] == 'public'
        assert len(data['tables']) == 1
        assert data['tables'][0]['name'] == 'users'
        assert len(data['tables'][0]['columns']) == 1
        assert data['tables'][0]['columns'][0]['name'] == 'id'
        assert data['tables'][0]['columns'][0]['type'] == 'integer'

    def test_analyze_table_relationships(self, introspector):
        """Test analyzing table relationships."""
        introspector.inspector.get_table_names.return_value = ['users', 'posts']
        
        def mock_find_related_tables(table, schema=None):
            if table == 'users':
                return [('posts', 'references')]
            elif table == 'posts':
                return [('users', 'referenced_by')]
            return []
        
        with patch.object(introspector, 'find_related_tables', side_effect=mock_find_related_tables):
            relationships = introspector.analyze_table_relationships('public')
        
        assert relationships['users'] == ['posts']
        assert relationships['posts'] == ['users']

    def test_introspect_schema_full(self, introspector):
        """Test full schema introspection."""
        # Mock all inspector methods
        introspector.inspector.get_table_names.return_value = ['users']
        introspector.inspector.get_view_names.return_value = ['user_stats']
        introspector.inspector.get_sequence_names.return_value = ['user_id_seq']
        
        # Mock table introspection
        table_info = TableInfo('users', schema='public')
        with patch.object(introspector, 'introspect_table', return_value=table_info):
            schema_info = introspector.introspect_schema('public')
        
        assert schema_info.name == 'public'
        assert len(schema_info.tables) == 1
        assert schema_info.tables[0].name == 'users'
        assert schema_info.views == ['user_stats']
        assert schema_info.sequences == ['user_id_seq']

    def test_introspect_schema_with_errors(self, introspector):
        """Test schema introspection with errors handled gracefully."""
        introspector.inspector.get_table_names.side_effect = Exception("Schema not found")
        introspector.inspector.get_view_names.return_value = []  # Prevent len() error
        introspector.inspector.get_sequence_names.return_value = []
        
        # The method should handle errors gracefully and return empty schema
        schema_info = introspector.introspect_schema('nonexistent')
        
        assert schema_info.name == 'nonexistent'
        assert len(schema_info.tables) == 0
        assert len(schema_info.views) == 0