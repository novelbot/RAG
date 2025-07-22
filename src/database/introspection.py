"""
Database schema detection and introspection capabilities.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

from sqlalchemy import (
    Engine, MetaData, Table, Column, inspect, text,
    String, Integer, Float, Boolean, DateTime, Date, Time, Text, JSON
)
from sqlalchemy.engine import Inspector
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.schema import ForeignKey, Index, PrimaryKeyConstraint, UniqueConstraint
from loguru import logger

from src.core.exceptions import DatabaseError
from src.core.logging import LoggerMixin
from src.database.pool import AdvancedConnectionPool


class ColumnType(Enum):
    """Column data types."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    DATE = "date"
    TIME = "time"
    TEXT = "text"
    JSON = "json"
    BINARY = "binary"
    UNKNOWN = "unknown"


@dataclass
class ColumnInfo:
    """Information about a database column."""
    name: str
    type: ColumnType
    nullable: bool = True
    default: Any = None
    primary_key: bool = False
    foreign_key: Optional[str] = None
    unique: bool = False
    indexed: bool = False
    autoincrement: bool = False
    comment: Optional[str] = None
    max_length: Optional[int] = None
    precision: Optional[int] = None
    scale: Optional[int] = None


@dataclass
class IndexInfo:
    """Information about a database index."""
    name: str
    columns: List[str]
    unique: bool = False
    type: Optional[str] = None
    comment: Optional[str] = None


@dataclass
class ForeignKeyInfo:
    """Information about a foreign key constraint."""
    name: str
    columns: List[str]
    referred_table: str
    referred_columns: List[str]
    on_delete: Optional[str] = None
    on_update: Optional[str] = None


@dataclass
class TableInfo:
    """Information about a database table."""
    name: str
    schema: Optional[str] = None
    columns: List[ColumnInfo] = field(default_factory=list)
    primary_keys: List[str] = field(default_factory=list)
    foreign_keys: List[ForeignKeyInfo] = field(default_factory=list)
    indexes: List[IndexInfo] = field(default_factory=list)
    unique_constraints: List[List[str]] = field(default_factory=list)
    comment: Optional[str] = None
    row_count: Optional[int] = None


@dataclass
class SchemaInfo:
    """Information about a database schema."""
    name: str
    tables: List[TableInfo] = field(default_factory=list)
    views: List[str] = field(default_factory=list)
    sequences: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    procedures: List[str] = field(default_factory=list)


class DatabaseIntrospector(LoggerMixin):
    """Database schema introspection and analysis."""
    
    def __init__(self, pool: AdvancedConnectionPool):
        self.pool = pool
        self.engine = pool.engine
        self.inspector = inspect(self.engine)
        self._type_mapping = self._build_type_mapping()
        
    def _build_type_mapping(self) -> Dict[str, ColumnType]:
        """Build mapping of SQLAlchemy types to ColumnType enum."""
        return {
            'VARCHAR': ColumnType.STRING,
            'CHAR': ColumnType.STRING,
            'TEXT': ColumnType.TEXT,
            'STRING': ColumnType.STRING,
            'INTEGER': ColumnType.INTEGER,
            'INT': ColumnType.INTEGER,
            'BIGINT': ColumnType.INTEGER,
            'SMALLINT': ColumnType.INTEGER,
            'TINYINT': ColumnType.INTEGER,
            'DECIMAL': ColumnType.FLOAT,
            'NUMERIC': ColumnType.FLOAT,
            'FLOAT': ColumnType.FLOAT,
            'DOUBLE': ColumnType.FLOAT,
            'REAL': ColumnType.FLOAT,
            'BOOLEAN': ColumnType.BOOLEAN,
            'BOOL': ColumnType.BOOLEAN,
            'DATE': ColumnType.DATE,
            'TIME': ColumnType.TIME,
            'DATETIME': ColumnType.DATETIME,
            'TIMESTAMP': ColumnType.DATETIME,
            'JSON': ColumnType.JSON,
            'JSONB': ColumnType.JSON,
            'BLOB': ColumnType.BINARY,
            'BINARY': ColumnType.BINARY,
            'VARBINARY': ColumnType.BINARY,
        }
    
    def get_schemas(self) -> List[str]:
        """Get all schema names."""
        try:
            schemas = self.inspector.get_schema_names()
            self.logger.debug(f"Found {len(schemas)} schemas")
            return schemas
        except Exception as e:
            self.logger.error(f"Failed to get schemas: {e}")
            return []
    
    def get_tables(self, schema: Optional[str] = None) -> List[str]:
        """Get all table names in a schema."""
        try:
            tables = self.inspector.get_table_names(schema=schema)
            self.logger.debug(f"Found {len(tables)} tables in schema {schema or 'default'}")
            return tables
        except Exception as e:
            self.logger.error(f"Failed to get tables: {e}")
            return []
    
    def get_views(self, schema: Optional[str] = None) -> List[str]:
        """Get all view names in a schema."""
        try:
            views = self.inspector.get_view_names(schema=schema)
            self.logger.debug(f"Found {len(views)} views in schema {schema or 'default'}")
            return views
        except Exception as e:
            self.logger.error(f"Failed to get views: {e}")
            return []
    
    def introspect_table(self, table_name: str, schema: Optional[str] = None) -> TableInfo:
        """
        Introspect a table and return detailed information.
        
        Args:
            table_name: Name of the table
            schema: Schema name (optional)
            
        Returns:
            TableInfo object with table details
        """
        try:
            table_info = TableInfo(name=table_name, schema=schema)
            
            # Get columns
            columns = self.inspector.get_columns(table_name, schema=schema)
            table_info.columns = [self._build_column_info(col) for col in columns]
            
            # Get primary keys
            pk_info = self.inspector.get_pk_constraint(table_name, schema=schema)
            table_info.primary_keys = pk_info.get('constrained_columns', [])
            
            # Mark primary key columns
            for col in table_info.columns:
                if col.name in table_info.primary_keys:
                    col.primary_key = True
            
            # Get foreign keys
            fk_info = self.inspector.get_foreign_keys(table_name, schema=schema)
            table_info.foreign_keys = [self._build_foreign_key_info(fk) for fk in fk_info]
            
            # Mark foreign key columns
            for fk in table_info.foreign_keys:
                for col_name in fk.columns:
                    col = next((c for c in table_info.columns if c.name == col_name), None)
                    if col:
                        col.foreign_key = f"{fk.referred_table}.{'.'.join(fk.referred_columns)}"
            
            # Get indexes
            indexes = self.inspector.get_indexes(table_name, schema=schema)
            table_info.indexes = [self._build_index_info(idx) for idx in indexes]
            
            # Mark indexed columns
            for idx in table_info.indexes:
                for col_name in idx.columns:
                    col = next((c for c in table_info.columns if c.name == col_name), None)
                    if col:
                        col.indexed = True
                        if idx.unique:
                            col.unique = True
            
            # Get unique constraints
            unique_constraints = self.inspector.get_unique_constraints(table_name, schema=schema)
            table_info.unique_constraints = [uc.get('column_names', []) for uc in unique_constraints]
            
            # Get table comment
            table_info.comment = self._get_table_comment(table_name, schema)
            
            # Get row count (optional, can be expensive)
            table_info.row_count = self._get_table_row_count(table_name, schema)
            
            self.logger.debug(f"Introspected table {table_name} with {len(table_info.columns)} columns")
            return table_info
            
        except Exception as e:
            self.logger.error(f"Failed to introspect table {table_name}: {e}")
            raise DatabaseError(f"Table introspection failed: {e}")
    
    def _build_column_info(self, column_data: Dict[str, Any]) -> ColumnInfo:
        """Build ColumnInfo from SQLAlchemy column data."""
        col_type = self._map_column_type(column_data['type'])
        
        return ColumnInfo(
            name=column_data['name'],
            type=col_type,
            nullable=column_data.get('nullable', True),
            default=column_data.get('default'),
            autoincrement=column_data.get('autoincrement', False),
            comment=column_data.get('comment'),
            max_length=getattr(column_data['type'], 'length', None),
            precision=getattr(column_data['type'], 'precision', None),
            scale=getattr(column_data['type'], 'scale', None)
        )
    
    def _map_column_type(self, sqlalchemy_type) -> ColumnType:
        """Map SQLAlchemy type to ColumnType enum."""
        type_name = str(sqlalchemy_type).upper()
        
        # Check for exact matches first
        for sql_type, column_type in self._type_mapping.items():
            if sql_type in type_name:
                return column_type
        
        # Default to unknown
        return ColumnType.UNKNOWN
    
    def _build_foreign_key_info(self, fk_data: Dict[str, Any]) -> ForeignKeyInfo:
        """Build ForeignKeyInfo from SQLAlchemy foreign key data."""
        return ForeignKeyInfo(
            name=fk_data.get('name', ''),
            columns=fk_data.get('constrained_columns', []),
            referred_table=fk_data.get('referred_table', ''),
            referred_columns=fk_data.get('referred_columns', []),
            on_delete=fk_data.get('options', {}).get('ondelete'),
            on_update=fk_data.get('options', {}).get('onupdate')
        )
    
    def _build_index_info(self, index_data: Dict[str, Any]) -> IndexInfo:
        """Build IndexInfo from SQLAlchemy index data."""
        return IndexInfo(
            name=index_data.get('name', ''),
            columns=index_data.get('column_names', []),
            unique=index_data.get('unique', False),
            type=index_data.get('type')
        )
    
    def _get_table_comment(self, table_name: str, schema: Optional[str] = None) -> Optional[str]:
        """Get table comment if available."""
        try:
            # This is database-specific and may not work on all databases
            table_info = self.inspector.get_table_comment(table_name, schema=schema)
            return table_info.get('text')
        except Exception:
            return None
    
    def _get_table_row_count(self, table_name: str, schema: Optional[str] = None, max_count: int = 1000000) -> Optional[int]:
        """Get approximate row count for a table."""
        try:
            full_table_name = f"{schema}.{table_name}" if schema else table_name
            
            with self.pool.get_connection() as conn:
                # Use a limit to avoid counting huge tables
                result = conn.execute(text(f"SELECT COUNT(*) FROM {full_table_name} LIMIT {max_count}"))
                return result.scalar()
        except Exception as e:
            self.logger.debug(f"Failed to get row count for {table_name}: {e}")
            return None
    
    def introspect_schema(self, schema: Optional[str] = None) -> SchemaInfo:
        """
        Introspect an entire schema.
        
        Args:
            schema: Schema name (optional)
            
        Returns:
            SchemaInfo object with schema details
        """
        try:
            schema_info = SchemaInfo(name=schema or "default")
            
            # Get tables
            table_names = self.get_tables(schema)
            schema_info.tables = [self.introspect_table(table, schema) for table in table_names]
            
            # Get views
            schema_info.views = self.get_views(schema)
            
            # Get sequences (if supported)
            try:
                schema_info.sequences = self.inspector.get_sequence_names(schema=schema)
            except Exception:
                schema_info.sequences = []
            
            self.logger.info(f"Introspected schema {schema or 'default'} with {len(schema_info.tables)} tables")
            return schema_info
            
        except Exception as e:
            self.logger.error(f"Failed to introspect schema {schema}: {e}")
            raise DatabaseError(f"Schema introspection failed: {e}")
    
    def find_tables_with_column(self, column_name: str, schema: Optional[str] = None) -> List[str]:
        """Find all tables that contain a specific column."""
        matching_tables = []
        
        try:
            tables = self.get_tables(schema)
            
            for table in tables:
                try:
                    columns = self.inspector.get_columns(table, schema=schema)
                    if any(col['name'] == column_name for col in columns):
                        matching_tables.append(table)
                except Exception as e:
                    self.logger.debug(f"Failed to check table {table}: {e}")
                    continue
            
            self.logger.debug(f"Found {len(matching_tables)} tables with column '{column_name}'")
            return matching_tables
            
        except Exception as e:
            self.logger.error(f"Failed to find tables with column {column_name}: {e}")
            return []
    
    def find_related_tables(self, table_name: str, schema: Optional[str] = None) -> List[Tuple[str, str]]:
        """
        Find tables related to the given table through foreign keys.
        
        Returns:
            List of (table_name, relationship_type) tuples
        """
        related_tables = []
        
        try:
            # Find tables that reference this table
            all_tables = self.get_tables(schema)
            
            for table in all_tables:
                if table == table_name:
                    continue
                    
                try:
                    fks = self.inspector.get_foreign_keys(table, schema=schema)
                    for fk in fks:
                        if fk.get('referred_table') == table_name:
                            related_tables.append((table, 'references'))
                except Exception as e:
                    self.logger.debug(f"Failed to check foreign keys for table {table}: {e}")
                    continue
            
            # Find tables that this table references
            try:
                fks = self.inspector.get_foreign_keys(table_name, schema=schema)
                for fk in fks:
                    referred_table = fk.get('referred_table')
                    if referred_table:
                        related_tables.append((referred_table, 'referenced_by'))
            except Exception as e:
                self.logger.debug(f"Failed to check foreign keys for table {table_name}: {e}")
            
            self.logger.debug(f"Found {len(related_tables)} related tables for {table_name}")
            return related_tables
            
        except Exception as e:
            self.logger.error(f"Failed to find related tables for {table_name}: {e}")
            return []
    
    def get_sample_data(self, table_name: str, schema: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get sample data from a table.
        
        Args:
            table_name: Name of the table
            schema: Schema name (optional)
            limit: Maximum number of rows to return
            
        Returns:
            List of dictionaries representing rows
        """
        try:
            full_table_name = f"{schema}.{table_name}" if schema else table_name
            
            with self.pool.get_connection() as conn:
                result = conn.execute(text(f"SELECT * FROM {full_table_name} LIMIT {limit}"))
                rows = result.fetchall()
                
                # Convert to list of dictionaries
                columns = result.keys()
                sample_data = []
                
                for row in rows:
                    row_dict = {}
                    for i, col in enumerate(columns):
                        value = row[i]
                        # Convert non-serializable types to strings
                        if isinstance(value, (bytes, bytearray)):
                            value = f"<binary data: {len(value)} bytes>"
                        elif hasattr(value, '__dict__'):
                            value = str(value)
                        row_dict[col] = value
                    sample_data.append(row_dict)
                
                self.logger.debug(f"Retrieved {len(sample_data)} sample rows from {table_name}")
                return sample_data
                
        except Exception as e:
            self.logger.error(f"Failed to get sample data from {table_name}: {e}")
            return []
    
    def export_schema_json(self, schema: Optional[str] = None, include_sample_data: bool = False) -> str:
        """
        Export schema information as JSON.
        
        Args:
            schema: Schema name (optional)
            include_sample_data: Whether to include sample data
            
        Returns:
            JSON string with schema information
        """
        try:
            schema_info = self.introspect_schema(schema)
            
            # Convert to dictionary for JSON serialization
            schema_dict = {
                'name': schema_info.name,
                'tables': [],
                'views': schema_info.views,
                'sequences': schema_info.sequences
            }
            
            for table in schema_info.tables:
                table_dict = {
                    'name': table.name,
                    'schema': table.schema,
                    'comment': table.comment,
                    'row_count': table.row_count,
                    'columns': [],
                    'primary_keys': table.primary_keys,
                    'foreign_keys': [],
                    'indexes': [],
                    'unique_constraints': table.unique_constraints
                }
                
                # Add columns
                for col in table.columns:
                    col_dict = {
                        'name': col.name,
                        'type': col.type.value,
                        'nullable': col.nullable,
                        'default': col.default,
                        'primary_key': col.primary_key,
                        'foreign_key': col.foreign_key,
                        'unique': col.unique,
                        'indexed': col.indexed,
                        'autoincrement': col.autoincrement,
                        'comment': col.comment,
                        'max_length': col.max_length,
                        'precision': col.precision,
                        'scale': col.scale
                    }
                    table_dict['columns'].append(col_dict)
                
                # Add foreign keys
                for fk in table.foreign_keys:
                    fk_dict = {
                        'name': fk.name,
                        'columns': fk.columns,
                        'referred_table': fk.referred_table,
                        'referred_columns': fk.referred_columns,
                        'on_delete': fk.on_delete,
                        'on_update': fk.on_update
                    }
                    table_dict['foreign_keys'].append(fk_dict)
                
                # Add indexes
                for idx in table.indexes:
                    idx_dict = {
                        'name': idx.name,
                        'columns': idx.columns,
                        'unique': idx.unique,
                        'type': idx.type,
                        'comment': idx.comment
                    }
                    table_dict['indexes'].append(idx_dict)
                
                # Add sample data if requested
                if include_sample_data:
                    table_dict['sample_data'] = self.get_sample_data(table.name, table.schema)
                
                schema_dict['tables'].append(table_dict)
            
            return json.dumps(schema_dict, indent=2, default=str)
            
        except Exception as e:
            self.logger.error(f"Failed to export schema JSON: {e}")
            raise DatabaseError(f"Schema export failed: {e}")
    
    def analyze_table_relationships(self, schema: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Analyze relationships between tables in a schema.
        
        Args:
            schema: Schema name (optional)
            
        Returns:
            Dictionary mapping table names to their related tables
        """
        try:
            relationships = {}
            tables = self.get_tables(schema)
            
            for table in tables:
                related = self.find_related_tables(table, schema)
                relationships[table] = [rel[0] for rel in related]
            
            self.logger.debug(f"Analyzed relationships for {len(tables)} tables")
            return relationships
            
        except Exception as e:
            self.logger.error(f"Failed to analyze table relationships: {e}")
            return {}