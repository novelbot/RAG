"""
Factory for creating RDB extractors for different database types.
"""

from typing import Dict, Type, Optional
from src.core.config import DatabaseConfig
from src.database.drivers import DatabaseType
from .base import BaseRDBExtractor, ExtractionConfig
from .exceptions import ExtractionConfigurationError


class RDBExtractorFactory:
    """
    Factory for creating RDB extractors based on database type.
    
    This factory uses the registry pattern to allow for easy extension
    with new database-specific extractors.
    """
    
    # Registry of extractor classes by database type
    _extractors: Dict[DatabaseType, Type[BaseRDBExtractor]] = {}
    
    @classmethod
    def register(cls, database_type: DatabaseType, extractor_class: Type[BaseRDBExtractor]) -> None:
        """
        Register an extractor class for a specific database type.
        
        Args:
            database_type: Database type
            extractor_class: Extractor class to register
        """
        cls._extractors[database_type] = extractor_class
    
    @classmethod
    def create(cls, config: ExtractionConfig) -> BaseRDBExtractor:
        """
        Create an extractor instance for the specified database type.
        
        Args:
            config: Extraction configuration
            
        Returns:
            RDB extractor instance
            
        Raises:
            ExtractionConfigurationError: If no extractor is registered for the database type
        """
        database_type = config.database_config.database_type
        
        if database_type not in cls._extractors:
            # If no specific extractor is registered, use the generic one
            from .generic import GenericRDBExtractor
            return GenericRDBExtractor(config)
        
        extractor_class = cls._extractors[database_type]
        return extractor_class(config)
    
    @classmethod
    def get_supported_types(cls) -> List[DatabaseType]:
        """
        Get list of supported database types.
        
        Returns:
            List of supported database types
        """
        return list(cls._extractors.keys())
    
    @classmethod
    def is_supported(cls, database_type: DatabaseType) -> bool:
        """
        Check if a database type is supported.
        
        Args:
            database_type: Database type to check
            
        Returns:
            True if supported, False otherwise
        """
        return database_type in cls._extractors