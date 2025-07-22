"""
File Metadata Type Definitions.

This module defines data classes for representing various types of file metadata,
including basic file system metadata and format-specific metadata.
"""

import stat
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union


@dataclass
class BasicFileMetadata:
    """Basic file system metadata available for all files."""
    size: int
    creation_time: datetime
    modification_time: datetime
    access_time: datetime
    permissions: str
    file_type: str
    owner_uid: int
    group_gid: int
    is_symlink: bool = False
    inode: Optional[int] = None
    device_id: Optional[int] = None
    hard_link_count: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "size": self.size,
            "creation_time": self.creation_time.isoformat(),
            "modification_time": self.modification_time.isoformat(),
            "access_time": self.access_time.isoformat(),
            "permissions": self.permissions,
            "file_type": self.file_type,
            "owner_uid": self.owner_uid,
            "group_gid": self.group_gid,
            "is_symlink": self.is_symlink,
            "inode": self.inode,
            "device_id": self.device_id,
            "hard_link_count": self.hard_link_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BasicFileMetadata":
        """Create from dictionary (for JSON deserialization)."""
        return cls(
            size=data["size"],
            creation_time=datetime.fromisoformat(data["creation_time"]),
            modification_time=datetime.fromisoformat(data["modification_time"]),
            access_time=datetime.fromisoformat(data["access_time"]),
            permissions=data["permissions"],
            file_type=data["file_type"],
            owner_uid=data["owner_uid"],
            group_gid=data["group_gid"],
            is_symlink=data.get("is_symlink", False),
            inode=data.get("inode"),
            device_id=data.get("device_id"),
            hard_link_count=data.get("hard_link_count")
        )


class FormatSpecificMetadata(ABC):
    """Abstract base class for format-specific metadata."""
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        pass
    
    @abstractmethod
    def to_flat_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary for indexing and search."""
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FormatSpecificMetadata":
        """Create from dictionary (for JSON deserialization)."""
        pass


@dataclass
class PDFMetadata(FormatSpecificMetadata):
    """Metadata specific to PDF files."""
    page_count: int
    title: Optional[str] = None
    author: Optional[str] = None
    creator: Optional[str] = None
    producer: Optional[str] = None
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None
    encrypted: bool = False
    version: Optional[str] = None
    has_form_fields: bool = False
    has_bookmarks: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "pdf",
            "page_count": self.page_count,
            "title": self.title,
            "author": self.author,
            "creator": self.creator,
            "producer": self.producer,
            "creation_date": self.creation_date.isoformat() if self.creation_date else None,
            "modification_date": self.modification_date.isoformat() if self.modification_date else None,
            "encrypted": self.encrypted,
            "version": self.version,
            "has_form_fields": self.has_form_fields,
            "has_bookmarks": self.has_bookmarks
        }
    
    def to_flat_dict(self) -> Dict[str, Any]:
        return {
            "format_type": "pdf",
            "pdf_page_count": self.page_count,
            "pdf_title": self.title,
            "pdf_author": self.author,
            "pdf_encrypted": self.encrypted,
            "pdf_version": self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PDFMetadata":
        return cls(
            page_count=data["page_count"],
            title=data.get("title"),
            author=data.get("author"),
            creator=data.get("creator"),
            producer=data.get("producer"),
            creation_date=datetime.fromisoformat(data["creation_date"]) if data.get("creation_date") else None,
            modification_date=datetime.fromisoformat(data["modification_date"]) if data.get("modification_date") else None,
            encrypted=data.get("encrypted", False),
            version=data.get("version"),
            has_form_fields=data.get("has_form_fields", False),
            has_bookmarks=data.get("has_bookmarks", False)
        )


@dataclass 
class WordMetadata(FormatSpecificMetadata):
    """Metadata specific to Word documents."""
    word_count: int
    page_count: int = 0
    character_count: int = 0
    character_count_with_spaces: int = 0
    paragraph_count: int = 0
    title: Optional[str] = None
    author: Optional[str] = None
    last_modified_by: Optional[str] = None
    creation_date: Optional[datetime] = None
    last_modified: Optional[datetime] = None
    language: Optional[str] = None
    has_tables: bool = False
    has_images: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "word",
            "word_count": self.word_count,
            "page_count": self.page_count,
            "character_count": self.character_count,
            "character_count_with_spaces": self.character_count_with_spaces,
            "paragraph_count": self.paragraph_count,
            "title": self.title,
            "author": self.author,
            "last_modified_by": self.last_modified_by,
            "creation_date": self.creation_date.isoformat() if self.creation_date else None,
            "last_modified": self.last_modified.isoformat() if self.last_modified else None,
            "language": self.language,
            "has_tables": self.has_tables,
            "has_images": self.has_images
        }
    
    def to_flat_dict(self) -> Dict[str, Any]:
        return {
            "format_type": "word",
            "word_count": self.word_count,
            "page_count": self.page_count,
            "character_count": self.character_count,
            "word_title": self.title,
            "word_author": self.author
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WordMetadata":
        return cls(
            word_count=data["word_count"],
            page_count=data.get("page_count", 0),
            character_count=data.get("character_count", 0),
            character_count_with_spaces=data.get("character_count_with_spaces", 0),
            paragraph_count=data.get("paragraph_count", 0),
            title=data.get("title"),
            author=data.get("author"),
            last_modified_by=data.get("last_modified_by"),
            creation_date=datetime.fromisoformat(data["creation_date"]) if data.get("creation_date") else None,
            last_modified=datetime.fromisoformat(data["last_modified"]) if data.get("last_modified") else None,
            language=data.get("language"),
            has_tables=data.get("has_tables", False),
            has_images=data.get("has_images", False)
        )


@dataclass
class ExcelMetadata(FormatSpecificMetadata):
    """Metadata specific to Excel files."""
    sheet_names: List[str]
    sheet_count: int
    total_rows: int = 0
    total_columns: int = 0
    has_formulas: bool = False
    has_charts: bool = False
    title: Optional[str] = None
    author: Optional[str] = None
    creation_date: Optional[datetime] = None
    last_modified: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "excel",
            "sheet_names": self.sheet_names,
            "sheet_count": self.sheet_count,
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "has_formulas": self.has_formulas,
            "has_charts": self.has_charts,
            "title": self.title,
            "author": self.author,
            "creation_date": self.creation_date.isoformat() if self.creation_date else None,
            "last_modified": self.last_modified.isoformat() if self.last_modified else None
        }
    
    def to_flat_dict(self) -> Dict[str, Any]:
        return {
            "format_type": "excel",
            "sheet_count": self.sheet_count,
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "excel_title": self.title,
            "excel_author": self.author
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExcelMetadata":
        return cls(
            sheet_names=data["sheet_names"],
            sheet_count=data["sheet_count"],
            total_rows=data.get("total_rows", 0),
            total_columns=data.get("total_columns", 0),
            has_formulas=data.get("has_formulas", False),
            has_charts=data.get("has_charts", False),
            title=data.get("title"),
            author=data.get("author"),
            creation_date=datetime.fromisoformat(data["creation_date"]) if data.get("creation_date") else None,
            last_modified=datetime.fromisoformat(data["last_modified"]) if data.get("last_modified") else None
        )


@dataclass
class TextMetadata(FormatSpecificMetadata):
    """Metadata specific to text files."""
    line_count: int
    word_count: int
    character_count: int
    encoding: str
    has_bom: bool = False
    language: Optional[str] = None
    line_ending_type: str = "unknown"  # "unix", "windows", "mac", "mixed"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "text",
            "line_count": self.line_count,
            "word_count": self.word_count,
            "character_count": self.character_count,
            "encoding": self.encoding,
            "has_bom": self.has_bom,
            "language": self.language,
            "line_ending_type": self.line_ending_type
        }
    
    def to_flat_dict(self) -> Dict[str, Any]:
        return {
            "format_type": "text",
            "line_count": self.line_count,
            "word_count": self.word_count,
            "character_count": self.character_count,
            "text_encoding": self.encoding
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextMetadata":
        return cls(
            line_count=data["line_count"],
            word_count=data["word_count"],
            character_count=data["character_count"],
            encoding=data["encoding"],
            has_bom=data.get("has_bom", False),
            language=data.get("language"),
            line_ending_type=data.get("line_ending_type", "unknown")
        )


@dataclass
class ImageMetadata(FormatSpecificMetadata):
    """Metadata specific to image files."""
    width: int
    height: int
    format: str
    color_mode: str
    has_transparency: bool = False
    dpi: Optional[Tuple[int, int]] = None
    bit_depth: Optional[int] = None
    compression: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "image",
            "width": self.width,
            "height": self.height,
            "format": self.format,
            "color_mode": self.color_mode,
            "has_transparency": self.has_transparency,
            "dpi": list(self.dpi) if self.dpi else None,
            "bit_depth": self.bit_depth,
            "compression": self.compression
        }
    
    def to_flat_dict(self) -> Dict[str, Any]:
        return {
            "format_type": "image",
            "image_width": self.width,
            "image_height": self.height,
            "image_format": self.format,
            "image_color_mode": self.color_mode
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageMetadata":
        return cls(
            width=data["width"],
            height=data["height"],
            format=data["format"],
            color_mode=data["color_mode"],
            has_transparency=data.get("has_transparency", False),
            dpi=tuple(data["dpi"]) if data.get("dpi") else None,
            bit_depth=data.get("bit_depth"),
            compression=data.get("compression")
        )


@dataclass
class FileMetadata:
    """Complete metadata for a file including basic and format-specific data."""
    path: Path
    basic_metadata: BasicFileMetadata
    format_metadata: Optional[FormatSpecificMetadata] = None
    extraction_timestamp: datetime = field(default_factory=datetime.now)
    extraction_errors: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Convert string path to Path object."""
        if isinstance(self.path, str):
            self.path = Path(self.path)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "path": str(self.path),
            "basic_metadata": self.basic_metadata.to_dict(),
            "format_metadata": self.format_metadata.to_dict() if self.format_metadata else None,
            "extraction_timestamp": self.extraction_timestamp.isoformat(),
            "extraction_errors": self.extraction_errors
        }
    
    def to_flat_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary for indexing and search."""
        flat = {
            "file_path": str(self.path),
            "file_name": self.path.name,
            "file_extension": self.path.suffix.lower(),
            "extraction_timestamp": self.extraction_timestamp.isoformat(),
            "has_errors": len(self.extraction_errors) > 0,
            "error_count": len(self.extraction_errors)
        }
        
        # Add basic metadata
        flat.update({
            f"basic_{k}": v for k, v in self.basic_metadata.to_dict().items()
        })
        
        # Add format metadata if available
        if self.format_metadata:
            flat.update(self.format_metadata.to_flat_dict())
        
        return flat
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileMetadata":
        """Create from dictionary (for JSON deserialization)."""
        format_metadata = None
        if data.get("format_metadata"):
            format_data = data["format_metadata"]
            format_type = format_data.get("type")
            
            if format_type == "pdf":
                format_metadata = PDFMetadata.from_dict(format_data)
            elif format_type == "word":
                format_metadata = WordMetadata.from_dict(format_data)
            elif format_type == "excel":
                format_metadata = ExcelMetadata.from_dict(format_data)
            elif format_type == "text":
                format_metadata = TextMetadata.from_dict(format_data)
            elif format_type == "image":
                format_metadata = ImageMetadata.from_dict(format_data)
        
        return cls(
            path=Path(data["path"]),
            basic_metadata=BasicFileMetadata.from_dict(data["basic_metadata"]),
            format_metadata=format_metadata,
            extraction_timestamp=datetime.fromisoformat(data["extraction_timestamp"]),
            extraction_errors=data.get("extraction_errors", [])
        )


@dataclass
class MetadataStats:
    """Statistics about extracted metadata."""
    total_files: int
    successful_extractions: int
    failed_extractions: int
    total_size: int
    file_type_distribution: Dict[str, int] = field(default_factory=dict)
    format_type_distribution: Dict[str, int] = field(default_factory=dict)
    average_file_size: float = 0.0
    largest_file: Optional[Path] = None
    smallest_file: Optional[Path] = None
    extraction_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_files": self.total_files,
            "successful_extractions": self.successful_extractions,
            "failed_extractions": self.failed_extractions,
            "total_size": self.total_size,
            "file_type_distribution": self.file_type_distribution,
            "format_type_distribution": self.format_type_distribution,
            "average_file_size": self.average_file_size,
            "largest_file": str(self.largest_file) if self.largest_file else None,
            "smallest_file": str(self.smallest_file) if self.smallest_file else None,
            "extraction_time": self.extraction_time
        }