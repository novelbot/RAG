"""
Document management models for the RAG server.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, BigInteger, Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column
from src.core.database import Base


class DocumentStatus(Enum):
    """Document processing status"""
    UPLOADING = "uploading"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    DELETED = "deleted"


class Document(Base):
    """Document model for storing document metadata"""
    
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(BigInteger, nullable=False)
    mime_type = Column(String(100), nullable=False)
    status = Column(SQLEnum(DocumentStatus), nullable=False, default=DocumentStatus.UPLOADING)
    
    # Metadata
    upload_date = Column(DateTime, default=datetime.utcnow)
    processed_date = Column(DateTime, nullable=True)
    doc_metadata = Column(JSON, nullable=True)
    
    # Processing information
    error_message = Column(Text, nullable=True)
    chunk_count = Column(Integer, nullable=True)
    vector_count = Column(Integer, nullable=True)
    
    # Access control
    owner_id = Column(Integer, nullable=True)
    access_tags = Column(JSON, nullable=True)
    
    def __repr__(self) -> str:
        return f"Document(id={self.id}, filename='{self.filename}', status='{self.status.value}')"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary format"""
        return {
            "id": self.id,
            "filename": self.filename,
            "original_filename": self.original_filename,
            "file_size": self.file_size,
            "mime_type": self.mime_type,
            "status": self.status.value,
            "upload_date": self.upload_date.isoformat() if self.upload_date else None,
            "processed_date": self.processed_date.isoformat() if self.processed_date else None,
            "metadata": self.doc_metadata or {},
            "chunk_count": self.chunk_count,
            "vector_count": self.vector_count,
            "error_message": self.error_message,
            "access_tags": self.access_tags or {}
        }
    
    @classmethod
    def from_file_info(cls, filename: str, file_path: str, file_size: int, 
                      mime_type: str, owner_id: Optional[int] = None) -> "Document":
        """Create document from file information"""
        return cls(
            filename=filename,
            original_filename=filename,
            file_path=file_path,
            file_size=file_size,
            mime_type=mime_type,
            owner_id=owner_id,
            status=DocumentStatus.UPLOADING
        )
    
    def mark_processed(self, chunk_count: int = 0, vector_count: int = 0, 
                      metadata: Optional[Dict[str, Any]] = None):
        """Mark document as successfully processed"""
        self.status = DocumentStatus.PROCESSED
        self.processed_date = datetime.utcnow()
        self.chunk_count = chunk_count
        self.vector_count = vector_count
        if metadata:
            self.doc_metadata = {**(self.doc_metadata or {}), **metadata}
    
    def mark_failed(self, error_message: str):
        """Mark document as failed to process"""
        self.status = DocumentStatus.FAILED
        self.error_message = error_message
        self.processed_date = datetime.utcnow()