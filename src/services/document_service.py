"""
Document management service for handling document operations.
"""

import os
import mimetypes
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc

from src.core.database import get_session
from src.core.logging import LoggerMixin
from src.models.document import Document, DocumentStatus
from src.file_system.scanner import DirectoryScanner
from src.file_system.metadata_extractor import MetadataExtractor


class DocumentService(LoggerMixin):
    """Service class for document management operations"""
    
    def __init__(self):
        """Initialize document service"""
        self.scanner = DirectoryScanner()
        self.metadata_extractor = MetadataExtractor()
        self.logger.info("Document service initialized")
    
    def list_documents(self, 
                      page: int = 1, 
                      limit: int = 20,
                      status_filter: Optional[str] = None,
                      search_query: Optional[str] = None,
                      sort_by: str = "upload_date",
                      sort_order: str = "desc") -> Dict[str, Any]:
        """
        List documents with pagination and filtering
        
        Args:
            page: Page number (1-based)
            limit: Number of documents per page
            status_filter: Filter by document status
            search_query: Search in filename
            sort_by: Field to sort by
            sort_order: Sort order (asc/desc)
            
        Returns:
            Dictionary with documents list and pagination info
        """
        try:
            with get_session() as session:
                query = session.query(Document)
                
                # Apply status filter
                if status_filter:
                    try:
                        status_enum = DocumentStatus(status_filter)
                        query = query.filter(Document.status == status_enum)
                    except ValueError:
                        self.logger.warning(f"Invalid status filter: {status_filter}")
                
                # Apply search query
                if search_query:
                    search_pattern = f"%{search_query}%"
                    query = query.filter(
                        or_(
                            Document.filename.ilike(search_pattern),
                            Document.original_filename.ilike(search_pattern)
                        )
                    )
                
                # Apply sorting
                if hasattr(Document, sort_by):
                    sort_column = getattr(Document, sort_by)
                    if sort_order.lower() == "desc":
                        query = query.order_by(desc(sort_column))
                    else:
                        query = query.order_by(asc(sort_column))
                else:
                    # Default sort by upload_date desc
                    query = query.order_by(desc(Document.upload_date))
                
                # Get total count
                total_count = query.count()
                
                # Apply pagination
                offset = (page - 1) * limit
                documents = query.offset(offset).limit(limit).all()
                
                # Calculate pagination info
                total_pages = ((total_count - 1) // limit) + 1 if total_count > 0 else 1
                has_next = page < total_pages
                has_prev = page > 1
                
                return {
                    "documents": [doc.to_dict() for doc in documents],
                    "pagination": {
                        "page": page,
                        "limit": limit,
                        "total_count": total_count,
                        "total_pages": total_pages,
                        "has_next": has_next,
                        "has_prev": has_prev
                    }
                }
        
        except Exception as e:
            self.logger.error(f"Error listing documents: {e}")
            raise
    
    def get_document(self, document_id: int) -> Optional[Document]:
        """
        Get a single document by ID
        
        Args:
            document_id: Document ID
            
        Returns:
            Document object or None if not found
        """
        try:
            with get_session() as session:
                document = session.query(Document).filter(Document.id == document_id).first()
                return document
        
        except Exception as e:
            self.logger.error(f"Error getting document {document_id}: {e}")
            raise
    
    def scan_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Scan directory for documents and update database
        
        Args:
            directory_path: Path to directory to scan
            
        Returns:
            List of scanned file information
        """
        try:
            directory = Path(directory_path)
            if not directory.exists() or not directory.is_dir():
                raise ValueError(f"Directory does not exist: {directory_path}")
            
            # Scan directory for supported files
            supported_extensions = {'.txt', '.pdf', '.doc', '.docx', '.md', '.rtf'}
            scanned_files = []
            
            for file_path in directory.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                    try:
                        # Get file info
                        file_size = file_path.stat().st_size
                        mime_type = mimetypes.guess_type(str(file_path))[0] or 'application/octet-stream'
                        
                        file_info = {
                            "filename": file_path.name,
                            "file_path": str(file_path.absolute()),
                            "file_size": file_size,
                            "mime_type": mime_type,
                            "relative_path": str(file_path.relative_to(directory))
                        }
                        
                        # Extract metadata if possible
                        try:
                            metadata = self.metadata_extractor.extract_metadata(str(file_path))
                            file_info["metadata"] = metadata
                        except Exception as meta_error:
                            self.logger.warning(f"Failed to extract metadata from {file_path}: {meta_error}")
                            file_info["metadata"] = {}
                        
                        scanned_files.append(file_info)
                        
                    except Exception as file_error:
                        self.logger.warning(f"Error processing file {file_path}: {file_error}")
                        continue
            
            self.logger.info(f"Scanned {len(scanned_files)} files from {directory_path}")
            return scanned_files
        
        except Exception as e:
            self.logger.error(f"Error scanning directory {directory_path}: {e}")
            raise
    
    def create_document(self, filename: str, file_path: str, file_size: int,
                       mime_type: str, owner_id: Optional[int] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> Document:
        """
        Create a new document record
        
        Args:
            filename: Original filename
            file_path: Path to the file
            file_size: File size in bytes
            mime_type: MIME type of the file
            owner_id: Owner user ID
            metadata: Additional metadata
            
        Returns:
            Created Document object
        """
        try:
            with get_session() as session:
                document = Document.from_file_info(
                    filename=filename,
                    file_path=file_path,
                    file_size=file_size,
                    mime_type=mime_type,
                    owner_id=owner_id
                )
                
                if metadata:
                    document.doc_metadata = metadata
                
                session.add(document)
                session.commit()
                session.refresh(document)
                
                self.logger.info(f"Created document record: {document.filename}")
                return document
        
        except Exception as e:
            self.logger.error(f"Error creating document: {e}")
            raise
    
    def update_document_status(self, document_id: int, status: DocumentStatus,
                              error_message: Optional[str] = None,
                              metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update document status
        
        Args:
            document_id: Document ID
            status: New status
            error_message: Error message if failed
            metadata: Additional metadata
            
        Returns:
            True if updated successfully
        """
        try:
            with get_session() as session:
                document = session.query(Document).filter(Document.id == document_id).first()
                if not document:
                    return False
                
                document.status = status
                if error_message:
                    document.error_message = error_message
                
                if metadata:
                    document.doc_metadata = {**(document.doc_metadata or {}), **metadata}
                
                if status == DocumentStatus.PROCESSED:
                    document.mark_processed(
                        chunk_count=metadata.get("chunk_count", 0) if metadata else 0,
                        vector_count=metadata.get("vector_count", 0) if metadata else 0
                    )
                elif status == DocumentStatus.FAILED:
                    document.mark_failed(error_message or "Processing failed")
                
                session.commit()
                self.logger.info(f"Updated document {document_id} status to {status.value}")
                return True
        
        except Exception as e:
            self.logger.error(f"Error updating document {document_id} status: {e}")
            raise
    
    def delete_document(self, document_id: int, remove_file: bool = False) -> bool:
        """
        Delete a document record and optionally the file
        
        Args:
            document_id: Document ID
            remove_file: Whether to remove the actual file
            
        Returns:
            True if deleted successfully
        """
        try:
            with get_session() as session:
                document = session.query(Document).filter(Document.id == document_id).first()
                if not document:
                    return False
                
                file_path = document.file_path
                
                # Remove from database
                session.delete(document)
                session.commit()
                
                # Remove file if requested
                if remove_file and file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        self.logger.info(f"Removed file: {file_path}")
                    except Exception as file_error:
                        self.logger.warning(f"Failed to remove file {file_path}: {file_error}")
                
                self.logger.info(f"Deleted document {document_id}")
                return True
        
        except Exception as e:
            self.logger.error(f"Error deleting document {document_id}: {e}")
            raise
    
    def get_document_stats(self) -> Dict[str, Any]:
        """
        Get document statistics
        
        Returns:
            Statistics dictionary
        """
        try:
            with get_session() as session:
                total_count = session.query(Document).count()
                
                status_counts = {}
                for status in DocumentStatus:
                    count = session.query(Document).filter(Document.status == status).count()
                    status_counts[status.value] = count
                
                # Get total file size
                total_size = session.query(Document).with_entities(
                    session.query(Document.file_size).subquery().c.file_size
                ).scalar() or 0
                
                return {
                    "total_documents": total_count,
                    "status_breakdown": status_counts,
                    "total_file_size": total_size
                }
        
        except Exception as e:
            self.logger.error(f"Error getting document stats: {e}")
            raise