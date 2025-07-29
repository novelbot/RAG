"""
Pydantic schemas for API request and response validation.
"""

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Union, Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator
from enum import Enum


# Configuration for all schemas
class BaseAPISchema(BaseModel):
    """Base schema with common configuration"""
    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        extra="forbid"
    )


# Authentication Schemas
class LoginRequest(BaseAPISchema):
    """Login request schema"""
    username: str = Field(..., min_length=3, max_length=50, description="Username or email")
    password: str = Field(..., min_length=1, description="Password")
    remember_me: bool = Field(False, description="Remember login session")


class RegisterRequest(BaseAPISchema):
    """User registration request schema"""
    username: str = Field(..., min_length=3, max_length=50, description="Desired username")
    password: str = Field(..., min_length=6, description="Password (minimum 6 characters)")
    email: Optional[str] = Field("", max_length=255, description="Email address (optional)")
    role: Optional[str] = Field("user", description="User role (defaults to 'user')")


class RegisterResponse(BaseAPISchema):
    """User registration response schema"""
    message: str = Field(..., description="Registration status message")
    user_id: Optional[str] = Field(None, description="Created user ID if successful")
    username: str = Field(..., description="Registered username")


class TokenResponse(BaseAPISchema):
    """Token response schema"""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")


class UserResponse(BaseAPISchema):
    """User response schema"""
    id: str = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    email: str = Field(..., description="Email address")
    roles: List[str] = Field(default_factory=list, description="User roles")
    is_active: bool = Field(..., description="Whether user is active")


# Query Processing Schemas
class QueryRequest(BaseAPISchema):
    """Query request schema"""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query or question")
    max_results: int = Field(10, ge=1, le=100, description="Maximum number of results to return")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional search filters")
    include_metadata: bool = Field(True, description="Include document metadata in results")


class SearchResult(BaseAPISchema):
    """Individual search result schema"""
    id: str = Field(..., description="Document ID")
    title: Optional[str] = Field(None, description="Document title")
    content: str = Field(..., description="Document content excerpt")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")


class SearchResponse(BaseAPISchema):
    """Search response schema"""
    query: str = Field(..., description="Original search query")
    results: List[SearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., ge=0, description="Total number of results found")
    search_time_ms: float = Field(..., ge=0, description="Search execution time in milliseconds")
    user_id: str = Field(..., description="ID of user who performed the search")


class AnswerSource(BaseAPISchema):
    """Source document for RAG answer"""
    document_id: str = Field(..., description="Source document ID")
    title: Optional[str] = Field(None, description="Document title")
    excerpt: str = Field(..., description="Relevant excerpt from document")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")


class RAGResponse(BaseAPISchema):
    """RAG (Retrieval-Augmented Generation) response schema"""
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="AI-generated answer")
    sources: List[AnswerSource] = Field(..., description="Source documents used for answer")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Processing metadata")
    user_id: str = Field(..., description="ID of user who asked the question")


class BatchSearchRequest(BaseAPISchema):
    """Batch search request schema"""
    queries: List[str] = Field(..., min_length=1, max_length=10, description="List of search queries")
    max_results: int = Field(5, ge=1, le=50, description="Maximum results per query")


class BatchSearchResponse(BaseAPISchema):
    """Batch search response schema"""
    batch_id: str = Field(..., description="Batch processing ID")
    queries_processed: int = Field(..., ge=0, description="Number of queries processed")
    results: List[Dict[str, Any]] = Field(..., description="Results for each query")
    user_id: str = Field(..., description="ID of user who submitted the batch")


class QueryHistory(BaseAPISchema):
    """Query history item schema"""
    id: str = Field(..., description="Query ID")
    query: str = Field(..., description="Search query or question")
    timestamp: datetime = Field(..., description="Query timestamp")
    type: Literal["search", "ask"] = Field(..., description="Type of query")


class QueryHistoryResponse(BaseAPISchema):
    """Query history response schema"""
    history: List[QueryHistory] = Field(..., description="Query history items")
    total: int = Field(..., ge=0, description="Total number of queries")
    limit: int = Field(..., ge=1, description="Number of items per page")
    offset: int = Field(..., ge=0, description="Number of items skipped")
    user_id: str = Field(..., description="User ID")


# Document Management Schemas
class DocumentStatus(str, Enum):
    """Document processing status"""
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    REPROCESSING = "reprocessing"


class DocumentUploadResponse(BaseAPISchema):
    """Document upload response schema"""
    document_id: str = Field(..., description="Generated document ID")
    filename: str = Field(..., description="Original filename")
    size: int = Field(..., ge=0, description="File size in bytes")
    status: DocumentStatus = Field(..., description="Processing status")
    message: str = Field(..., description="Status message")
    user_id: str = Field(..., description="ID of user who uploaded the document")


class BatchUploadResponse(BaseAPISchema):
    """Batch upload response schema"""
    batch_id: str = Field(..., description="Batch processing ID")
    uploaded_count: int = Field(..., ge=0, description="Number of files uploaded")
    documents: List[Dict[str, Any]] = Field(..., description="Upload results for each file")
    user_id: str = Field(..., description="User ID")


class DocumentInfo(BaseAPISchema):
    """Document information schema"""
    id: str = Field(..., description="Document ID")
    filename: str = Field(..., description="Original filename")
    status: DocumentStatus = Field(..., description="Processing status")
    upload_date: datetime = Field(..., description="Upload timestamp")
    size: int = Field(..., ge=0, description="File size in bytes")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")


class DocumentListResponse(BaseAPISchema):
    """Document list response schema"""
    documents: List[DocumentInfo] = Field(..., description="List of documents")
    total: int = Field(..., ge=0, description="Total number of documents")
    page: int = Field(..., ge=1, description="Current page number")
    limit: int = Field(..., ge=1, description="Items per page")
    total_pages: int = Field(..., ge=0, description="Total number of pages")
    user_id: str = Field(..., description="User ID")


class DocumentDetailResponse(BaseAPISchema):
    """Detailed document information schema"""
    id: str = Field(..., description="Document ID")
    filename: str = Field(..., description="Original filename")
    status: DocumentStatus = Field(..., description="Processing status")
    upload_date: datetime = Field(..., description="Upload timestamp")
    size: int = Field(..., ge=0, description="File size in bytes")
    chunks: Optional[int] = Field(None, ge=0, description="Number of text chunks")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    user_id: str = Field(..., description="User ID")


class DocumentDeleteResponse(BaseAPISchema):
    """Document deletion response schema"""
    message: str = Field(..., description="Deletion confirmation message")
    document_id: str = Field(..., description="Deleted document ID")


class DocumentReprocessResponse(BaseAPISchema):
    """Document reprocessing response schema"""
    document_id: str = Field(..., description="Document ID")
    status: DocumentStatus = Field(..., description="New processing status")
    message: str = Field(..., description="Reprocessing status message")
    user_id: str = Field(..., description="User ID")


# Monitoring and Health Schemas
class ComponentStatus(BaseAPISchema):
    """Component health status schema"""
    status: Literal["healthy", "unhealthy", "degraded"] = Field(..., description="Component status")
    last_check: datetime = Field(..., description="Last health check timestamp")
    response_time_ms: Optional[float] = Field(None, ge=0, description="Response time in milliseconds")


class HealthCheckResponse(BaseAPISchema):
    """Health check response schema"""
    status: Literal["healthy", "unhealthy", "degraded"] = Field(..., description="Overall system status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    response_time_ms: float = Field(..., ge=0, description="Total response time")
    components: Dict[str, ComponentStatus] = Field(..., description="Individual component statuses")
    version: str = Field(..., description="Application version")


class SimpleHealthResponse(BaseAPISchema):
    """Simple health check response schema"""
    status: Literal["healthy", "unhealthy"] = Field(..., description="Basic health status")
    timestamp: datetime = Field(..., description="Health check timestamp")


class SystemMetrics(BaseAPISchema):
    """System metrics schema"""
    cpu_usage_percent: float = Field(..., ge=0, le=100, description="CPU usage percentage")
    memory_usage_percent: float = Field(..., ge=0, le=100, description="Memory usage percentage")
    disk_usage_percent: float = Field(..., ge=0, le=100, description="Disk usage percentage")
    uptime_seconds: int = Field(..., ge=0, description="System uptime in seconds")


class ApplicationMetrics(BaseAPISchema):
    """Application metrics schema"""
    total_requests: int = Field(..., ge=0, description="Total number of requests")
    requests_per_minute: float = Field(..., ge=0, description="Requests per minute")
    average_response_time_ms: float = Field(..., ge=0, description="Average response time")
    error_rate_percent: float = Field(..., ge=0, le=100, description="Error rate percentage")
    active_connections: int = Field(..., ge=0, description="Active connections")


class DatabaseMetrics(BaseAPISchema):
    """Database metrics schema"""
    connection_pool_usage: int = Field(..., ge=0, description="Active connections in pool")
    connection_pool_size: int = Field(..., ge=0, description="Total connection pool size")
    query_count: int = Field(..., ge=0, description="Total number of queries")
    average_query_time_ms: float = Field(..., ge=0, description="Average query time")


class VectorDatabaseMetrics(BaseAPISchema):
    """Vector database metrics schema"""
    collection_count: int = Field(..., ge=0, description="Number of collections")
    total_vectors: int = Field(..., ge=0, description="Total number of vectors")
    index_status: Literal["ready", "building", "error"] = Field(..., description="Index status")
    search_latency_ms: float = Field(..., ge=0, description="Average search latency")


class LLMMetrics(BaseAPISchema):
    """LLM service metrics schema"""
    total_requests: int = Field(..., ge=0, description="Total LLM requests")
    successful_requests: int = Field(..., ge=0, description="Successful requests")
    average_latency_ms: float = Field(..., ge=0, description="Average response latency")
    token_usage: int = Field(..., ge=0, description="Total tokens used")


class MetricsResponse(BaseAPISchema):
    """System metrics response schema"""
    system: SystemMetrics = Field(..., description="System-level metrics")
    application: ApplicationMetrics = Field(..., description="Application-level metrics")
    database: DatabaseMetrics = Field(..., description="Database metrics")
    vector_database: VectorDatabaseMetrics = Field(..., description="Vector database metrics")
    llm_services: LLMMetrics = Field(..., description="LLM service metrics")
    timestamp: datetime = Field(..., description="Metrics collection timestamp")


class LogEntry(BaseAPISchema):
    """Log entry schema"""
    timestamp: datetime = Field(..., description="Log timestamp")
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(..., description="Log level")
    module: str = Field(..., description="Module name")
    message: str = Field(..., description="Log message")
    request_id: Optional[str] = Field(None, description="Request ID")


class LogsResponse(BaseAPISchema):
    """Logs response schema"""
    logs: List[LogEntry] = Field(..., description="Log entries")
    total: int = Field(..., ge=0, description="Total number of log entries")
    filters: Dict[str, Any] = Field(..., description="Applied filters")
    timestamp: datetime = Field(..., description="Response timestamp")


class UsageStatistics(BaseAPISchema):
    """Usage statistics schema"""
    total_queries: int = Field(..., ge=0, description="Total number of queries")
    search_queries: int = Field(..., ge=0, description="Number of search queries")
    ask_queries: int = Field(..., ge=0, description="Number of ask queries")
    documents_uploaded: int = Field(..., ge=0, description="Number of documents uploaded")
    average_query_time_ms: float = Field(..., ge=0, description="Average query time")
    total_tokens_used: int = Field(..., ge=0, description="Total tokens used")
    error_count: int = Field(..., ge=0, description="Number of errors")


class UsageStatsResponse(BaseAPISchema):
    """Usage statistics response schema"""
    period: Literal["1h", "24h", "7d", "30d"] = Field(..., description="Statistics period")
    user_id: str = Field(..., description="User ID")
    statistics: UsageStatistics = Field(..., description="Usage statistics")
    timestamp: datetime = Field(..., description="Statistics timestamp")


class AlertRequest(BaseAPISchema):
    """Alert creation request schema"""
    name: str = Field(..., min_length=1, max_length=100, description="Alert name")
    description: Optional[str] = Field(None, max_length=500, description="Alert description")
    condition: Dict[str, Any] = Field(..., description="Alert condition configuration")
    actions: List[Dict[str, Any]] = Field(..., description="Actions to take when alert fires")
    enabled: bool = Field(True, description="Whether alert is enabled")


class AlertResponse(BaseAPISchema):
    """Alert creation response schema"""
    message: str = Field(..., description="Success message")
    alert_id: str = Field(..., description="Created alert ID")


class ServiceInfo(BaseAPISchema):
    """Service information schema"""
    status: Literal["running", "stopped", "error"] = Field(..., description="Service status")
    uptime: Optional[str] = Field(None, description="Service uptime")
    version: Optional[str] = Field(None, description="Service version")
    response_time_ms: Optional[float] = Field(None, ge=0, description="Response time")
    connection_count: Optional[int] = Field(None, ge=0, description="Active connections")


class LLMProviderStatus(BaseAPISchema):
    """LLM provider status schema"""
    status: Literal["available", "unavailable", "degraded"] = Field(..., description="Provider status")
    latency_ms: float = Field(..., ge=0, description="Response latency")


class ServiceStatusResponse(BaseAPISchema):
    """Service status response schema"""
    services: Dict[str, ServiceInfo] = Field(..., description="Service status information")
    llm_providers: Optional[Dict[str, LLMProviderStatus]] = Field(None, description="LLM provider statuses")
    overall_status: Literal["operational", "degraded", "outage"] = Field(..., description="Overall system status")
    last_updated: datetime = Field(..., description="Last status update timestamp")


# Common Response Schemas
class MessageResponse(BaseAPISchema):
    """Simple message response schema"""
    message: str = Field(..., description="Response message")


class ErrorResponse(BaseAPISchema):
    """Error response schema"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Error timestamp")


class PaginationInfo(BaseAPISchema):
    """Pagination information schema"""
    page: int = Field(..., ge=1, description="Current page number")
    limit: int = Field(..., ge=1, le=100, description="Items per page")
    total: int = Field(..., ge=0, description="Total number of items")
    total_pages: int = Field(..., ge=0, description="Total number of pages")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_prev: bool = Field(..., description="Whether there is a previous page")


class PaginatedResponse(BaseAPISchema):
    """Generic paginated response schema"""
    items: List[Any] = Field(..., description="List of items")
    pagination: PaginationInfo = Field(..., description="Pagination information")


# Validation Helpers
@field_validator('*', mode='before')
@classmethod
def empty_str_to_none(cls, v):
    """Convert empty strings to None"""
    if v == '':
        return None
    return v


# Model configuration updates
def configure_schema_extra(schema: Dict[str, Any], model_type):
    """Add extra configuration to schema"""
    if "properties" in schema:
        for prop_name, prop_info in schema["properties"].items():
            if "description" not in prop_info:
                prop_info["description"] = f"{prop_name.replace('_', ' ').title()}"