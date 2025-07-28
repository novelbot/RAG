"""
Audit Logging System.

This module provides comprehensive audit logging for all access attempts,
permission changes, and security events with secure log storage and analysis.
"""

import json
import hashlib
import threading
from typing import Dict, List, Any, Optional, Set, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import Column, String, Integer, Boolean, DateTime, Text, ForeignKey, Index
from sqlalchemy.orm import relationship, Session
from sqlalchemy.sql import func

from src.core.database import Base
from src.core.mixins import TimestampMixin
from src.core.logging import LoggerMixin
from src.database.base import DatabaseManager
from .exceptions import AuditLoggingError


class AuditEventType(Enum):
    """Types of audit events."""
    # Authentication events
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILED = "login_failed"
    LOGOUT = "logout"
    TOKEN_REFRESH = "token_refresh"
    TOKEN_REVOKED = "token_revoked"
    PASSWORD_CHANGED = "password_changed"
    ACCOUNT_LOCKED = "account_locked"
    ACCOUNT_UNLOCKED = "account_unlocked"
    
    # Authorization events
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_REVOKED = "permission_revoked"
    ROLE_ASSIGNED = "role_assigned"
    ROLE_REMOVED = "role_removed"
    
    # Group events
    GROUP_CREATED = "group_created"
    GROUP_DELETED = "group_deleted"
    GROUP_UPDATED = "group_updated"
    GROUP_MEMBER_ADDED = "group_member_added"
    GROUP_MEMBER_REMOVED = "group_member_removed"
    GROUP_ROLE_ASSIGNED = "group_role_assigned"
    GROUP_ROLE_REVOKED = "group_role_revoked"
    
    # Resource access events
    RESOURCE_ACCESSED = "resource_accessed"
    RESOURCE_CREATED = "resource_created"
    RESOURCE_UPDATED = "resource_updated"
    RESOURCE_DELETED = "resource_deleted"
    RESOURCE_SHARED = "resource_shared"
    RESOURCE_UNSHARED = "resource_unshared"
    
    # System events
    SYSTEM_STARTED = "system_started"
    SYSTEM_STOPPED = "system_stopped"
    CONFIGURATION_CHANGED = "configuration_changed"
    SECURITY_POLICY_CHANGED = "security_policy_changed"
    
    # Suspicious activity
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    BRUTE_FORCE_ATTACK = "brute_force_attack"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    UNAUTHORIZED_ACCESS = "unauthorized_access"


class AuditSeverity(Enum):
    """Severity levels for audit events."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuditResult(Enum):
    """Result of audited operation."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    BLOCKED = "blocked"


@dataclass
class AuditContext:
    """Context information for audit events."""
    user_id: Optional[int] = None
    username: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None
    client_info: Optional[Dict[str, Any]] = None
    additional_data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "session_id": self.session_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "request_id": self.request_id,
            "client_info": self.client_info,
            "additional_data": self.additional_data
        }


class AuditLog(Base, TimestampMixin):
    """
    Audit log entry model.
    
    Stores comprehensive audit information with integrity verification.
    """
    
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True)
    
    # Event identification
    event_type = Column(String(100), nullable=False, index=True)
    severity = Column(String(20), default=AuditSeverity.MEDIUM.value, nullable=False)
    result = Column(String(20), default=AuditResult.SUCCESS.value, nullable=False)
    
    # Event details
    message = Column(Text, nullable=False)
    description = Column(Text)
    resource_type = Column(String(100))
    resource_id = Column(String(255))
    resource_name = Column(String(255))
    
    # Context information
    user_id = Column(Integer, ForeignKey('users.id'))
    username = Column(String(100))
    session_id = Column(String(255))
    ip_address = Column(String(45))  # IPv6 compatible
    user_agent = Column(Text)
    request_id = Column(String(255))
    
    # Metadata
    audit_metadata = Column(Text)  # JSON field for additional data
    
    # Integrity and security
    checksum = Column(String(64), nullable=False)  # SHA-256 hash
    previous_log_id = Column(Integer, ForeignKey('audit_logs.id'))
    chain_hash = Column(String(64))  # Hash chain for integrity
    
    # Timestamps
    event_timestamp = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", foreign_keys=[user_id])
    previous_log = relationship("AuditLog", remote_side=[id])
    
    # Indexes for performance
    __table_args__ = (
        Index("idx_audit_event_type", "event_type"),
        Index("idx_audit_user_id", "user_id"),
        Index("idx_audit_timestamp", "event_timestamp"),
        Index("idx_audit_severity", "severity"),
        Index("idx_audit_result", "result"),
        Index("idx_audit_resource", "resource_type", "resource_id"),
        Index("idx_audit_ip", "ip_address"),
    )
    
    def __repr__(self) -> str:
        return f"<AuditLog(id={self.id}, event_type='{self.event_type}', user_id={self.user_id})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "event_type": self.event_type,
            "severity": self.severity,
            "result": self.result,
            "message": self.message,
            "description": self.description,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "resource_name": self.resource_name,
            "user_id": self.user_id,
            "username": self.username,
            "session_id": self.session_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "request_id": self.request_id,
            "metadata": json.loads(self.audit_metadata) if self.audit_metadata else {},
            "checksum": self.checksum,
            "previous_log_id": self.previous_log_id,
            "chain_hash": self.chain_hash,
            "event_timestamp": self.event_timestamp.isoformat(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class AuditStatistics:
    """Audit statistics summary."""
    total_events: int
    events_by_type: Dict[str, int]
    events_by_severity: Dict[str, int]
    events_by_result: Dict[str, int]
    events_by_user: Dict[int, int]
    recent_events: List[Dict[str, Any]]
    suspicious_activity_count: int
    failed_login_attempts: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "total_events": self.total_events,
            "events_by_type": self.events_by_type,
            "events_by_severity": self.events_by_severity,
            "events_by_result": self.events_by_result,
            "events_by_user": self.events_by_user,
            "recent_events": self.recent_events,
            "suspicious_activity_count": self.suspicious_activity_count,
            "failed_login_attempts": self.failed_login_attempts
        }


class AuditLogger(LoggerMixin):
    """
    Comprehensive Audit Logging System.
    
    Provides secure audit logging with integrity verification,
    suspicious activity detection, and comprehensive reporting.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize audit logger.
        
        Args:
            db_manager: Database manager
        """
        self.db_manager = db_manager
        self._lock = threading.Lock()
        
        # Configuration
        self.enable_integrity_chain = True
        self.enable_suspicious_detection = True
        self.retention_days = 365  # 1 year default
        self.max_chain_length = 10000
        
        # Suspicious activity tracking
        self.failed_login_attempts: Dict[str, List[datetime]] = {}
        self.suspicious_ips: Set[str] = set()
        self.rate_limit_window = 300  # 5 minutes
        self.max_attempts_per_window = 5
        
        # Last log for chaining
        self.last_log_id: Optional[int] = None
        self.last_chain_hash: Optional[str] = None
        
        # Initialize last log info
        self._initialize_chain_info()
        
        self.logger.info("Audit Logger initialized successfully")
    
    def _get_session(self) -> Session:
        """Get database session."""
        return self.db_manager.get_session()
    
    def _initialize_chain_info(self) -> None:
        """Initialize chain information from the last log entry."""
        try:
            with self._get_session() as session:
                last_log = session.query(AuditLog).order_by(AuditLog.id.desc()).first()
                if last_log:
                    self.last_log_id = last_log.id
                    self.last_chain_hash = last_log.chain_hash
                else:
                    self.last_log_id = None
                    self.last_chain_hash = None
        except Exception as e:
            self.logger.error(f"Failed to initialize chain info: {e}")
    
    def _generate_checksum(self, log_data: Dict[str, Any]) -> str:
        """Generate SHA-256 checksum for log integrity."""
        try:
            # Create deterministic string representation
            sorted_data = json.dumps(log_data, sort_keys=True, default=str)
            return hashlib.sha256(sorted_data.encode()).hexdigest()
        except Exception as e:
            self.logger.error(f"Failed to generate checksum: {e}")
            return hashlib.sha256(str(log_data).encode()).hexdigest()
    
    def _generate_chain_hash(self, current_checksum: str, previous_hash: Optional[str]) -> str:
        """Generate hash chain for integrity verification."""
        try:
            if previous_hash:
                combined = f"{previous_hash}{current_checksum}"
            else:
                combined = current_checksum
            return hashlib.sha256(combined.encode()).hexdigest()
        except Exception as e:
            self.logger.error(f"Failed to generate chain hash: {e}")
            return hashlib.sha256(current_checksum.encode()).hexdigest()
    
    def log_event(self, 
                  event_type: AuditEventType,
                  message: str,
                  context: Optional[AuditContext] = None,
                  severity: AuditSeverity = AuditSeverity.MEDIUM,
                  result: AuditResult = AuditResult.SUCCESS,
                  resource_type: Optional[str] = None,
                  resource_id: Optional[str] = None,
                  resource_name: Optional[str] = None,
                  description: Optional[str] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> Optional[AuditLog]:
        """
        Log an audit event.
        
        Args:
            event_type: Type of audit event
            message: Event message
            context: Audit context information
            severity: Event severity
            result: Event result
            resource_type: Type of resource involved
            resource_id: ID of resource involved
            resource_name: Name of resource involved
            description: Detailed description
            metadata: Additional metadata
            
        Returns:
            Created audit log entry or None if failed
        """
        try:
            with self._lock:
                # Prepare log data
                log_data = {
                    "event_type": event_type.value,
                    "severity": severity.value,
                    "result": result.value,
                    "message": message,
                    "description": description,
                    "resource_type": resource_type,
                    "resource_id": resource_id,
                    "resource_name": resource_name,
                    "event_timestamp": datetime.utcnow().isoformat(),
                    "metadata": metadata or {}
                }
                
                # Add context information
                if context:
                    log_data.update(context.to_dict())
                
                # Generate integrity hash
                checksum = self._generate_checksum(log_data)
                
                # Generate chain hash
                chain_hash = self._generate_chain_hash(checksum, self.last_chain_hash)
                
                # Create audit log entry
                with self._get_session() as session:
                    audit_log = AuditLog(
                        event_type=event_type.value,
                        severity=severity.value,
                        result=result.value,
                        message=message,
                        description=description,
                        resource_type=resource_type,
                        resource_id=resource_id,
                        resource_name=resource_name,
                        user_id=context.user_id if context else None,
                        username=context.username if context else None,
                        session_id=context.session_id if context else None,
                        ip_address=context.ip_address if context else None,
                        user_agent=context.user_agent if context else None,
                        request_id=context.request_id if context else None,
                        metadata=json.dumps(metadata) if metadata else None,
                        checksum=checksum,
                        previous_log_id=self.last_log_id,
                        chain_hash=chain_hash if self.enable_integrity_chain else None
                    )
                    
                    session.add(audit_log)
                    session.commit()
                    
                    # Update chain info
                    self.last_log_id = audit_log.id
                    self.last_chain_hash = chain_hash
                    
                    # Check for suspicious activity
                    if self.enable_suspicious_detection:
                        self._check_suspicious_activity(audit_log)
                    
                    self.logger.debug(f"Logged audit event: {event_type.value} for user {context.user_id if context else 'system'}")
                    return audit_log
                    
        except Exception as e:
            self.logger.error(f"Failed to log audit event: {e}")
            return None
    
    def _check_suspicious_activity(self, audit_log: AuditLog) -> None:
        """Check for suspicious activity patterns."""
        try:
            # Check for failed login attempts
            if audit_log.event_type == AuditEventType.LOGIN_FAILED.value:
                self._track_failed_login(audit_log.ip_address, audit_log.username)
            
            # Check for privilege escalation attempts
            if audit_log.event_type == AuditEventType.PERMISSION_GRANTED.value:
                self._check_privilege_escalation(audit_log)
            
            # Check for unusual access patterns
            if audit_log.event_type == AuditEventType.RESOURCE_ACCESSED.value:
                self._check_unusual_access(audit_log)
                
        except Exception as e:
            self.logger.error(f"Failed to check suspicious activity: {e}")
    
    def _track_failed_login(self, ip_address: Optional[str], username: Optional[str]) -> None:
        """Track failed login attempts for suspicious activity detection."""
        if not ip_address:
            return
        
        try:
            current_time = datetime.utcnow()
            key = f"{ip_address}:{username or 'unknown'}"
            
            # Initialize tracking for this key
            if key not in self.failed_login_attempts:
                self.failed_login_attempts[key] = []
            
            # Add current attempt
            self.failed_login_attempts[key].append(current_time)
            
            # Clean old attempts
            cutoff_time = current_time - timedelta(seconds=self.rate_limit_window)
            self.failed_login_attempts[key] = [
                attempt for attempt in self.failed_login_attempts[key]
                if attempt > cutoff_time
            ]
            
            # Check if threshold exceeded
            if len(self.failed_login_attempts[key]) >= self.max_attempts_per_window:
                self.suspicious_ips.add(ip_address)
                
                # Log suspicious activity
                self.log_event(
                    event_type=AuditEventType.BRUTE_FORCE_ATTACK,
                    message=f"Potential brute force attack detected from {ip_address}",
                    context=AuditContext(
                        ip_address=ip_address,
                        username=username,
                        additional_data={
                            "failed_attempts": len(self.failed_login_attempts[key]),
                            "time_window": self.rate_limit_window
                        }
                    ),
                    severity=AuditSeverity.HIGH,
                    result=AuditResult.BLOCKED
                )
                
        except Exception as e:
            self.logger.error(f"Failed to track failed login: {e}")
    
    def _check_privilege_escalation(self, audit_log: AuditLog) -> None:
        """Check for potential privilege escalation."""
        try:
            # This is a simplified check - in practice, you'd implement more sophisticated detection
            metadata = json.loads(audit_log.metadata) if audit_log.metadata else {}
            
            # Check for admin permissions being granted
            if 'admin' in metadata.get('permissions', []):
                self.log_event(
                    event_type=AuditEventType.PRIVILEGE_ESCALATION,
                    message=f"Admin privileges granted to user {audit_log.user_id}",
                    context=AuditContext(
                        user_id=audit_log.user_id,
                        username=audit_log.username,
                        ip_address=audit_log.ip_address,
                        additional_data=metadata
                    ),
                    severity=AuditSeverity.HIGH,
                    result=AuditResult.SUCCESS
                )
                
        except Exception as e:
            self.logger.error(f"Failed to check privilege escalation: {e}")
    
    def _check_unusual_access(self, audit_log: AuditLog) -> None:
        """Check for unusual access patterns."""
        try:
            # Check for access from suspicious IPs
            if audit_log.ip_address in self.suspicious_ips:
                self.log_event(
                    event_type=AuditEventType.SUSPICIOUS_ACTIVITY,
                    message=f"Resource access from suspicious IP {audit_log.ip_address}",
                    context=AuditContext(
                        user_id=audit_log.user_id,
                        username=audit_log.username,
                        ip_address=audit_log.ip_address,
                        additional_data={"original_event_id": audit_log.id}
                    ),
                    severity=AuditSeverity.MEDIUM,
                    result=AuditResult.SUCCESS
                )
                
        except Exception as e:
            self.logger.error(f"Failed to check unusual access: {e}")
    
    def verify_log_integrity(self, log_id: int) -> bool:
        """
        Verify the integrity of a specific log entry.
        
        Args:
            log_id: Log entry ID to verify
            
        Returns:
            True if log integrity is valid
        """
        try:
            with self._get_session() as session:
                log_entry = session.query(AuditLog).filter(AuditLog.id == log_id).first()
                if not log_entry:
                    return False
                
                # Reconstruct log data
                log_data = {
                    "event_type": log_entry.event_type,
                    "severity": log_entry.severity,
                    "result": log_entry.result,
                    "message": log_entry.message,
                    "description": log_entry.description,
                    "resource_type": log_entry.resource_type,
                    "resource_id": log_entry.resource_id,
                    "resource_name": log_entry.resource_name,
                    "event_timestamp": log_entry.event_timestamp.isoformat(),
                    "metadata": json.loads(log_entry.metadata) if log_entry.metadata else {},
                    "user_id": log_entry.user_id,
                    "username": log_entry.username,
                    "session_id": log_entry.session_id,
                    "ip_address": log_entry.ip_address,
                    "user_agent": log_entry.user_agent,
                    "request_id": log_entry.request_id
                }
                
                # Verify checksum
                calculated_checksum = self._generate_checksum(log_data)
                if calculated_checksum != log_entry.checksum:
                    self.logger.warning(f"Checksum mismatch for log {log_id}")
                    return False
                
                # Verify chain hash if enabled
                if self.enable_integrity_chain and log_entry.chain_hash:
                    previous_hash = None
                    if log_entry.previous_log_id:
                        previous_log = session.query(AuditLog).filter(
                            AuditLog.id == log_entry.previous_log_id
                        ).first()
                        if previous_log:
                            previous_hash = previous_log.chain_hash
                    
                    calculated_chain_hash = self._generate_chain_hash(calculated_checksum, previous_hash)
                    if calculated_chain_hash != log_entry.chain_hash:
                        self.logger.warning(f"Chain hash mismatch for log {log_id}")
                        return False
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to verify log integrity: {e}")
            return False
    
    def get_audit_statistics(self, 
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> AuditStatistics:
        """
        Get audit statistics for a time period.
        
        Args:
            start_date: Start date for statistics
            end_date: End date for statistics
            
        Returns:
            Audit statistics
        """
        try:
            with self._get_session() as session:
                query = session.query(AuditLog)
                
                # Apply date filters
                if start_date:
                    query = query.filter(AuditLog.event_timestamp >= start_date)
                if end_date:
                    query = query.filter(AuditLog.event_timestamp <= end_date)
                
                # Get all logs
                logs = query.all()
                
                # Calculate statistics
                total_events = len(logs)
                events_by_type = {}
                events_by_severity = {}
                events_by_result = {}
                events_by_user = {}
                
                for log in logs:
                    # By type
                    events_by_type[log.event_type] = events_by_type.get(log.event_type, 0) + 1
                    
                    # By severity
                    events_by_severity[log.severity] = events_by_severity.get(log.severity, 0) + 1
                    
                    # By result
                    events_by_result[log.result] = events_by_result.get(log.result, 0) + 1
                    
                    # By user
                    if log.user_id:
                        events_by_user[log.user_id] = events_by_user.get(log.user_id, 0) + 1
                
                # Get recent events
                recent_logs = session.query(AuditLog).order_by(
                    AuditLog.event_timestamp.desc()
                ).limit(20).all()
                
                recent_events = [log.to_dict() for log in recent_logs]
                
                # Count suspicious activities
                suspicious_activity_count = len([
                    log for log in logs
                    if log.event_type in [
                        AuditEventType.SUSPICIOUS_ACTIVITY.value,
                        AuditEventType.BRUTE_FORCE_ATTACK.value,
                        AuditEventType.PRIVILEGE_ESCALATION.value,
                        AuditEventType.UNAUTHORIZED_ACCESS.value
                    ]
                ])
                
                failed_login_attempts = len([
                    log for log in logs
                    if log.event_type == AuditEventType.LOGIN_FAILED.value
                ])
                
                return AuditStatistics(
                    total_events=total_events,
                    events_by_type=events_by_type,
                    events_by_severity=events_by_severity,
                    events_by_result=events_by_result,
                    events_by_user=events_by_user,
                    recent_events=recent_events,
                    suspicious_activity_count=suspicious_activity_count,
                    failed_login_attempts=failed_login_attempts
                )
                
        except Exception as e:
            self.logger.error(f"Failed to get audit statistics: {e}")
            raise AuditLoggingError(f"Failed to get audit statistics: {e}")
    
    def search_logs(self, 
                    event_type: Optional[AuditEventType] = None,
                    user_id: Optional[int] = None,
                    ip_address: Optional[str] = None,
                    severity: Optional[AuditSeverity] = None,
                    result: Optional[AuditResult] = None,
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None,
                    limit: int = 100) -> List[AuditLog]:
        """
        Search audit logs with filters.
        
        Args:
            event_type: Filter by event type
            user_id: Filter by user ID
            ip_address: Filter by IP address
            severity: Filter by severity
            result: Filter by result
            start_date: Filter by start date
            end_date: Filter by end date
            limit: Maximum number of results
            
        Returns:
            List of matching audit logs
        """
        try:
            with self._get_session() as session:
                query = session.query(AuditLog)
                
                # Apply filters
                if event_type:
                    query = query.filter(AuditLog.event_type == event_type.value)
                if user_id:
                    query = query.filter(AuditLog.user_id == user_id)
                if ip_address:
                    query = query.filter(AuditLog.ip_address == ip_address)
                if severity:
                    query = query.filter(AuditLog.severity == severity.value)
                if result:
                    query = query.filter(AuditLog.result == result.value)
                if start_date:
                    query = query.filter(AuditLog.event_timestamp >= start_date)
                if end_date:
                    query = query.filter(AuditLog.event_timestamp <= end_date)
                
                # Order by timestamp descending and limit results
                query = query.order_by(AuditLog.event_timestamp.desc()).limit(limit)
                
                return query.all()
                
        except Exception as e:
            self.logger.error(f"Failed to search logs: {e}")
            return []
    
    def cleanup_old_logs(self, retention_days: Optional[int] = None) -> int:
        """
        Clean up old audit logs based on retention policy.
        
        Args:
            retention_days: Number of days to retain logs
            
        Returns:
            Number of logs deleted
        """
        try:
            days_to_keep = retention_days or self.retention_days
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            with self._get_session() as session:
                # Count logs to be deleted
                count = session.query(AuditLog).filter(
                    AuditLog.event_timestamp < cutoff_date
                ).count()
                
                # Delete old logs
                session.query(AuditLog).filter(
                    AuditLog.event_timestamp < cutoff_date
                ).delete()
                
                session.commit()
                
                self.logger.info(f"Cleaned up {count} old audit logs")
                return count
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup old logs: {e}")
            return 0
    
    def export_logs(self, 
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None,
                    format: str = "json") -> str:
        """
        Export audit logs to various formats.
        
        Args:
            start_date: Start date for export
            end_date: End date for export
            format: Export format (json, csv)
            
        Returns:
            Exported logs as string
        """
        try:
            with self._get_session() as session:
                query = session.query(AuditLog)
                
                # Apply date filters
                if start_date:
                    query = query.filter(AuditLog.event_timestamp >= start_date)
                if end_date:
                    query = query.filter(AuditLog.event_timestamp <= end_date)
                
                logs = query.order_by(AuditLog.event_timestamp.desc()).all()
                
                if format.lower() == "json":
                    return json.dumps([log.to_dict() for log in logs], indent=2, default=str)
                elif format.lower() == "csv":
                    import csv
                    import io
                    
                    output = io.StringIO()
                    writer = csv.DictWriter(output, fieldnames=[
                        'id', 'event_type', 'severity', 'result', 'message',
                        'user_id', 'username', 'ip_address', 'event_timestamp'
                    ])
                    
                    writer.writeheader()
                    for log in logs:
                        writer.writerow({
                            'id': log.id,
                            'event_type': log.event_type,
                            'severity': log.severity,
                            'result': log.result,
                            'message': log.message,
                            'user_id': log.user_id,
                            'username': log.username,
                            'ip_address': log.ip_address,
                            'event_timestamp': log.event_timestamp.isoformat()
                        })
                    
                    return output.getvalue()
                else:
                    raise ValueError(f"Unsupported export format: {format}")
                    
        except Exception as e:
            self.logger.error(f"Failed to export logs: {e}")
            raise AuditLoggingError(f"Failed to export logs: {e}")
    
    def is_ip_suspicious(self, ip_address: str) -> bool:
        """Check if an IP address is marked as suspicious."""
        return ip_address in self.suspicious_ips
    
    def clear_suspicious_ip(self, ip_address: str) -> None:
        """Clear an IP address from the suspicious list."""
        self.suspicious_ips.discard(ip_address)
        self.logger.info(f"Cleared suspicious IP: {ip_address}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get audit system health information."""
        try:
            with self._get_session() as session:
                total_logs = session.query(AuditLog).count()
                recent_logs = session.query(AuditLog).filter(
                    AuditLog.event_timestamp >= datetime.utcnow() - timedelta(hours=24)
                ).count()
                
                return {
                    "total_logs": total_logs,
                    "recent_logs_24h": recent_logs,
                    "suspicious_ips": len(self.suspicious_ips),
                    "integrity_chain_enabled": self.enable_integrity_chain,
                    "suspicious_detection_enabled": self.enable_suspicious_detection,
                    "retention_days": self.retention_days,
                    "last_log_id": self.last_log_id,
                    "chain_hash_present": self.last_chain_hash is not None
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get system health: {e}")
            return {"error": str(e)}