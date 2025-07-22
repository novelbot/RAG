"""
Pipeline monitoring and metrics collection.
"""

import time
import asyncio
import psutil
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque

from src.core.logging import LoggerMixin


class PipelineStatus(Enum):
    """Pipeline status states."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PipelineAlert:
    """Pipeline alert/notification."""
    id: str
    severity: AlertSeverity
    title: str
    message: str
    component: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceMetrics:
    """System resource metrics."""
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    memory_percent: float = 0.0
    disk_io_mb: float = 0.0
    network_io_mb: float = 0.0
    process_count: int = 0
    thread_count: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PerformanceMetrics:
    """Pipeline performance metrics."""
    throughput_items_per_second: float = 0.0
    average_latency_seconds: float = 0.0
    p95_latency_seconds: float = 0.0
    p99_latency_seconds: float = 0.0
    error_rate_percent: float = 0.0
    success_rate_percent: float = 100.0
    total_processed: int = 0
    total_errors: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CostMetrics:
    """Cost tracking metrics."""
    embedding_api_cost: float = 0.0
    storage_cost: float = 0.0
    compute_cost: float = 0.0
    total_cost: float = 0.0
    cost_per_document: float = 0.0
    estimated_monthly_cost: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ComponentHealth:
    """Health status of a pipeline component."""
    component_name: str
    healthy: bool
    last_check: datetime
    error_count: int = 0
    uptime_seconds: float = 0.0
    last_error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PipelineMetrics(LoggerMixin):
    """
    Comprehensive pipeline metrics collection and monitoring.
    
    Tracks performance, resource usage, costs, and health status
    across all pipeline components.
    """
    
    def __init__(self, monitoring_interval: float = 30.0, history_retention_hours: int = 24):
        """
        Initialize pipeline metrics.
        
        Args:
            monitoring_interval: How often to collect metrics (seconds)
            history_retention_hours: How long to keep historical metrics
        """
        self.monitoring_interval = monitoring_interval
        self.history_retention = timedelta(hours=history_retention_hours)
        
        # Current metrics
        self.status = PipelineStatus.INITIALIZING
        self.start_time = datetime.utcnow()
        self.last_updated = datetime.utcnow()
        
        # Historical metrics (time-series data)
        self.resource_history: deque = deque(maxlen=int(history_retention_hours * 3600 / monitoring_interval))
        self.performance_history: deque = deque(maxlen=int(history_retention_hours * 3600 / monitoring_interval))
        self.cost_history: deque = deque(maxlen=int(history_retention_hours * 3600 / monitoring_interval))
        
        # Component tracking
        self.component_health: Dict[str, ComponentHealth] = {}
        self.component_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Alerts and notifications
        self.active_alerts: Dict[str, PipelineAlert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        
        # Performance tracking
        self.latency_samples: deque = deque(maxlen=10000)  # Last 10k samples for percentiles
        self.error_events: deque = deque(maxlen=1000)
        
        # Counters
        self.total_documents_processed = 0
        self.total_errors = 0
        self.total_api_calls = 0
        self.total_cost = 0.0
        
        # Background monitoring task
        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_monitoring = False
    
    async def start_monitoring(self) -> None:
        """Start background monitoring."""
        if self._is_monitoring:
            return
        
        self._is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.status = PipelineStatus.RUNNING
        self.logger.info("Pipeline monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._is_monitoring = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.status = PipelineStatus.STOPPED
        self.logger.info("Pipeline monitoring stopped")
    
    def record_document_processed(self, processing_time: float, success: bool = True, cost: float = 0.0):
        """
        Record a processed document.
        
        Args:
            processing_time: Time taken to process the document
            success: Whether processing was successful
            cost: Cost associated with processing
        """
        self.total_documents_processed += 1
        self.latency_samples.append(processing_time)
        self.total_cost += cost
        
        if not success:
            self.total_errors += 1
            self.error_events.append(datetime.utcnow())
        
        self.last_updated = datetime.utcnow()
    
    def record_api_call(self, provider: str, cost: float, tokens: int = 0):
        """
        Record an API call.
        
        Args:
            provider: API provider name
            cost: Cost of the API call
            tokens: Number of tokens used
        """
        self.total_api_calls += 1
        self.total_cost += cost
        
        # Track per-provider metrics
        if provider not in self.component_metrics:
            self.component_metrics[provider] = {
                "api_calls": 0,
                "total_cost": 0.0,
                "total_tokens": 0
            }
        
        self.component_metrics[provider]["api_calls"] += 1
        self.component_metrics[provider]["total_cost"] += cost
        self.component_metrics[provider]["total_tokens"] += tokens
    
    def update_component_health(self, component_name: str, healthy: bool, error: Optional[str] = None, **metadata):
        """
        Update health status of a component.
        
        Args:
            component_name: Name of the component
            healthy: Whether the component is healthy
            error: Error message if unhealthy
            **metadata: Additional metadata
        """
        if component_name not in self.component_health:
            self.component_health[component_name] = ComponentHealth(
                component_name=component_name,
                healthy=healthy,
                last_check=datetime.utcnow(),
                uptime_seconds=0.0
            )
        
        component = self.component_health[component_name]
        previous_health = component.healthy
        
        component.healthy = healthy
        component.last_check = datetime.utcnow()
        component.metadata.update(metadata)
        
        if not healthy:
            component.error_count += 1
            component.last_error = error
            
            # Create alert for health degradation
            if previous_health:
                self._create_alert(
                    AlertSeverity.ERROR,
                    f"Component {component_name} Unhealthy",
                    f"Component {component_name} has become unhealthy: {error}",
                    component_name
                )
        else:
            # Resolve alert if component is healthy again
            if not previous_health:
                self._resolve_component_alerts(component_name)
        
        # Update uptime
        if hasattr(self, '_component_start_times'):
            start_time = self._component_start_times.get(component_name, self.start_time)
            component.uptime_seconds = (datetime.utcnow() - start_time).total_seconds()
    
    def _create_alert(self, severity: AlertSeverity, title: str, message: str, component: str, **metadata):
        """Create a new alert."""
        alert_id = f"{component}_{int(time.time())}"
        alert = PipelineAlert(
            id=alert_id,
            severity=severity,
            title=title,
            message=message,
            component=component,
            timestamp=datetime.utcnow(),
            metadata=metadata
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Log the alert
        log_method = {
            AlertSeverity.INFO: self.logger.info,
            AlertSeverity.WARNING: self.logger.warning,
            AlertSeverity.ERROR: self.logger.error,
            AlertSeverity.CRITICAL: self.logger.critical
        }.get(severity, self.logger.info)
        
        log_method(f"ALERT [{severity.value.upper()}] {title}: {message}")
    
    def _resolve_component_alerts(self, component: str):
        """Resolve all active alerts for a component."""
        resolved_alerts = []
        
        for alert_id, alert in self.active_alerts.items():
            if alert.component == component and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.utcnow()
                resolved_alerts.append(alert_id)
        
        for alert_id in resolved_alerts:
            del self.active_alerts[alert_id]
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while self._is_monitoring:
            try:
                await self._collect_metrics()
                await asyncio.sleep(self.monitoring_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _collect_metrics(self):
        """Collect current metrics."""
        current_time = datetime.utcnow()
        
        # Collect resource metrics
        resource_metrics = self._collect_resource_metrics()
        self.resource_history.append(resource_metrics)
        
        # Collect performance metrics
        performance_metrics = self._collect_performance_metrics()
        self.performance_history.append(performance_metrics)
        
        # Collect cost metrics
        cost_metrics = self._collect_cost_metrics()
        self.cost_history.append(cost_metrics)
        
        # Clean up old data
        self._cleanup_old_data(current_time)
        
        # Check for alerts
        self._check_alert_conditions(resource_metrics, performance_metrics)
    
    def _collect_resource_metrics(self) -> ResourceMetrics:
        """Collect system resource metrics."""
        try:
            process = psutil.Process()
            
            return ResourceMetrics(
                cpu_percent=psutil.cpu_percent(),
                memory_mb=process.memory_info().rss / 1024 / 1024,
                memory_percent=process.memory_percent(),
                process_count=len(psutil.pids()),
                thread_count=process.num_threads(),
                timestamp=datetime.utcnow()
            )
        except Exception as e:
            self.logger.warning(f"Failed to collect resource metrics: {e}")
            return ResourceMetrics()
    
    def _collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect performance metrics."""
        try:
            # Calculate throughput (items per second)
            runtime_seconds = (datetime.utcnow() - self.start_time).total_seconds()
            throughput = self.total_documents_processed / max(runtime_seconds, 1)
            
            # Calculate latency percentiles
            latency_list = list(self.latency_samples)
            if latency_list:
                latency_list.sort()
                avg_latency = sum(latency_list) / len(latency_list)
                p95_latency = latency_list[int(0.95 * len(latency_list))] if latency_list else 0
                p99_latency = latency_list[int(0.99 * len(latency_list))] if latency_list else 0
            else:
                avg_latency = p95_latency = p99_latency = 0
            
            # Calculate error rate
            total_operations = self.total_documents_processed
            error_rate = (self.total_errors / max(total_operations, 1)) * 100
            success_rate = 100 - error_rate
            
            return PerformanceMetrics(
                throughput_items_per_second=throughput,
                average_latency_seconds=avg_latency,
                p95_latency_seconds=p95_latency,
                p99_latency_seconds=p99_latency,
                error_rate_percent=error_rate,
                success_rate_percent=success_rate,
                total_processed=total_operations,
                total_errors=self.total_errors,
                timestamp=datetime.utcnow()
            )
        except Exception as e:
            self.logger.warning(f"Failed to collect performance metrics: {e}")
            return PerformanceMetrics()
    
    def _collect_cost_metrics(self) -> CostMetrics:
        """Collect cost metrics."""
        try:
            # Calculate cost per document
            cost_per_doc = self.total_cost / max(self.total_documents_processed, 1)
            
            # Estimate monthly cost based on current rate
            runtime_hours = (datetime.utcnow() - self.start_time).total_seconds() / 3600
            hourly_cost = self.total_cost / max(runtime_hours, 1)
            monthly_cost = hourly_cost * 24 * 30  # Rough monthly estimate
            
            return CostMetrics(
                embedding_api_cost=sum(
                    metrics.get("total_cost", 0) 
                    for component, metrics in self.component_metrics.items()
                    if "embedding" in component.lower()
                ),
                total_cost=self.total_cost,
                cost_per_document=cost_per_doc,
                estimated_monthly_cost=monthly_cost,
                timestamp=datetime.utcnow()
            )
        except Exception as e:
            self.logger.warning(f"Failed to collect cost metrics: {e}")
            return CostMetrics()
    
    def _cleanup_old_data(self, current_time: datetime):
        """Clean up old data based on retention policy."""
        cutoff_time = current_time - self.history_retention
        
        # Clean up error events
        self.error_events = deque(
            [event for event in self.error_events if event > cutoff_time],
            maxlen=1000
        )
    
    def _check_alert_conditions(self, resource_metrics: ResourceMetrics, performance_metrics: PerformanceMetrics):
        """Check for alert conditions."""
        # Memory usage alert
        if resource_metrics.memory_percent > 90:
            self._create_alert(
                AlertSeverity.CRITICAL,
                "High Memory Usage",
                f"Memory usage is {resource_metrics.memory_percent:.1f}%",
                "system"
            )
        elif resource_metrics.memory_percent > 80:
            self._create_alert(
                AlertSeverity.WARNING,
                "Elevated Memory Usage",
                f"Memory usage is {resource_metrics.memory_percent:.1f}%",
                "system"
            )
        
        # Error rate alert
        if performance_metrics.error_rate_percent > 20:
            self._create_alert(
                AlertSeverity.ERROR,
                "High Error Rate",
                f"Error rate is {performance_metrics.error_rate_percent:.1f}%",
                "pipeline"
            )
        elif performance_metrics.error_rate_percent > 10:
            self._create_alert(
                AlertSeverity.WARNING,
                "Elevated Error Rate",
                f"Error rate is {performance_metrics.error_rate_percent:.1f}%",
                "pipeline"
            )
        
        # Latency alert
        if performance_metrics.p95_latency_seconds > 60:  # 1 minute
            self._create_alert(
                AlertSeverity.WARNING,
                "High Latency",
                f"95th percentile latency is {performance_metrics.p95_latency_seconds:.1f}s",
                "pipeline"
            )
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        uptime_seconds = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            "status": self.status.value,
            "uptime_seconds": uptime_seconds,
            "start_time": self.start_time.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "total_documents_processed": self.total_documents_processed,
            "total_errors": self.total_errors,
            "total_api_calls": self.total_api_calls,
            "total_cost": self.total_cost,
            "active_alerts": len(self.active_alerts),
            "healthy_components": sum(1 for comp in self.component_health.values() if comp.healthy),
            "total_components": len(self.component_health)
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.performance_history:
            return {"status": "no_data"}
        
        latest = self.performance_history[-1]
        
        # Calculate trends if we have enough data
        trend_data = {}
        if len(self.performance_history) >= 2:
            prev = self.performance_history[-2]
            trend_data = {
                "throughput_trend": latest.throughput_items_per_second - prev.throughput_items_per_second,
                "latency_trend": latest.average_latency_seconds - prev.average_latency_seconds,
                "error_rate_trend": latest.error_rate_percent - prev.error_rate_percent
            }
        
        return {
            "current_performance": {
                "throughput_items_per_second": latest.throughput_items_per_second,
                "average_latency_seconds": latest.average_latency_seconds,
                "p95_latency_seconds": latest.p95_latency_seconds,
                "p99_latency_seconds": latest.p99_latency_seconds,
                "error_rate_percent": latest.error_rate_percent,
                "success_rate_percent": latest.success_rate_percent
            },
            "trends": trend_data,
            "totals": {
                "processed": latest.total_processed,
                "errors": latest.total_errors
            }
        }
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get resource usage summary."""
        if not self.resource_history:
            return {"status": "no_data"}
        
        latest = self.resource_history[-1]
        
        return {
            "current_usage": {
                "cpu_percent": latest.cpu_percent,
                "memory_mb": latest.memory_mb,
                "memory_percent": latest.memory_percent,
                "process_count": latest.process_count,
                "thread_count": latest.thread_count
            },
            "timestamp": latest.timestamp.isoformat()
        }
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost summary."""
        if not self.cost_history:
            return {"status": "no_data"}
        
        latest = self.cost_history[-1]
        
        return {
            "current_costs": {
                "total_cost": latest.total_cost,
                "cost_per_document": latest.cost_per_document,
                "estimated_monthly_cost": latest.estimated_monthly_cost,
                "embedding_api_cost": latest.embedding_api_cost
            },
            "provider_breakdown": {
                provider: metrics.get("total_cost", 0)
                for provider, metrics in self.component_metrics.items()
            },
            "timestamp": latest.timestamp.isoformat()
        }
    
    def get_alerts(self, severity: Optional[AlertSeverity] = None, component: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get alerts with optional filtering."""
        alerts = []
        
        for alert in self.active_alerts.values():
            if severity and alert.severity != severity:
                continue
            if component and alert.component != component:
                continue
            
            alerts.append({
                "id": alert.id,
                "severity": alert.severity.value,
                "title": alert.title,
                "message": alert.message,
                "component": alert.component,
                "timestamp": alert.timestamp.isoformat(),
                "resolved": alert.resolved,
                "metadata": alert.metadata
            })
        
        return alerts
    
    def get_component_health_summary(self) -> Dict[str, Any]:
        """Get component health summary."""
        return {
            component.component_name: {
                "healthy": component.healthy,
                "last_check": component.last_check.isoformat(),
                "error_count": component.error_count,
                "uptime_seconds": component.uptime_seconds,
                "last_error": component.last_error,
                "metadata": component.metadata
            }
            for component in self.component_health.values()
        }
    
    def export_metrics(self, format: str = "json") -> Union[str, Dict[str, Any]]:
        """Export all metrics in specified format."""
        data = {
            "status": self.get_current_status(),
            "performance": self.get_performance_summary(),
            "resources": self.get_resource_summary(),
            "costs": self.get_cost_summary(),
            "alerts": self.get_alerts(),
            "component_health": self.get_component_health_summary(),
            "export_timestamp": datetime.utcnow().isoformat()
        }
        
        if format.lower() == "json":
            import json
            return json.dumps(data, indent=2)
        else:
            return data