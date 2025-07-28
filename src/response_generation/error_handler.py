"""
Comprehensive Error Handling and Timeout Management for LLM Calls.
"""

import time
import asyncio
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone, timedelta

from src.core.logging import LoggerMixin
from src.response_generation.base import ResponseGeneratorConfig
from src.response_generation.exceptions import (
    ResponseGenerationError, TimeoutError, ProviderUnavailableError,
    ContextTooLongError, ResponseQualityError
)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"                    # Minor issues, can continue
    MEDIUM = "medium"              # Moderate issues, retry recommended  
    HIGH = "high"                  # Serious issues, fallback needed
    CRITICAL = "critical"          # Critical issues, abort operation


class ErrorCategory(Enum):
    """Error categories for classification."""
    TIMEOUT = "timeout"            # Request timeout errors
    RATE_LIMIT = "rate_limit"      # Rate limiting errors
    PROVIDER = "provider"          # Provider-specific errors
    NETWORK = "network"            # Network connectivity errors
    AUTHENTICATION = "authentication"  # Authentication/authorization errors
    CONTENT = "content"            # Content-related errors
    VALIDATION = "validation"      # Validation errors
    SYSTEM = "system"              # System-level errors
    UNKNOWN = "unknown"            # Unclassified errors


class RetryStrategy(Enum):
    """Retry strategies for failed operations."""
    NONE = "none"                  # No retry
    IMMEDIATE = "immediate"        # Immediate retry
    LINEAR = "linear"              # Linear backoff
    EXPONENTIAL = "exponential"    # Exponential backoff
    CUSTOM = "custom"              # Custom retry logic


@dataclass
class ErrorPolicy:
    """Policy for handling specific error types."""
    error_types: List[type]
    severity: ErrorSeverity
    category: ErrorCategory
    retry_strategy: RetryStrategy
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    should_fallback: bool = True
    should_circuit_break: bool = False
    custom_handler: Optional[Callable] = None


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3
    enable_half_open: bool = True


@dataclass
class TimeoutConfig:
    """Configuration for timeout management."""
    default_timeout: float = 30.0
    slow_timeout: float = 60.0
    fast_timeout: float = 15.0
    ensemble_timeout: float = 90.0
    enable_adaptive_timeout: bool = True
    timeout_multiplier: float = 1.5


@dataclass
class ErrorEvent:
    """Record of an error event."""
    timestamp: datetime
    error_type: str
    error_message: str
    severity: ErrorSeverity
    category: ErrorCategory
    provider: Optional[str] = None
    model: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    resolution: Optional[str] = None


class CircuitBreaker:
    """Circuit breaker implementation for provider reliability."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half-open
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half-open"
            else:
                raise ProviderUnavailableError("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    async def call_async(self, func: Callable, *args, **kwargs):
        """Execute async function with circuit breaker protection."""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half-open"
            else:
                raise ProviderUnavailableError("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if not self.last_failure_time:
            return True
        
        time_since_failure = datetime.now(timezone.utc) - self.last_failure_time
        return time_since_failure.total_seconds() >= self.config.recovery_timeout
    
    def _on_success(self):
        """Handle successful operation."""
        if self.state == "half-open":
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = "closed"
                self.failure_count = 0
                self.success_count = 0
        elif self.state == "closed":
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now(timezone.utc)
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = "open"
        elif self.state == "half-open":
            self.state = "open"


class ErrorHandler(LoggerMixin):
    """
    Comprehensive Error Handling and Timeout Management System.
    
    Features:
    - Circuit breaker pattern for provider reliability
    - Adaptive timeout management
    - Retry logic with multiple strategies
    - Error classification and severity assessment
    - Fallback strategies and graceful degradation
    - Error monitoring and reporting
    - Provider-specific error handling
    """
    
    def __init__(self, config: ResponseGeneratorConfig):
        """
        Initialize Error Handler.
        
        Args:
            config: Response generator configuration
        """
        self.config = config
        
        # Initialize timeout configuration
        self.timeout_config = TimeoutConfig(
            default_timeout=config.single_timeout,
            ensemble_timeout=config.ensemble_timeout
        )
        
        # Initialize circuit breaker configuration
        self.circuit_breaker_config = CircuitBreakerConfig()
        
        # Error policies for different error types
        self.error_policies = self._initialize_error_policies()
        
        # Circuit breakers for each provider
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Error history and statistics
        self.error_history: List[ErrorEvent] = []
        self.error_stats: Dict[str, int] = {}
        
        self.logger.info("ErrorHandler initialized successfully")
    
    def handle_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        retry_count: int = 0
    ) -> Dict[str, Any]:
        """
        Handle and classify an error.
        
        Args:
            error: The exception that occurred
            context: Context information about the error
            retry_count: Current retry attempt count
            
        Returns:
            Error handling decision with recommendations
        """
        # Classify the error
        severity, category = self._classify_error(error)
        
        # Find appropriate policy
        policy = self._find_error_policy(error, category)
        
        # Record error event
        error_event = ErrorEvent(
            timestamp=datetime.now(timezone.utc),
            error_type=type(error).__name__,
            error_message=str(error),
            severity=severity,
            category=category,
            provider=context.get("provider"),
            model=context.get("model"),
            context=context,
            retry_count=retry_count
        )
        
        self.error_history.append(error_event)
        self._update_error_stats(error_event)
        
        # Determine handling strategy
        decision = self._make_handling_decision(error, policy, retry_count, context)
        
        self.logger.warning(
            f"Handled {category.value} error: {error} "
            f"(severity: {severity.value}, decision: {decision['action']})"
        )
        
        return decision
    
    def should_retry(
        self,
        error: Exception,
        retry_count: int,
        context: Dict[str, Any]
    ) -> Tuple[bool, float]:
        """
        Determine if operation should be retried and calculate delay.
        
        Args:
            error: The exception that occurred
            retry_count: Current retry attempt count
            context: Context information
            
        Returns:
            Tuple of (should_retry, delay_seconds)
        """
        # Find error policy
        _, category = self._classify_error(error)
        policy = self._find_error_policy(error, category)
        
        # Check retry limits
        if retry_count >= policy.max_retries:
            return False, 0.0
        
        # Check circuit breaker status
        provider = context.get("provider")
        if provider and self._is_circuit_breaker_open(provider):
            return False, 0.0
        
        # Calculate delay based on strategy
        delay = self._calculate_retry_delay(
            policy.retry_strategy,
            retry_count,
            policy.base_delay,
            policy.max_delay
        )
        
        return True, delay
    
    def get_timeout(
        self,
        operation_type: str,
        provider: Optional[str] = None,
        complexity: Optional[int] = None
    ) -> float:
        """
        Get appropriate timeout for operation.
        
        Args:
            operation_type: Type of operation (single, ensemble, etc.)
            provider: Provider name for adaptive timeout
            complexity: Operation complexity score
            
        Returns:
            Timeout in seconds
        """
        # Base timeout by operation type
        base_timeout = {
            "single": self.timeout_config.default_timeout,
            "ensemble": self.timeout_config.ensemble_timeout,
            "fast": self.timeout_config.fast_timeout,
            "slow": self.timeout_config.slow_timeout
        }.get(operation_type, self.timeout_config.default_timeout)
        
        # Apply adaptive timeout if enabled
        if self.timeout_config.enable_adaptive_timeout:
            base_timeout = self._apply_adaptive_timeout(
                base_timeout, provider, complexity
            )
        
        return base_timeout
    
    def with_timeout(
        self,
        func: Callable,
        timeout: float,
        *args,
        **kwargs
    ):
        """
        Execute function with timeout (synchronous).
        
        Args:
            func: Function to execute
            timeout: Timeout in seconds
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            TimeoutError: If operation times out
        """
        # For synchronous operations, we'll use a simple timeout approach
        # In practice, you might want to use threading or process-based timeouts
        start_time = time.time()
        
        try:
            # Note: This is a simplified timeout implementation
            # Real implementation would need proper thread/process management
            result = func(*args, **kwargs)
            
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(
                    f"Operation timed out after {elapsed:.2f}s (limit: {timeout}s)",
                    timeout
                )
            
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(
                    f"Operation timed out after {elapsed:.2f}s (limit: {timeout}s)",
                    timeout
                )
            raise
    
    async def with_timeout_async(
        self,
        func: Callable,
        timeout: float,
        *args,
        **kwargs
    ):
        """
        Execute async function with timeout.
        
        Args:
            func: Async function to execute
            timeout: Timeout in seconds
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            TimeoutError: If operation times out
        """
        try:
            return await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Async operation timed out after {timeout}s",
                timeout
            )
    
    def with_circuit_breaker(
        self,
        func: Callable,
        provider: str,
        *args,
        **kwargs
    ):
        """
        Execute function with circuit breaker protection (synchronous).
        
        Args:
            func: Function to execute
            provider: Provider name for circuit breaker
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        circuit_breaker = self._get_circuit_breaker(provider)
        return circuit_breaker.call(func, *args, **kwargs)
    
    async def with_circuit_breaker_async(
        self,
        func: Callable,
        provider: str,
        *args,
        **kwargs
    ):
        """
        Execute async function with circuit breaker protection.
        
        Args:
            func: Async function to execute
            provider: Provider name for circuit breaker
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        circuit_breaker = self._get_circuit_breaker(provider)
        return await circuit_breaker.call_async(func, *args, **kwargs)
    
    def _classify_error(self, error: Exception) -> Tuple[ErrorSeverity, ErrorCategory]:
        """Classify error by severity and category."""
        
        error_type = type(error)
        error_message = str(error).lower()
        
        # Classify by error type
        if isinstance(error, TimeoutError) or 'timeout' in error_message:
            return ErrorSeverity.MEDIUM, ErrorCategory.TIMEOUT
        
        elif isinstance(error, ProviderUnavailableError):
            return ErrorSeverity.HIGH, ErrorCategory.PROVIDER
        
        elif isinstance(error, ContextTooLongError):
            return ErrorSeverity.LOW, ErrorCategory.CONTENT
        
        elif isinstance(error, ResponseQualityError):
            return ErrorSeverity.LOW, ErrorCategory.VALIDATION
        
        elif 'rate limit' in error_message or 'quota' in error_message:
            return ErrorSeverity.MEDIUM, ErrorCategory.RATE_LIMIT
        
        elif 'authentication' in error_message or 'unauthorized' in error_message:
            return ErrorSeverity.HIGH, ErrorCategory.AUTHENTICATION
        
        elif 'network' in error_message or 'connection' in error_message:
            return ErrorSeverity.MEDIUM, ErrorCategory.NETWORK
        
        elif isinstance(error, (ValueError, TypeError)):
            return ErrorSeverity.HIGH, ErrorCategory.VALIDATION
        
        elif isinstance(error, (MemoryError, OSError)):
            return ErrorSeverity.CRITICAL, ErrorCategory.SYSTEM
        
        else:
            return ErrorSeverity.MEDIUM, ErrorCategory.UNKNOWN
    
    def _find_error_policy(self, error: Exception, category: ErrorCategory) -> ErrorPolicy:
        """Find appropriate error policy for error type."""
        
        error_type = type(error)
        
        # Look for specific error type policy
        for policy in self.error_policies:
            if error_type in policy.error_types:
                return policy
        
        # Look for category-based policy
        for policy in self.error_policies:
            if policy.category == category:
                return policy
        
        # Return default policy
        return self._get_default_policy()
    
    def _make_handling_decision(
        self,
        error: Exception,
        policy: ErrorPolicy,
        retry_count: int,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make decision on how to handle the error."""
        
        decision = {
            "action": "raise",  # raise, retry, fallback, ignore
            "delay": 0.0,
            "should_circuit_break": policy.should_circuit_break,
            "should_fallback": policy.should_fallback,
            "message": str(error)
        }
        
        # Check if we should retry
        should_retry, delay = self.should_retry(error, retry_count, context)
        
        if should_retry and policy.retry_strategy != RetryStrategy.NONE:
            decision["action"] = "retry"
            decision["delay"] = delay
        
        elif policy.should_fallback and self.config.enable_graceful_degradation:
            decision["action"] = "fallback"
        
        elif policy.severity == ErrorSeverity.LOW:
            decision["action"] = "ignore"
        
        # Apply circuit breaker if needed
        if policy.should_circuit_break:
            provider = context.get("provider")
            if provider:
                circuit_breaker = self._get_circuit_breaker(provider)
                circuit_breaker._on_failure()
        
        return decision
    
    def _calculate_retry_delay(
        self,
        strategy: RetryStrategy,
        retry_count: int,
        base_delay: float,
        max_delay: float
    ) -> float:
        """Calculate delay for retry based on strategy."""
        
        if strategy == RetryStrategy.IMMEDIATE:
            return 0.0
        
        elif strategy == RetryStrategy.LINEAR:
            delay = base_delay * (retry_count + 1)
        
        elif strategy == RetryStrategy.EXPONENTIAL:
            delay = base_delay * (2 ** retry_count)
        
        else:  # Default to linear
            delay = base_delay * (retry_count + 1)
        
        return min(delay, max_delay)
    
    def _apply_adaptive_timeout(
        self,
        base_timeout: float,
        provider: Optional[str],
        complexity: Optional[int]
    ) -> float:
        """Apply adaptive timeout based on provider performance and complexity."""
        
        timeout = base_timeout
        
        # Adjust based on provider performance
        if provider and provider in self.error_stats:
            error_rate = self._calculate_provider_error_rate(provider)
            if error_rate > 0.2:  # High error rate
                timeout *= self.timeout_config.timeout_multiplier
        
        # Adjust based on complexity
        if complexity:
            if complexity > 7:  # High complexity
                timeout *= 1.5
            elif complexity < 3:  # Low complexity
                timeout *= 0.8
        
        return timeout
    
    def _get_circuit_breaker(self, provider: str) -> CircuitBreaker:
        """Get or create circuit breaker for provider."""
        if provider not in self.circuit_breakers:
            self.circuit_breakers[provider] = CircuitBreaker(
                self.circuit_breaker_config
            )
        return self.circuit_breakers[provider]
    
    def _is_circuit_breaker_open(self, provider: str) -> bool:
        """Check if circuit breaker is open for provider."""
        if provider not in self.circuit_breakers:
            return False
        return self.circuit_breakers[provider].state == "open"
    
    def _update_error_stats(self, error_event: ErrorEvent) -> None:
        """Update error statistics."""
        key = f"{error_event.category.value}_{error_event.error_type}"
        self.error_stats[key] = self.error_stats.get(key, 0) + 1
        
        # Keep only recent errors (last 1000)
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]
    
    def _calculate_provider_error_rate(self, provider: str) -> float:
        """Calculate error rate for provider in recent time window."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)
        
        recent_errors = [
            event for event in self.error_history
            if (event.timestamp > cutoff_time and 
                event.provider == provider)
        ]
        
        if not recent_errors:
            return 0.0
        
        # This is a simplified calculation
        # In practice, you'd want to track total requests as well
        return len(recent_errors) / 100.0  # Assuming 100 requests per hour
    
    def _initialize_error_policies(self) -> List[ErrorPolicy]:
        """Initialize default error handling policies."""
        
        policies = [
            # Timeout errors
            ErrorPolicy(
                error_types=[TimeoutError, asyncio.TimeoutError],
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.TIMEOUT,
                retry_strategy=RetryStrategy.LINEAR,
                max_retries=2,
                base_delay=1.0,
                should_fallback=True
            ),
            
            # Rate limit errors
            ErrorPolicy(
                error_types=[],  # Detected by message content
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.RATE_LIMIT,
                retry_strategy=RetryStrategy.EXPONENTIAL,
                max_retries=3,
                base_delay=5.0,
                max_delay=60.0,
                should_fallback=True
            ),
            
            # Provider errors
            ErrorPolicy(
                error_types=[ProviderUnavailableError],
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.PROVIDER,
                retry_strategy=RetryStrategy.LINEAR,
                max_retries=1,
                base_delay=2.0,
                should_fallback=True,
                should_circuit_break=True
            ),
            
            # Content errors
            ErrorPolicy(
                error_types=[ContextTooLongError],
                severity=ErrorSeverity.LOW,
                category=ErrorCategory.CONTENT,
                retry_strategy=RetryStrategy.NONE,
                max_retries=0,
                should_fallback=True
            ),
            
            # Authentication errors
            ErrorPolicy(
                error_types=[],
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.AUTHENTICATION,
                retry_strategy=RetryStrategy.NONE,
                max_retries=0,
                should_fallback=False
            ),
            
            # System errors
            ErrorPolicy(
                error_types=[MemoryError, OSError],
                severity=ErrorSeverity.CRITICAL,
                category=ErrorCategory.SYSTEM,
                retry_strategy=RetryStrategy.NONE,
                max_retries=0,
                should_fallback=True
            )
        ]
        
        return policies
    
    def _get_default_policy(self) -> ErrorPolicy:
        """Get default error policy for unclassified errors."""
        return ErrorPolicy(
            error_types=[Exception],
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.UNKNOWN,
            retry_strategy=RetryStrategy.LINEAR,
            max_retries=2,
            base_delay=1.0,
            should_fallback=True
        )
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        
        # Calculate statistics
        total_errors = len(self.error_history)
        
        if total_errors == 0:
            return {"total_errors": 0}
        
        # Error breakdown by category
        category_breakdown = {}
        severity_breakdown = {}
        provider_breakdown = {}
        
        for event in self.error_history:
            # Category breakdown
            cat = event.category.value
            category_breakdown[cat] = category_breakdown.get(cat, 0) + 1
            
            # Severity breakdown
            sev = event.severity.value
            severity_breakdown[sev] = severity_breakdown.get(sev, 0) + 1
            
            # Provider breakdown
            if event.provider:
                provider_breakdown[event.provider] = provider_breakdown.get(event.provider, 0) + 1
        
        # Circuit breaker states
        circuit_breaker_states = {
            provider: cb.state
            for provider, cb in self.circuit_breakers.items()
        }
        
        return {
            "total_errors": total_errors,
            "category_breakdown": category_breakdown,
            "severity_breakdown": severity_breakdown,
            "provider_breakdown": provider_breakdown,
            "circuit_breaker_states": circuit_breaker_states,
            "recent_errors": [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "type": event.error_type,
                    "category": event.category.value,
                    "severity": event.severity.value,
                    "provider": event.provider
                }
                for event in self.error_history[-10:]  # Last 10 errors
            ]
        }
    
    def get_handler_info(self) -> Dict[str, Any]:
        """Get error handler configuration and status."""
        return {
            "timeout_config": {
                "default_timeout": self.timeout_config.default_timeout,
                "ensemble_timeout": self.timeout_config.ensemble_timeout,
                "enable_adaptive_timeout": self.timeout_config.enable_adaptive_timeout
            },
            "circuit_breaker_config": {
                "failure_threshold": self.circuit_breaker_config.failure_threshold,
                "recovery_timeout": self.circuit_breaker_config.recovery_timeout,
                "success_threshold": self.circuit_breaker_config.success_threshold
            },
            "error_policies_count": len(self.error_policies),
            "circuit_breakers_active": len(self.circuit_breakers),
            "total_errors_recorded": len(self.error_history)
        }