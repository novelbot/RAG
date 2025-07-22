"""
Authentication and Authorization exceptions.
"""

from src.core.exceptions import RAGException


class AuthenticationError(RAGException):
    """Base exception for authentication errors."""
    def __init__(self, message: str, details=None):
        super().__init__(message, 401, details)


class AuthorizationError(RAGException):
    """Base exception for authorization errors."""
    def __init__(self, message: str, details=None):
        super().__init__(message, 403, details)


class TokenExpiredError(AuthenticationError):
    """Exception raised when a token has expired."""
    pass


class InvalidTokenError(AuthenticationError):
    """Exception raised when a token is invalid or malformed."""
    pass


class InsufficientPermissionsError(AuthorizationError):
    """Exception raised when a user lacks required permissions."""
    pass


class InvalidCredentialsError(AuthenticationError):
    """Exception raised when login credentials are invalid."""
    pass


class UserNotFoundError(AuthenticationError):
    """Exception raised when a user is not found."""
    pass


class UserAlreadyExistsError(AuthenticationError):
    """Exception raised when trying to create a user that already exists."""
    pass


class RoleNotFoundError(AuthorizationError):
    """Exception raised when a role is not found."""
    pass


class PermissionNotFoundError(AuthorizationError):
    """Exception raised when a permission is not found."""
    pass


class TokenBlacklistedError(AuthenticationError):
    """Exception raised when a token has been blacklisted."""
    pass