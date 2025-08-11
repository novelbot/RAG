"""
Test configuration helper for HTTP/HTTPS support.
"""

import os
import ssl
import httpx
from typing import Optional, Dict, Any


def get_server_url(path: str = "") -> str:
    """
    Get the server URL based on SSL configuration.
    
    Args:
        path: API path to append to the base URL
        
    Returns:
        Complete URL with protocol, host, port, and path
    """
    ssl_enabled = os.getenv("SSL_ENABLED", "false").lower() in ("true", "1", "yes")
    
    if ssl_enabled:
        protocol = "https"
        port = int(os.getenv("HTTPS_PORT", "8443"))
    else:
        protocol = "http"
        port = int(os.getenv("API_PORT", "8000"))
    
    host = os.getenv("API_HOST", "localhost")
    
    # Remove leading slash from path if present
    if path and path.startswith("/"):
        path = path[1:]
    
    return f"{protocol}://{host}:{port}/{path}" if path else f"{protocol}://{host}:{port}"


def get_client_kwargs() -> Dict[str, Any]:
    """
    Get httpx client kwargs based on SSL configuration.
    
    Returns:
        Dictionary of kwargs for httpx.AsyncClient or httpx.Client
    """
    ssl_enabled = os.getenv("SSL_ENABLED", "false").lower() in ("true", "1", "yes")
    
    kwargs = {}
    
    if ssl_enabled:
        # For self-signed certificates in development, disable SSL verification
        # In production, you would want to verify certificates properly
        if os.getenv("APP_ENV", "development") == "development":
            # Create an SSL context that doesn't verify certificates
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            kwargs["verify"] = ssl_context
        else:
            # In production, use proper certificate verification
            ca_cert = os.getenv("SSL_CA_CERT_FILE")
            if ca_cert and os.path.exists(ca_cert):
                kwargs["verify"] = ca_cert
            else:
                kwargs["verify"] = True
    
    return kwargs


def create_test_client() -> httpx.AsyncClient:
    """
    Create an httpx AsyncClient configured for testing.
    
    Returns:
        Configured httpx.AsyncClient
    """
    base_url = get_server_url()
    client_kwargs = get_client_kwargs()
    
    return httpx.AsyncClient(
        base_url=base_url,
        timeout=httpx.Timeout(30.0),
        **client_kwargs
    )


def create_sync_test_client() -> httpx.Client:
    """
    Create an httpx Client configured for testing (synchronous).
    
    Returns:
        Configured httpx.Client
    """
    base_url = get_server_url()
    client_kwargs = get_client_kwargs()
    
    return httpx.Client(
        base_url=base_url,
        timeout=httpx.Timeout(30.0),
        **client_kwargs
    )


# Convenience functions for common API endpoints
def get_auth_url() -> str:
    """Get the authentication endpoint URL."""
    return get_server_url("api/v1/auth/login")


def get_chat_url() -> str:
    """Get the chat endpoint URL."""
    return get_server_url("api/v1/episode/chat")


def get_stream_url() -> str:
    """Get the streaming chat endpoint URL."""
    return get_server_url("api/v1/episode/chat/stream")


def get_health_url() -> str:
    """Get the health check endpoint URL."""
    return get_server_url("health")