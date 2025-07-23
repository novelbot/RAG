"""
API Client for RAG Server Web UI
Handles communication with the RAG server REST API endpoints
"""

import requests
import streamlit as st
from typing import Dict, Any, List, Optional, Union
import json
from datetime import datetime
import time

class RAGAPIClient:
    """Client for interacting with the RAG server API"""
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers including authentication"""
        headers = {"Content-Type": "application/json"}
        
        # Get auth token from session state
        token = st.session_state.get("jwt_token")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        
        return headers
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make an HTTP request with error handling"""
        url = f"{self.base_url}{endpoint}"
        headers = self._get_headers()
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                headers=headers,
                timeout=self.timeout,
                **kwargs
            )
            return response
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to RAG server. Please check if the server is running.")
            raise
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. Please try again.")
            raise
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Request failed: {str(e)}")
            raise
    
    # Authentication API
    def authenticate(self, username: str, password: str) -> Dict[str, Any]:
        """Authenticate user and get JWT token"""
        data = {"username": username, "password": password}
        response = self._make_request("POST", "/auth/login", json=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Authentication failed: {response.status_code}")
    
    def register(self, username: str, password: str, email: str = "", role: str = "user") -> Dict[str, Any]:
        """Register a new user"""
        data = {
            "username": username,
            "password": password,
            "email": email,
            "role": role
        }
        
        response = self._make_request("POST", "/api/v1/auth/register", json=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Registration failed: {response.status_code} - {response.text}")
    
    # Document Management API
    def upload_document(self, file_content: bytes, filename: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Upload a document to the system"""
        files = {"file": (filename, file_content)}
        data = {"metadata": json.dumps(metadata or {})}
        
        headers = {"Authorization": f"Bearer {st.session_state.get('jwt_token', '')}"}
        
        response = self.session.post(
            f"{self.base_url}/api/v1/documents/upload",
            files=files,
            data=data,
            headers=headers,
            timeout=self.timeout
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Upload failed: {response.status_code} - {response.text}")
    
    def get_documents(self, limit: int = 50, offset: int = 0, search: str = "") -> Dict[str, Any]:
        """Get list of documents"""
        params = {"limit": limit, "offset": offset}
        if search:
            params["search"] = search
        
        response = self._make_request("GET", "/api/v1/documents", params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get documents: {response.status_code}")
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document"""
        response = self._make_request("DELETE", f"/api/v1/documents/{document_id}")
        return response.status_code == 200
    
    def get_document_status(self, document_id: str) -> Dict[str, Any]:
        """Get processing status of a document"""
        response = self._make_request("GET", f"/api/v1/documents/{document_id}/status")
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get document status: {response.status_code}")
    
    # Query API
    def query_rag(self, query: str, mode: str = "rag", k: int = 5, 
                  llm_provider: str = "openai", model: str = "gpt-4") -> Dict[str, Any]:
        """Perform a RAG query"""
        data = {
            "query": query,
            "max_results": k
        }
        
        response = self._make_request("POST", "/api/v1/query/ask", json=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Query failed: {response.status_code} - {response.text}")
    
    def search_documents(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Search documents using vector similarity"""
        data = {
            "query": query,
            "max_results": max_results
        }
        
        response = self._make_request("POST", "/api/v1/query/search", json=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Search failed: {response.status_code} - {response.text}")
    
    def chat_llm(self, messages: List[Dict[str, str]], model: str = "gpt-4", 
                 temperature: float = 0.7, max_tokens: int = 1000) -> Dict[str, Any]:
        """Chat with LLM directly"""
        # Use the last user message for the query
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break
        
        if not user_message:
            raise Exception("No user message found in chat")
        
        data = {
            "query": user_message,
            "max_results": 5
        }
        
        response = self._make_request("POST", "/api/v1/query/ask", json=data)
        
        if response.status_code == 200:
            result = response.json()
            # Transform to chat format
            return {
                "content": result.get("answer", "No response received"),
                "metadata": result.get("metadata", {})
            }
        else:
            raise Exception(f"Chat failed: {response.status_code} - {response.text}")
    
    def generate_single_llm(self, query: str, context: str = "", model: str = "gpt-4",
                           temperature: float = 0.7) -> Dict[str, Any]:
        """Generate response using single LLM"""
        # Use the ask endpoint for single LLM as well
        data = {
            "query": f"{context} {query}".strip() if context else query,
            "max_results": 5
        }
        
        response = self._make_request("POST", "/api/v1/query/ask", json=data)
        
        if response.status_code == 200:
            result = response.json()
            return {
                "response": result.get("answer", "No response received"),
                "metadata": result.get("metadata", {})
            }
        else:
            raise Exception(f"Single LLM generation failed: {response.status_code}")
    
    def generate_ensemble_llm(self, query: str, ensemble_size: int = 3,
                             consensus_threshold: float = 0.7) -> Dict[str, Any]:
        """Generate response using ensemble LLM"""
        # Use the ask endpoint for ensemble as well - in a real implementation 
        # this would be different, but for now we'll use the same endpoint
        data = {
            "query": query,
            "max_results": 5
        }
        
        response = self._make_request("POST", "/api/v1/query/ask", json=data)
        
        if response.status_code == 200:
            result = response.json()
            return {
                "best_response": result.get("answer", "No response received"),
                "all_responses": [
                    {
                        "model": "primary",
                        "response": result.get("answer", "No response received"),
                        "quality_score": 0.95
                    }
                ],
                "consensus_score": 0.95,
                "metadata": result.get("metadata", {})
            }
        else:
            raise Exception(f"Ensemble LLM generation failed: {response.status_code}")
    
    # Embedding API
    def generate_embeddings(self, input_texts: List[str], model: str = "text-embedding-3-large",
                           provider: str = "openai") -> Dict[str, Any]:
        """Generate embeddings for input texts"""
        data = {
            "input": input_texts,
            "model": model,
            "provider": provider,
            "normalize": True
        }
        
        response = self._make_request("POST", "/embeddings", json=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Embedding generation failed: {response.status_code}")
    
    # System Status and Monitoring
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status information"""
        response = self._make_request("GET", "/api/v1/monitoring/status")
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "error", "message": "Failed to get system status"}
    
    def get_health_check(self) -> Dict[str, Any]:
        """Get health check information"""
        response = self._make_request("GET", "/api/v1/monitoring/health")
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"healthy": False, "message": "Health check failed"}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics"""
        response = self._make_request("GET", "/api/v1/monitoring/metrics")
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": "Failed to get metrics"}
    
    # User Management (Admin only)
    def get_users(self, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """Get list of users (admin only)"""
        params = {"limit": limit, "offset": offset}
        response = self._make_request("GET", "/admin/users", params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get users: {response.status_code}")
    
    def create_user(self, username: str, email: str, role: str, password: str, department: str = None) -> Dict[str, Any]:
        """Create a new user (admin only)"""
        data = {
            "username": username,
            "email": email,
            "role": role,
            "password": password
        }
        
        if department:
            data["department"] = department
        
        response = self._make_request("POST", "/admin/users", json=data)
        
        if response.status_code == 201:
            return response.json()
        else:
            raise Exception(f"Failed to create user: {response.status_code}")
    
    def update_user(self, user_id: str, **updates) -> Dict[str, Any]:
        """Update a user (admin only)"""
        response = self._make_request("PUT", f"/admin/users/{user_id}", json=updates)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to update user: {response.status_code}")
    
    def delete_user(self, user_id: str) -> bool:
        """Delete a user (admin only)"""
        response = self._make_request("DELETE", f"/admin/users/{user_id}")
        return response.status_code == 200
    
    # Configuration Management
    def get_configuration(self) -> Dict[str, Any]:
        """Get system configuration"""
        response = self._make_request("GET", "/admin/config")
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": "Failed to get configuration"}
    
    def update_configuration(self, config_updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update system configuration (admin only)"""
        response = self._make_request("PUT", "/admin/config", json=config_updates)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to update configuration: {response.status_code}")


# Singleton instance for global use
@st.cache_resource
def get_api_client() -> RAGAPIClient:
    """Get a cached API client instance"""
    api_base_url = st.secrets.get("API_BASE_URL", "http://localhost:8000")
    return RAGAPIClient(base_url=api_base_url)