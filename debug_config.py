#!/usr/bin/env python3
"""
Debug configuration settings to check why /docs is not working
"""
import os
import sys
sys.path.append('.')

# Set environment variables
os.environ.update({
    'APP_ENV': 'development',
    'DEBUG': 'true',
    'DB_HOST': 'localhost',
    'DB_PORT': '3306',
    'DB_USER': 'mysql',
    'DB_PASSWORD': 'novelbotisbestie',
    'DB_NAME': 'ragdb',
    'MILVUS_HOST': 'localhost',
    'MILVUS_PORT': '19530',
    'LLM_PROVIDER': 'ollama',
    'LLM_MODEL': 'gemma3:27b-it-q8_0',
    'EMBEDDING_PROVIDER': 'ollama',
    'EMBEDDING_MODEL': 'jeffh/intfloat-multilingual-e5-large-instruct:f32',
    'SECRET_KEY': 'test-secret-key',
})

print("🔧 Debugging Configuration...")

try:
    from src.core.config import get_config
    config = get_config()
    
    print(f"App Name: {config.app_name}")
    print(f"Version: {config.version}")
    print(f"Environment: {config.environment}")
    print(f"Debug: {config.debug}")
    print(f"API Debug: {config.api.debug}")
    print(f"API Reload: {config.api.reload}")
    
    # Check what /docs setting would be
    docs_url = "/docs" if config.debug else None
    redoc_url = "/redoc" if config.debug else None
    
    print(f"\nFastAPI Settings:")
    print(f"docs_url: {docs_url}")
    print(f"redoc_url: {redoc_url}")
    
    # Test creating FastAPI app
    print(f"\n🚀 Testing FastAPI App Creation...")
    from fastapi import FastAPI
    
    app = FastAPI(
        title=config.app_name,
        version=config.version,
        description="RAG Server Test",
        docs_url="/docs",  # Force enable for testing
        redoc_url="/redoc",  # Force enable for testing
    )
    
    print(f"✅ FastAPI app created successfully!")
    print(f"App title: {app.title}")
    print(f"App version: {app.version}")
    print(f"Docs URL: {app.docs_url}")
    print(f"ReDoc URL: {app.redoc_url}")
    
    # Check available routes
    routes = [route.path for route in app.routes]
    print(f"\nDefault routes: {routes}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()