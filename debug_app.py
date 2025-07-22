#!/usr/bin/env python3
"""
Debug the actual RAG server app creation to find /docs issue
"""
import os
import sys
sys.path.append('.')

# Set environment variables first
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

print("🔧 Debugging RAG Server App Creation...")

try:
    # Import and create app
    from src.core.app import create_app
    from src.core.config import get_config
    
    print("✅ Imports successful")
    
    # Get config first
    config = get_config()
    print(f"Config debug mode: {config.debug}")
    
    # Create the app
    print("Creating FastAPI app...")
    app = create_app()
    
    print(f"✅ App created successfully!")
    print(f"App title: {app.title}")
    print(f"App version: {app.version}")
    print(f"Docs URL: {app.docs_url}")
    print(f"ReDoc URL: {app.redoc_url}")
    
    # List all routes
    print("\n📋 Available routes:")
    for route in app.routes:
        if hasattr(route, 'path'):
            print(f"  {route.path} - {getattr(route, 'methods', 'N/A')}")
    
    # Check middleware stack
    print("\n🔧 Middleware stack:")
    for i, middleware in enumerate(app.user_middleware):
        print(f"  {i+1}. {middleware.cls.__name__}")
    
    print("\n✅ App analysis complete!")
    
except Exception as e:
    print(f"❌ Error creating app: {e}")
    import traceback
    traceback.print_exc()