#!/usr/bin/env python3
"""
Debug route registration in the actual running server
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

print("🔧 Debugging route registration...")

try:
    from src.core.app import create_app
    
    # Create the actual app used by the server
    app = create_app()
    
    print("🗺️  All registered routes:")
    for route in app.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            print(f"  {route.path:<30} {route.methods}")
        elif hasattr(route, 'path'):
            print(f"  {route.path:<30} [Static/Other]")
    
    # Check if docs routes exist
    docs_routes = [r for r in app.routes if hasattr(r, 'path') and '/docs' in r.path]
    print(f"\n📚 Docs-related routes found: {len(docs_routes)}")
    for route in docs_routes:
        print(f"  {route.path} - {getattr(route, 'methods', 'N/A')}")
    
    # Check OpenAPI configuration
    print(f"\n⚙️  OpenAPI Configuration:")
    print(f"  Title: {app.title}")
    print(f"  Version: {app.version}")  
    print(f"  Docs URL: {app.docs_url}")
    print(f"  ReDoc URL: {app.redoc_url}")
    print(f"  OpenAPI URL: {app.openapi_url}")
    
    # Try to generate OpenAPI schema
    try:
        schema = app.openapi()
        print(f"  OpenAPI Schema: Generated ({len(str(schema))} chars)")
    except Exception as e:
        print(f"  OpenAPI Schema: Failed to generate - {e}")
    
    print("\n✅ Route debugging complete!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()