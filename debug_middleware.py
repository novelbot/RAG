#!/usr/bin/env python3
"""
Debug middleware configuration
"""
import os
import sys
sys.path.append('.')

# Set environment variables
os.environ.update({
    'APP_ENV': 'development',
    'DEBUG': 'true'
})

print("🔧 Debugging Middleware Configuration...")

try:
    from src.api.middleware import AuthenticationMiddleware
    
    # Create middleware instance
    middleware = AuthenticationMiddleware(None)  # app=None for testing
    
    print("🔒 AuthenticationMiddleware exempt paths:")
    for path in sorted(middleware.exempt_paths):
        print(f"  {path}")
    
    # Test path checking
    test_paths = ['/docs', '/redoc', '/openapi.json', '/api/v1/auth/login']
    print(f"\n🧪 Path exemption test:")
    for path in test_paths:
        is_exempt = middleware._is_exempt_path(path)
        status = "✅ EXEMPT" if is_exempt else "❌ REQUIRES AUTH"
        print(f"  {path:<20} {status}")
    
    print("\n✅ Middleware debugging complete!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()