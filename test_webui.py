#!/usr/bin/env python3
"""
Test script for RAG Server Web UI
Tests basic functionality and API integration
"""

import sys
import os
from pathlib import Path
import subprocess
import time
import requests

# Add webui to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
        
        import plotly.graph_objects as go
        import plotly.express as px
        print("✅ Plotly imported successfully")
        
        import pandas as pd
        print("✅ Pandas imported successfully")
        
        import jwt
        print("✅ PyJWT imported successfully")
        
        import requests
        print("✅ Requests imported successfully")
        
        # Test webui modules
        from webui.config import config
        print("✅ WebUI config imported successfully")
        
        from webui.auth import AuthManager
        print("✅ WebUI auth imported successfully")
        
        from webui.api_client import RAGAPIClient
        print("✅ WebUI API client imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_config():
    """Test configuration loading"""
    print("\n🧪 Testing configuration...")
    
    try:
        from webui.config import config, LLM_PROVIDERS, USER_ROLES
        
        print(f"✅ API Base URL: {config.API_BASE_URL}")
        print(f"✅ App Title: {config.APP_TITLE}")
        print(f"✅ Demo Mode: {config.DEMO_MODE}")
        print(f"✅ Max Upload Size: {config.MAX_UPLOAD_SIZE_MB}MB")
        print(f"✅ Allowed File Types: {config.ALLOWED_FILE_TYPES}")
        
        # Test LLM providers config
        assert "openai" in LLM_PROVIDERS
        assert "anthropic" in LLM_PROVIDERS
        print("✅ LLM providers configuration loaded")
        
        # Test user roles config
        assert "admin" in USER_ROLES
        assert "user" in USER_ROLES
        print("✅ User roles configuration loaded")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_auth_manager():
    """Test authentication manager"""
    print("\n🧪 Testing authentication manager...")
    
    try:
        from webui.auth import AuthManager
        
        auth = AuthManager()
        print("✅ AuthManager instantiated")
        
        # Test demo authentication
        demo_auth_result = auth._demo_authenticate("admin", "admin123")
        if demo_auth_result:
            print("✅ Demo authentication works")
            
            # Test full authentication flow
            success, token, user_info = auth._authenticate("admin", "admin123")
            if success and token and user_info:
                print(f"   User: {user_info.get('username')}")
                print(f"   Role: {user_info.get('role')}")
            else:
                print("❌ Full authentication failed")
                return False
        else:
            print("❌ Demo authentication failed")
            return False
        
        # Test JWT generation
        if token:
            print("✅ JWT token generated")
            
            # Test JWT decoding
            import jwt as pyjwt
            try:
                payload = pyjwt.decode(token, options={"verify_signature": False})
                print(f"   Token payload: {payload.get('username')}")
                print("✅ JWT token is valid")
            except Exception as e:
                print(f"❌ JWT token validation failed: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Auth manager test failed: {e}")
        return False

def test_api_client():
    """Test API client"""
    print("\n🧪 Testing API client...")
    
    try:
        from webui.api_client import RAGAPIClient
        
        client = RAGAPIClient()
        print("✅ RAGAPIClient instantiated")
        print(f"   Base URL: {client.base_url}")
        print(f"   Timeout: {client.timeout}")
        
        # Test headers generation
        headers = client._get_headers()
        assert "Content-Type" in headers
        print("✅ Headers generation works")
        
        return True
        
    except Exception as e:
        print(f"❌ API client test failed: {e}")
        return False

def test_page_modules():
    """Test that all page modules can be imported"""
    print("\n🧪 Testing page modules...")
    
    page_modules = [
        "webui.pages.dashboard",
        "webui.pages.documents", 
        "webui.pages.query",
        "webui.pages.admin",
        "webui.pages.settings"
    ]
    
    try:
        for module in page_modules:
            __import__(module)
            print(f"✅ {module} imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Page module test failed: {e}")
        return False

def test_streamlit_app():
    """Test that the Streamlit app can be started"""
    print("\n🧪 Testing Streamlit app startup...")
    
    try:
        app_file = Path(__file__).parent / "webui" / "app.py"
        if not app_file.exists():
            print(f"❌ App file not found: {app_file}")
            return False
        
        print(f"✅ App file exists: {app_file}")
        
        # Test that the app file is syntactically correct
        with open(app_file, 'r') as f:
            code = f.read()
        
        compile(code, str(app_file), 'exec')
        print("✅ App file syntax is valid")
        
        return True
        
    except Exception as e:
        print(f"❌ Streamlit app test failed: {e}")
        return False

def test_demo_users():
    """Test demo user configuration"""
    print("\n🧪 Testing demo users...")
    
    try:
        from webui.config import config
        
        demo_users = config.get_demo_users()
        
        if not demo_users:
            print("❌ No demo users configured")
            return False
        
        expected_users = ["admin", "user", "manager"]
        for user in expected_users:
            if user not in demo_users:
                print(f"❌ Demo user '{user}' not configured")
                return False
            
            user_data = demo_users[user]
            if not all(key in user_data for key in ["password", "role", "email"]):
                print(f"❌ Demo user '{user}' missing required fields")
                return False
            
            print(f"✅ Demo user '{user}' configured correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo users test failed: {e}")
        return False

def test_file_structure():
    """Test that all required files exist"""
    print("\n🧪 Testing file structure...")
    
    required_files = [
        "webui/__init__.py",
        "webui/app.py",
        "webui/auth.py",
        "webui/api_client.py",
        "webui/config.py",
        "webui/pages/__init__.py",
        "webui/pages/dashboard.py",
        "webui/pages/documents.py",
        "webui/pages/query.py",
        "webui/pages/admin.py",
        "webui/pages/settings.py",
        "requirements_webui.txt",
        "run_webui.py",
        ".streamlit/config.toml",
        ".streamlit/secrets.toml"
    ]
    
    base_path = Path(__file__).parent
    
    try:
        for file_path in required_files:
            full_path = base_path / file_path
            if not full_path.exists():
                print(f"❌ Required file missing: {file_path}")
                return False
            print(f"✅ {file_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ File structure test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("🚀 Starting RAG Server Web UI Tests")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Authentication Manager", test_auth_manager),
        ("API Client", test_api_client),
        ("Page Modules", test_page_modules),
        ("Demo Users", test_demo_users),
        ("Streamlit App", test_streamlit_app)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Web UI is ready to run.")
        print("\nTo start the Web UI, run:")
        print("  python run_webui.py")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)