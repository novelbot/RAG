#!/usr/bin/env python3
"""
Test script for auth token refresh endpoint functionality.
Tests the implemented token refresh logic.
"""

import asyncio
import httpx
import json
import sys
import time
import threading
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.app import create_app
import uvicorn

class AuthRefreshTester:
    def __init__(self, base_url: str = "http://localhost:8002"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        self.access_token = None
        self.refresh_token = None
    
    async def test_login_and_get_tokens(self, username: str = "admin", password: str = "admin123"):
        """Login and get initial tokens."""
        print(f"\n=== Testing Login to Get Initial Tokens ===")
        
        try:
            response = await self.client.post(
                f"{self.base_url}/api/v1/auth/login",
                json={"username": username, "password": password}
            )
            
            print(f"Login Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                self.access_token = data.get("access_token")
                self.refresh_token = data.get("refresh_token")
                
                print(f"✓ Login successful")
                print(f"✓ Access token received: {self.access_token[:30]}..." if self.access_token else "No access token")
                print(f"✓ Refresh token received: {self.refresh_token[:30]}..." if self.refresh_token else "No refresh token")
                print(f"✓ Token type: {data.get('token_type')}")
                print(f"✓ Expires in: {data.get('expires_in')} seconds")
                
                return True
            else:
                print(f"✗ Login failed: {response.text}")
                return False
                
        except Exception as e:
            print(f"✗ Login error: {e}")
            return False
    
    async def test_token_refresh_valid(self):
        """Test token refresh with valid refresh token."""
        print(f"\n=== Testing Token Refresh with Valid Token ===")
        
        if not self.refresh_token:
            print("✗ No refresh token available for testing")
            return False
        
        try:
            # Use the refresh token to get new tokens
            response = await self.client.post(
                f"{self.base_url}/api/v1/auth/refresh",
                json={"refresh_token": self.refresh_token}
            )
            
            print(f"Refresh Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                new_access_token = data.get("access_token")
                new_refresh_token = data.get("refresh_token")
                
                print(f"✓ Token refresh successful")
                print(f"✓ New access token: {new_access_token[:30]}..." if new_access_token else "No new access token")
                print(f"✓ New refresh token: {new_refresh_token[:30]}..." if new_refresh_token else "No new refresh token")
                print(f"✓ Token type: {data.get('token_type')}")
                print(f"✓ Expires in: {data.get('expires_in')} seconds")
                
                # Check if tokens are different from original (indicating rotation)
                if new_access_token != self.access_token:
                    print("✓ Access token was rotated (new token generated)")
                else:
                    print("ℹ Access token is the same (no rotation)")
                
                # Update our tokens for further testing
                self.access_token = new_access_token
                self.refresh_token = new_refresh_token
                
                return True
            else:
                print(f"✗ Token refresh failed: {response.text}")
                return False
                
        except Exception as e:
            print(f"✗ Token refresh error: {e}")
            return False
    
    async def test_token_refresh_invalid(self):
        """Test token refresh with invalid refresh token."""
        print(f"\n=== Testing Token Refresh with Invalid Token ===")
        
        try:
            # Use an invalid refresh token
            invalid_token = "invalid_refresh_token_12345"
            response = await self.client.post(
                f"{self.base_url}/api/v1/auth/refresh",
                json={"refresh_token": invalid_token}
            )
            
            print(f"Invalid Token Refresh Status: {response.status_code}")
            
            if response.status_code == 401:
                data = response.json()
                print(f"✓ Invalid token correctly rejected")
                print(f"✓ Error message: {data.get('detail')}")
                return True
            else:
                print(f"✗ Expected 401, got {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            print(f"✗ Invalid token test error: {e}")
            return False
    
    async def test_token_refresh_demo_token(self):
        """Test that old demo token logic is removed."""
        print(f"\n=== Testing Demo Token Should Not Work ===")
        
        try:
            # Try the old demo token that should no longer work
            demo_token = "demo_refresh_token"
            response = await self.client.post(
                f"{self.base_url}/api/v1/auth/refresh",
                json={"refresh_token": demo_token}
            )
            
            print(f"Demo Token Refresh Status: {response.status_code}")
            
            if response.status_code == 401:
                data = response.json()
                print(f"✓ Demo token correctly rejected (old logic removed)")
                print(f"✓ Error message: {data.get('detail')}")
                return True
            else:
                print(f"✗ Demo token should be rejected but got {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            print(f"✗ Demo token test error: {e}")
            return False
    
    async def test_refreshed_token_usage(self):
        """Test that refreshed tokens can be used for API calls."""
        print(f"\n=== Testing Refreshed Token Usage ===")
        
        if not self.access_token:
            print("✗ No access token available for testing")
            return False
        
        try:
            # Use the refreshed token to call /me endpoint
            response = await self.client.get(
                f"{self.base_url}/api/v1/auth/me",
                headers={"Authorization": f"Bearer {self.access_token}"}
            )
            
            print(f"Token Usage Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Refreshed token works for API calls")
                print(f"✓ User ID: {data.get('id')}")
                print(f"✓ Username: {data.get('username')}")
                print(f"✓ Roles: {data.get('roles')}")
                return True
            else:
                print(f"✗ Refreshed token failed for API call: {response.text}")
                return False
                
        except Exception as e:
            print(f"✗ Token usage test error: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all token refresh tests."""
        print("🚀 Starting Token Refresh API Tests")
        print("=" * 60)
        
        tests = [
            ("Login and Get Tokens", self.test_login_and_get_tokens()),
            ("Valid Token Refresh", self.test_token_refresh_valid()),
            ("Invalid Token Refresh", self.test_token_refresh_invalid()),
            ("Demo Token Rejection", self.test_token_refresh_demo_token()),
            ("Refreshed Token Usage", self.test_refreshed_token_usage())
        ]
        
        results = []
        for test_name, test_coro in tests:
            print(f"\n🧪 Running {test_name}...")
            try:
                result = await test_coro
                results.append((test_name, result))
            except Exception as e:
                print(f"✗ {test_name} failed with exception: {e}")
                results.append((test_name, False))
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 TOKEN REFRESH TEST SUMMARY")
        print("=" * 60)
        
        passed = 0
        failed = 0
        
        for test_name, result in results:
            status = "✓ PASSED" if result else "✗ FAILED"
            print(f"{status:<10} {test_name}")
            if result:
                passed += 1
            else:
                failed += 1
        
        print("-" * 60)
        print(f"Total: {len(results)} tests, {passed} passed, {failed} failed")
        
        return failed == 0
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

def start_test_server():
    """Start the test server on port 8002."""
    print("🚀 Starting test server on port 8002...")
    
    app = create_app()
    
    def run_server():
        uvicorn.run(app, host="127.0.0.1", port=8002, log_level="warning")
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait for server to start
    time.sleep(3)
    print("✅ Test server started at http://127.0.0.1:8002")
    
    return server_thread

async def run_auth_refresh_tests():
    """Run the auth refresh API tests."""
    tester = AuthRefreshTester()
    
    try:
        success = await tester.run_all_tests()
        
        if success:
            print("\n✅ All token refresh tests passed!")
            print("Token refresh endpoint is working correctly.")
        else:
            print("\n❌ Some token refresh tests failed.")
        
        return success
        
    finally:
        await tester.close()

if __name__ == "__main__":
    print("Auth Token Refresh Test Suite")
    print("Testing the implemented token refresh logic")
    print()
    
    # Start server
    server_thread = start_test_server()
    
    # Run tests
    try:
        success = asyncio.run(run_auth_refresh_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
        sys.exit(1)