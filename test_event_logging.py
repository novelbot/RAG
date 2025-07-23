#!/usr/bin/env python3
"""
Test script to verify event logging functionality.
Tests login, logout, document upload/delete events.
"""

import asyncio
import aiohttp
import json
from pathlib import Path
import time

BASE_URL = "http://localhost:8000/api/v1"

async def test_login_logout():
    """Test login and logout event logging."""
    print("Testing login/logout events...")
    
    async with aiohttp.ClientSession() as session:
        # Test login
        login_data = {
            "username": "admin",
            "password": "admin123"
        }
        
        async with session.post(f"{BASE_URL}/auth/login", json=login_data) as resp:
            if resp.status == 200:
                result = await resp.json()
                token = result.get("access_token")
                print(f"✅ Login successful, token: {token[:20]}...")
                
                # Wait a bit
                await asyncio.sleep(2)
                
                # Test logout
                headers = {"Authorization": f"Bearer {token}"}
                async with session.post(f"{BASE_URL}/auth/logout", headers=headers) as logout_resp:
                    if logout_resp.status == 200:
                        print("✅ Logout successful")
                    else:
                        print(f"❌ Logout failed: {logout_resp.status}")
                        
            else:
                print(f"❌ Login failed: {resp.status}")
                print(await resp.text())

async def test_document_operations():
    """Test document upload and delete event logging."""
    print("\nTesting document operations...")
    
    async with aiohttp.ClientSession() as session:
        # First login to get token
        login_data = {
            "username": "user",
            "password": "user123"
        }
        
        async with session.post(f"{BASE_URL}/auth/login", json=login_data) as resp:
            if resp.status != 200:
                print("❌ Failed to login for document test")
                return
                
            result = await resp.json()
            token = result.get("access_token")
            headers = {"Authorization": f"Bearer {token}"}
            
            # Create a test file
            test_content = "This is a test document for event logging."
            
            # Test document upload
            data = aiohttp.FormData()
            data.add_field('file', 
                          test_content.encode(), 
                          filename='test_document.txt',
                          content_type='text/plain')
            
            async with session.post(f"{BASE_URL}/documents/upload", 
                                  data=data, headers=headers) as upload_resp:
                if upload_resp.status == 200:
                    upload_result = await upload_resp.json()
                    document_id = upload_result.get("document_id")
                    print(f"✅ Document upload successful, ID: {document_id}")
                    
                    # Wait a bit
                    await asyncio.sleep(2)
                    
                    # Test document deletion
                    async with session.delete(f"{BASE_URL}/documents/{document_id}", 
                                           headers=headers) as delete_resp:
                        if delete_resp.status == 200:
                            print("✅ Document deletion successful")
                        else:
                            print(f"❌ Document deletion failed: {delete_resp.status}")
                            
                else:
                    print(f"❌ Document upload failed: {upload_resp.status}")
                    print(await upload_resp.text())

async def check_recent_activity():
    """Check recent activity to see if events were logged."""
    print("\nChecking recent activity...")
    
    async with aiohttp.ClientSession() as session:
        # Login as admin to check activity
        login_data = {
            "username": "admin",
            "password": "admin123"
        }
        
        async with session.post(f"{BASE_URL}/auth/login", json=login_data) as resp:
            if resp.status != 200:
                print("❌ Failed to login to check activity")
                return
                
            result = await resp.json()
            token = result.get("access_token")
            headers = {"Authorization": f"Bearer {token}"}
            
            # Get recent activity
            async with session.get(f"{BASE_URL}/monitoring/metrics/recent-activity", 
                                 headers=headers) as activity_resp:
                if activity_resp.status == 200:
                    activities = await activity_resp.json()
                    print(f"✅ Retrieved {len(activities)} recent activities:")
                    
                    for i, activity in enumerate(activities[:10]):  # Show first 10
                        print(f"  {i+1}. {activity['time']} - {activity['user']} - {activity['action']}")
                        
                else:
                    print(f"❌ Failed to get recent activity: {activity_resp.status}")

async def main():
    """Run all tests."""
    print("🧪 Testing Event Logging System")
    print("=" * 50)
    
    try:
        # Test authentication events
        await test_login_logout()
        
        # Test document events
        await test_document_operations()
        
        # Wait for events to be processed
        print("\nWaiting for events to be processed...")
        await asyncio.sleep(3)
        
        # Check recent activity
        await check_recent_activity()
        
        print("\n✅ Event logging tests completed!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())