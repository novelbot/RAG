#!/usr/bin/env python3
"""
Test script to verify that API endpoints return real data.
"""

import requests
import json
import sys
import time

def test_api_endpoints():
    """Test the monitoring API endpoints to verify real data."""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing API endpoints for real data...\n")
    
    # Test endpoints
    endpoints = [
        ("/api/v1/monitoring/status", "System Status"),
        ("/api/v1/monitoring/health", "Health Check"),
        ("/api/v1/monitoring/metrics", "Metrics"),
        ("/api/v1/monitoring/metrics/recent-activity", "Recent Activity"),
        ("/api/v1/monitoring/metrics/query-trends", "Query Trends"),
        ("/api/v1/monitoring/metrics/user-activity", "User Activity")
    ]
    
    results = {}
    
    for endpoint, name in endpoints:
        print(f"📡 Testing {name} ({endpoint})...")
        
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                results[endpoint] = {
                    "status": "✅ SUCCESS",
                    "data": data,
                    "size": len(json.dumps(data))
                }
                print(f"   Status: ✅ {response.status_code}")
                print(f"   Data size: {len(json.dumps(data))} bytes")
                
                # Analyze data content
                if "status" in data:
                    print(f"   Overall status: {data.get('status', 'N/A')}")
                
                if "services" in data:
                    services = data["services"]
                    print(f"   Services checked: {len(services)}")
                    for service_name, service_data in services.items():
                        if isinstance(service_data, dict) and "status" in service_data:
                            print(f"     - {service_name}: {service_data['status']}")
                
                if "application_metrics" in data:
                    app_metrics = data["application_metrics"]
                    print(f"   Application metrics:")
                    for key, value in app_metrics.items():
                        print(f"     - {key}: {value}")
                
                if isinstance(data, list) and name == "Recent Activity":
                    print(f"   Recent activities: {len(data)} events")
                    for i, activity in enumerate(data[:3]):
                        print(f"     {i+1}. {activity.get('action', 'Unknown')} by {activity.get('user', 'Unknown')}")
                        
            else:
                results[endpoint] = {
                    "status": f"❌ ERROR {response.status_code}",
                    "error": response.text[:200]
                }
                print(f"   Status: ❌ {response.status_code}")
                print(f"   Error: {response.text[:100]}...")
                
        except requests.exceptions.ConnectionError:
            results[endpoint] = {
                "status": "🔌 CONNECTION ERROR",
                "error": "Cannot connect to server"
            }
            print(f"   Status: 🔌 Cannot connect to {base_url}")
            print("   Make sure the FastAPI server is running!")
            
        except Exception as e:
            results[endpoint] = {
                "status": f"❌ EXCEPTION",
                "error": str(e)
            }
            print(f"   Status: ❌ Exception: {e}")
        
        print()
    
    # Summary
    print("=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    success_count = 0
    for endpoint, result in results.items():
        status = result["status"]
        print(f"{status} {endpoint}")
        if "SUCCESS" in status:
            success_count += 1
    
    print(f"\n✅ {success_count}/{len(endpoints)} endpoints working")
    
    if success_count == 0:
        print("\n🚨 NO ENDPOINTS WORKING")
        print("Make sure to start the FastAPI server first:")
        print("   uv run main.py")
        return False
    elif success_count == len(endpoints):
        print("\n🎉 ALL ENDPOINTS WORKING WITH REAL DATA!")
        return True
    else:
        print(f"\n⚠️  PARTIAL SUCCESS ({success_count}/{len(endpoints)})")
        return False


def test_with_sample_data():
    """Test with the sample database we created."""
    print("🗄️  Testing with sample database...\n")
    
    # First, let's check if our test database exists
    import os
    if os.path.exists("test_metrics.db"):
        print("✅ Sample database found: test_metrics.db")
        
        # Let's manually query the database to show what data is available
        import sqlite3
        conn = sqlite3.connect("test_metrics.db")
        conn.row_factory = sqlite3.Row
        
        print("\n📊 Sample data in database:")
        
        # Check documents
        cursor = conn.execute("SELECT COUNT(*) FROM document_events WHERE event_type='upload'")
        doc_count = cursor.fetchone()[0]
        print(f"   📄 Documents: {doc_count}")
        
        # Check queries
        cursor = conn.execute("SELECT COUNT(*) FROM query_logs")
        query_count = cursor.fetchone()[0]
        print(f"   🔍 Queries: {query_count}")
        
        # Check users
        cursor = conn.execute("SELECT COUNT(DISTINCT user_id) FROM user_sessions")
        user_count = cursor.fetchone()[0]
        print(f"   👥 Users: {user_count}")
        
        # Check recent events
        cursor = conn.execute("SELECT COUNT(*) FROM system_events")
        event_count = cursor.fetchone()[0]
        print(f"   📝 System events: {event_count}")
        
        # Show recent events
        cursor = conn.execute("""
            SELECT description, user_id, timestamp 
            FROM system_events 
            ORDER BY timestamp DESC 
            LIMIT 3
        """)
        events = cursor.fetchall()
        print(f"\n   Recent events:")
        for event in events:
            print(f"     - {event[0]} ({event[1] or 'system'})")
        
        conn.close()
        
        print("\n💡 To use this data in your API:")
        print("   1. Copy test_metrics.db to metrics.db")
        print("   2. Update your FastAPI app to use the metrics system")
        print("   3. Start the server and test the endpoints")
        
    else:
        print("❌ Sample database not found")
        print("Run: python sync_metrics_test.py to create sample data")


def main():
    """Main test function."""
    print("🚀 API Endpoint Data Verification\n")
    
    # Test if server is running and endpoints work
    server_working = test_api_endpoints()
    
    print("\n" + "="*60)
    
    # Show what sample data is available
    test_with_sample_data()
    
    if not server_working:
        print("\n🔧 TO FIX:")
        print("1. Start your FastAPI server: uv run main.py")
        print("2. Make sure metrics system is integrated")
        print("3. Copy sample data: cp test_metrics.db metrics.db")
        
    return 0 if server_working else 1


if __name__ == "__main__":
    exit(main())