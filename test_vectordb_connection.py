#!/usr/bin/env python3
"""
Vector Database Connection Test
Tests connection to Milvus vector database using .env configuration
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from pymilvus import connections, Collection, utility

def load_environment():
    """Load environment variables from .env file"""
    env_path = Path(__file__).parent / '.env'
    if not env_path.exists():
        print(f"❌ .env file not found at {env_path}")
        return False
    
    load_dotenv(env_path)
    return True

def test_milvus_connection():
    """Test connection to Milvus vector database"""
    
    # Get Milvus configuration from environment
    host = os.getenv('MILVUS_HOST', 'localhost')
    port = int(os.getenv('MILVUS_PORT', '19530'))
    user = os.getenv('MILVUS_USER', '')
    password = os.getenv('MILVUS_PASSWORD', '')
    
    print(f"🔧 Testing Milvus connection...")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   User: {user if user else 'None'}")
    print(f"   Password: {'***' if password else 'None'}")
    print()
    
    try:
        # Connect to Milvus
        connection_params = {
            'alias': 'test_connection',
            'host': host,
            'port': port
        }
        
        # Add authentication if provided
        if user and password:
            connection_params['user'] = user
            connection_params['password'] = password
        
        connections.connect(**connection_params)
        print("✅ Connection established successfully!")
        
        # Test basic operations
        print("\n🔍 Testing basic operations...")
        
        # List collections using the specific connection
        collections = utility.list_collections(using='test_connection')
        print(f"   Collections found: {len(collections)}")
        for collection_name in collections:
            print(f"   - {collection_name}")
        
        # Get server version
        try:
            version = utility.get_server_version(using='test_connection')
            print(f"   Server version: {version}")
        except:
            print("   Server version: Not available")
        
        # Test connection status
        connected_aliases = connections.list_connections()
        print(f"   Active connections: {connected_aliases}")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        
        # Provide troubleshooting hints
        print("\n💡 Troubleshooting tips:")
        print("   1. Check if Milvus server is running")
        print("   2. Verify host and port configuration")
        print("   3. Check firewall settings")
        print("   4. Validate username/password if using authentication")
        
        return False
    
    finally:
        # Clean up connection
        try:
            connections.disconnect('test_connection')
        except:
            pass

def main():
    """Main test function"""
    print("🚀 Vector Database Connection Test")
    print("=" * 50)
    
    # Load environment variables
    if not load_environment():
        sys.exit(1)
    
    # Test Milvus connection
    success = test_milvus_connection()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed! Vector database is accessible.")
        sys.exit(0)
    else:
        print("❌ Connection test failed. Please check configuration.")
        sys.exit(1)

if __name__ == "__main__":
    main()