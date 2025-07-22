#!/usr/bin/env python3
"""
Test the new MySQL server connection
"""
import os
import sys
sys.path.append('.')

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

async def test_mysql_connection():
    """Test the new MySQL server connection"""
    print("🔧 Testing New MySQL Server Connection...")
    print(f"Host: {os.getenv('DB_HOST')}")
    print(f"Port: {os.getenv('DB_PORT')}")
    print(f"Database: {os.getenv('DB_NAME')}")
    print(f"User: {os.getenv('DB_USER')}")
    print("Password: [HIDDEN]")
    print()
    
    try:
        import pymysql
        
        # Test connection with new settings
        connection = pymysql.connect(
            host=os.getenv('DB_HOST'),
            port=int(os.getenv('DB_PORT')),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_NAME'),
            connect_timeout=10  # 10 second timeout
        )
        
        print("✅ Connection successful!")
        
        # Test basic query
        with connection.cursor() as cursor:
            cursor.execute("SELECT VERSION() as version, DATABASE() as database_name, USER() as current_user")
            result = cursor.fetchone()
            
        print(f"📊 Server Info:")
        print(f"   MySQL Version: {result[0]}")
        print(f"   Current Database: {result[1]}")
        print(f"   Current User: {result[2]}")
        
        # Test permissions - try to show tables
        try:
            with connection.cursor() as cursor:
                cursor.execute("SHOW TABLES")
                tables = cursor.fetchall()
                
            print(f"📋 Available Tables: {len(tables)} found")
            for i, table in enumerate(tables[:5]):  # Show first 5 tables
                print(f"   {i+1}. {table[0]}")
            if len(tables) > 5:
                print(f"   ... and {len(tables) - 5} more tables")
                
        except Exception as e:
            print(f"⚠️  Could not list tables: {e}")
        
        connection.close()
        print("\n🎉 MySQL server connection test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\nPossible issues:")
        print("1. Server is not accessible from this network")
        print("2. Incorrect credentials")
        print("3. Firewall blocking connection")
        print("4. Database server is down")
        return False

async def test_rag_config_loading():
    """Test if RAG server loads the new config correctly"""
    print("\n🔧 Testing RAG Server Config Loading...")
    
    try:
        from src.core.config import get_config
        config = get_config()
        
        print(f"✅ Config loaded successfully!")
        print(f"📊 Database Config:")
        print(f"   Host: {config.database.host}")
        print(f"   Port: {config.database.port}")
        print(f"   Database: {config.database.name}")
        print(f"   User: {config.database.user}")
        print(f"   Driver: {config.database.driver}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False

if __name__ == "__main__":
    import asyncio
    
    print("🚀 MySQL Server Connection Test")
    print("=" * 50)
    
    # Test connection
    mysql_ok = asyncio.run(test_mysql_connection())
    
    # Test config loading
    config_ok = asyncio.run(test_rag_config_loading())
    
    print("\n" + "=" * 50)
    print("📊 FINAL RESULTS")
    print("=" * 50)
    print(f"MySQL Connection: {'✅ PASS' if mysql_ok else '❌ FAIL'}")
    print(f"Config Loading:   {'✅ PASS' if config_ok else '❌ FAIL'}")
    
    if mysql_ok and config_ok:
        print("\n🎉 All tests passed! Your RAG server should work with the new MySQL server.")
    else:
        print("\n⚠️  Some tests failed. Check the issues above.")