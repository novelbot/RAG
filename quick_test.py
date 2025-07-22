#!/usr/bin/env python3
import asyncio
import subprocess
import httpx
import time

async def quick_test():
    print("🚀 Quick test of fixed /docs endpoint...")
    
    # Start server
    process = subprocess.Popen(['uv', 'run', 'python', 'main.py'], 
                              stdout=subprocess.DEVNULL, 
                              stderr=subprocess.DEVNULL)
    
    # Wait for startup
    await asyncio.sleep(3)
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Test /docs
            response = await client.get("http://localhost:8000/docs")
            if response.status_code == 200:
                print("✅ /docs is now working! Status: 200")
            else:
                print(f"❌ /docs still failing. Status: {response.status_code}")
                print(f"Response: {response.text[:100]}...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Kill server
        process.terminate()
        process.wait()

if __name__ == "__main__":
    asyncio.run(quick_test())