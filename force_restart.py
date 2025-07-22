#!/usr/bin/env python3
"""
Force restart the server and test monitoring endpoint
"""
import subprocess
import time
import requests
import signal
import os
import json

def kill_existing_processes():
    """Kill any existing main.py processes"""
    try:
        subprocess.run(["pkill", "-f", "main.py"], check=False)
        print("✅ Killed existing processes")
        time.sleep(2)
    except:
        pass

def clear_cache():
    """Clear Python cache"""
    try:
        subprocess.run(["find", ".", "-name", "__pycache__", "-type", "d", "-exec", "rm", "-rf", "{}", "+"], 
                      check=False, cwd="/Users/danielim/CS/2025summer/novelbot_RAG_server")
        print("✅ Cleared Python cache")
    except:
        pass

def start_server():
    """Start the server in background"""
    try:
        process = subprocess.Popen(
            ["uv", "run", "python", "main.py"],
            cwd="/Users/danielim/CS/2025summer/novelbot_RAG_server",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print("✅ Started new server process")
        return process
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return None

def test_endpoint():
    """Test the monitoring endpoint"""
    url = "http://localhost:8000/api/v1/monitoring/status"
    
    for attempt in range(10):  # Try for 10 seconds
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                llm_providers = data.get('services', {}).get('llm_providers', {})
                
                print(f"✅ Endpoint accessible (attempt {attempt + 1})")
                print("📊 LLM Providers:")
                for provider, info in llm_providers.items():
                    print(f"   {provider}: {info.get('status', 'N/A')}")
                    if 'model' in info:
                        print(f"     Model: {info['model']}")
                    if 'latency_ms' in info:
                        print(f"     Latency: {info['latency_ms']}ms")
                
                return True
            else:
                print(f"   Status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"   Connection failed (attempt {attempt + 1}): {e}")
        
        time.sleep(1)
    
    return False

def main():
    print("🔄 Force restarting RAG Server...")
    
    # Step 1: Kill existing processes
    kill_existing_processes()
    
    # Step 2: Clear cache
    clear_cache()
    
    # Step 3: Start new server
    process = start_server()
    if not process:
        return
    
    # Step 4: Wait for startup
    print("⏳ Waiting for server startup...")
    time.sleep(5)
    
    # Step 5: Test endpoint
    print("🧪 Testing monitoring endpoint...")
    if test_endpoint():
        print("\n🎉 SUCCESS! Server is running with updated code!")
    else:
        print("\n❌ FAILED! Server may not be running or endpoint is not accessible")
    
    # Keep server running
    try:
        print("\n📝 Server is now running. Check:")
        print("   http://localhost:8000/api/v1/monitoring/status")
        print("   Press Ctrl+C to stop the server")
        
        process.wait()  # Wait for process to complete
    except KeyboardInterrupt:
        print("\n🛑 Stopping server...")
        process.terminate()
        process.wait()

if __name__ == "__main__":
    main()