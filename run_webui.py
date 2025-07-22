#!/usr/bin/env python3
"""
RAG Server Web UI Runner
Launches the Streamlit web interface for the RAG server
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import streamlit
        import plotly
        import pandas
        import jwt
        import requests
        print("✅ All required dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Installing requirements...")
        
        # Install dependencies
        requirements_file = Path(__file__).parent / "requirements_webui.txt"
        if requirements_file.exists():
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ])
            return True
        else:
            print("❌ requirements_webui.txt not found")
            return False

def main():
    """Main function to run the web UI"""
    parser = argparse.ArgumentParser(description="RAG Server Web UI")
    parser.add_argument(
        "--port", 
        type=int, 
        default=8501, 
        help="Port to run the web UI on (default: 8501)"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default="localhost", 
        help="Host to bind to (default: localhost)"
    )
    parser.add_argument(
        "--reload", 
        action="store_true", 
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--check-deps", 
        action="store_true", 
        help="Only check dependencies without running the app"
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    if args.check_deps:
        print("✅ Dependency check completed successfully")
        return
    
    # Set up environment
    webui_dir = Path(__file__).parent / "webui"
    app_file = webui_dir / "app.py"
    
    if not app_file.exists():
        print(f"❌ Web UI application file not found: {app_file}")
        sys.exit(1)
    
    # Build streamlit command
    cmd = [
        "streamlit", "run", str(app_file),
        "--server.port", str(args.port),
        "--server.address", args.host,
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        "--theme.base", "light"
    ]
    
    # Add reload option for development
    if args.reload:
        cmd.extend(["--server.fileWatcherType", "watchdog"])
    
    # Set environment variables
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent)
    
    print("🚀 Starting RAG Server Web UI...")
    print(f"🌐 URL: http://{args.host}:{args.port}")
    print(f"📁 App file: {app_file}")
    
    if args.reload:
        print("🔄 Auto-reload enabled")
    
    print("\nDemo Credentials:")
    print("👤 Admin: username=admin, password=admin123")
    print("👤 User: username=user, password=user123")
    print("\nPress Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Run streamlit
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()