"""
Main entry point for the RAG server application.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from src.core.app import run_server

def main():
    """Main function to run the RAG server"""
    # Load .env file explicitly
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Loaded environment from: {env_path}")
        print(f"   DB_HOST: {os.getenv('DB_HOST', 'NOT_SET')}")
        print(f"   LLM_PROVIDER: {os.getenv('LLM_PROVIDER', 'NOT_SET')}")
        print(f"   LLM_MODEL: {os.getenv('LLM_MODEL', 'NOT_SET')}")
    else:
        print(f"⚠️  .env file not found at: {env_path}")
    
    run_server()

if __name__ == "__main__":
    main()