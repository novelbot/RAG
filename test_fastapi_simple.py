#!/usr/bin/env python3
"""
Simple FastAPI test to isolate the /docs issue
"""
import os
import sys
sys.path.append('.')

from fastapi import FastAPI
import uvicorn

# Set environment variables
os.environ.update({
    'APP_ENV': 'development',
    'DEBUG': 'true'
})

# Create a simple FastAPI app without middleware
app = FastAPI(
    title="Simple RAG Server Test",
    version="1.0.0", 
    description="Testing /docs endpoint",
    docs_url="/docs",
    redoc_url="/redoc"
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Simple FastAPI test"}

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy"}

@app.get("/test")
async def test():
    """Test endpoint"""
    return {"test": "success"}

if __name__ == "__main__":
    print("🚀 Starting simple FastAPI test server...")
    print("Available endpoints:")
    print("- http://localhost:8003/")
    print("- http://localhost:8003/health")  
    print("- http://localhost:8003/test")
    print("- http://localhost:8003/docs")
    print("- http://localhost:8003/redoc")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8003,
        log_level="info"
    )