#!/usr/bin/env python3
"""
Minimal FastAPI test to confirm /docs works
"""
from fastapi import FastAPI
import uvicorn

# Create minimal FastAPI app
app = FastAPI(
    title="Minimal Test",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

@app.get("/")
async def root():
    return {"message": "Minimal test"}

@app.get("/health")  
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    print("🚀 Starting minimal FastAPI test...")
    print("Check: http://localhost:8005/docs")
    uvicorn.run(app, host="0.0.0.0", port=8005)