"""
Test UI routes for streaming endpoint testing.
"""
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from pathlib import Path

router = APIRouter(prefix="", tags=["Test UI"])

@router.get("/test", response_class=HTMLResponse)
async def test_streaming_page(request: Request):
    """Serve the streaming test page"""
    template_path = Path(__file__).parent.parent.parent.parent / "templates" / "test_streaming.html"
    
    with open(template_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    return HTMLResponse(content=html_content)