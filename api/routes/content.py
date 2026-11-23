"""Educational guide and static content endpoints."""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter(prefix="/api/content", tags=["content"])
CONTENT_DIR = Path(__file__).resolve().parents[2] / "content"


@router.get("/guide/{language}")
def get_guide(language: str):
    if language not in {"pt", "en"}:
        raise HTTPException(status_code=404, detail="Guide language not found")
    path = CONTENT_DIR / f"intro_{language}.json"
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/assets/au-reference")
def get_au_reference():
    path = CONTENT_DIR / "assets" / "Au-Reference.jpg"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="AU reference image not found")
    return FileResponse(path, media_type="image/jpeg", filename="Au-Reference.jpg")
