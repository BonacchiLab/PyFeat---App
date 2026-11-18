"""Local FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import content, extraction, system

app = FastAPI(title="FEDET API", version="0.1.0", description="Local facial-expression data extraction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
app.include_router(system.router)
app.include_router(content.router)
app.include_router(extraction.router)


@app.get("/health")
def health():
    return {"status": "ok"}
