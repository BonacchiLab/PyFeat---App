"""Validated API request and response models."""

from typing import Literal
from pydantic import BaseModel, Field


class StimulusWindow(BaseModel):
    nome: str = Field(min_length=1)
    inicio: float = Field(ge=0)
    fim: float = Field(ge=0)


class TaskStatus(BaseModel):
    task_id: str
    status: Literal["queued", "running", "completed", "failed"]
    progress: float = Field(ge=0, le=1)
    message: str
    created_at: str
    updated_at: str
    error: str | None = None
