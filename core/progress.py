"""In-memory progress tracking for background extraction tasks."""

from datetime import datetime, timezone
from threading import Lock
from typing import Any


class TaskProgress:
    def __init__(self) -> None:
        self._lock = Lock()
        self._tasks: dict[str, dict[str, Any]] = {}

    def create(self, task_id: str) -> None:
        with self._lock:
            self._tasks[task_id] = {
                "task_id": task_id, "status": "queued", "progress": 0.0,
                "message": "Queued", "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(), "error": None,
                "result_path": None,
            }

    def update(self, task_id: str, **values: Any) -> None:
        with self._lock:
            if task_id not in self._tasks:
                return
            self._tasks[task_id].update(values)
            self._tasks[task_id]["updated_at"] = datetime.now(timezone.utc).isoformat()

    def get(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            task = self._tasks.get(task_id)
            return dict(task) if task else None


task_progress = TaskProgress()
