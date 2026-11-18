"""Video extraction tasks, SSE progress, and Parquet downloads."""

import json
import asyncio
import tempfile
import time
import uuid
from pathlib import Path
from typing import List

import cv2
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from core.detector import detect_video
from core.processing import apply_stimulus_metadata, calculate_skip_frames, dataframe_to_parquet_bytes, prepare_frame_dataframe
from core.progress import task_progress
from api.schemas import StimulusWindow

router = APIRouter(prefix="/api/extractions", tags=["extractions"])
RESULTS_DIR = Path(tempfile.gettempdir()) / "fedet-results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov"}


@router.post("", status_code=202)
async def start_extraction(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    subject_id: str = Form(..., min_length=1),
    sampling_mode: str = Form("fast_fps"),
    target_fps: float = Form(1.0, gt=0),
    stimuli_json: str = Form("[]"),
):
    filename = Path(file.filename or "video").name
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Supported formats: MP4, AVI, MOV")
    if sampling_mode not in {"all_frames", "fast_fps", "custom_fps"}:
        raise HTTPException(status_code=422, detail="Invalid sampling_mode")
    try:
        stimulus_payload = json.loads(stimuli_json)
        stimuli = [StimulusWindow.model_validate(item).model_dump() for item in stimulus_payload]
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail="stimuli_json must be a JSON array of stimulus windows") from exc
    video_bytes = await file.read()
    if not video_bytes:
        raise HTTPException(status_code=400, detail="Uploaded video is empty")
    task_id = str(uuid.uuid4())
    task_progress.create(task_id)
    background_tasks.add_task(_process_video, task_id, filename, video_bytes, subject_id.strip(), sampling_mode, target_fps, stimuli)
    return {"task_id": task_id, "status_url": f"/api/extractions/{task_id}", "events_url": f"/api/extractions/{task_id}/events"}


def _process_video(task_id, filename, video_bytes, subject_id, sampling_mode, target_fps, stimuli):
    video_path = None
    try:
        task_progress.update(task_id, status="running", message="Reading video metadata", progress=0.01)
        suffix = Path(filename).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_video:
            temp_video.write(video_bytes)
            video_path = Path(temp_video.name)
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if not fps or fps <= 0:
            fps = 30.0
        skip_frames = 1 if sampling_mode == "all_frames" else calculate_skip_frames(fps, target_fps if sampling_mode == "custom_fps" else 1.0)

        def update_progress(batch_number, total_batches):
            ratio = batch_number / total_batches if total_batches else 0.0
            task_progress.update(task_id, progress=min(0.95, 0.05 + ratio * 0.9), message=f"Extracting frames ({batch_number}/{total_batches})")

        task_progress.update(task_id, message="Running Py-Feat extraction", progress=0.05)
        prediction = detect_video(video_path, skip_frames, update_progress)
        raw_df = prediction.to_pandas() if hasattr(prediction, "to_pandas") else prediction
        output_df = prepare_frame_dataframe(raw_df, fps, sampling_mode)
        if stimuli:
            output_df = apply_stimulus_metadata(output_df, stimuli)
        output_df["subject_id"] = subject_id
        output_df["input_file"] = filename
        output_path = RESULTS_DIR / f"{task_id}.parquet"
        output_path.write_bytes(dataframe_to_parquet_bytes(output_df))
        task_progress.update(task_id, status="completed", progress=1.0, message="Extraction completed", result_path=str(output_path), rows=len(output_df))
    except Exception as exc:
        task_progress.update(task_id, status="failed", message="Extraction failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        if video_path and video_path.exists():
            video_path.unlink(missing_ok=True)


@router.get("/{task_id}")
def get_task(task_id: str):
    task = task_progress.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    task.pop("result_path", None)
    return task


@router.get("/{task_id}/events")
async def task_events(task_id: str):
    if task_progress.get(task_id) is None:
        raise HTTPException(status_code=404, detail="Task not found")

    async def event_stream():
        while True:
            task = task_progress.get(task_id)
            if task is None:
                break
            yield f"event: progress\ndata: {json.dumps(task)}\n\n"
            if task["status"] in {"completed", "failed"}:
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/{task_id}/download")
def download_result(task_id: str):
    task = task_progress.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task["status"] != "completed" or not task.get("result_path"):
        raise HTTPException(status_code=409, detail="Extraction is not complete")
    path = Path(task["result_path"])
    return FileResponse(path, media_type="application/vnd.apache.parquet", filename=f"{task_id}_pyfeat.parquet")
