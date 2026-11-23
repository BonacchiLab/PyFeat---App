"""Video extraction tasks, SSE progress, and Parquet downloads."""

import json
import asyncio
import math
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import List

import cv2
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from core.detector import detect_video
from core.processing import add_dominant_emotion, apply_stimulus_metadata, calculate_skip_frames, dataframe_to_parquet_bytes, prepare_frame_dataframe
from core.progress import task_progress
from api.schemas import StimulusWindow

router = APIRouter(prefix="/api/extractions", tags=["extractions"])
RESULTS_DIR = Path(tempfile.gettempdir()) / "fedet-results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov"}
PYFEAT_VIDEO_OUTPUT_SIZE = 700


def _subject_filename(subject_id: str) -> str:
    safe_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", subject_id.strip()).rstrip(" .")
    return f"{safe_name or 'subject'}.parquet"


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


@router.post("/metadata")
async def video_metadata(file: UploadFile = File(...)):
    filename = Path(file.filename or "video").name
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Supported formats: MP4, AVI, MOV")
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix.lower()) as temp_video:
            temp_video.write(await file.read())
            temp_path = Path(temp_video.name)
        cap = cv2.VideoCapture(str(temp_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        fps = fps if fps > 0 else 30.0
        return {"fps": fps, "frame_count": frame_count, "duration_seconds": frame_count / fps if frame_count else 0.0}
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)


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
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if not fps or fps <= 0:
            fps = 30.0
        resize_scale = PYFEAT_VIDEO_OUTPUT_SIZE / max(video_width, video_height) if video_width and video_height else 1.0
        analysis_width = int(video_width * resize_scale) if video_width else None
        analysis_height = int(video_height * resize_scale) if video_height else None
        skip_frames = 1 if sampling_mode == "all_frames" else calculate_skip_frames(fps, target_fps if sampling_mode == "custom_fps" else 1.0)
        expected_rows = int(math.ceil(frame_count / skip_frames)) if frame_count else 0

        def update_progress(batch_number, total_batches):
            ratio = batch_number / total_batches if total_batches else 0.0
            task_progress.update(task_id, progress=min(0.95, 0.05 + ratio * 0.9), message=f"Extracting frames ({batch_number}/{total_batches})")

        task_progress.update(task_id, message="Running Py-Feat extraction", progress=0.05)
        prediction = detect_video(video_path, skip_frames, update_progress)
        raw_df = prediction.to_pandas() if hasattr(prediction, "to_pandas") else prediction
        output_df = prepare_frame_dataframe(raw_df, fps, sampling_mode)
        output_df = add_dominant_emotion(output_df)
        if stimuli:
            output_df = apply_stimulus_metadata(output_df, stimuli)
        output_df["subject_id"] = subject_id
        output_df["input_file"] = filename
        output_path = RESULTS_DIR / f"{task_id}.parquet"
        output_path.write_bytes(dataframe_to_parquet_bytes(output_df))
        detected_frames = int(pd.to_numeric(output_df["frame"], errors="coerce").nunique()) if "frame" in output_df.columns else len(output_df)
        task_progress.update(task_id, status="completed", progress=1.0, message="Extraction completed", result_path=str(output_path), rows=len(output_df), expected_rows=expected_rows, detected_frames=detected_frames, stimuli=stimuli, subject_id=subject_id, sampling_mode=sampling_mode, fps=float(fps), analysis_width=analysis_width, analysis_height=analysis_height)
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
    task.pop("stimuli", None)
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
    return FileResponse(path, media_type="application/vnd.apache.parquet", filename=_subject_filename(task.get("subject_id", task_id)))


@router.get("/{task_id}/frames")
def get_frame_data(task_id: str, offset: int = 0, limit: int = 500):
    task = task_progress.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task["status"] != "completed" or not task.get("result_path"):
        raise HTTPException(status_code=409, detail="Extraction is not complete")
    if task.get("sampling_mode") != "all_frames":
        raise HTTPException(status_code=409, detail={"message": "Frame-synchronized inspection requires all_frames sampling.", "sampling_mode": task.get("sampling_mode")})
    path = Path(task["result_path"])
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Result file not found")
    safe_offset = max(0, offset)
    safe_limit = min(max(1, limit), 1000)
    df = pd.read_parquet(path)
    emotion_columns = [column for column in ("anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral") if column in df.columns and pd.api.types.is_numeric_dtype(df[column])]
    au_columns = sorted((column for column in df.columns if re.fullmatch(r"AU\d+", column)), key=lambda column: int(column[2:]))
    landmark_columns = [column for column in df.columns if re.fullmatch(r"[xy]_\d+", column)]
    landmark_ids = sorted({int(column[2:]) for column in landmark_columns})
    box_columns = [column for column in ("FaceRectX", "FaceRectY", "FaceRectWidth", "FaceRectHeight", "FaceScore") if column in df.columns]
    selected = [column for column in ("frame", "time_sec", "approx_time", "dominant_emotion", *emotion_columns, *au_columns, *box_columns, *landmark_columns) if column in df.columns]
    if "frame" in df.columns:
        frame_numbers = df["frame"].drop_duplicates().iloc[safe_offset:safe_offset + safe_limit].tolist()
        total_frames = int(df["frame"].nunique(dropna=False))
    else:
        frame_numbers = list(range(safe_offset, min(safe_offset + safe_limit, len(df))))
        total_frames = len(df)
    if "frame" in df.columns:
        page_mask = df["frame"].isin(frame_numbers)
        if any(pd.isna(frame_number) for frame_number in frame_numbers):
            page_mask |= df["frame"].isna()
        page_df = df.loc[page_mask, selected]
        page_groups = page_df.groupby("frame", sort=False, dropna=False)
    else:
        page_groups = ((frame_number, df.iloc[[frame_number]][selected]) for frame_number in frame_numbers)
    frames = []
    for frame_number, group in page_groups:
        records = json.loads(group[selected].to_json(orient="records"))
        detections = []
        for record in records:
            landmarks = []
            for landmark_id in landmark_ids:
                x, y = record.get(f"x_{landmark_id}"), record.get(f"y_{landmark_id}")
                if x is not None and y is not None:
                    landmarks.append({"x": x, "y": y})
            dominant = record.get("dominant_emotion")
            dominant_emotions = dominant.split(" + ") if isinstance(dominant, str) else []
            box = {column: record.get(column) for column in box_columns} or None
            detections.append({
                "dominant_emotion": dominant,
                "dominant_emotions": dominant_emotions,
                "emotions": {column: record.get(column) for column in emotion_columns},
                "action_units": {column: record.get(column) for column in au_columns},
                "face_box": box,
                "landmarks": landmarks or None,
            })
        first = records[0] if records else {}
        frames.append({
            "frame": None if pd.isna(frame_number) else frame_number,
            "time_sec": first.get("time_sec"),
            "approx_time": first.get("approx_time"),
            "dominant_emotion": first.get("dominant_emotion"),
            "dominant_emotions": first.get("dominant_emotion", "").split(" + ") if isinstance(first.get("dominant_emotion"), str) else [],
            "emotions": {column: first.get(column) for column in emotion_columns},
            "action_units": {column: first.get(column) for column in au_columns},
            "landmarks": detections[0]["landmarks"] if detections else None,
            "detections": detections,
        })
    return {"sampling_mode": "all_frames", "total": total_frames, "offset": safe_offset, "limit": safe_limit, "frames": frames, "has_landmarks": bool(landmark_ids), "coordinate_width": task.get("analysis_width"), "coordinate_height": task.get("analysis_height")}


@router.get("/{task_id}/preview")
def preview_result(task_id: str, limit: int = 50, threshold: float = 0.9, include_timeline: bool = True):
    task = task_progress.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task["status"] != "completed" or not task.get("result_path"):
        raise HTTPException(status_code=409, detail="Extraction is not complete")
    path = Path(task["result_path"])
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Result file not found")
    df = pd.read_parquet(path)
    safe_limit = min(max(limit, 1), 200)
    records = json.loads(df.head(safe_limit).to_json(orient="records", date_format="iso"))
    scores = pd.to_numeric(df["FaceScore"], errors="coerce") if "FaceScore" in df.columns else pd.Series(dtype="float64")
    time_column = next((column for column in ("time_sec", "time_start_sec", "frame") if column in df.columns), None)
    numeric_columns = [column for column in df.select_dtypes(include="number").columns if column != time_column]
    chart_step = max(1, (len(df) + 599) // 600)
    chart_df = df.iloc[::chart_step].copy()
    chart_points = json.loads(chart_df[[time_column, *numeric_columns]].to_json(orient="records")) if time_column else []
    au_columns = sorted((column for column in numeric_columns if column.startswith("AU") and column[2:].isdigit()), key=lambda column: int(column[2:]))
    emotion_names = ["neutral", "happiness", "sadness", "anger", "fear", "disgust", "surprise"]
    emotion_columns = [column for column in emotion_names if column in numeric_columns]
    if include_timeline and "dominant_emotion" in df.columns and emotion_columns:
        timeline_columns = [column for column in ("frame", "approx_time", "time_sec", "time_start_sec", "dominant_emotion", *emotion_columns) if column in df.columns]
        timeline = []
        for row in json.loads(df[timeline_columns].to_json(orient="records")):
            dominant = row.get("dominant_emotion")
            dominant_emotions = dominant.split(" + ") if isinstance(dominant, str) else []
            dominant_score = row.get(dominant_emotions[0]) if dominant_emotions else None
            timestamp = row.get("approx_time")
            if timestamp is None:
                timestamp = row.get("time_sec", row.get("time_start_sec"))
            timeline.append({
                "frame": row.get("frame"),
                "timestamp": timestamp,
                "dominant_emotion": dominant,
                "dominant_emotions": dominant_emotions,
                "dominant_score": dominant_score,
                "emotions": {column: row.get(column) for column in emotion_columns},
            })
    else:
        timeline = []
    pose_columns = [column for column in ("Pitch", "Roll", "Yaw") if column in numeric_columns]
    stimulus_statistics = {}
    if time_column:
        times = pd.to_numeric(df[time_column], errors="coerce")
        task_stimuli = task.get("stimuli", [])
        for stimulus in task_stimuli:
            start, end = sorted((float(stimulus["inicio"]), float(stimulus["fim"])))
            before_rows = times.ge(start - 5) & times.lt(start)
            during_rows = times.ge(start) & times.lt(end)
            metrics = {}
            for column in [*au_columns, *emotion_columns, *pose_columns]:
                values = pd.to_numeric(df[column], errors="coerce")
                before = values.loc[before_rows].mean()
                during = values.loc[during_rows].mean()
                metrics[column] = {
                    "before": float(before) if pd.notna(before) else None,
                    "during": float(during) if pd.notna(during) else None,
                }
            stimulus_statistics[stimulus["nome"]] = metrics
    return {
        "columns": list(df.columns),
        "rows": len(df),
        "preview": records,
        "face_score_median": float(scores.median()) if scores.notna().any() else None,
        "face_score_missing": int(scores.isna().sum()) if len(scores) else None,
        "face_score_flagged": int(scores.lt(threshold).sum()) if len(scores) else None,
        "detected_frames": task.get("detected_frames", len(df)),
        "expected_rows": task.get("expected_rows", len(df)),
        "time_column": time_column,
        "chart_points": chart_points,
        "chart_columns": {"action_units": au_columns, "emotions": emotion_columns, "head_pose": pose_columns},
        "emotion_timeline": timeline,
        "stimulus_statistics": stimulus_statistics,
    }
