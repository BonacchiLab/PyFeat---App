# FEDET

FEDET (Facial Expression Data Extraction Tool) is a local research application for extracting and exploring facial-expression measures from videos with [Py-Feat](https://py-feat.org/). The frontend is React, TypeScript, and Vite; FastAPI runs video processing in the local Python backend.

## Run locally

Use Python 3.11 and Node.js LTS. Install Python requirements in the project virtual environment and frontend dependencies once:

```powershell
python -m pip install -r requirements.txt
cd frontend
npm install
```

Start the API from the project root:

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

In a second terminal, start the frontend:

```powershell
cd frontend
npm run dev
```

Open the local address printed by Vite (usually `http://localhost:5173`). The FastAPI documentation is at `http://127.0.0.1:8000/docs`.

## Current workflow

Upload MP4, AVI, or MOV videos, assign subject IDs, choose frame sampling, and optionally configure stimulus names and windows shared across or specific to each video. The frontend submits one extraction task per video to the API, shows progress, previews the first 50 output rows, displays quality summaries and interactive plots, and downloads one Parquet file per video.

The React interface includes a guide, FaceScore quality view, signal plots, AU heatmap, before/during stimulus comparison, and event-aligned trajectories. Sampling supports approximately 1 FPS, all frames, or a custom FPS. The sidebar estimates output rows and processing time from video metadata and observed processing history when available. The similarity matrix was intentionally removed.

Exports contain Py-Feat predictions plus `subject_id` and `input_file`. Time columns depend on sampling mode: frame-based results include `time_sec`; approximately 1 FPS results are aggregated by second. The migration preserves the pipeline's scientific calculations, output column names, and data types.

## Verify the frontend build

From the `frontend/` directory, run:

```powershell
npm run build
```

During development, restart the API after backend code changes so the running server loads the current implementation. Previous parity checks against Streamlit are summarized in `development.md`.

## Project layout

- `frontend/`: React, TypeScript, and Vite interface.
- `api/`: FastAPI endpoints for extraction tasks, progress, metadata, previews, and downloads.
- `core/`: Py-Feat detector, processing, analytics, and task progress modules.
- `requirements.txt` and `frontend/package.json`: Python and frontend dependencies.
