# Development notes

## Goals

FEDET should let researchers and students extract, preprocess, preview, clean, and visualize facial-expression data from videos without programming. It should support varied research designs and remain straightforward to use.

## Current implementation

- A local FastAPI backend accepts video extraction tasks, runs Py-Feat, reports progress, and serves Parquet downloads, previews, video metadata, and quality summaries.
- Core detector, processing, analytics, and task-progress code is separated into `core/` modules. The processing behavior and output columns were carried over from the original Streamlit workflow.
- The React/TypeScript/Vite frontend supports multiple videos, subject IDs, every-frame/approximately 1 FPS/custom-FPS sampling, and shared or per-video stimulus windows.
- After existing frame preparation and optional per-second aggregation, `core.processing.add_dominant_emotion` adds a `dominant_emotion` label to each output row. Original Py-Feat emotion scores and other columns remain unchanged; the new label is included in Parquet exports.
- The extraction preview exposes an `emotion_timeline` containing each output row's frame, timestamp, dominant label, dominant score, and available emotion scores. The React frontend renders one colored, selectable block per row with a legend and score details.
- For `all_frames` extractions, `GET /api/extractions/{task_id}/frames` serves paginated frame details (maximum 1,000 frames per request), grouping multiple face detections under their source frame and including face boxes, emotion scores, AUs, timestamps, and paired `x_N`/`y_N` landmarks when present. The local React video inspector loads 500 frames at a time, seeks when a frame block is selected, tracks playback time, and can overlay face boxes, landmarks, and emotion/AU summaries. Canvas coordinates use the source video dimensions; the video element preserves its intrinsic aspect ratio. Other sampling modes display a clear limitation notice; the endpoint rejects frame-synchronized requests for them.
- The frontend displays extraction progress, a 50-row preview, FaceScore quality information, signal plots, an AU heatmap, before/during comparisons, and event-aligned trajectories.
- The Streamlit interface has been removed after React/API verification. Recording similarity was intentionally removed.
- The React sidebar estimates sampled rows and processing time using the selected sampling mode, video metadata, and observed processing history where available; it uses a four-second-per-sampled-frame fallback without history.

## Verification status

- Ran the two short test videos through the React/API flow in approximately 1 FPS, all-frames, and custom-2-FPS modes, with a shared stimulus window from 1 to 2 seconds.
- Before removing Streamlit, compared React and Streamlit approximately-1-FPS Parquet results: row counts, ordered columns, data types, and numeric values matched. The only differing value was the temporary source-video path recorded in `input`, generated separately for each upload.
- Before removing Streamlit, checked stimulus before/during statistics from all six mode/video combinations against its analytics calculation; the values matched.
- Ran `npm run build` successfully after the frontend changes and `git diff --check` without whitespace errors.
- During parity checks, run both interfaces against the same current API process and use identical video settings. Restart FastAPI after backend changes; an already-running server can continue serving older code.

## Dominant emotion selection

- `add_dominant_emotion` in `core/processing.py` looks for the numeric columns among `anger`, `disgust`, `fear`, `happiness`, `sadness`, `surprise`, and `neutral`.
- For each output row, it selects every emotion column tied for the largest available numeric score. A single winner is stored by name, such as `happiness`; a tie is stored as a ` + `-separated label, such as `anger + fear`. This is a direct maximum comparison over the existing Py-Feat values; it does not normalize, threshold, smooth, or otherwise modify the scores.
- If all available scores in a row are missing, `dominant_emotion` is null. For ties, the stored names follow the configured column order: anger, disgust, fear, happiness, sadness, surprise, neutral.
- Classification runs after the existing sampling/aggregation step. Thus, all-frames and custom-FPS outputs are labeled per sampled frame; the approximately 1 FPS mode labels each existing per-second aggregate row based on its already-aggregated emotion scores. The aggregation calculation itself is unchanged.
- The API includes this field in `emotion_timeline`; the UI's frame blocks represent output rows/samples, not necessarily every source-video frame in modes that sample or aggregate.

## Remaining work - MOST IMPORTANT

- Continue reviewing visual and functional parity for less common workflows, error states, and differing video characteristics.
- Review the API/React boundary and error handling, and provide a simple way to start the local frontend and backend together.

## Future directions

- Investigate Py-Feat version support, including model availability, output columns, behavior, and installation compatibility.
- Decide whether emotion categorization should use a baseline/confidence threshold before assigning a category (for example, only label an emotion when its score meets a chosen criterion) or always retain the emotion with the highest score. Current behavior always uses the maximum available score and preserves exact ties as multiple labels; no baseline is applied.
- Explore predominant-emotion epochs, including smoothing and minimum duration. Frame/sample-level dominant-emotion labels, including all tied maximum emotions, are already implemented; all-missing rows remain unlabeled.
- Add CSV and XML export alongside Parquet.
- Add dataset reduction controls for rows and columns, with undo and permanent access to the original raw dataset.
- Consider integrating Tomás's eye-tracking pipeline after the main application is structured and stable.

Feature priorities are set by the project owner as work proceeds.
