# Development notes

## Goals

FEDET should let researchers and students extract, preprocess, preview, clean, and visualize facial-expression data from videos without programming. It should support varied research designs and remain straightforward to use.

## Current implementation

- A local FastAPI backend accepts video extraction tasks, runs Py-Feat, reports progress, and serves Parquet downloads, previews, video metadata, and quality summaries.
- Core detector, processing, analytics, and task-progress code is separated into `core/` modules. Processing behavior and output columns are preserved from the Streamlit pipeline.
- The React/TypeScript/Vite frontend supports multiple videos, subject IDs, every-frame/approximately 1 FPS/custom-FPS sampling, and shared or per-video stimulus windows.
- The frontend displays extraction progress, a 50-row preview, FaceScore quality information, signal plots, an AU heatmap, before/during comparisons, and event-aligned trajectories.
- The Streamlit app remains as a legacy reference while the React frontend is brought to full parity. Recording similarity was intentionally removed.
- The React sidebar estimates sampled rows and processing time using the selected sampling mode, video metadata, and observed processing history where available; it uses a four-second-per-sampled-frame fallback without history.

## Verification status

- Ran the two short test videos through the React/API flow in approximately 1 FPS, all-frames, and custom-2-FPS modes, with a shared stimulus window from 1 to 2 seconds.
- Compared the matching React and Streamlit approximately-1-FPS Parquet results: row counts, ordered columns, data types, and numeric values matched. The only differing value was the temporary source-video path recorded in `input`, which is generated separately for each upload.
- Checked stimulus before/during statistics from all six mode/video combinations against the Streamlit analytics calculation; the values matched.
- Ran `npm run build` successfully after the frontend changes and `git diff --check` without whitespace errors.
- For repeatable comparisons, start both interfaces against the same current API process and use identical video settings. Restart FastAPI after backend changes; an already-running server can continue serving older code.

## Remaining migration work - MOST IMPORTANT

- Continue reviewing visual and functional parity for less common workflows, error states, and differing video characteristics.
- Review the API/React boundary and error handling, and provide a simple way to start the local frontend and backend together.
- Remove the Streamlit interface and its obsolete code only after the replacement is complete and stable.

## Future directions

- Investigate Py-Feat version support, including model availability, output columns, behavior, and installation compatibility.
- Explore predominant-emotion epochs. Decide tie handling, low-confidence/missing predictions, smoothing, and minimum duration before implementation.
- Add CSV and XML export alongside Parquet.
- Add dataset reduction controls for rows and columns, with undo and permanent access to the original raw dataset.
- Consider integrating Tomás's eye-tracking pipeline after the main application is structured and stable.

Feature priorities are set by the project owner as work proceeds.
