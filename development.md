# Development notes

## Goals

FEDET should let researchers and students extract, preprocess, preview, clean, and visualize facial-expression data from videos without programming. It should support varied research designs and remain straightforward to use.

## Current implementation

- A local FastAPI backend accepts video extraction tasks, runs Py-Feat, reports progress, and serves Parquet downloads, previews, video metadata, and quality summaries.
- Core detector, processing, analytics, and task-progress code is separated into `core/` modules. Processing behavior and output columns are preserved from the Streamlit pipeline.
- The React/TypeScript/Vite frontend supports multiple videos, subject IDs, every-frame/approximately 1 FPS/custom-FPS sampling, and shared or per-video stimulus windows.
- The frontend displays extraction progress, a 50-row preview, FaceScore quality information, signal plots, an AU heatmap, before/during comparisons, and event-aligned trajectories.
- The Streamlit app remains as a legacy interface while the React frontend is brought to parity. Recording similarity was intentionally removed.

## Remaining migration work

- Finish matching the useful Streamlit workflow and visual behavior in React, then verify the full multi-video flow.
- Review the API/React boundary and error handling, and provide a simple way to start the local frontend and backend together.
- Remove the Streamlit interface and its obsolete code only after the replacement is complete and stable.

## Future directions

- Investigate Py-Feat version support, including model availability, output columns, behavior, and installation compatibility.
- Explore predominant-emotion epochs. Decide tie handling, low-confidence/missing predictions, smoothing, and minimum duration before implementation.
- Add CSV and XML export alongside Parquet.
- Add dataset reduction controls for rows and columns, with undo and permanent access to the original raw dataset.
- Consider integrating Tomás's eye-tracking pipeline after the main application is structured and stable.

Feature priorities are set by the project owner as work proceeds.
