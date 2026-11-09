# Development notes

This document records the intended direction of FEDET and helps understand what is implemented, what should come next, and which decisions are still open. 

## Primary goal

Make facial-expression data work accessible to researchers and students who do not have the programming experience, confidence, or time to write their own analysis code. The interface should provide a simple way to extract facial-expression data from videos, preprocess it, and explore useful visualizations.

The primary goal is met when a researcher or student can use the interface to extract facial-expression data from videos, preprocess it, and obtain useful visualizations through a straightforward workflow.

## Secondary goals

- Keep the user interface simple and easy to learn.
- Support a broad range of research and experimental designs.
- Offer flexible analysis settings, especially for frame sampling and data cleaning, rather than assuming a single experimental workflow.

The accessibility part means that the interface should be kept usable without programming experience or knowledge.


## Current implementation

- Upload multiple MP4, AVI, or MOV videos and assign a subject ID to each.
- Select every-frame, approximately 1 FPS, or custom-FPS sampling.
- Optionally annotate stimulus names and time intervals relative to each video.
- Run Py-Feat extraction and preprocess per-frame results or aggregate sampled results by second.
- Inspect detection-quality summaries, Action Unit, emotion, and head-pose signals, an Action Unit heatmap, and stimulus-based comparisons across recordings.
- Download a separate Parquet file for each processed video.

The interface and processing logic currently live in `app.py`. The detector uses Py-Feat's default model configuration and chooses CUDA when available, otherwise CPU. 

## Next steps

The order below **does not reflects the current priority order**. 

### 1. Investigate Py-Feat version support

Determine whether and how the application should support different Py-Feat versions. Compare relevant model availability, output columns, processing behavior, installation constraints, and compatibility with the current Python and dependency setup.

**Still need to decide this, investigate the pros and cons**

### 2. Explore predominant-emotion epochs

Investigate ways to categorize each frame by its predominant emotion, group frame-level categories into temporal “epochs,” and visualize those epochs clearly.

Before implementation, decide how to handle ties, low-confidence or missing predictions, temporal smoothing, and the minimum duration of an epoch. The exact categorization and visualization approach is still up to decision.

### 3. Add more output formats

Allow users to export processed datasets in additional formats, including CSV and XML, alongside the existing Parquet export.


### 4. Add dataset reduction options

Let users reduce the output dataset by removing columns and/or rows that are not relevant to their analysis.

**Important:** Even if the user decides to reduce the dataset, there should always be both an "undo" function that undoes the last action of the user in the data reducing set of actions. Also, the original raw dataset should always remain available for download.

### 5. Improve the interface

Continue refining the visual design and workflow so the interface is attractive, clear, and easy to use for people with little or no programming experience. Keep the goal of flexible analysis settings in view as controls and workflows evolve.

### 6. Consider eye-tracking

Treat eye-tracking as a possible future direction. Later, after the main objective of the interface is concluded and well structured, work alongside Tomás to merge his eye-tracking pipeline to this interface.


