# FEDET

FEDET (Facial Expression Data Extraction Tool) is a Streamlit research interface for extracting and exploring facial-expression measures from videos with [Py-Feat](https://py-feat.org/).

**Current status:** prototype. Video processing, visualizations, and Parquet export are implemented. 

## Run locally

The current development environment uses Python 3.11.9. Other Python versions have not been validated (at least for now).

### Windows

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Current workflow

1. Upload one or more MP4, AVI, or MOV videos and assign each a subject ID.
2. Choose to analyze every frame, approximately one frame per second, or a custom sampling rate.
3. Optionally enter stimulus names and start/end times in seconds relative to each video.
4. Run extraction, inspect the plots and data preview, and download one Parquet file per video.

The interface includes detection-quality summaries, Action Unit, emotion, and head-pose plots, an Action Unit heatmap, and stimulus comparisons across recordings. Stimulus analyses include before/during summaries, event-aligned trajectories, and recording similarity.

Exports contain Py-Feat predictions and the `subject_id` and `input_file` identifiers. The time-related columns depend on sampling mode: frame-based output includes `time_sec`; the approximately 1 FPS mode aggregates observations by second.

## Code and assets

- `app.py` contains the Streamlit interface, detector setup, sampling and stimulus processing, plots, and Parquet export.
- `requirements.txt` pins the Python dependencies.
- `Au-Reference.jpg` is displayed in the in-app introduction guide.
