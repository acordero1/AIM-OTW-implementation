# AIM OTW implementation

Online Time Warping (Dixon, 2005) with aubio pitch features, for following a live performance through a reference recording.

## Files
- `otw.py`: the OTW tracker (`OTW` class). Pass `verbose=True` for per-step debug output.
- `extract_pitch.py`: pitch per frame with aubio (`HOP_SIZE` = 256 samples).
- `dtw.py`, `run_otw.py`, `run_librosa.py`, `plot_comparison.py`, `compare.py`: earlier experiments and plots.
- `evaluate.py`: runs OTW on every test pair in `audio/`, compares it to offline DTW and to a known truth, prints the errors and saves `otw_results.png`.
- `sheet_to_audio.py`: turns a score (MusicXML or a text note list) in `scores/` into a reference/live audio pair plus a truth file in `audio/`.

## Setup
aubio does not build on Python 3.14 and crashes with NumPy 2. Use Python 3.11 and `numpy<2`.

    python3.11 -m venv venv && source venv/bin/activate
    pip install -r requirements.txt

## Evaluate
    python evaluate.py

Every pair in `audio/` needs all three files: `<name>_ref.wav`, `<name>_live.wav` and `<name>_truth.csv`. The truth file lists note start
times in the live audio and the matching times in the reference. In the graphs the black line is the truth, the green
dashed line is offline DTW (librosa, using the same cost as `OTW.cost`) and the red line is OTW.

## Making test audio
Edit the settings at the top of `sheet_to_audio.py` (`SCORE_FILE`, `NAME`, `REF_BPM`, `LIVE_BPM`, `BEAT_UNIT`,
`START_MEASURE`, `END_MEASURE`) and run `python sheet_to_audio.py`. Trills and grace notes in MusicXML are played.
