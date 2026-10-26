# AIM OTW implementation

We are implementing largely based from the the Dixon paper. We intend to follow a live performance through a reference recording.

Note that aubio does not build on Python 3.14 and crashes with NumPy 2, so we used Python 3.11 and `numpy<2`.

Currently every pair in `audio/` has `<name>_ref.wav`, `<name>_live.wav` and `<name>_truth.csv`. Truth csv matches locations in ref and truth (note start times in the live audio matched with times in the reference)

In the error graphs the black line is the truth, the green dashed line is offline DTW and the red line is OTW.
