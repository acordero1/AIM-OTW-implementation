import numpy as np
import aubio
import librosa
import matplotlib.pyplot as plt
from extract_pitch import extract_pitch
FILE = "C major scale.wav"
BUFFER = 2048   # must match extract_pitch
HOP = 256       # must match extract_pitch

# aubio
aubio_f0 = np.array(extract_pitch(FILE))
aubio_f0[aubio_f0 == 0.0] = np.nan          # treat 0 as unvoiced for comparison

# librosa
y, sr = librosa.load(FILE, sr=None)

yin_f0 = librosa.yin(
    y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"),
    sr=sr, frame_length=BUFFER, hop_length=HOP,
)

pyin_f0, voiced_flag, voiced_prob = librosa.pyin(
    y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"),
    sr=sr, frame_length=BUFFER, hop_length=HOP,
)  # unvoiced frames come back as NaN


n = min(len(aubio_f0), len(yin_f0), len(pyin_f0))
aubio_f0, yin_f0, pyin_f0 = aubio_f0[:n], yin_f0[:n], pyin_f0[:n]
t = np.arange(n) * HOP / sr

# plot
plt.figure(figsize=(12, 5))
plt.plot(t, aubio_f0, label="aubio yinfast", lw=1)
plt.plot(t, yin_f0, label="librosa yin", lw=1, alpha=0.7)
plt.plot(t, pyin_f0, label="librosa pyin", lw=1, alpha=0.7)
plt.xlabel("Time (s)"); plt.ylabel("Hz"); plt.legend()
plt.show()

# compare in cents: (100 cents = 1 semitone)
def cents_diff(a, b):
    return 1200 * np.log2(a / b)

mask = ~np.isnan(aubio_f0) & ~np.isnan(pyin_f0)
d = cents_diff(aubio_f0[mask], pyin_f0[mask])
print(f"aubio vs pyin: median |err| = {np.median(np.abs(d)):.1f} cents, "
      f"voiced overlap = {mask.mean():.0%}") #voiced overlap: how many of the frames are voiced in both methods
#LIMITATION: this compares frames across both methods, which optimizes for similarity, not necessarily truth. This is simply comparing to librosa's method.