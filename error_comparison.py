import glob
import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from extract_pitch import HOP_SIZE_GLOBAL, extract_pitch
from otw import OTW

AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio")

# testing our algo from otw.py
def run_otw(ref, live):
    tracker = OTW(ref, max_run_count=3, c=100)
    arr = []
    for entry in live:
        arr.append(tracker.update(entry))

    return np.array(arr)

# haven't checked our dtw yet so this is librosa's dtw
def run_offline_dtw(ref, live):
    u = np.array(live)[:, None] # live is col and ref is row such that every live frame an be compared with every reference frame
    v = np.array(ref)[None, :]
    cost = np.abs(np.log2(u + 1e-6) - np.log2(v + 1e-6))
    cost = np.where(np.isnan(u) | np.isnan(v), 0.2, cost)
    path = librosa.sequence.dtw(C=cost, step_sizes_sigma=np.array([[1, 1], [0, 1], [1, 0]]))[1]
    best_ref_frame = {}

    for live_frame, ref_frame in path[::-1]:
        best_ref_frame[live_frame] = ref_frame

    arr = []

    for i in range(len(live)):
        arr.append(best_ref_frame[i])

    return np.array(arr)


def print_error(label, err): #error that quantifies gap between  position of tracker vs position of truth
    print(f"{label}: mean={1000 * err.mean()} ms, median={1000 * np.median(err)} ms, max={1000 * err.max()} ms")


def plot_results(results):
    cols = 3
    rows = (len(results) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for ax, (name, (live_times, otw, offline, truth)) in zip(axes.flat, results.items()):
        ax.plot(live_times, truth, "k", lw=2, label="truth")
        ax.plot(live_times, offline, "g--", label="offline DTW")
        ax.plot(live_times, otw, "r", label="OTW")
        ax.set_title(name)
        ax.set_xlabel("live time (s)")
        ax.set_ylabel("reference position (s)")

    for ax in axes.flat[len(results):]:
        ax.axis("off")
    axes.flat[0].legend()
    fig.tight_layout()
    # fig.savefig("results.png", dpi=110)
    # print("\nSaved results to results.png")
    
    plt.show()


def main():
    results = {}
    for ref_path in sorted(glob.glob(os.path.join(AUDIO_DIR, "*_ref.wav"))):
        name = os.path.basename(ref_path).replace("_ref.wav", "")
        live_path = os.path.join(AUDIO_DIR, name + "_live.wav")
        truth_path = os.path.join(AUDIO_DIR, name + "_truth.csv")

        seconds_per_frame = HOP_SIZE_GLOBAL / sf.info(ref_path).samplerate
        ref_pitches = extract_pitch(ref_path)
        live_pitches = extract_pitch(live_path)

        otw = run_otw(ref_pitches, live_pitches) * seconds_per_frame
        offline = run_offline_dtw(ref_pitches, live_pitches) * seconds_per_frame

        live_times = np.arange(len(live_pitches)) * seconds_per_frame
        truth_table = np.loadtxt(truth_path, delimiter=",", skiprows=1)
        truth = np.interp(live_times, truth_table[:, 0], truth_table[:, 1])

        print(name)
        print_error("OTW vs truth", np.abs(otw - truth))
        print_error("offline DTW vs truth", np.abs(offline - truth))
        results[name] = (live_times, otw, offline, truth)

    plot_results(results)


if __name__ == "__main__":
    main()
