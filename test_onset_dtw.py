import numpy as np
import librosa
import matplotlib.pyplot as plt

from extract_pitch import extract_pitch
from extract_onset import extract_onset
from otw import OTW


def extract_features(file):
    pitches = np.asarray(extract_pitch(file), dtype=float).reshape(-1)
    onsets = np.asarray(extract_onset(file), dtype=float).reshape(-1)
    #match lengths
    if(len(onsets)>len(pitches)):
        onsets = onsets[:len(pitches)]
    elif(len(onsets)<len(pitches)):
        onsets = np.pad(onsets, (0, len(pitches) - len(onsets)))
    #normalize onset strngth
    onsets = np.where(np.isfinite(onsets), onsets, 0.0)
    onsets = np.log1p(np.maximum(onsets, 0.0))
    scale = np.percentile(onsets, 95)
    onsets=onsets/scale
    return pitches, onsets

def run_dtw(file1, file2):
    seqA =  np.asarray(extract_pitch(file1), dtype=float)[:, None]
    seqB =  np.asarray(extract_pitch(file2), dtype=float)[None, :]
    eps = 1e-8
    finite = np.isfinite(seqA) & np.isfinite(seqB)
    both_valid = finite & (seqA > eps) & (seqB > eps)
    both_silent = finite & (seqA <= eps) & (seqB <= eps)
    C = np.full((seqA.shape[0], seqB.shape[1]), 0.2)
    C[both_silent] = 0.0
    pitch_diff = np.abs(np.log2(seqA + 1e-6) - np.log2(seqB + 1e-6))  
    C[both_valid] = pitch_diff[both_valid]
    D, pathaubio = librosa.sequence.dtw(C=C)
    pathaubio = pathaubio[::-1]
    return D, pathaubio


def run_otw_mode(ref_features, live_features, cost_mode):
    otw = OTW(ref_features, c=100, max_run_count=3, cost_mode=cost_mode,)
    for u in live_features:
        otw.update(u)
    path = np.asarray(otw.get_path(), dtype=int)

    print(f"\nOTW mode: {cost_mode}")
    print("Reference length:", len(ref_features))
    print("Live length:", len(live_features))
    print("Final t:", otw.t)
    print("Final j:", otw.j)
    print("Path length:", len(path))

    return path




def plot_otw_dtw_comparison(D_pitch, dtw_pitch_path, otw_paths, live_length, ref_length,):
    modes = [
        ("pitch", "OTW Pitch vs Pitch DTW"),
        ("onset", "OTW Onset vs Pitch DTW"),
        ("combined", "OTW Combined vs Pitch DTW"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    for ax, (mode, title) in zip(axes, modes):
        image = ax.imshow(D_pitch, origin="lower", aspect="auto", cmap="viridis", interpolation="nearest")
        ax.plot( dtw_pitch_path[:, 1], dtw_pitch_path[:, 0], color="cyan", linewidth=1.5, label="Offline DTW (pitch)")
        path = otw_paths[mode]
        ax.plot(path[:, 1], path[:, 0], color="red", linewidth=1.2, label=f"OTW ({mode})")
        ax.set_title(title)
        ax.set_xlabel("Reference index j")
        ax.set_xlim(0, max(ref_length - 1, 1))
        ax.set_ylim(0, max(live_length - 1, 1))
        ax.grid(False)
        ax.legend(loc="best")
    axes[0].set_ylabel("Live index t")

    fig.suptitle("OTW paths compared with offline pitch DTW", y=1.02)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Main comparison
# ------------------------------------------------------------

def plot_otw_dtw(file1, file2):
    live_pitch, live_onset = extract_features(file1)
    ref_pitch, ref_onset = extract_features(file2)
    print("Reference length:", len(ref_pitch))
    print("Live length:", len(live_pitch))

    #offline dtw
    D_pitch, dtw_pitch_path = run_dtw(file1, file2)
    print("Offline DTW done")
    print("DTW path start (live, ref):", dtw_pitch_path[0])
    print("DTW path end (live, ref):", dtw_pitch_path[-1])

    #otw
    pitch_path = run_otw_mode(ref_pitch, live_pitch, cost_mode="pitch")
    onset_path = run_otw_mode(ref_onset, live_onset, cost_mode="onset")

    ref_combined = np.column_stack((ref_pitch, ref_onset))
    live_combined = np.column_stack((live_pitch, live_onset))
    combined_path = run_otw_mode(ref_combined, live_combined, cost_mode="combined")
    
    #plotting
    otw_paths = {"pitch": pitch_path, "onset": onset_path, "combined": combined_path}
    plot_otw_dtw_comparison(D_pitch, dtw_pitch_path, otw_paths, live_length=len(live_pitch), ref_length=len(ref_pitch))

    return {
        "D_pitch": D_pitch,
        "dtw_pitch_path": dtw_pitch_path,
        "otw_pitch_path": pitch_path,
        "otw_onset_path": onset_path,
        "otw_combined_path": combined_path,
    }



if __name__ == "__main__":
    plot_otw_dtw(
        r"oboe_audios/Etude1_mixed.wav",
        r"oboe_audios/Etude1_normal.wav",
    )

    # plot_otw_dtw(
    #     r"oboe_audios/Etude2_fast.wav",
    #     r"oboe_audios/Etude2_normal.wav",
    # )