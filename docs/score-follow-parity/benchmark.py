"""Benchmark this repo's OTW (and, read-only, score-follow's PitchFollower) against truth csvs.

Run from the repo root:  python docs/score-follow-parity/benchmark.py [--sf]
Error = |predicted ref time - truth ref time| at each live frame, both on the same time convention.
"""
import glob, os, sys, time
import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from extract_pitch import HOP_SIZE_GLOBAL, extract_pitch
from otw import OTW

SF = "/Users/jonathon/Projects/score-follow/packages"


def truth_at(truth_path, live_times):
    tt = np.loadtxt(truth_path, delimiter=",", skiprows=1)
    return np.interp(live_times, tt[:, 0], tt[:, 1])


def report(name, err, secs, nframes):
    print(f"{name:20s} mean={1000*err.mean():7.0f}ms median={1000*np.median(err):6.0f}ms "
          f"<250ms={100*np.mean(err<.25):5.1f}%  {1000*secs/nframes:.3f} ms/frame")
    return err


def run_local(ref_path, live_path, truth_path):
    sr = sf.info(ref_path).samplerate
    ref, live = extract_pitch(ref_path), extract_pitch(live_path)
    t0 = time.perf_counter()
    tr = OTW(ref, max_run_count=3, c=100)
    pos = np.array([tr.update(u) for u in live])
    dt = time.perf_counter() - t0
    lt = np.arange(len(live)) * HOP_SIZE_GLOBAL / sr
    return np.abs(pos * HOP_SIZE_GLOBAL / sr - truth_at(truth_path, lt)), dt, len(live)


def run_sf(ref_path, live_path, truth_path):
    sys.path.insert(0, SF + "/dixon-pitch")
    from dixon_pitch import PitchFollower
    x, sr = sf.read(live_path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    f = PitchFollower(ref_path)
    t0 = time.perf_counter()
    pos = f.run(x)
    dt = time.perf_counter() - t0
    lt = (np.arange(len(pos)) * f.hop + f.hop - f.extractor.win / 2) / sr
    return np.abs(pos / sr - truth_at(truth_path, lt)), dt, len(pos)


if __name__ == "__main__":
    runners = [("this repo", run_local)] + ([("score-follow pitch", run_sf)] if "--sf" in sys.argv else [])
    allerr = {n: [] for n, _ in runners}
    for rp in sorted(glob.glob(os.path.join(ROOT, "audio", "*_ref.wav"))):
        name = os.path.basename(rp)[:-8]
        lp, tp = rp.replace("_ref", "_live"), rp.replace("_ref.wav", "_truth.csv")
        print(name)
        for rn, fn in runners:
            e, dt, n = fn(rp, lp, tp)
            allerr[rn].append((e, dt, n))
            report("  " + rn, e, dt, n)
    print("\nOVERALL (pooled frames; mean of per-case medians)")
    for rn, _ in runners:
        e = np.concatenate([a for a, _, _ in allerr[rn]])
        med = np.mean([np.median(a) for a, _, _ in allerr[rn]])
        dt, n = sum(d for _, d, _ in allerr[rn]), sum(n for _, _, n in allerr[rn])
        print(f"{rn:20s} mean={1000*e.mean():7.0f}ms case-median-avg={1000*med:6.0f}ms "
              f"<250ms={100*np.mean(e<.25):5.1f}%  {1000*dt/n:.3f} ms/frame")
