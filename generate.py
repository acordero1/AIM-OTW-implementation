import os

import numpy as np
import soundfile as sf

SR = 44100
OUT = "audio"
os.makedirs(OUT, exist_ok=True)


def tone(midi, length, timbre, detune_cents=0.0, vibrato=False):#note
    n = int(length * SR)
    t = np.arange(n) / SR
    freq = 440.0 * 2 ** ((midi - 69 + detune_cents / 100.0) / 12)
    phase = 2 * np.pi * freq * t

    if vibrato:
        phase = phase + 0.35 * np.sin(2 * np.pi * 5.5 * t) * np.minimum(1, t / 0.3)

    if timbre == "plucked":
        harmonics = [1.0, 0.5, 0.3, 0.15, 0.08]
        wave = sum(a * np.sin(h * phase) for h, a in enumerate(harmonics, start=1))
        env = np.exp(-t * 3.5)
    elif timbre == "bowed":
        wave = sum(np.sin(h * phase) / h for h in range(1, 12))
        env = np.minimum(1, t / 0.08)
    elif timbre == "oboe":
        wave = sum(a * np.sin(h * phase) for h, a in enumerate([1.0, 0.6, 0.7, 0.4, 0.25, 0.15], start=1))
        env = np.minimum(1, t / 0.03)
    else:
        wave = sum(a * np.sin(h * phase) for h, a in enumerate([1.0, 0.4, 0.2], start=1))
        env = np.ones(n)

    attack = np.minimum(1, t / 0.008)
    release = np.clip((length - t) / 0.02, 0, 1)

    return (wave * env * attack * release).astype(np.float32)


def render(notes, durations, timbre, rng, gap=0.04, detune_sd=0.0, vibrato=False, noise=0.0): #melody
    pieces = []
    starts = []
    now = 0.0

    for midi, length in zip(notes, durations):
        starts.append(now)
        n = int(length * SR)

        if midi is None:
            pieces.append(np.zeros(n, dtype=np.float32))
        else:
            detune = rng.normal(0, detune_sd) if detune_sd else 0.0
            piece = tone(midi, length, timbre, detune, vibrato)
            piece[max(0, int((length - gap) * SR)):] = 0.0
            pieces.append(piece)

        now += length

    starts.append(now)

    audio = np.concatenate(pieces)
    audio = 0.5 * audio / max(1e-9, np.abs(audio).max())

    if noise:
        audio = audio + rng.normal(0, noise, len(audio))

    return audio.astype(np.float32), np.array(starts)


def tempo_factors(n, kind, rng): #random stretcher/squeezer to simulate human playing
    k = np.arange(n)

    if kind == "slower":
        base = np.linspace(1.0, 1.3, n)
    elif kind == "faster":
        base = np.linspace(1.0, 0.75, n)
    elif kind == "wobble":
        base = 1.0 + 0.2 * np.sin(k / 5.0)
    elif kind == "late_rit":
        base = np.where(k < 0.7 * n, 1.0, 1.0 + 0.8 * (k - 0.7 * n) / (0.3 * n))
    else:
        base = np.ones(n)

    return base * (1 + rng.normal(0, 0.04, n))


C_MAJOR = [60, 62, 64, 65, 67, 69, 71, 72]

PIECES = {
    "scale": dict(notes=C_MAJOR + C_MAJOR[::-1][1:], beats=[1] * 15, spb=0.5, timbre="sustained", gap=0.05, live=dict(tempo="slower", detune=10, noise=0.003)),
    "spaced_notes": dict(notes=[67, None, 72, None, 64, None, 69, None, 71, None, 60, None, 65, None, 74, None], beats=[1, 1.2, 1, 0.8, 1.5, 1.0, 1, 1.5, 1, 1, 2, 0.7, 1, 1.2, 1.5, 0.5], spb=0.45, timbre="plucked", gap=0.03, live=dict(tempo="wobble", detune=15, noise=0.004)),
    "melody_repeats": dict(notes=[64, 64, 65, 67, 67, 65, 64, 62, 60, 60, 62, 64, 64, 62, 62, 64, 64, 65, 67, 67, 65, 64, 62, 60, 60, 62, 64, 62, 60, 60], beats=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.5, 0.5, 2] * 2, spb=0.4, timbre="sustained", gap=0.05, live=dict(tempo="faster", detune=10, noise=0.004)),
    "arpeggios": dict(notes=[60, 64, 67, 72, 67, 64, 60, 65, 69, 72, 77, 72, 69, 65, 62, 67, 71, 74, 71, 67, 62, 60, 64, 67, 72, 76, 79, 76, 72, 67], beats=[0.5] * 30, spb=0.4, timbre="plucked", gap=0.03, live=dict(tempo="wobble", detune=12, noise=0.005)),
    "cello_range": dict(notes=[48, 55, 52, 50, 55, 59, 57, 55, 53, 48, 50, 52, 55, 60, 59, 55], beats=[2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 3], spb=0.5, timbre="bowed", gap=0.06, live=dict(tempo="late_rit", detune=20, noise=0.004, vibrato=True)),
    "mixed_rests": dict(notes=[60, 62, 64, None, 67, 67, None, 72, 71, 69, None, None, 65, 64, 62, None, 60, 64, 67, 72], beats=[0.5, 0.5, 1, 1, 0.5, 0.5, 1.5, 1, 0.5, 1, 1, 1, 0.5, 0.5, 1, 1.5, 1, 1, 1, 2], spb=0.4, timbre="bowed", gap=0.04, live=dict(tempo="slower", detune=15, noise=0.005, vibrato=True)),
    "score1_cantabile": dict(notes=[65, 77, 76, 79, 77, 71, 74, 72, 80, 82, 81, 84, 72, 82, 81, 79, 77, 76, 74, 77, 74, 73, None, 72, 74, 76, 77, 79, 80, 81, 82, 81, 84, None, 69, 70, 72, 74, 76, 77, 79, 81, 79, 82, None, 86, 84, 80, 82, 81, 76, 79, 77, 71, 74, 72, 68, 70, 69, 64, 65], beats=[1, 1.5, 0.5, 0.5, 0.5, 0.3333, 0.3333, 1.8333, 0.5, 0.5, 0.5, 1, 1.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 1.5, 0.25, 0.25, 2, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 1, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 1, 0.5, 0.5, 0.5, 0.5, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 2], spb=60 / 72, timbre="oboe", gap=0.04, live=dict(tempo="slower", detune=10, noise=0.004, vibrato=True)),
    "score2_andante": dict(notes=[67, 70, 72, 74, 79, 74, 82, 81, 79, 78, 79, 78, 79, 78, 79, 78, 79, 76, 78, 79, 81, 72, 75, 74, 69, 70, 72, 70, 72, 70, 72, 70, 72, 69, 70, 72, 74, 69, 74, 77, 75, 74, 79, 78, 79, 81, 74, 82, 81, 82, 84, 77, 86, 84, 82, 84, 82, 81, 82, 81, 79, 81, 82, 81, 79, 81, 79, 78, 79, 78, 79, 78, 79, 78, 79, 78, 79, 78, 79, 78, 79, 78, 79, 76, 78, 79, None, 77, 74, 75, 77, 82, 84, 82, 81, 79, 77, 74, 75, 77, 86, 84, 82, 81, 82, 83, 84, 75, 79, 77, 75, 74, 70, 72, 74, 79, 75, 74, 72, 75, 74, 70, 67, 69, 67, 66, 67, 70, 69, 67], beats=[0.5, 0.375, 0.125, 0.5, 0.5, 0.5, 1, 0.25, 0.25, 0.0625, 0.0625, 0.0625, 0.0625, 0.035, 0.03, 0.03, 0.03, 0.0625, 0.0625, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.0625, 0.0625, 0.0625, 0.0625, 0.035, 0.03, 0.03, 0.03, 0.0625, 0.0625, 0.25, 0.25, 0.25, 0.1875, 0.0625, 0.25, 0.25, 0.5, 0.375, 0.125, 0.5, 0.5, 0.5, 0.375, 0.125, 0.5, 0.5, 0.5, 0.375, 0.125, 0.5, 0.375, 0.125, 0.5, 0.375, 0.125, 0.5, 0.375, 0.125, 0.5, 0.375, 0.125, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.035, 0.03, 0.03, 0.03, 0.0625, 0.0625, 1, 1, 0.5, 0.375, 0.125, 0.5, 0.4375, 0.0625, 1, 0.5, 0.5, 0.5, 0.375, 0.125, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.375, 0.125, 0.5, 1, 0.375, 0.125, 0.5, 1, 0.375, 0.125, 0.25, 0.0625, 0.0625, 0.0625, 0.0625, 0.25, 0.25, 2], spb=60 / 44, timbre="oboe", gap=0.02, live=dict(tempo="slower", detune=10, noise=0.004, vibrato=True)),
}

for i, (name, p) in enumerate(PIECES.items()):
    if os.path.exists(os.path.join(OUT, f"{name}_ref.wav")):
        print(f"{name}: already exists, skipping")
        continue

    rng = np.random.default_rng(100 + i)
    ref_dur = np.array(p["beats"]) * p["spb"]
    ref, ref_starts = render(p["notes"], ref_dur, p["timbre"], rng, p["gap"])
    live_cfg = p["live"]
    live_dur = ref_dur * tempo_factors(len(ref_dur), live_cfg["tempo"], rng)
    live, live_starts = render(p["notes"], live_dur, p["timbre"], rng, p["gap"], detune_sd=live_cfg["detune"], vibrato=live_cfg.get("vibrato", False), noise=live_cfg["noise"])
    
    sf.write(os.path.join(OUT, f"{name}_ref.wav"), ref, SR)
    sf.write(os.path.join(OUT, f"{name}_live.wav"), live, SR)

    with open(os.path.join(OUT, f"{name}_truth.csv"), "w") as f:
        f.write("live_sec,ref_sec\n")
        for a, b in zip(live_starts, ref_starts):
            f.write(f"{a:.4f},{b:.4f}\n")

    print(f"{name}: ref {len(ref) / SR:.1f}s, live {len(live) / SR:.1f}s, {len(p['notes'])} events")
