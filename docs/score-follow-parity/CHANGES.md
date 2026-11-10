# Closing the gap with score-follow's `dixon-pitch`

Reference (read-only): `/Users/jonathon/Projects/score-follow/packages/dixon-pitch/dixon_pitch.py`
(`OnlineTimeWarp`, `_pitch_cost`, `_decide`, `step`). Only `otw.py` was changed in this repo.
Feature extraction (`extract_pitch.py`), `c=100`, `max_run_count=3` and the public API
(`OTW(...).update(u) -> j`, `get_path()`) are unchanged.

## Results

`benchmark.py` runs every `audio/*_ref/_live` pair against its `_truth.csv`. Error is
|predicted reference time - truth reference time| per live frame. Raw output is in
`baseline.txt` (before) and `after.txt` (after). Environment: Python 3.11, numpy 1.26,
numba installed (score-follow uses it; this repo does not need it).

| | mean err | mean of per-case medians | within 250 ms | time per frame |
|---|---|---|---|---|
| this repo, before | 3621 ms | 1983 ms | 7.7% | 0.52 ms |
| this repo, after | 202 ms | 83 ms | 78.5% | 0.13 ms |
| score-follow `PitchFollower` | 241 ms | 92 ms | 74.4% | 0.07 ms |

Per case (see `after.txt`), this repo has the lower median error in 7 of 9 cases. It is
slightly worse on `arpeggios` (median 20 vs 10 ms, both 100% within 250 ms) and
`melody_repeats` (median 81 vs 46 ms, 92.1% vs 92.9% within 250 ms). Speed is still behind
score-follow (see below).
Caveats: one set of 9 clips with a truth that is note-onset interpolated; the score-follow
hop is 512 samples vs 256 here, so "per frame" timings are not per audio second
(0.13 ms per 5.8 ms of audio here, 0.07 ms per 11.6 ms there, with the first case in
score-follow including numba compilation).

## Changes (all in `otw.py`)

### 1. Every live frame is consumed (the main fix)
*Was:* a "column" decision advanced only the reference pointer `j` and left the new live frame in
`pending_live`. Each column step grew the backlog by one, so `t` fell further behind the live
stream and the reported `j` described old audio. This is the large lag (seconds) in `baseline.txt`.
*Now:* `update()` treats a "column" decision as "both" for consuming the new frame
(`decided` vs `inc`), as score-follow's `step` does (`_add_row` always runs; `_add_column`
runs when `inc != ROW`).
*Alone this helped but was not enough* (median 1983 -> 740 ms), see 2 and 3.

### 2. Run-count bookkeeping uses the decided move
*Why:* after change 1 a forced-row column step would be labelled "both" and reset the run
counter, so `max_run_count` never limited column runs. The label now stays "column" when the
decision was column (score-follow `_record(inc)` with the decided `inc`).

### 3. Reference catches up without new input
*Why:* with `max_run_count=3` the reference can only move 3 columns per live frame
otherwise, so it cannot keep up after a rest or a fast passage. After the normal step,
`update()` loops `add_column()` while `get_inc() == "column"`, the same as the
`while self._decide() == COLUMN` loop in score-follow's `step`. The run-count rule ends the loop.
*Measured after 1+2+3:* 202 ms mean error, 78.5% within 250 ms (before: 3621 ms, 7.7%).

### 4. Pitch cost follows score-follow's `_pitch_cost`
*Was:* `|log2(u) - log2(v)|` in octaves, unbounded; NaN on either side cost 0.2 (including
unvoiced vs unvoiced, which is a perfect match).
*Now (`cost`):* semitone distance saturating at 6, divided by 6 (range 0-1); voiced vs unvoiced = 1;
unvoiced vs unvoiced = 0. Unbounded distance let a single octave error dominate a path, and
charging 0.2 for matched silence made rests look like mismatches.
*Note:* on its own this change made accuracy worse (median 1983 -> 2655 ms) while the lag bug
existed; it pays off only together with 1-3. I kept it because it matches score-follow and is
the better-defined cost, not because the benchmark isolates it. I did not run the ablation of
1-3 without 4.

### 5. Speed (no change in output)
Accuracy numbers were identical before and after each step below (202 ms / 83 ms / 78.5%).
- `cost` is vectorised over a whole new row or column (was one Python call per cell, 943k calls
  on Etude1). 0.71 -> 0.29 ms/frame.
- `get_inc` scans the frontier row/column with one numpy `argmin` each instead of a Python loop of
  `path_cost` calls (`norm_cost` helper). `argmin` returns the first minimum, the same tie rule
  as the old strict `<` loop. `path_cost` is left in place but is no longer used.
- `evaluate_path_cost` takes the precomputed cell cost and picks the predecessor with scalar
  compares instead of building a list and calling `min` with a lambda (same tie order: left, up,
  diagonal). 0.29 -> 0.13 ms/frame.
- `add_column()` helper replaces the duplicated column-fill code.

## Differences from score-follow deliberately not adopted
Each would be a bigger change and accuracy already matches, so they were left out:
- start gate (`start_on_active`: hold at the first voiced reference frame until the first voiced live frame),
- symmetric weights (diagonal costs 2x) and `(t - oi) + (k - ok) + 2` normalisation (this repo
  normalises by tracked path length `L`),
- ring-buffer window and numba kernels (the remaining ~2x speed gap per frame),
- search width 256 and hop 512 (here `c=100`, hop 256),
- yinfft pitch in MIDI (here yinfast in Hz).

## Behaviour notes
- `get_path()` now has more rows than live frames because catch-up column moves add rows.
  `update()` still returns one `j` per live frame, so `run_otw.py` and `error_comparison.py` work unchanged.
- `evaluate_path_cost(t, j, u)` now takes the cell cost, not the live pitch. Nothing else in the repo calls it.

## Reproduce
```
python docs/score-follow-parity/benchmark.py        # this repo
python docs/score-follow-parity/benchmark.py --sf   # also score-follow dixon-pitch (read-only import)
```
`aubio` 0.4.9 builds on Apple clang only with numpy<2 and
`CFLAGS="-Wno-error=incompatible-function-pointer-types" pip install --no-build-isolation aubio`.
