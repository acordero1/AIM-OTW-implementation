from extract_pitch import extract_pitch
from otw import OTW
import numpy as np

def run_otw(file, ref_file):
    ref_pitches = extract_pitch(ref_file)
    live_pitches = extract_pitch(file)

    # live_pitches = np.interp( #test if length mismatch is issue
    # np.linspace(0, 1, len(ref_pitches)),
    # np.linspace(0, 1, len(live_pitches)),
    # live_pitches
    # )

    otw = OTW(ref_pitches, c=100, max_run_count=3)
    print(len(live_pitches))
    frames = []
    costs=[]
    acc_costs=[]
    for u in live_pitches:
        j = otw.update(u)
        frames.append(j)
        costs.append(otw.D_curr.copy())
        acc_costs.append(otw.D_prev.copy())
    path = otw.get_path()

    print("len_ref =", len(ref_pitches), "final_j =", otw.get_path()[-1,1], "final_t =", otw.get_path()[-1,0])
    p = np.asarray(otw.get_path())
    # atomic moves are successive differences
    d = np.diff(p, axis=0)
    col_moves = np.sum(d[:,1] > 0)
    row_moves = np.sum(d[:,0] > 0)
    both_moves = np.sum((d[:,0] > 0) & (d[:,1] > 0))
    print("col,row,both:", col_moves, row_moves, both_moves, "len(path):", len(p))

    # compute run lengths of same inc
    incs=[]
    for dd in d:
        if dd[1]>0 and dd[0]==0: incs.append("col")
        elif dd[0]>0 and dd[1]==0: incs.append("row")
        elif dd[0]>0 and dd[1]>0: incs.append("both")
    from collections import Counter
    print(Counter(incs))
    # run-lengths
    runs=[]
    cur=incs[0]; cnt=1
    for x in incs[1:]:
        if x==cur: cnt+=1
        else:
            runs.append((cur,cnt)); cur=x; cnt=1
    runs.append((cur,cnt))
    # show top long runs
    print(sorted([r for r in runs if r[1]>5], key=lambda x:-x[1])[:10])

        # after run_otw
    print("len_ref =", len(ref_pitches), "len_live =", len(live_pitches))
    p = otw.get_path()
    d = np.diff(p, axis=0)
    col_moves = np.sum(d[:,1] > 0)
    row_moves = np.sum(d[:,0] > 0)
    print("final_j, final_t =", p[-1,1], p[-1,0])
    print("col,row,both:", col_moves, row_moves, np.sum((d[:,0]>0)&(d[:,1]>0)), "len(path):", len(p))


    return live_pitches, frames, costs, acc_costs, path

