from run_otw import run_otw
from run_librosa import librosa_pitch
import numpy as np
import matplotlib.pyplot as plt
from extract_pitch import extract_pitch
from dtw import dtw

def plot_otw_dtw(file1, file2):
    pitches, frames, costs, acc_costs, path = run_otw(file1, file2)
    Cmat = np.vstack(costs)
    Dmat = np.vstack(acc_costs)
    otw_path = np.array(path)
    print("OTW done")
    print(len(path))


    # libC, libD, path= librosa_pitch(file1, file2)
    # print("DTW done")

    seqA=extract_pitch(file1)
    seqB=extract_pitch(file2)
    C, D, pathaubdio=dtw(seqA, seqB)
    pathaubdio=np.array(pathaubdio)

    plt.imshow(D, origin="lower", aspect="auto", cmap="viridis")
    plt.plot(otw_path[:,1], otw_path[:,0], color="red")
    # plt.plot(path[:,1], path[:,0], color="cyan", label="DTW")
    plt.plot(pathaubdio[:,1], pathaubdio[:,0], color="green", label="DTW")
    plt.title("OTW Cost Matrix")
    plt.legend
    plt.xlabel("Reference Index j")
    plt.ylabel("Time Index t")
    plt.show()
    return

plot_otw_dtw("1.wav", "2.wav")
