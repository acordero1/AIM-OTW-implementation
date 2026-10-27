from run_otw import run_otw
from run_librosa import librosa_pitch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from extract_pitch import extract_pitch
from dtw import dtw

def plot_otw_dtw(file1, file2):
    pitches, frames, costs, path = run_otw(file1, file2)
    Cmat = np.vstack(costs)
    otw_path = np.array(path)
    print("OTW done")
    print(len(path))


    # libC, libD, path= librosa_pitch(file1, file2)
    print("DTW done")

    seqA=np.asarray(extract_pitch(file1))
    seqB=np.asarray(extract_pitch(file2))

    # C, D, pathaubdio=dtw(seqA, seqB)
    # pathaubdio=np.array(pathaubdio)
    seqA = np.array(seqA)[:, None] # live is col and ref is row such that every live frame an be compared with every reference frame
    seqB = np.array(seqB)[None, :]
    C = np.abs(np.log2(seqA + 1e-6) - np.log2(seqB + 1e-6))
    C = np.where(np.isnan(seqA) | np.isnan(seqB), 0.2, C)  
    D, pathaubio = librosa.sequence.dtw(C=C)
    pathaubio = pathaubio[::-1]

    seqB=seqB.T
    print("Reference length:", len(seqA))
    print("Live length:", len(seqB))

    print("DTW path start:", pathaubio[0])
    print("DTW path end:", pathaubio[-1])

    print("Expected endpoint:", (len(seqA) - 1, len(seqB) - 1))

    plt.imshow(D, origin="lower", aspect="auto", cmap="viridis")
    plt.plot(otw_path[:,1], otw_path[:,0], color="red", label="OTW")
    # plt.plot(path[:,1], path[:,0], color="cyan", label="Librosa")
    plt.plot(pathaubio[:,1], pathaubio[:,0], color="cyan", label="DTW")
    plt.title("OTW vs DTW Cost Matrix")
    plt.legend()
    plt.xlabel("Reference Index j")
    plt.ylabel("Time Index t")
    plt.show()
    return

if __name__ == "__main__":
    plot_otw_dtw(r"oboe_audios\Etude1_normal.wav", r"oboe_audios/Etude1_mixed.wav")
    # plot_otw_dtw(r"oboe_audios\Etude2_normal.wav", r"oboe_audios\Etude2_fast.wav")

