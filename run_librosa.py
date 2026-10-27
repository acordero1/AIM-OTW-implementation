import librosa
import numpy as np
from dtw import dtw 

def librosa_pitch(filename1, filename2):
    x1, sr1 = librosa.load(filename1, sr=None)
    x2, sr2 = librosa.load(filename2, sr=None)

    f01, voiced_flag, voiced_prob = librosa.pyin(x1, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
        sr=sr1, frame_length=2048, hop_length=256, center=False)
    f01 = np.nan_to_num(f01)

    f02, voiced_flag, voiced_prob = librosa.pyin(x2, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
            sr=sr2, frame_length=2048, hop_length=256, center=False)
    f02 = np.nan_to_num(f02)

    C = np.abs(f01[:, None] - f02[None, :])
    D, path = librosa.sequence.dtw(C=C)
    path=np.array(path)
    return C, D, path

   
