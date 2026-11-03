import aubio
import math
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from extract_pitch import extract_pitch

HOP_SIZE_GLOBAL = 256

def extract_onset(filename):
    buffersize = 2048
    hopsize = HOP_SIZE_GLOBAL

    source = aubio.source(filename, samplerate=0, hop_size=hopsize)
    samplerate = source.samplerate
    # print("samplerate:", samplerate)

    onset_o = aubio.onset("default", buffersize, hopsize, samplerate)
    onset_o.set_threshold(0.3)
    onset_o.set_silence(-60.0)

    onset_strengths = []
    onset_events = []
    frame=0

    while True:
        samples, read = source()

        if(read<hopsize): 
            break

        is_onset = onset_o(samples)
        strength = onset_o.get_descriptor()
        onset_strengths.append(float(strength))
        onset_events.append(bool(is_onset[0]))
        frame+=1

    return np.asarray(onset_strengths), np.asarray(onset_events)

