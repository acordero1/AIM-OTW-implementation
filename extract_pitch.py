import aubio
import math
def extract_pitch(filename):
    buffersize = 2048
    hopsize = 256
    tolerance = 0.15

    source = aubio.source(filename, samplerate=0, hop_size=hopsize)
    samplerate = source.samplerate
    # print("samplerate:", samplerate)

    pitch_o = aubio.pitch("yinfast", buffersize, hopsize, samplerate)
    pitch_o.set_unit("Hz")
    pitch_o.set_tolerance(tolerance)
    pitch_o.set_silence(-60.0)

    pitches = []
    confidences = []

    while True:
        samples, read = source()

        if(read<hopsize): 
            break

        pitch = pitch_o(samples)[0]
        confidence = pitch_o.get_confidence()
        if(confidence<0.5): 
            pitch = math.nan
        pitches.append(pitch)
        confidences.append(confidence)

    # print(pitches)
    # print(confidences)
    return pitches
