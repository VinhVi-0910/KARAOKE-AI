import librosa
import numpy as np

class AutoTuner:
    def tune(self, y, sr, pitch_hz):
        notes = librosa.hz_to_midi(pitch_hz)
        corrected = np.round(notes)
        shift = np.nan_to_num(corrected - notes)

        tuned = librosa.effects.pitch_shift(
            y, sr, n_steps=np.mean(shift)
        )
        return tuned
