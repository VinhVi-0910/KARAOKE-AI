import crepe
import numpy as np
import librosa

class PitchDetector:
    def __init__(self, confidence_threshold=0.5, crepe_sr=16000, step_size=10):
        self.confidence_threshold = confidence_threshold
        self.crepe_sr = crepe_sr
        self.step_size = step_size

    def detect(self, y, sr):
        # Resample to CREPE's expected sample rate
        if sr != self.crepe_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.crepe_sr)
            sr = self.crepe_sr

        # Ensure float32
        y = y.astype(np.float32)

        time, freq, confidence, _ = crepe.predict(
            y, sr, viterbi=True, step_size=self.step_size
        )

        freq[confidence < self.confidence_threshold] = np.nan
        return time, freq
