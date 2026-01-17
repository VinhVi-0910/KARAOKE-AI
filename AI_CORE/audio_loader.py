import librosa
import numpy as np

class AudioLoader:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate

    def load(self, path):
        y, sr = librosa.load(path, sr=self.sample_rate, mono=True)
        y = librosa.util.normalize(y)
        return y, sr
