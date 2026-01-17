from audio_loader import AudioLoader
from pitch_detector import PitchDetector
from visualizer import Visualizer

loader = AudioLoader()
detector = PitchDetector()
viz = Visualizer()

y, sr = loader.load("voice.wav")
time, pitch = detector.detect(y, sr)
viz.plot_pitch(time, pitch)
