import matplotlib.pyplot as plt

class Visualizer:
    def plot_pitch(self, time, pitch):
        plt.figure(figsize=(12, 4))
        plt.plot(time, pitch)
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.show()
