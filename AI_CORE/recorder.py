"""Real-time audio recording and processing.

Handles microphone input capture and streaming to processing pipeline.
"""
import numpy as np
from typing import Callable, Optional
import threading
import queue
import sounddevice as sd


class RealTimeRecorder:
    """Captures real-time audio from microphone and streams to processing callback.
    
    Usage:
        def on_chunk(chunk, sr):
            print(f"Chunk: {chunk.shape} at {sr} Hz")
        
        recorder = RealTimeRecorder(callback=on_chunk)
        recorder.start()
        # ... recording happens ...
        recorder.stop()
        full_audio = recorder.get_audio()
    """
    
    def __init__(
        self,
        callback: Optional[Callable] = None,
        sample_rate: int = 44100,
        chunk_duration: float = 0.5,
        channels: int = 1,
        device: Optional[int] = None
    ):
        """Initialize recorder.
        
        Args:
            callback: function(chunk, sr) called on each audio chunk
            sample_rate: recording sample rate (Hz)
            chunk_duration: duration of each chunk (seconds)
            channels: number of audio channels
            device: audio device index (None = default)
        """
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        self.channels = channels
        self.device = device
        self.callback = callback
        
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.stream = None
    
    def _audio_callback(self, indata, frames, time, status):
        """Callback for sounddevice stream."""
        if status:
            print(f"Recording error: {status}")
        
        # Copy audio chunk
        chunk = indata[:, 0] if self.channels == 1 else indata.copy()
        self.audio_queue.put(chunk.copy())
        
        # Call user callback if provided
        if self.callback:
            try:
                self.callback(chunk, self.sample_rate)
            except Exception as e:
                print(f"Callback error: {e}")
    
    def start(self) -> None:
        """Start recording."""
        if self.is_recording:
            return
        
        self.is_recording = True
        self.stream = sd.InputStream(
            device=self.device,
            channels=self.channels,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            callback=self._audio_callback
        )
        self.stream.start()
    
    def stop(self) -> None:
        """Stop recording."""
        if not self.is_recording:
            return
        
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
    
    def get_audio(self) -> np.ndarray:
        """Get all recorded audio as single array.
        
        Returns:
            audio array (mono)
        """
        chunks = []
        while not self.audio_queue.empty():
            try:
                chunk = self.audio_queue.get_nowait()
                chunks.append(chunk)
            except queue.Empty:
                break
        
        if chunks:
            return np.concatenate(chunks)
        return np.array([])
    
    def is_recording_now(self) -> bool:
        """Check if currently recording."""
        return self.is_recording


class AudioBuffer:
    """Thread-safe ring buffer for streaming audio."""
    
    def __init__(self, max_duration: float = 30.0, sample_rate: int = 44100):
        """Initialize buffer.
        
        Args:
            max_duration: maximum duration to keep (seconds)
            sample_rate: sample rate (Hz)
        """
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.buffer = np.zeros(self.max_samples)
        self.write_pos = 0
        self.lock = threading.Lock()
    
    def write(self, data: np.ndarray) -> None:
        """Write audio data to buffer."""
        with self.lock:
            n_samples = len(data)
            
            # Handle wrap-around
            if self.write_pos + n_samples <= self.max_samples:
                self.buffer[self.write_pos:self.write_pos + n_samples] = data
            else:
                # Split into two parts
                first_part = self.max_samples - self.write_pos
                self.buffer[self.write_pos:] = data[:first_part]
                self.buffer[:n_samples - first_part] = data[first_part:]
            
            self.write_pos = (self.write_pos + n_samples) % self.max_samples
    
    def read(self, duration: float) -> np.ndarray:
        """Read most recent N seconds from buffer.
        
        Args:
            duration: duration to read (seconds)
        
        Returns:
            audio array
        """
        with self.lock:
            n_samples = min(int(duration * self.sample_rate), self.max_samples)
            
            if self.write_pos >= n_samples:
                return self.buffer[self.write_pos - n_samples:self.write_pos].copy()
            else:
                # Wrap around
                part1 = self.buffer[self.write_pos - n_samples:]
                part2 = self.buffer[:self.write_pos]
                return np.concatenate([part1, part2])
