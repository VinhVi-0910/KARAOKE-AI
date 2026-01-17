#!/usr/bin/env python3
"""Test script for AI_CORE pipeline using synthetic audio."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import librosa
import tempfile
import scipy.io.wavfile
from .audio_loader import AudioLoader
from .pitch_detector import PitchDetector
from .scorer import Scorer
from .visualizer import Visualizer
from .engine import Engine

def create_synthetic_audio(duration=2.0, sr=44100, freq=440.0):
    """Create a simple sine wave audio for testing."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * freq * t)
    return y, sr

def main():
    print("Testing AI_CORE pipeline with synthetic audio...")

    # Create synthetic audio (A4 note, 440 Hz)
    y, sr = create_synthetic_audio()
    print(f"Created synthetic audio: {len(y)} samples at {sr} Hz")

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_path = f.name
        scipy.io.wavfile.write(temp_path, sr, (y * 32767).astype(np.int16))  # Convert to int16

    try:
        # Test Engine (full pipeline)
        engine = Engine()
        result = engine.analyze(temp_path, visualise=False)
        print(f"Engine result: time shape {result['time'].shape}, pitch shape {result['pitch'].shape}")
        print(f"Mean detected pitch: {np.nanmean(result['pitch']):.2f} Hz")

        # Test scoring with target
        target_pitch = np.full_like(result['pitch'], 440.0)
        score = Scorer().score_pitch(result['pitch'], target_pitch)
        print(f"Score against 440 Hz target: {score:.2f}")

        print("Pipeline test completed successfully!")
    finally:
        os.unlink(temp_path)

if __name__ == "__main__":
    main()