"""AI_CORE package initializer.

Exports core classes for convenient imports and exposes a package version.
"""
__version__ = "0.1.0"

from .audio_loader import AudioLoader
from .autotuner import AutoTuner
from .pitch_detector import PitchDetector
from .scorer import Scorer
from .visualizer import Visualizer
from .engine import Engine
from .song_manager import Song, SongManager
from .recorder import RealTimeRecorder, AudioBuffer
from .performance import PerformanceAnalyzer, PitchMetrics, VibratoMetrics, ScoreReport
from .session import KaraokeSession, SessionManager, SessionResult
from .karaoke_app import KaraokeApp

__all__ = [
    "AudioLoader",
    "AutoTuner",
    "PitchDetector",
    "Scorer",
    "Visualizer",
    "Engine",
    "Song",
    "SongManager",
    "RealTimeRecorder",
    "AudioBuffer",
    "PerformanceAnalyzer",
    "PitchMetrics",
    "VibratoMetrics",
    "ScoreReport",
    "KaraokeSession",
    "SessionManager",
    "SessionResult",
    "KaraokeApp",
]
