"""Engine orchestrator for the karaoke AI pipeline.

This module provides a small, well-documented Engine class that wires
the loader, pitch detector, scorer and visualizer together so you can
run the typical processing flow from one place.
"""
from typing import Optional, Dict, Any
import numpy as np

from .audio_loader import AudioLoader
from .pitch_detector import PitchDetector
from .scorer import Scorer
from .visualizer import Visualizer
from .autotuner import AutoTuner


class Engine:
    """Orchestrates the karaoke AI flow.

    Responsibilities:
    - load audio
    - detect pitch/time
    - optionally apply autotune
    - compute score against target pitch (if provided)
    - optionally visualize results

    The engine keeps dependencies injectable which helps testing.
    """

    def __init__(
        self,
        loader: Optional[AudioLoader] = None,
        detector: Optional[PitchDetector] = None,
        scorer: Optional[Scorer] = None,
        visualizer: Optional[Visualizer] = None,
        autotuner: Optional[AutoTuner] = None,
    ) -> None:
        self.loader = loader or AudioLoader()
        self.detector = detector or PitchDetector()
        self.scorer = scorer or Scorer()
        self.visualizer = visualizer or Visualizer()
        self.autotuner = autotuner

    def analyze(self, path: str, target_pitch: Optional[np.ndarray] = None, visualise: bool = False) -> Dict[str, Any]:
        """Run full analysis on a file.

        Args:
            path: path to audio file
            target_pitch: optional array of target pitch values (Hz) aligned with detector output
            visualise: when True, show plots

        Returns:
            dict with keys: time, pitch, score (when target provided), tuned_audio (when autotuner present)
        """
        import numpy as np

        # Load
        y, sr = self.loader.load(path)

        # Detect
        time, pitch = self.detector.detect(y, sr)

        result: Dict[str, Any] = {"time": time, "pitch": pitch}

        # Score if target provided
        if target_pitch is not None:
            # ensure shapes align; caller is responsible for alignment
            score = self.scorer.score_pitch(pitch, target_pitch)
            result["score"] = score

        # Autotune if available
        if self.autotuner is not None:
            # The autotuner expects an array of pitch values (or a representative pitch)
            try:
                tuned = self.autotuner.tune(y, sr, pitch)
                result["tuned_audio"] = tuned
            except Exception:
                # Keep engine robust: do not raise on autotune failure
                result["tuned_audio"] = None

        if visualise:
            try:
                self.visualizer.plot_pitch(time, pitch)
            except Exception:
                pass

        return result
