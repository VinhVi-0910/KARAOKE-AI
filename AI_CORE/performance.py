"""Performance analysis and detailed scoring for karaoke.

Provides comprehensive pitch accuracy metrics, vibrato detection, and performance reports.
"""
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
import json


@dataclass
class PitchMetrics:
    """Detailed pitch accuracy metrics."""
    
    mean_error_hz: float  # mean absolute pitch error (Hz)
    std_error_hz: float  # standard deviation of error (Hz)
    accuracy: float  # percentage of frames within tolerance (0-100)
    stability: float  # measure of pitch stability (0-1, higher is more stable)
    voiced_frames: int  # number of non-NaN frames
    total_frames: int  # total frames analyzed


@dataclass
class VibratoMetrics:
    """Vibrato detection metrics."""
    
    detected: bool  # whether vibrato was detected
    frequency_hz: Optional[float] = None  # vibrato frequency (Hz)
    depth_cents: Optional[float] = None  # vibrato depth in cents
    coverage: float = 0.0  # percentage of song with vibrato


@dataclass
class ScoreReport:
    """Complete performance report."""
    
    total_score: float  # overall score (0-100)
    pitch_metrics: PitchMetrics
    vibrato_metrics: VibratoMetrics
    duration: float  # recording duration (seconds)
    notes: Dict[str, str] = None  # feedback notes
    
    def __post_init__(self):
        if self.notes is None:
            self.notes = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_score": self.total_score,
            "pitch_metrics": asdict(self.pitch_metrics),
            "vibrato_metrics": asdict(self.vibrato_metrics),
            "duration": self.duration,
            "notes": self.notes
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class PerformanceAnalyzer:
    """Analyzes pitch performance and generates detailed reports.
    
    Computes:
    - Pitch accuracy metrics
    - Stability and consistency
    - Vibrato detection
    - Overall score with feedback
    """
    
    def __init__(
        self,
        pitch_tolerance_cents: float = 100.0,
        vibrato_freq_range: Tuple[float, float] = (4.0, 8.0),
        min_vibrato_depth: float = 30.0
    ):
        """Initialize analyzer.
        
        Args:
            pitch_tolerance_cents: allowed pitch error in cents (100 cents = 1 semitone)
            vibrato_freq_range: expected vibrato frequency range (Hz)
            min_vibrato_depth: minimum vibrato depth to detect (cents)
        """
        self.pitch_tolerance_cents = pitch_tolerance_cents
        self.vibrato_freq_range = vibrato_freq_range
        self.min_vibrato_depth = min_vibrato_depth
    
    def analyze_pitch_accuracy(
        self,
        user_pitch: np.ndarray,
        target_pitch: np.ndarray,
        time: np.ndarray
    ) -> PitchMetrics:
        """Analyze pitch accuracy against target.
        
        Args:
            user_pitch: detected user pitch (Hz)
            target_pitch: target pitch (Hz)
            time: time array (seconds)
        
        Returns:
            PitchMetrics object
        """
        # Mask valid frames (non-NaN)
        mask = ~np.isnan(user_pitch) & ~np.isnan(target_pitch)
        
        if np.sum(mask) == 0:
            return PitchMetrics(
                mean_error_hz=np.nan,
                std_error_hz=np.nan,
                accuracy=0.0,
                stability=0.0,
                voiced_frames=0,
                total_frames=len(user_pitch)
            )
        
        # Compute error in Hz
        user_valid = user_pitch[mask]
        target_valid = target_pitch[mask]
        error_hz = np.abs(user_valid - target_valid)
        
        # Convert to cents for tolerance check
        error_cents = 1200 * np.log2(user_valid / target_valid)
        error_cents = np.abs(error_cents)
        
        # Accuracy: percentage within tolerance
        accuracy = 100.0 * np.sum(error_cents < self.pitch_tolerance_cents) / len(error_cents)
        
        # Stability: inverse of coefficient of variation
        # Lower variation = higher stability
        variation = np.std(error_hz) / (np.mean(error_hz) + 1e-6)
        stability = 1.0 / (1.0 + variation)  # map to 0-1
        
        return PitchMetrics(
            mean_error_hz=float(np.mean(error_hz)),
            std_error_hz=float(np.std(error_hz)),
            accuracy=float(accuracy),
            stability=float(stability),
            voiced_frames=int(np.sum(mask)),
            total_frames=int(len(user_pitch))
        )
    
    def detect_vibrato(
        self,
        pitch: np.ndarray,
        time: np.ndarray
    ) -> VibratoMetrics:
        """Detect vibrato in pitch contour.
        
        Simple vibrato detection using pitch residual oscillation.
        
        Args:
            pitch: pitch contour (Hz)
            time: time array (seconds)
        
        Returns:
            VibratoMetrics object
        """
        # Remove NaNs
        mask = ~np.isnan(pitch)
        if np.sum(mask) < 100:
            return VibratoMetrics(detected=False)
        
        valid_pitch = pitch[mask]
        valid_time = time[mask]
        
        # Smooth pitch contour (remove vibrato)
        from scipy import signal
        try:
            smoothed = signal.savgol_filter(valid_pitch, window_length=min(51, len(valid_pitch)//2*2+1), polyorder=3)
        except Exception:
            smoothed = valid_pitch
        
        # Compute residual (vibrato component)
        residual = valid_pitch - smoothed
        
        # Simple autocorrelation to find vibrato frequency
        max_lag = int(0.5 * len(residual))  # max 0.5 seconds
        
        if max_lag < 10:
            return VibratoMetrics(detected=False)
        
        autocorr = np.correlate(residual - np.mean(residual), residual - np.mean(residual), mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]
        
        # Find peaks in autocorrelation
        peaks, _ = signal.find_peaks(autocorr[:max_lag], height=0.3)
        
        if len(peaks) == 0:
            return VibratoMetrics(detected=False)
        
        # Find vibrato frequency
        best_lag = peaks[0]
        sr = 1.0 / np.mean(np.diff(valid_time))
        vibrato_freq = sr / best_lag
        
        # Check if in expected range
        if not (self.vibrato_freq_range[0] <= vibrato_freq <= self.vibrato_freq_range[1]):
            return VibratoMetrics(detected=False)
        
        # Measure vibrato depth (cents)
        vibrato_depth = 1200 * np.log2(np.max(residual) / np.mean(smoothed))
        
        if np.abs(vibrato_depth) < self.min_vibrato_depth:
            return VibratoMetrics(detected=False)
        
        # Coverage: percentage of signal with significant oscillation
        residual_std = np.std(residual)
        coverage = 100.0 * np.sum(np.abs(residual) > residual_std) / len(residual)
        
        return VibratoMetrics(
            detected=True,
            frequency_hz=float(vibrato_freq),
            depth_cents=float(np.abs(vibrato_depth)),
            coverage=float(coverage)
        )
    
    def generate_report(
        self,
        user_pitch: np.ndarray,
        target_pitch: np.ndarray,
        time: np.ndarray,
        scorer=None
    ) -> ScoreReport:
        """Generate comprehensive performance report.
        
        Args:
            user_pitch: detected user pitch (Hz)
            target_pitch: target pitch (Hz)
            time: time array (seconds)
            scorer: optional Scorer instance for overall score
        
        Returns:
            ScoreReport object
        """
        # Analyze pitch accuracy
        pitch_metrics = self.analyze_pitch_accuracy(user_pitch, target_pitch, time)
        
        # Detect vibrato
        vibrato_metrics = self.detect_vibrato(user_pitch, time)
        
        # Compute overall score
        if scorer:
            base_score = scorer.score_pitch(user_pitch, target_pitch)
        else:
            # Default: use accuracy as base score
            base_score = pitch_metrics.accuracy
        
        # Bonuses/penalties
        vibrato_bonus = 5.0 if vibrato_metrics.detected else 0.0  # bonus for vibrato
        stability_bonus = pitch_metrics.stability * 10.0  # up to +10 for stability
        
        total_score = min(100.0, base_score + vibrato_bonus + stability_bonus)
        total_score = max(0.0, total_score)
        
        # Generate feedback notes
        notes = self._generate_feedback(pitch_metrics, vibrato_metrics, total_score)
        
        duration = time[-1] - time[0] if len(time) > 0 else 0.0
        
        return ScoreReport(
            total_score=float(total_score),
            pitch_metrics=pitch_metrics,
            vibrato_metrics=vibrato_metrics,
            duration=float(duration),
            notes=notes
        )
    
    def _generate_feedback(
        self,
        pitch_metrics: PitchMetrics,
        vibrato_metrics: VibratoMetrics,
        score: float
    ) -> Dict[str, str]:
        """Generate feedback notes based on metrics."""
        notes = {}
        
        # Accuracy feedback
        if pitch_metrics.accuracy >= 80:
            notes["pitch_accuracy"] = "Excellent pitch accuracy!"
        elif pitch_metrics.accuracy >= 60:
            notes["pitch_accuracy"] = "Good pitch accuracy. Try to stay closer to the target."
        elif pitch_metrics.accuracy >= 40:
            notes["pitch_accuracy"] = "Fair pitch accuracy. Focus on hitting the right notes."
        else:
            notes["pitch_accuracy"] = "Pitch accuracy needs improvement. Listen carefully to the backing track."
        
        # Stability feedback
        if pitch_metrics.stability >= 0.8:
            notes["stability"] = "Great pitch stability!"
        elif pitch_metrics.stability >= 0.5:
            notes["stability"] = "Good stability. Try to reduce pitch wavering."
        else:
            notes["stability"] = "Work on maintaining stable pitch."
        
        # Vibrato feedback
        if vibrato_metrics.detected:
            notes["vibrato"] = f"Nice vibrato detected! ({vibrato_metrics.frequency_hz:.1f} Hz, {vibrato_metrics.depth_cents:.0f} cents)"
        
        # Overall feedback
        if score >= 90:
            notes["overall"] = "Outstanding performance!"
        elif score >= 75:
            notes["overall"] = "Great job! Keep it up!"
        elif score >= 60:
            notes["overall"] = "Good effort! Practice more to improve."
        else:
            notes["overall"] = "Keep practicing! You'll improve with time."
        
        return notes
