"""Main karaoke application orchestrator.

Coordinates all components (song manager, recorder, engine, analyzer, session manager)
for end-to-end karaoke experience.
"""
from typing import Optional, Dict, Any
from .engine import Engine
from .song_manager import SongManager, Song
from .recorder import RealTimeRecorder
from .performance import PerformanceAnalyzer, ScoreReport
from .session import SessionManager, SessionResult, KaraokeSession
from .pitch_detector import PitchDetector
from .scorer import Scorer
import numpy as np


class KaraokeApp:
    """Main application class for karaoke AI.
    
    Workflow:
    1. User starts session
    2. User selects a song
    3. User records their performance
    4. App analyzes pitch and generates score
    5. Results saved to session
    6. User can review stats
    """
    
    def __init__(
        self,
        song_library_path: Optional[str] = None,
        session_storage_path: str = "./karaoke_sessions"
    ):
        """Initialize karaoke app.
        
        Args:
            song_library_path: path to song library folder
            session_storage_path: path to store session data
        """
        # Core components
        self.engine = Engine()
        self.song_manager = SongManager(library_path=song_library_path)
        self.session_manager = SessionManager(storage_path=session_storage_path)
        self.performance_analyzer = PerformanceAnalyzer()
        
        # State
        self.current_session: Optional[KaraokeSession] = None
        self.current_song: Optional[Song] = None
        self.user_pitch: Optional[np.ndarray] = None
        self.user_time: Optional[np.ndarray] = None
        self.target_pitch: Optional[np.ndarray] = None
        self.target_time: Optional[np.ndarray] = None
    
    def start_session(self, username: str) -> KaraokeSession:
        """Start a new karaoke session.
        
        Args:
            username: user name
        
        Returns:
            KaraokeSession object
        """
        self.current_session = self.session_manager.create_session(username)
        print(f"[KaraokeApp] Session started for {username}")
        return self.current_session
    
    def end_session(self) -> Optional[KaraokeSession]:
        """End current session and save results.
        
        Returns:
            completed KaraokeSession or None
        """
        if not self.current_session:
            print("[KaraokeApp] No active session")
            return None
        
        session = self.session_manager.end_session()
        print(f"[KaraokeApp] Session ended. Average score: {session.get_average_score():.1f}")
        return session
    
    def select_song(self, song_id: str) -> Song:
        """Select a song for the next performance.
        
        Args:
            song_id: song identifier
        
        Returns:
            Song object
        
        Raises:
            ValueError if song not found
        """
        song = self.song_manager.get_song(song_id)
        if not song:
            raise ValueError(f"Song {song_id} not found in library")
        
        self.current_song = song
        
        # Load target pitch if available, else extract it
        target_data = self.song_manager.get_target_pitch(song_id)
        if target_data:
            self.target_time, self.target_pitch = target_data
        else:
            print(f"[KaraokeApp] Extracting target pitch for {song.title}...")
            self.target_time, self.target_pitch = self.song_manager.extract_target_pitch_from_audio(
                song_id, self.engine.detector
            )
        
        print(f"[KaraokeApp] Selected: {song.title} by {song.artist}")
        return song
    
    def record_performance(
        self,
        duration: Optional[float] = None,
        on_chunk=None
    ) -> np.ndarray:
        """Record user's karaoke performance.
        
        Args:
            duration: duration to record (seconds), or None for manual stop
            on_chunk: optional callback(chunk, sr) for real-time processing
        
        Returns:
            recorded audio array
        """
        if not self.current_song:
            raise RuntimeError("No song selected. Call select_song() first.")
        
        print(f"[KaraokeApp] Recording... (duration: {duration or 'until stop'}s)")
        
        recorder = RealTimeRecorder(
            callback=on_chunk,
            sample_rate=self.engine.loader.sample_rate
        )
        
        recorder.start()
        
        if duration:
            import time
            time.sleep(duration)
        # else: wait for manual stop (user should call recorder.stop())
        
        recorder.stop()
        
        audio = recorder.get_audio()
        self.user_audio = audio
        print(f"[KaraokeApp] Recording complete: {len(audio) / self.engine.loader.sample_rate:.1f}s")
        
        return audio
    
    def analyze_performance(self, visualize: bool = False) -> ScoreReport:
        """Analyze recorded performance against target.
        
        Args:
            visualize: whether to display plots
        
        Returns:
            ScoreReport object
        
        Raises:
            RuntimeError if no recording or no target pitch
        """
        if not self.current_song:
            raise RuntimeError("No song selected.")
        
        if not hasattr(self, 'user_audio') or len(self.user_audio) == 0:
            raise RuntimeError("No recorded audio. Call record_performance() first.")
        
        if self.target_pitch is None:
            raise RuntimeError("No target pitch available.")
        
        print(f"[KaraokeApp] Analyzing performance...")
        
        # Detect pitch from recording
        sr = self.engine.loader.sample_rate
        user_time, user_pitch = self.engine.detector.detect(self.user_audio, sr)
        self.user_time = user_time
        self.user_pitch = user_pitch
        
        # Align arrays if needed (interpolate to same length)
        if len(user_pitch) != len(self.target_pitch):
            # Interpolate user pitch to target time grid
            user_pitch = self._align_pitch(
                user_pitch, user_time,
                self.target_pitch, self.target_time
            )
        
        # Generate comprehensive report
        report = self.performance_analyzer.generate_report(
            user_pitch, self.target_pitch, self.target_time,
            scorer=self.engine.scorer
        )
        
        if visualize:
            self.engine.visualizer.plot_pitch(self.target_time, self.target_pitch)
            self.engine.visualizer.plot_pitch(user_time, user_pitch)
        
        print(f"[KaraokeApp] Score: {report.total_score:.1f}/100")
        for key, note in report.notes.items():
            print(f"  {key}: {note}")
        
        return report
    
    def save_performance(self, report: ScoreReport) -> SessionResult:
        """Save performance to current session.
        
        Args:
            report: ScoreReport object
        
        Returns:
            SessionResult object
        """
        if not self.current_session:
            raise RuntimeError("No active session.")
        
        if not self.current_song:
            raise RuntimeError("No selected song.")
        
        from datetime import datetime
        
        result = SessionResult(
            song_id=self.current_song.id,
            song_title=self.current_song.title,
            timestamp=datetime.now().isoformat(),
            score=report.total_score,
            accuracy=report.pitch_metrics.accuracy,
            stability=report.pitch_metrics.stability,
            duration=report.duration,
            notes=report.notes
        )
        
        self.current_session.add_result(result)
        print(f"[KaraokeApp] Result saved to session.")
        
        return result
    
    def perform_song(
        self,
        song_id: str,
        record_duration: Optional[float] = None,
        visualize: bool = False
    ) -> ScoreReport:
        """Complete workflow: select → record → analyze → save.
        
        Args:
            song_id: song ID
            record_duration: recording duration (seconds)
            visualize: whether to display plots
        
        Returns:
            ScoreReport with results
        """
        # Select song
        self.select_song(song_id)
        
        # Record (need manual stop or duration)
        self.record_performance(duration=record_duration)
        
        # Analyze
        report = self.analyze_performance(visualize=visualize)
        
        # Save to session
        if self.current_session:
            self.save_performance(report)
        
        return report
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session.
        
        Returns:
            dict with session stats
        """
        if not self.current_session:
            return {"status": "no active session"}
        
        return {
            "session_id": self.current_session.session_id,
            "username": self.current_session.username,
            "num_songs": len(self.current_session.results),
            "average_score": self.current_session.get_average_score(),
            "average_accuracy": self.current_session.get_average_accuracy(),
            "results": [
                {
                    "song": r.song_title,
                    "score": r.score,
                    "accuracy": r.accuracy
                }
                for r in self.current_session.results
            ]
        }
    
    def get_user_stats(self, username: str) -> Dict[str, Any]:
        """Get user's performance statistics.
        
        Args:
            username: user name
        
        Returns:
            dict with stats
        """
        return self.session_manager.get_user_stats(username)
    
    def get_user_top_songs(self, username: str, limit: int = 10) -> list:
        """Get user's top performing songs.
        
        Args:
            username: user name
            limit: max songs to return
        
        Returns:
            list of song stats
        """
        return self.session_manager.get_top_songs(username, limit)
    
    @staticmethod
    def _align_pitch(
        user_pitch: np.ndarray,
        user_time: np.ndarray,
        target_pitch: np.ndarray,
        target_time: np.ndarray
    ) -> np.ndarray:
        """Interpolate user pitch to match target time grid.
        
        Args:
            user_pitch, user_time: detected pitch and time
            target_pitch, target_time: target pitch and time
        
        Returns:
            user_pitch interpolated to target_time
        """
        from scipy.interpolate import interp1d
        
        # Remove NaNs for interpolation
        valid_mask = ~np.isnan(user_pitch)
        
        if np.sum(valid_mask) < 2:
            # Not enough points, return zeros
            return np.full_like(target_pitch, np.nan)
        
        # Create interpolation function
        f = interp1d(
            user_time[valid_mask],
            user_pitch[valid_mask],
            kind='linear',
            fill_value='extrapolate',
            bounds_error=False
        )
        
        # Interpolate to target time grid
        aligned = f(target_time)
        
        # Clip to valid range
        aligned[aligned < 50] = np.nan  # too low
        aligned[aligned > 2000] = np.nan  # too high
        
        return aligned
