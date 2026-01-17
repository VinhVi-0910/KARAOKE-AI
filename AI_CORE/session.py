"""Karaoke session management.

Manages karaoke sessions, saves/loads results, and tracks user performance.
"""
import json
import os
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class SessionResult:
    """Result of a single karaoke performance."""
    
    song_id: str
    song_title: str
    timestamp: str  # ISO format datetime
    score: float
    accuracy: float  # percent
    stability: float  # 0-1
    duration: float  # seconds
    notes: Dict[str, str] = None  # feedback
    
    def __post_init__(self):
        if self.notes is None:
            self.notes = {}
    
    def to_dict(self) -> Dict:
        return asdict(self)


class KaraokeSession:
    """Represents a single karaoke session (multiple songs)."""
    
    def __init__(self, session_id: str, username: str):
        """Initialize session.
        
        Args:
            session_id: unique session identifier
            username: user name
        """
        self.session_id = session_id
        self.username = username
        self.start_time = datetime.now().isoformat()
        self.results: List[SessionResult] = []
    
    def add_result(self, result: SessionResult) -> None:
        """Add a song performance result."""
        self.results.append(result)
    
    def get_average_score(self) -> float:
        """Get average score across all songs."""
        if not self.results:
            return 0.0
        return np.mean([r.score for r in self.results])
    
    def get_average_accuracy(self) -> float:
        """Get average accuracy across all songs."""
        if not self.results:
            return 0.0
        return np.mean([r.accuracy for r in self.results])
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "username": self.username,
            "start_time": self.start_time,
            "num_songs": len(self.results),
            "average_score": self.get_average_score(),
            "average_accuracy": self.get_average_accuracy(),
            "results": [r.to_dict() for r in self.results]
        }
    
    def to_json(self) -> str:
        """Convert to JSON."""
        return json.dumps(self.to_dict(), indent=2)


class SessionManager:
    """Manages karaoke sessions and performance history.
    
    Responsibilities:
    - create and manage sessions
    - save/load session data
    - query performance history
    - generate statistics
    """
    
    def __init__(self, storage_path: str = "./karaoke_sessions"):
        """Initialize session manager.
        
        Args:
            storage_path: path to store session files
        """
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        self.current_session: Optional[KaraokeSession] = None
        self.sessions: Dict[str, KaraokeSession] = {}
        self._load_sessions()
    
    def create_session(self, username: str) -> KaraokeSession:
        """Create a new session.
        
        Args:
            username: user name
        
        Returns:
            KaraokeSession object
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"{username}_{timestamp}"
        
        session = KaraokeSession(session_id, username)
        self.current_session = session
        self.sessions[session_id] = session
        
        return session
    
    def get_current_session(self) -> Optional[KaraokeSession]:
        """Get current active session."""
        return self.current_session
    
    def end_session(self) -> Optional[KaraokeSession]:
        """End current session and save it."""
        if not self.current_session:
            return None
        
        session = self.current_session
        self.save_session(session)
        self.current_session = None
        
        return session
    
    def save_session(self, session: KaraokeSession) -> str:
        """Save session to disk.
        
        Args:
            session: KaraokeSession object
        
        Returns:
            path to saved file
        """
        filepath = os.path.join(self.storage_path, f"{session.session_id}.json")
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(session.to_json())
        
        return filepath
    
    def load_session(self, session_id: str) -> Optional[KaraokeSession]:
        """Load session from disk.
        
        Args:
            session_id: session identifier
        
        Returns:
            KaraokeSession object or None
        """
        if session_id in self.sessions:
            return self.sessions[session_id]
        
        filepath = os.path.join(self.storage_path, f"{session_id}.json")
        if not os.path.exists(filepath):
            return None
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            session = KaraokeSession(data["session_id"], data["username"])
            session.start_time = data["start_time"]
            
            for result_data in data.get("results", []):
                result = SessionResult(
                    song_id=result_data["song_id"],
                    song_title=result_data["song_title"],
                    timestamp=result_data["timestamp"],
                    score=result_data["score"],
                    accuracy=result_data["accuracy"],
                    stability=result_data["stability"],
                    duration=result_data["duration"],
                    notes=result_data.get("notes", {})
                )
                session.add_result(result)
            
            self.sessions[session_id] = session
            return session
        except Exception as e:
            print(f"Error loading session: {e}")
            return None
    
    def _load_sessions(self) -> None:
        """Load all sessions from storage."""
        if not os.path.exists(self.storage_path):
            return
        
        for filename in os.listdir(self.storage_path):
            if filename.endswith(".json"):
                session_id = filename[:-5]  # remove .json
                self.load_session(session_id)
    
    def get_user_history(self, username: str) -> List[KaraokeSession]:
        """Get all sessions for a user.
        
        Args:
            username: user name
        
        Returns:
            list of KaraokeSession objects
        """
        return [s for s in self.sessions.values() if s.username == username]
    
    def get_user_stats(self, username: str) -> Dict:
        """Get performance statistics for a user.
        
        Args:
            username: user name
        
        Returns:
            dict with stats
        """
        sessions = self.get_user_history(username)
        
        if not sessions:
            return {
                "username": username,
                "total_sessions": 0,
                "total_songs": 0,
                "average_score": 0.0,
                "average_accuracy": 0.0,
                "best_score": 0.0,
                "worst_score": 0.0
            }
        
        all_scores = []
        all_accuracies = []
        
        for session in sessions:
            for result in session.results:
                all_scores.append(result.score)
                all_accuracies.append(result.accuracy)
        
        return {
            "username": username,
            "total_sessions": len(sessions),
            "total_songs": sum(len(s.results) for s in sessions),
            "average_score": float(np.mean(all_scores)) if all_scores else 0.0,
            "average_accuracy": float(np.mean(all_accuracies)) if all_accuracies else 0.0,
            "best_score": float(np.max(all_scores)) if all_scores else 0.0,
            "worst_score": float(np.min(all_scores)) if all_scores else 0.0
        }
    
    def get_top_songs(self, username: str, limit: int = 10) -> List[Dict]:
        """Get user's top performing songs.
        
        Args:
            username: user name
            limit: number of songs to return
        
        Returns:
            list of dicts with song info and stats
        """
        sessions = self.get_user_history(username)
        
        song_stats = {}
        for session in sessions:
            for result in session.results:
                if result.song_id not in song_stats:
                    song_stats[result.song_id] = {
                        "song_id": result.song_id,
                        "song_title": result.song_title,
                        "attempts": 0,
                        "avg_score": 0.0,
                        "best_score": 0.0
                    }
                
                stats = song_stats[result.song_id]
                stats["attempts"] += 1
                stats["best_score"] = max(stats["best_score"], result.score)
                stats["avg_score"] = (stats["avg_score"] * (stats["attempts"] - 1) + result.score) / stats["attempts"]
        
        # Sort by best score
        sorted_songs = sorted(song_stats.values(), key=lambda x: x["best_score"], reverse=True)
        return sorted_songs[:limit]
