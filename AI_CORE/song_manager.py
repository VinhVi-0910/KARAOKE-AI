"""Song management for karaoke app.

Handles loading songs, storing metadata, and managing target pitch data.
"""
import os
import json
import numpy as np
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict
import librosa


@dataclass
class Song:
    """Represents a karaoke song with metadata and target pitch."""
    
    id: str  # unique song identifier
    title: str
    artist: str
    audio_path: str  # path to backing track or original song
    target_pitch: Optional[np.ndarray] = None  # target pitch array (Hz)
    target_time: Optional[np.ndarray] = None  # time array for target pitch
    duration: float = 0.0  # song duration in seconds
    bpm: Optional[float] = None  # beats per minute (optional)
    key: Optional[str] = None  # musical key (optional)
    metadata: Dict[str, Any] = None  # additional metadata
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SongManager:
    """Manages song library and operations.
    
    Responsibilities:
    - load song files and extract metadata
    - store/load target pitch data
    - query and filter songs
    """
    
    def __init__(self, library_path: Optional[str] = None):
        """Initialize song manager.
        
        Args:
            library_path: path to folder containing songs and metadata
        """
        self.library_path = library_path
        self.songs: Dict[str, Song] = {}
        
        if library_path and os.path.exists(library_path):
            self._load_library()
    
    def add_song(self, song: Song) -> None:
        """Add a song to the library."""
        if not os.path.exists(song.audio_path):
            raise FileNotFoundError(f"Audio file not found: {song.audio_path}")
        
        # Store duration info
        try:
            y, sr = librosa.load(song.audio_path, sr=None)
            song.duration = librosa.get_duration(y=y, sr=sr)
        except Exception:
            pass  # If we can't load, keep duration as 0
        
        self.songs[song.id] = song
    
    def get_song(self, song_id: str) -> Optional[Song]:
        """Retrieve a song by ID."""
        return self.songs.get(song_id)
    
    def list_songs(self) -> List[Song]:
        """Return all songs in library."""
        return list(self.songs.values())
    
    def search_by_title(self, title: str) -> List[Song]:
        """Search songs by title (case-insensitive partial match)."""
        term = title.lower()
        return [s for s in self.songs.values() if term in s.title.lower()]
    
    def search_by_artist(self, artist: str) -> List[Song]:
        """Search songs by artist (case-insensitive partial match)."""
        term = artist.lower()
        return [s for s in self.songs.values() if term in s.artist.lower()]
    
    def save_target_pitch(self, song_id: str, time: np.ndarray, pitch: np.ndarray) -> None:
        """Store target pitch for a song."""
        if song_id not in self.songs:
            raise ValueError(f"Song {song_id} not in library")
        
        song = self.songs[song_id]
        song.target_time = time
        song.target_pitch = pitch
    
    def get_target_pitch(self, song_id: str) -> Optional[tuple]:
        """Get target pitch for a song.
        
        Returns:
            (time, pitch) arrays or None if not available
        """
        song = self.get_song(song_id)
        if song and song.target_pitch is not None:
            return song.target_time, song.target_pitch
        return None
    
    def extract_target_pitch_from_audio(self, song_id: str, detector) -> tuple:
        """Auto-extract target pitch from song audio using pitch detector.
        
        Args:
            song_id: ID of song
            detector: PitchDetector instance
        
        Returns:
            (time, pitch) arrays
        """
        song = self.get_song(song_id)
        if not song:
            raise ValueError(f"Song {song_id} not in library")
        
        # Load audio
        y, sr = librosa.load(song.audio_path, sr=None)
        
        # Detect pitch
        time, pitch = detector.detect(y, sr)
        
        # Save it
        self.save_target_pitch(song_id, time, pitch)
        
        return time, pitch
    
    def _load_library(self) -> None:
        """Load songs and metadata from library directory."""
        if not self.library_path or not os.path.isdir(self.library_path):
            return
        
        # Look for songs.json metadata file
        metadata_file = os.path.join(self.library_path, "songs.json")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for song_data in data.get("songs", []):
                        # Reconstruct audio path relative to library
                        audio_path = song_data.get("audio_path")
                        if audio_path and not os.path.isabs(audio_path):
                            audio_path = os.path.join(self.library_path, audio_path)
                        
                        song = Song(
                            id=song_data.get("id"),
                            title=song_data.get("title"),
                            artist=song_data.get("artist"),
                            audio_path=audio_path,
                            bpm=song_data.get("bpm"),
                            key=song_data.get("key"),
                            metadata=song_data.get("metadata", {})
                        )
                        
                        # Load target pitch if exists
                        pitch_file = os.path.join(self.library_path, f"{song.id}_pitch.npy")
                        time_file = os.path.join(self.library_path, f"{song.id}_time.npy")
                        if os.path.exists(pitch_file) and os.path.exists(time_file):
                            song.target_pitch = np.load(pitch_file)
                            song.target_time = np.load(time_file)
                        
                        if os.path.exists(song.audio_path):
                            self.add_song(song)
            except Exception as e:
                print(f"Error loading library: {e}")
    
    def save_library(self, path: str) -> None:
        """Save library and all target pitch data to disk.
        
        Creates:
        - songs.json: metadata for all songs
        - {song_id}_pitch.npy: pitch arrays
        - {song_id}_time.npy: time arrays
        """
        os.makedirs(path, exist_ok=True)
        
        # Save metadata
        songs_data = []
        for song in self.songs.values():
            songs_data.append({
                "id": song.id,
                "title": song.title,
                "artist": song.artist,
                "audio_path": os.path.relpath(song.audio_path, path),  # relative path
                "bpm": song.bpm,
                "key": song.key,
                "metadata": song.metadata
            })
            
            # Save pitch arrays
            if song.target_pitch is not None:
                np.save(os.path.join(path, f"{song.id}_pitch.npy"), song.target_pitch)
            if song.target_time is not None:
                np.save(os.path.join(path, f"{song.id}_time.npy"), song.target_time)
        
        with open(os.path.join(path, "songs.json"), "w", encoding="utf-8") as f:
            json.dump({"songs": songs_data}, f, indent=2, ensure_ascii=False)
