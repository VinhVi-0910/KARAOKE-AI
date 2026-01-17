"""
# AI Karaoke App - Quick Start Guide

## Installation

### 1. Setup Python Environment
```bash
# Navigate to project folder
cd KARAOKE_AI

# Create virtual environment (if not already done)
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Windows CMD:
.venv\Scripts\activate.bat
```

### 2. Install Dependencies
```bash
cd AI_CORE
pip install -r requirements.txt

# Or install individually:
pip install numpy librosa crepe matplotlib scipy sounddevice tensorflow-cpu
```

## Quick Start (5 minutes)

### Option 1: Run Demo
```bash
python -m AI_CORE.example_karaoke_app
```

This creates demo songs and runs through a complete session.

### Option 2: Use in Your Code
```python
from AI_CORE import KaraokeApp, Song
import numpy as np

# 1. Create app
app = KaraokeApp(
    song_library_path="./my_songs",
    session_storage_path="./sessions"
)

# 2. Add a song
song = Song(
    id="song1",
    title="Happy Birthday",
    artist="Traditional",
    audio_path="./audio/happy_birthday.wav",
    bpm=120,
    key="C"
)
app.song_manager.add_song(song)

# 3. Start session
session = app.start_session("john_doe")

# 4. Perform song (complete workflow)
report = app.perform_song(
    song_id="song1",
    record_duration=30,  # 30 second recording
    visualize=False
)

# 5. View results
print(f"Score: {report.total_score:.1f}/100")
print(f"Accuracy: {report.pitch_metrics.accuracy:.1f}%")
print(f"Notes:")
for key, note in report.notes.items():
    print(f"  {key}: {note}")

# 6. View session summary
summary = app.get_session_summary()
print(f"Session: {summary['num_songs']} songs, avg score: {summary['average_score']:.1f}")

# 7. End session
app.end_session()

# 8. View user stats
stats = app.get_user_stats("john_doe")
print(f"All-time: {stats['total_songs']} songs, best: {stats['best_score']:.1f}")

# 9. View top songs
top = app.get_user_top_songs("john_doe", limit=5)
for song_stat in top:
    print(f"  {song_stat['song_title']}: {song_stat['best_score']:.1f}")
```

## Component Examples

### Just Record Audio
```python
from AI_CORE import RealTimeRecorder

recorder = RealTimeRecorder(sample_rate=44100)
recorder.start()
# ... wait or sleep ...
recorder.stop()
audio = recorder.get_audio()
```

### Just Detect Pitch
```python
from AI_CORE import PitchDetector
import librosa

detector = PitchDetector()
y, sr = librosa.load("song.wav")
time, pitch = detector.detect(y, sr)
```

### Just Analyze Performance
```python
from AI_CORE import PerformanceAnalyzer
import numpy as np

analyzer = PerformanceAnalyzer()
metrics = analyzer.analyze_pitch_accuracy(user_pitch, target_pitch, time)
report = analyzer.generate_report(user_pitch, target_pitch, time)
```

### Just Manage Songs
```python
from AI_CORE import SongManager, Song

sm = SongManager()

# Add songs
song = Song(id="song1", title="Title", artist="Artist", audio_path="...")
sm.add_song(song)

# Search
results = sm.search_by_title("song")

# Extract target pitch
time, pitch = sm.extract_target_pitch_from_audio("song1", detector)

# Save library
sm.save_library("./song_library")
```

## Key Concepts

### Score Calculation
- Base score: % of frames within pitch tolerance
- Stability bonus: up to +10 for consistent pitch
- Vibrato bonus: +5 if vibrato detected
- Final: clamped to [0, 100]

### Pitch Metrics
- **Accuracy**: % of frames within ±100 cents of target
- **Stability**: Measure of pitch consistency (0-1)
- **Mean Error**: Average Hz difference
- **Voiced Frames**: Number of detected pitch frames

### Vibrato
- Detected if 4-8 Hz, depth ≥ 30 cents
- Coverage: % of song with vibrato
- Adds realism to assessment

## Configuration

### PitchDetector Parameters
```python
detector = PitchDetector(
    confidence_threshold=0.5,    # Min confidence to accept frame
    crepe_sr=16000,              # CREPE's sample rate
    step_size=10                 # Detection every 10ms
)
```

### PerformanceAnalyzer Parameters
```python
analyzer = PerformanceAnalyzer(
    pitch_tolerance_cents=100.0,           # ±100 cents = 1 semitone
    vibrato_freq_range=(4.0, 8.0),        # Expected vibrato frequency
    min_vibrato_depth=30.0                 # Minimum vibrato depth
)
```

### RealTimeRecorder Parameters
```python
recorder = RealTimeRecorder(
    sample_rate=44100,           # Recording quality
    chunk_duration=0.5,          # Chunk size (seconds)
    channels=1,                  # Mono recording
    device=None                  # Auto-detect device
)
```

## Tips

### For Best Results
1. **Use good microphone**: USB condenser mic recommended
2. **Quiet environment**: Minimize background noise
3. **Backing track**: Good quality accompaniment helps
4. **Proper distance**: 6-12 inches from microphone
5. **Warm up**: Voice quality improves with warm-up

### Performance Tuning
- Increase `confidence_threshold` if too many false positives
- Decrease `pitch_tolerance_cents` for stricter scoring
- Adjust `vibrato_freq_range` for different singing styles

## Troubleshooting

### "ModuleNotFoundError: No module named 'tensorflow'"
→ Run: `pip install tensorflow-cpu`

### "ModuleNotFoundError: No module named 'sounddevice'"
→ Run: `pip install sounddevice`

### "No audio device found"
→ Check microphone connection
→ Use `sd.query_devices()` to list available devices
→ Pass `device=<index>` to RealTimeRecorder

### Low pitch detection accuracy
→ Increase confidence_threshold
→ Use higher quality audio
→ Try different microphone
→ Adjust CREPE's step_size for finer resolution

### Sessions not saving
→ Check `session_storage_path` directory exists
→ Ensure write permissions to directory
→ Call `end_session()` to save

## Next Steps

### For Development
- Add more audio features (spectral, rhythm)
- Implement web UI (Flask/Django + React)
- Add database storage (SQLite/PostgreSQL)
- Create mobile app (Flutter/React Native)
- Add leaderboards and multiplayer

### For Users
- Build song library
- Practice regularly
- Track improvements
- Try different genres
- Share with friends

## API Reference

See class docstrings for detailed parameters:
- `KaraokeApp.start_session()`
- `KaraokeApp.select_song()`
- `KaraokeApp.record_performance()`
- `KaraokeApp.analyze_performance()`
- `SongManager.extract_target_pitch_from_audio()`
- `PerformanceAnalyzer.generate_report()`
- `SessionManager.get_user_stats()`

## Support
For issues, check:
1. README.md for architecture overview
2. Code docstrings for parameter details
3. example_karaoke_app.py for complete workflow
4. test_pipeline.py for basic testing

Happy singing! 🎤🎵
"""
