"""
# AI_CORE Karaoke App - Complete Architecture

## Overview
Comprehensive AI-powered karaoke application built with Python. The system handles audio processing, pitch detection, real-time recording, performance analysis, and session management.

## Architecture Layers

### 1. Audio Processing Layer
- **AudioLoader**: Loads and normalizes audio files
- **PitchDetector**: Detects pitch using CREPE (with resampling to 16kHz)
- **AutoTuner**: Applies pitch correction/shifting
- **Visualizer**: Plots pitch contours and analysis results

### 2. Song Management
- **Song**: Data class representing a karaoke song
  - Stores metadata: title, artist, audio path, duration, BPM, key
  - Stores target pitch/time arrays for reference
  
- **SongManager**: Manages song library
  - Load/save songs and metadata to disk (JSON + numpy arrays)
  - Search by title or artist
  - Extract and store target pitch from audio
  - Methods: `add_song()`, `get_song()`, `search_by_title()`, `extract_target_pitch_from_audio()`

### 3. Real-Time Recording
- **RealTimeRecorder**: Captures microphone input
  - Uses `sounddevice` for cross-platform audio input
  - Streams audio to callback for real-time processing
  - Methods: `start()`, `stop()`, `get_audio()`, `is_recording_now()`
  
- **AudioBuffer**: Thread-safe ring buffer
  - Keeps last N seconds of audio
  - Useful for streaming analysis

### 4. Performance Analysis
- **PitchMetrics**: Data class for pitch accuracy metrics
  - Mean/std error (Hz)
  - Accuracy percentage
  - Stability measure (0-1)
  - Voiced frame count
  
- **VibratoMetrics**: Detects vibrato
  - Frequency (Hz), depth (cents), coverage (%)
  - Uses autocorrelation on pitch residual
  
- **PerformanceAnalyzer**: Comprehensive analysis
  - Computes pitch accuracy against target
  - Detects vibrato in performance
  - Generates overall score with feedback
  - Methods: `analyze_pitch_accuracy()`, `detect_vibrato()`, `generate_report()`
  
- **ScoreReport**: Complete performance report
  - Total score (0-100)
  - Pitch metrics and vibrato metrics
  - Textual feedback notes

### 5. Session Management
- **SessionResult**: Single song performance result
  - Song ID, title, timestamp
  - Score, accuracy, stability
  - Duration and feedback notes
  
- **KaraokeSession**: Single karaoke session (multiple songs)
  - Session ID, username, start time
  - Stores list of results
  - Methods: `add_result()`, `get_average_score()`, `to_json()`
  
- **SessionManager**: Manage all sessions
  - Create/load/save sessions to disk
  - Query user history and statistics
  - Get top performing songs
  - Methods: `create_session()`, `save_session()`, `get_user_stats()`, `get_top_songs()`

### 6. Main Application Orchestrator
- **KaraokeApp**: High-level application
  - Coordinates all components
  - Main workflow: start session → select song → record → analyze → save
  - User-facing methods:
    - `start_session(username)`: Create new session
    - `select_song(song_id)`: Load song and target pitch
    - `record_performance(duration)`: Capture user singing
    - `analyze_performance(visualize)`: Compute score and metrics
    - `perform_song(song_id)`: Complete end-to-end workflow
    - `get_user_stats(username)`: View performance history
    - `get_user_top_songs(username)`: Get best songs

## File Structure
```
AI_CORE/
├── __init__.py                 # Package exports
├── audio_loader.py             # Audio loading utilities
├── pitch_detector.py           # CREPE-based pitch detection
├── autotuner.py                # Pitch shifting/correction
├── scorer.py                   # Basic scoring (HZ difference)
├── visualizer.py               # Matplotlib plotting
├── engine.py                   # Pipeline orchestrator
├── song_manager.py             # Song library management
├── recorder.py                 # Real-time recording
├── performance.py              # Detailed performance analysis
├── session.py                  # Session and history management
├── karaoke_app.py              # Main application
├── requirements.txt            # Dependencies
├── example_karaoke_app.py      # Complete usage example
└── test_pipeline.py            # Basic pipeline test
```

## Dependencies
- **numpy**: Array operations
- **librosa**: Audio processing (loading, resampling, time-stretching)
- **crepe**: State-of-art pitch detection (uses TensorFlow)
- **scipy**: Scientific computing (interpolation, signal processing)
- **matplotlib**: Plotting
- **sounddevice**: Cross-platform audio input
- **tensorflow-cpu**: CREPE inference

## Usage Example

### Simple Workflow
```python
from AI_CORE import KaraokeApp, Song

# Initialize
app = KaraokeApp()

# Add songs to library
song = Song(
    id="mysong",
    title="My Song",
    artist="Artist",
    audio_path="/path/to/audio.wav"
)
app.song_manager.add_song(song)

# Start session
session = app.start_session("username")

# Perform a song
report = app.perform_song("mysong", record_duration=30)
print(f"Score: {report.total_score:.1f}/100")

# End session and view stats
app.end_session()
stats = app.get_user_stats("username")
```

### Component Usage
```python
# Just use the Engine (basic pipeline)
from AI_CORE import Engine
engine = Engine()
result = engine.analyze("audio.wav", visualise=True)

# Or use individual components
from AI_CORE import PitchDetector, PerformanceAnalyzer, RealTimeRecorder
detector = PitchDetector()
analyzer = PerformanceAnalyzer()
recorder = RealTimeRecorder()

# Record
recorder.start()
# ... user sings ...
recorder.stop()
audio = recorder.get_audio()

# Analyze
time, pitch = detector.detect(audio, sr=44100)
metrics = analyzer.analyze_pitch_accuracy(pitch, target_pitch, time)
```

## Performance Scoring

### Score Components
1. **Pitch Accuracy** (0-100): Percentage of frames within ±100 cents of target
2. **Stability Bonus** (0-10): Based on pitch contour consistency
3. **Vibrato Bonus** (+5): Detected natural vibrato (4-8 Hz, 30+ cents)

### Feedback Categories
- **Pitch Accuracy**: How close to target notes
- **Stability**: Consistency of pitch over time
- **Vibrato**: Detection and quality
- **Overall**: Encouraging message based on score

## Future Enhancements
1. **Beat/Rhythm Analysis**: Measure timing accuracy against backing track
2. **Spectral Analysis**: Analyze voice quality, brightness, formants
3. **Emotion Detection**: Detect emotional expression in performance
4. **Real-time Feedback**: Display score and corrections during singing
5. **Song Recommendations**: Suggest songs based on user's vocal range
6. **Duet Mode**: Score two singers simultaneously
7. **Web/Mobile UI**: Build frontend for browser/mobile access
8. **Database**: Replace file-based storage with SQL database
9. **Multiplayer**: Compete with other users
10. **AI Coach**: Generate personalized training recommendations

## Testing

Run the example:
```bash
python -m AI_CORE.example_karaoke_app
```

Or run quick pipeline test:
```bash
python -m AI_CORE.test_pipeline
```

## Notes
- CREPE model is downloaded on first use (~50MB)
- Sample rate is normalized to 44100 Hz for consistency
- Pitch is detected at 10ms intervals (step_size=10)
- Vibrato detection is disabled if < 10 voiced frames
- All times and durations are in seconds
- Scores are clamped to [0, 100] range
"""
