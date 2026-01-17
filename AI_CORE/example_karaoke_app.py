"""Example usage of KaraokeApp for complete end-to-end workflow.

This demo shows how to:
1. Create a karaoke app
2. Add songs to library
3. Start a session
4. Perform karaoke
5. View results and stats
"""

import numpy as np
import librosa
import scipy.io.wavfile
import tempfile
import os
from AI_CORE import KaraokeApp, Song


def create_demo_songs():
    """Create and load demo songs."""
    temp_dir = tempfile.mkdtemp()
    
    # Create synthetic backing tracks
    songs = []
    
    # Song 1: A major scale (low)
    for freq_hz, note_name in [(220, "A3"), (246.94, "B3"), (277.18, "C#4")]:
        y_backing = librosa.tone(freq_hz, sr=44100, length=44100*3)  # 3 seconds
        sr = 44100
        
        # Save
        path = os.path.join(temp_dir, f"{note_name}_backing.wav")
        scipy.io.wavfile.write(path, sr, (y_backing * 0.8 * 32767).astype(np.int16))
        
        song = Song(
            id=f"demo_{note_name}",
            title=f"Demo Song {note_name}",
            artist="AI Karaoke",
            audio_path=path,
            bpm=120,
            key="A"
        )
        songs.append(song)
    
    return songs, temp_dir


def main():
    """Run karaoke demo."""
    print("=== AI Karaoke App Demo ===\n")
    
    # Initialize app
    app = KaraokeApp()
    
    # Add demo songs to library
    print("Step 1: Loading songs into library...")
    songs, temp_dir = create_demo_songs()
    for song in songs:
        app.song_manager.add_song(song)
        print(f"  Added: {song.title}")
    
    print(f"\nAvailable songs: {len(app.song_manager.list_songs())}")
    
    # Start session
    print("\nStep 2: Starting session...")
    session = app.start_session("demo_user")
    print(f"  Session ID: {session.session_id}")
    
    # Perform first song (with synthetic "user" recording)
    print("\nStep 3: Performing first song...")
    song_id = songs[0].id
    
    try:
        app.select_song(song_id)
        
        # Simulate recording (just use target pitch with small random noise)
        print("  Recording (simulated)...")
        sr = app.engine.loader.sample_rate
        duration = 3.0
        y_backing = librosa.tone(220, sr=sr, length=int(sr*duration))
        target_time, target_pitch = app.engine.detector.detect(y_backing, sr)
        # Add some noise to simulate imperfect performance
        user_pitch = target_pitch + np.random.normal(0, 5, len(target_pitch))
        user_pitch = np.clip(user_pitch, 100, 2000)
        
        app.user_pitch = user_pitch
        app.user_time = target_time
        app.target_pitch = target_pitch
        app.target_time = target_time
        app.user_audio = np.zeros(int(sr * duration))  # dummy audio
        
        # Analyze
        print("  Analyzing performance...")
        report = app.analyze_performance(visualize=False)
        print(f"  Score: {report.total_score:.1f}/100")
        
        # Save to session
        result = app.save_performance(report)
        
    except Exception as e:
        print(f"  Error: {e}")
    
    # Get session summary
    print("\nStep 4: Session Summary")
    summary = app.get_session_summary()
    print(f"  Songs performed: {summary['num_songs']}")
    print(f"  Average score: {summary['average_score']:.1f}/100")
    print(f"  Average accuracy: {summary['average_accuracy']:.1f}%")
    
    # Simulate more performances
    print("\nStep 5: Simulating more performances...")
    for i in range(1, len(songs)):
        song_id = songs[i].id
        try:
            app.select_song(song_id)
            
            # Simulate recording
            sr = app.engine.loader.sample_rate
            duration = 3.0
            y_backing = librosa.tone(246.94, sr=sr, length=int(sr*duration))
            target_time, target_pitch = app.engine.detector.detect(y_backing, sr)
            user_pitch = target_pitch + np.random.normal(0, 8, len(target_pitch))
            user_pitch = np.clip(user_pitch, 100, 2000)
            
            app.user_pitch = user_pitch
            app.user_time = target_time
            app.target_pitch = target_pitch
            app.target_time = target_time
            app.user_audio = np.zeros(int(sr * duration))
            
            report = app.analyze_performance(visualize=False)
            result = app.save_performance(report)
            print(f"  {songs[i].title}: {report.total_score:.1f}/100")
            
        except Exception as e:
            print(f"  Error in song {i}: {e}")
    
    # End session and view stats
    print("\nStep 6: Ending session and viewing stats...")
    app.end_session()
    
    stats = app.get_user_stats("demo_user")
    print(f"  Total sessions: {stats['total_sessions']}")
    print(f"  Total songs: {stats['total_songs']}")
    print(f"  Average score: {stats['average_score']:.1f}/100")
    print(f"  Best score: {stats['best_score']:.1f}/100")
    
    top_songs = app.get_user_top_songs("demo_user", limit=3)
    print("\n  Top songs:")
    for song_stat in top_songs:
        print(f"    {song_stat['song_title']}: {song_stat['best_score']:.1f} (attempts: {song_stat['attempts']})")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
