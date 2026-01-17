import numpy as np

class Scorer:
    def score_pitch(self, user_pitch, target_pitch):
        # Ensure arrays are numpy arrays
        user_pitch = np.asarray(user_pitch)
        target_pitch = np.asarray(target_pitch)

        # Check shape alignment
        if user_pitch.shape != target_pitch.shape:
            raise ValueError(f"Shape mismatch: user_pitch {user_pitch.shape} vs target_pitch {target_pitch.shape}")

        # Combined mask for both arrays
        mask = ~np.isnan(user_pitch) & ~np.isnan(target_pitch)

        diff = np.abs(user_pitch[mask] - target_pitch[mask])

        if len(diff) == 0:
            return 0.0

        score = 100 - np.mean(diff) * 0.5
        return max(0.0, min(100.0, score))
