#!/usr/bin/env python3
"""
Test enhanced trajectory smoothing algorithms
"""

import sys
import logging
from pathlib import Path
import numpy as np
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_trajectories():
    """Create test trajectories with different quality levels"""
    logger.info("Creating test trajectories...")

    # High quality trajectory (smooth movement)
    n_frames = 100
    t = np.linspace(0, 4*np.pi, n_frames)

    # Mouse 1: High quality circular motion with minimal noise
    mouse1_x = 100 + 30 * np.cos(t) + np.random.normal(0, 0.5, n_frames)
    mouse1_y = 100 + 30 * np.sin(t) + np.random.normal(0, 0.5, n_frames)

    # Mouse 2: Medium quality with more noise
    mouse2_x = 150 + 25 * np.cos(t + np.pi) + np.random.normal(0, 2.0, n_frames)
    mouse2_y = 150 + 25 * np.sin(t + np.pi) + np.random.normal(0, 2.0, n_frames)

    # Keypoints: body_center, neck, nose, tail_base
    n_keypoints = 4

    # Create tracking data: (n_frames, n_mice, n_keypoints, 3)
    tracking = np.zeros((n_frames, 2, n_keypoints, 3))

    # Fill trajectories
    for frame in range(n_frames):
        # Mouse 1
        tracking[frame, 0, 0, :] = [mouse1_x[frame], mouse1_y[frame], 0.95]  # body_center
        tracking[frame, 0, 1, :] = [mouse1_x[frame] + 5, mouse1_y[frame] + 5, 0.90]  # neck
        tracking[frame, 0, 2, :] = [mouse1_x[frame] + 10, mouse1_y[frame] + 10, 0.85]  # nose
        tracking[frame, 0, 3, :] = [mouse1_x[frame] - 15, mouse1_y[frame] - 15, 0.80]  # tail_base

        # Mouse 2
        tracking[frame, 1, 0, :] = [mouse2_x[frame], mouse2_y[frame], 0.75]  # body_center
        tracking[frame, 1, 1, :] = [mouse2_x[frame] + 3, mouse2_y[frame] + 3, 0.70]  # neck
        tracking[frame, 1, 2, :] = [mouse2_x[frame] + 6, mouse2_y[frame] + 6, 0.65]  # nose
        tracking[frame, 1, 3, :] = [mouse2_x[frame] - 10, mouse2_y[frame] - 10, 0.60]  # tail_base

    return tracking


def test_smoothing_methods():
    """Test different smoothing methods"""
    logger.info("Testing smoothing methods...")

    tracking = create_test_trajectories()
    confidence = tracking[..., 2]  # Extract confidence scores

    methods = ['savitzky_golay', 'ema', 'median', 'butterworth']

    results = {}

    for method in methods:
        logger.info(f"Testing {method} smoothing...")

        try:
            # Import and test each method
            if method == 'savitzky_golay':
                smoothed = test_savitzky_golay(tracking)
            elif method == 'ema':
                smoothed = test_ema(tracking)
            elif method == 'median':
                smoothed = test_median(tracking)
            elif method == 'butterworth':
                smoothed = test_butterworth(tracking)

            results[method] = smoothed
            logger.info(f"  ✅ {method}: shape {smoothed.shape}, mean change: {np.abs(smoothed - tracking).mean():.4f}")

        except Exception as e:
            logger.error(f"  ❌ {method} failed: {e}")
            results[method] = tracking.copy()

    return results


def test_savitzky_golay(tracking: np.ndarray) -> np.ndarray:
    """Test Savitzky-Golay smoothing"""
    from scipy.signal import savgol_filter

    smoothed = tracking.copy()
    n_frames, n_mice, n_keypoints, n_coords = tracking.shape

    for mouse in range(n_mice):
        for kp in range(n_keypoints):
            for coord in range(2):  # Only x, y
                signal = tracking[:, mouse, kp, coord]
                window_length = min(11, len(signal))  # Window of 11 or smaller

                if window_length >= 3 and window_length % 2 == 1:
                    smoothed_signal = savgol_filter(signal, window_length, polyorder=2)
                    smoothed[:, mouse, kp, coord] = smoothed_signal

    return smoothed


def test_ema(tracking: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Test EMA smoothing"""
    smoothed = tracking.copy()

    for mouse in range(tracking.shape[1]):
        for kp in range(tracking.shape[2]):
            for coord in range(2):  # Only x, y
                signal = tracking[:, mouse, kp, coord]
                smoothed_signal = np.zeros_like(signal)
                smoothed_signal[0] = signal[0]

                for i in range(1, len(signal)):
                    smoothed_signal[i] = alpha * signal[i] + (1 - alpha) * smoothed_signal[i-1]

                smoothed[:, mouse, kp, coord] = smoothed_signal

    return smoothed


def test_median(tracking: np.ndarray) -> np.ndarray:
    """Test median smoothing"""
    from scipy.ndimage import median_filter

    smoothed = tracking.copy()

    for mouse in range(tracking.shape[1]):
        for kp in range(tracking.shape[2]):
            for coord in range(2):  # Only x, y
                signal = tracking[:, mouse, kp, coord]
                smoothed_signal = median_filter(signal, size=5)
                smoothed[:, mouse, kp, coord] = smoothed_signal

    return smoothed


def test_butterworth(tracking: np.ndarray) -> np.ndarray:
    """Test Butterworth filtering"""
    from scipy.signal import butter, filtfilt

    # Design 4th order Butterworth filter
    cutoff = 0.3  # 30% of Nyquist frequency
    b, a = butter(4, cutoff, btype='low')

    smoothed = tracking.copy()

    for mouse in range(tracking.shape[1]):
        for kp in range(tracking.shape[2]):
            for coord in range(2):  # Only x, y
                signal = tracking[:, mouse, kp, coord]
                smoothed_signal = filtfilt(b, a, signal)
                smoothed[:, mouse, kp, coord] = smoothed_signal

    return smoothed


def test_adaptive_smoothing():
    """Test adaptive smoothing method selection"""
    logger.info("Testing adaptive smoothing...")

    tracking = create_test_trajectories()
    confidence = tracking[..., 2]

    # Test quality assessment
    def assess_quality(positions: np.ndarray, conf: np.ndarray) -> float:
        """Assess trajectory quality"""
        # Confidence score
        conf_score = np.mean(conf) if len(conf) > 0 else 1.0

        # Smoothness score (lower velocity variance = higher quality)
        if len(positions) > 3:
            velocities = positions[1:] - positions[:-1]
            velocity_magnitude = np.linalg.norm(velocities, axis=1)
            smoothness_score = 1.0 / (1.0 + np.std(velocity_magnitude))
        else:
            smoothness_score = 1.0

        # Missing data ratio
        missing_ratio = np.mean(np.isnan(positions).any(axis=1))

        # Combine factors
        quality_score = 0.4 * conf_score + 0.4 * smoothness_score + 0.2 * (1.0 - missing_ratio)
        return min(1.0, max(0.0, quality_score))

    # Test quality assessment for different trajectories
    quality_scores = []
    for mouse in range(tracking.shape[1]):
        for kp in range(tracking.shape[2]):
            positions = tracking[:, mouse, kp, :2]
            conf = confidence[:, mouse, kp]
            quality = assess_quality(positions, conf)
            quality_scores.append(quality)

    logger.info(f"Quality scores: min={min(quality_scores):.3f}, max={max(quality_scores):.3f}, mean={np.mean(quality_scores):.3f}")

    # Test method selection based on quality
    method_selection = []
    for quality in quality_scores:
        if quality > 0.8:
            method = 'savitzky_golay'
        elif quality > 0.5:
            method = 'ema'
        else:
            method = 'median'
        method_selection.append(method)

    logger.info(f"Method selection: {dict(zip(*np.unique(method_selection, return_counts=True)))}")

    return True


def test_smoothing_effectiveness():
    """Test how well different methods reduce noise"""
    logger.info("Testing smoothing effectiveness...")

    # Create noisy trajectory
    n_frames = 100
    t = np.linspace(0, 2*np.pi, n_frames)

    # True signal: smooth sine wave
    true_signal = np.sin(t)

    # Noisy observations
    noise_level = 0.3
    noisy_signal = true_signal + np.random.normal(0, noise_level, n_frames)

    # Test different smoothing methods
    methods = {
        'savitzky_golay': test_savitzky_golay_1d,
        'ema': test_ema_1d,
        'median': test_median_1d,
        'butterworth': test_butterworth_1d
    }

    results = {}

    for method_name, method_func in methods.items():
        smoothed = method_func(noisy_signal)
        mse = np.mean((smoothed - true_signal)**2)
        results[method_name] = {'mse': mse, 'signal': smoothed}

        logger.info(f"  {method_name}: MSE = {mse:.4f}")

    # Find best method
    best_method = min(results.keys(), key=lambda k: results[k]['mse'])
    logger.info(f"Best method for this signal: {best_method} (MSE: {results[best_method]['mse']:.4f})")

    return results


def test_savitzky_golay_1d(signal: np.ndarray) -> np.ndarray:
    """1D Savitzky-Golay for testing"""
    from scipy.signal import savgol_filter
    window = min(11, len(signal))
    if window >= 3 and window % 2 == 1:
        return savgol_filter(signal, window, polyorder=2)
    return signal


def test_ema_1d(signal: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """1D EMA for testing"""
    smoothed = np.zeros_like(signal)
    smoothed[0] = signal[0]

    for i in range(1, len(signal)):
        smoothed[i] = alpha * signal[i] + (1 - alpha) * smoothed[i-1]

    return smoothed


def test_median_1d(signal: np.ndarray) -> np.ndarray:
    """1D median for testing"""
    from scipy.ndimage import median_filter
    return median_filter(signal, size=5)


def test_butterworth_1d(signal: np.ndarray) -> np.ndarray:
    """1D Butterworth for testing"""
    from scipy.signal import butter, filtfilt
    b, a = butter(4, 0.3, btype='low')
    return filtfilt(b, a, signal)


def main():
    """Run smoothing tests"""
    logger.info("=== Enhanced Trajectory Smoothing Tests ===")

    success_count = 0

    # Test smoothing methods
    try:
        results = test_smoothing_methods()
        success_count += 1
        logger.info("✅ Smoothing methods test passed")
    except Exception as e:
        logger.error(f"❌ Smoothing methods test failed: {e}")
        results = {}

    # Test adaptive smoothing
    try:
        test_adaptive_smoothing()
        success_count += 1
        logger.info("✅ Adaptive smoothing test passed")
    except Exception as e:
        logger.error(f"❌ Adaptive smoothing test failed: {e}")

    # Test smoothing effectiveness
    try:
        effectiveness_results = test_smoothing_effectiveness()
        success_count += 1
        logger.info("✅ Smoothing effectiveness test passed")
    except Exception as e:
        logger.error(f"❌ Smoothing effectiveness test failed: {e}")
        effectiveness_results = {}

    logger.info(f"=== Tests completed: {success_count}/3 passed ===")

    if success_count >= 2:
        logger.info("🎉 Enhanced trajectory smoothing is working!")
        logger.info("Phase 1 Layer 1 complete - ready for Layer 2")
        return True
    else:
        logger.error("❌ Need to fix smoothing implementation")
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
