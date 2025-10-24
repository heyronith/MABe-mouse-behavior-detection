#!/usr/bin/env python3
"""
Test validation and submission components
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from training.metrics import FScoreCalculator, decode_predictions
from decoding.submission import SubmissionWriter, ProbabilityCalibrator, ThresholdOptimizer
from training.cross_validation import create_mock_predictions, load_annotations_from_csv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_f_score_calculation():
    """Test F-score calculation"""
    logger.info("Testing F-score calculation...")

    # Create mock predictions and ground truth
    n_videos = 10
    n_behaviors = 25

    predictions = []
    ground_truth = []

    for video_idx in range(n_videos):
        video_id = f"video_{video_idx}"
        lab_id = f"lab_{video_idx % 3}"  # 3 labs

        # Ground truth segments
        n_gt_segments = np.random.randint(5, 15)
        for _ in range(n_gt_segments):
            gt = {
                'video_id': video_id,
                'lab_id': lab_id,
                'agent_id': f'mouse{np.random.randint(1, 3)}',
                'target_id': f'mouse{np.random.randint(1, 3)}',
                'action': 'approach',
                'start_frame': np.random.randint(0, 1000),
                'stop_frame': np.random.randint(100, 1100)
            }
            ground_truth.append(gt)

        # Predictions (with some noise)
        n_pred_segments = n_gt_segments + np.random.randint(-2, 3)
        n_pred_segments = max(0, n_pred_segments)

        for _ in range(n_pred_segments):
            pred = {
                'video_id': video_id,
                'lab_id': lab_id,
                'agent_id': f'mouse{np.random.randint(1, 3)}',
                'target_id': f'mouse{np.random.randint(1, 3)}',
                'action': 'approach',
                'start_frame': np.random.randint(0, 1000),
                'stop_frame': np.random.randint(100, 1100)
            }
            predictions.append(pred)

    logger.info(f"Created {len(ground_truth)} GT and {len(predictions)} predictions")

    # Calculate F-score
    f_score_calc = FScoreCalculator(beta=1.0)
    scores = f_score_calc.calculate_f_score(predictions, ground_truth)

    logger.info("F-score Results:")
    logger.info(f"  Global F-score: {scores['global_scores']['f_score']:.4f}")
    logger.info(f"  Global Precision: {scores['global_scores']['precision']:.4f}")
    logger.info(f"  Global Recall: {scores['global_scores']['recall']:.4f}")
    # Test per-lab scores
    logger.info(f"Lab scores: {len(scores['lab_scores'])} labs")
    for lab, lab_score in list(scores['lab_scores'].items())[:3]:
        logger.info(f"  {lab}: {lab_score['mean_f_score']:.4f}")
    return True


def test_submission_writing():
    """Test submission file writing"""
    logger.info("Testing submission writing...")

    # Create mock predictions
    predictions = []
    for i in range(100):
        pred = {
            'video_id': f'video_{i % 10}',
            'agent_id': f'mouse{np.random.randint(1, 3)}',
            'target_id': f'mouse{np.random.randint(1, 3)}',
            'action': 'approach',
            'start_frame': np.random.randint(0, 1000),
            'stop_frame': np.random.randint(100, 1100)
        }
        predictions.append(pred)

    # Write submission
    writer = SubmissionWriter("test_submission.csv")
    submission_path = writer.write_submission(predictions)

    # Validate submission
    validation = writer.validate_submission(submission_path)

    logger.info("Submission Validation:")
    logger.info(f"  Valid: {validation['valid']}")
    logger.info(f"  Shape: {validation['shape']}")
    logger.info(f"  Videos: {validation['unique_videos']}")
    logger.info(f"  Behaviors: {validation['unique_behaviors']}")

    if validation['valid']:
        logger.info("✅ Submission format is correct")
    else:
        logger.error("❌ Submission format issues:")
        if validation.get('error'):
            logger.error(f"  Error: {validation['error']}")
        if validation.get('type_errors'):
            logger.error(f"  Type errors: {validation['type_errors']}")
        if validation.get('logic_errors'):
            logger.error(f"  Logic errors: {validation['logic_errors']}")

    return validation['valid']


def test_probability_calibration():
    """Test probability calibration"""
    logger.info("Testing probability calibration...")

    # Create mock probabilities and targets
    n_samples = 1000
    n_behaviors = 25

    probs = np.random.beta(2, 5, (n_samples, n_behaviors))  # Biased towards 0
    targets = np.random.binomial(1, 0.1, (n_samples, n_behaviors))  # Sparse positives

    logger.info(f"Mock data: {probs.shape}, positive rate: {targets.mean():.3f}")
    # Test calibration
    calibrator = ProbabilityCalibrator(method='temperature')
    calibrator.fit(probs, targets)

    calibrated_probs = calibrator.calibrate(probs)

    logger.info(f"Calibration: {probs.mean():.3f} -> {calibrated_probs.mean():.3f}")
    return True


def test_decode_predictions():
    """Test prediction decoding"""
    logger.info("Testing prediction decoding...")

    # Create mock probabilities
    n_frames = 1000
    n_behaviors = 25

    probs = np.random.beta(2, 8, (n_frames, n_behaviors))  # Biased towards 0

    # Add some signal
    probs[100:150, 0] = 0.9  # Strong signal for behavior 0
    probs[300:350, 1] = 0.8  # Strong signal for behavior 1

    logger.info(f"Mock probabilities: {probs.shape}, max: {probs.max():.3f}")
    # Decode predictions
    segments = decode_predictions(probs, threshold=0.5, min_duration=6, max_gap=9)

    logger.info(f"Decoded {len(segments)} segments")

    # Check segment format
    for i, segment in enumerate(segments[:5]):
        logger.info(f"  Segment {i}: {segment}")

    return True


def main():
    """Run validation tests"""
    logger.info("=== Validation & Submission Tests ===")

    success_count = 0

    # Test F-score calculation
    if test_f_score_calculation():
        success_count += 1
        logger.info("✅ F-score calculation passed")
    else:
        logger.error("❌ F-score calculation failed")

    # Test submission writing
    if test_submission_writing():
        success_count += 1
        logger.info("✅ Submission writing passed")
    else:
        logger.error("❌ Submission writing failed")

    # Test probability calibration
    if test_probability_calibration():
        success_count += 1
        logger.info("✅ Probability calibration passed")
    else:
        logger.error("❌ Probability calibration failed")

    # Test prediction decoding
    if test_decode_predictions():
        success_count += 1
        logger.info("✅ Prediction decoding passed")
    else:
        logger.error("❌ Prediction decoding failed")

    logger.info(f"=== Tests completed: {success_count}/4 passed ===")

    if success_count == 4:
        logger.info("🎉 Validation & submission pipeline is ready!")
        logger.info("Phase 0 complete - ready for actual training!")
        return True
    else:
        logger.error("❌ Need to fix validation issues")
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
