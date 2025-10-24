#!/usr/bin/env python3
"""
Test feature engineering with mock data
Tests the geometric and social feature extraction
"""

import sys
import logging
from pathlib import Path
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.features import FeatureExtractor
from data.preprocessing import TrajectorySmoother, ArenaNormalizer, EgocentricTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mock_tracking_data():
    """Create realistic mock tracking data based on CSV analysis"""
    logger.info("Creating mock tracking data...")

    # Based on CSV analysis:
    # - 21 labs with different video counts
    # - ~76 behaviors per video
    # - Variable keypoints (5-18 per mouse)
    # - 4 mice per experiment

    batch_size = 2
    n_frames = 64
    n_mice = 2  # Start with 2 mice for simplicity
    n_keypoints = 10  # Based on CSV analysis

    # Create tracking data: (batch, frames, mice, keypoints, 3)
    # 3 = (x, y, confidence)
    tracking = torch.zeros(batch_size, n_frames, n_mice, n_keypoints, 3)

    # Simulate mouse movement patterns
    for batch in range(batch_size):
        for mouse in range(n_mice):
            # Create realistic trajectories
            t = torch.linspace(0, 4*np.pi, n_frames)

            # Mouse moving in a circle with some noise
            center_x = 100 + mouse * 50
            center_y = 100

            # Circular motion with noise
            x = center_x + 30 * torch.cos(t + mouse * np.pi) + torch.randn(n_frames) * 5
            y = center_y + 30 * torch.sin(t + mouse * np.pi) + torch.randn(n_frames) * 5

            # Set positions (simulate body_center, neck, nose, tail_base)
            tracking[batch, :, mouse, 0, :2] = torch.stack([x, y], dim=1)  # body_center
            tracking[batch, :, mouse, 6, :2] = torch.stack([x + 10 * torch.cos(t), y + 10 * torch.sin(t)], dim=1)  # neck/nose
            tracking[batch, :, mouse, 7, :2] = torch.stack([x - 20 * torch.cos(t), y - 20 * torch.sin(t)], dim=1)  # tail_base

            # Set confidence to 1.0 for all keypoints
            tracking[batch, :, mouse, :, 2] = 1.0

    logger.info(f"Created tracking data: {tracking.shape}")
    return tracking


def test_trajectory_smoothing():
    """Test trajectory smoothing"""
    logger.info("Testing trajectory smoothing...")

    try:
        tracking = create_mock_tracking_data()

        # Test different smoothing methods
        methods = ['savitzky_golay', 'ema', 'median']

        for method in methods:
            smoother = TrajectorySmoother(method=method, window=5)
            smoothed = smoother.smooth(tracking)

            # Check that smoothing doesn't break the data
            assert smoothed.shape == tracking.shape
            assert not torch.isnan(smoothed).any()

            logger.info(f"✅ {method} smoothing: {smoothed.shape}")

        return True

    except Exception as e:
        logger.error(f"Error in trajectory smoothing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_preprocessing():
    """Test preprocessing pipeline"""
    logger.info("Testing preprocessing pipeline...")

    try:
        tracking = create_mock_tracking_data()

        # Test arena normalization
        normalizer = ArenaNormalizer(center=True, scale=True)
        normalized = normalizer.normalize(tracking)

        logger.info(f"Arena normalization: {normalized.shape}")

        # Test egocentric transformation
        transformer = EgocentricTransformer()
        transformed = transformer.transform(tracking)

        logger.info(f"Egocentric transformation: {transformed.shape}")

        return True

    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_feature_extraction():
    """Test feature extraction"""
    logger.info("Testing feature extraction...")

    try:
        tracking = create_mock_tracking_data()

        # Create minimal config
        config = type('Config', (), {})()

        # Test feature extractor
        feature_extractor = FeatureExtractor(config)
        features = feature_extractor.extract_features(tracking)

        logger.info(f"Features extracted: {features.shape}")
        logger.info(f"Feature stats - min: {features.min().item():.3f}, max: {features.max().item():.3f}, mean: {features.mean().item():.3f}")

        # Check feature dimensions make sense
        batch_size, n_frames, n_features = features.shape
        expected_features_per_mouse = 3  # speed, acceleration, angular velocity
        expected_geometric_per_mouse = 2  # body length, head orientation
        expected_social_pairs = 3  # pairwise distances
        expected_facing_scores = 1  # per pair

        max_expected = n_mice * (expected_features_per_mouse + expected_geometric_per_mouse) + 3 * expected_social_pairs + 1 * expected_facing_scores

        logger.info(f"Expected max features: ~{max_expected}, got: {n_features}")

        return True

    except Exception as e:
        logger.error(f"Error in feature extraction: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_model_integration():
    """Test integration with model"""
    logger.info("Testing model integration...")

    try:
        tracking = create_mock_tracking_data()

        # Extract features
        config = type('Config', (), {})()
        feature_extractor = FeatureExtractor(config)
        features = feature_extractor.extract_features(tracking)

        # Test with model
        from models.tcn import MABeModel

        model_config = type('ModelConfig', (), {
            'n_features': features.shape[-1],
            'n_behaviors': 25,  # Based on CSV analysis
            'hidden_dim': 256,
            'layers': 3,
            'dilation': [1, 2, 4],
            'kernel_size': 3,
            'dropout': 0.1
        })()
        config.model = model_config

        model = MABeModel(config)

        # Forward pass
        with torch.no_grad():
            logits = model(tracking)

        logger.info(f"Model output: {logits.shape}")
        logger.info(f"Output range: {logits.min().item():.3f} to {logits.max().item():.3f}")

        # Test loss
        from training.trainer import EffectiveNumLoss

        # Create mock annotations (sparse)
        batch_size, n_frames, _ = logits.shape
        annotations = torch.zeros(batch_size, n_frames, model_config.n_behaviors)

        # Add some mock positive examples
        annotations[0, 10:15, 0] = 1.0  # Some behavior
        annotations[0, 20:25, 1] = 1.0  # Another behavior

        loss_fn = EffectiveNumLoss()
        loss = loss_fn(logits, annotations)
        logger.info(f"Loss: {loss.item():.4f}")

        return True

    except Exception as e:
        logger.error(f"Error in model integration: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Run all tests"""
    logger.info("=== Feature Engineering Tests ===")

    success_count = 0

    # Test trajectory smoothing
    if test_trajectory_smoothing():
        success_count += 1
        logger.info("✅ Trajectory smoothing test passed")
    else:
        logger.error("❌ Trajectory smoothing test failed")

    # Test preprocessing
    if test_preprocessing():
        success_count += 1
        logger.info("✅ Preprocessing test passed")
    else:
        logger.error("❌ Preprocessing test failed")

    # Test feature extraction
    if test_feature_extraction():
        success_count += 1
        logger.info("✅ Feature extraction test passed")
    else:
        logger.error("❌ Feature extraction test failed")

    # Test model integration
    if test_model_integration():
        success_count += 1
        logger.info("✅ Model integration test passed")
    else:
        logger.error("❌ Model integration test failed")

    logger.info(f"=== Tests completed: {success_count}/4 passed ===")

    if success_count == 4:
        logger.info("🎉 Feature engineering pipeline is working!")
        logger.info("Ready for Phase 0 Layer 3 (Validation & Submission)")
        return True
    else:
        logger.error("❌ Need to fix feature engineering issues")
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
