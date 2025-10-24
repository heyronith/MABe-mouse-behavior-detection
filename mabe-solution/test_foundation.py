#!/usr/bin/env python3
"""
Test script for MABe Phase 0 Layer 1 - Foundation
Tests data loading, feature extraction, and basic model forward pass
"""

import sys
import logging
from pathlib import Path
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.loader import MABeDataset
from data.features import FeatureExtractor
from models.tcn import MABeModel
from utils import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_data_loading():
    """Test basic data loading functionality"""
    logger.info("Testing data loading...")

    config = load_config()

    # Create a small dataset with limited data
    dataset = MABeDataset(
        csv_path=config.data.train_csv,
        tracking_dir=config.data.train_tracking_dir,
        annotation_dir=config.data.train_annotation_dir,
        window_size=128,  # Small window for testing
        overlap=0.5,
        is_train=True
    )

    logger.info(f"Dataset size: {len(dataset)} windows")
    logger.info(f"Sample video info: {dataset.video_info[0] if dataset.video_info else 'None'}")

    # Test loading a single sample
    if len(dataset) > 0:
        sample = dataset[0]
        logger.info(f"Sample keys: {sample.keys()}")
        logger.info(f"Tracking shape: {sample['tracking'].shape}")
        if 'annotations' in sample:
            logger.info(f"Annotations shape: {sample['annotations'].shape}")

        return True
    else:
        logger.warning("No data loaded - check file paths")
        return False


def test_feature_extraction():
    """Test feature extraction"""
    logger.info("Testing feature extraction...")

    config = load_config()
    feature_extractor = FeatureExtractor(config)

    # Create dummy tracking data
    batch_size, n_frames, n_mice, n_keypoints = 2, 64, 2, 10
    tracking = torch.randn(batch_size, n_frames, n_mice, n_keypoints, 3)

    # Extract features
    features = feature_extractor.extract_features(tracking)
    logger.info(f"Features shape: {features.shape}")

    return True


def test_model_forward():
    """Test model forward pass"""
    logger.info("Testing model forward pass...")

    config = load_config()
    model = MABeModel(config)

    # Create dummy input
    batch_size, n_frames, n_mice, n_keypoints = 2, 64, 2, 10
    tracking = torch.randn(batch_size, n_frames, n_mice, n_keypoints, 3)

    # Forward pass
    with torch.no_grad():
        logits = model(tracking)

    logger.info(f"Model output shape: {logits.shape}")
    logger.info(f"Output range: {logits.min().item():.3f} to {logits.max().item():.3f}")

    return True


def main():
    """Run all tests"""
    logger.info("=== MABe Foundation Tests ===")

    success_count = 0

    # Test data loading
    if test_data_loading():
        success_count += 1
        logger.info("✅ Data loading test passed")
    else:
        logger.error("❌ Data loading test failed")

    # Test feature extraction
    if test_feature_extraction():
        success_count += 1
        logger.info("✅ Feature extraction test passed")
    else:
        logger.error("❌ Feature extraction test failed")

    # Test model forward
    if test_model_forward():
        success_count += 1
        logger.info("✅ Model forward test passed")
    else:
        logger.error("❌ Model forward test failed")

    logger.info(f"=== Tests completed: {success_count}/3 passed ===")

    if success_count == 3:
        logger.info("🎉 Foundation is ready for training!")
        return True
    else:
        logger.error("❌ Foundation needs fixes before proceeding")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
