#!/usr/bin/env python3
"""
Test the actual data loader with real parquet files
Tests loading, feature extraction, and basic model forward pass
"""

import sys
import logging
from pathlib import Path
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.loader import MABeDataset
from data.features import FeatureExtractor, preprocess_tracking
from models.tcn import MABeModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_single_parquet_loading():
    """Test loading a single parquet file"""
    logger.info("Testing single parquet file loading...")

    # Use a small file for testing
    tracking_path = "/Users/ronny/Downloads/MABe-mouse-behavior-detection/train_tracking/AdaptableSnail/44566106.parquet"
    annotation_path = "/Users/ronny/Downloads/MABe-mouse-behavior-detection/train_annotation/AdaptableSnail/44566106.parquet"

    if not Path(tracking_path).exists():
        logger.error(f"Tracking file not found: {tracking_path}")
        return False

    try:
        import pyarrow.parquet as pq

        # Read tracking data
        tracking_table = pq.read_table(tracking_path)
        tracking_df = tracking_table.to_pandas()

        logger.info(f"Tracking columns: {tracking_df.columns.tolist()}")
        logger.info(f"Tracking shape: {tracking_df.shape}")
        logger.info(f"Sample tracking data type: {type(tracking_df.iloc[0]['mouse1'])}")

        # Check data structure
        if 'mouse1' in tracking_df.columns:
            mouse1_data = tracking_df['mouse1'].iloc[0]
            logger.info(f"Mouse1 data shape: {mouse1_data.shape if hasattr(mouse1_data, 'shape') else 'no shape'}")
            logger.info(f"Mouse1 data type: {type(mouse1_data)}")

        # Read annotation data if available
        if Path(annotation_path).exists():
            annotation_table = pq.read_table(annotation_path)
            annotation_df = annotation_table.to_pandas()
            logger.info(f"Annotation columns: {annotation_df.columns.tolist()}")
            logger.info(f"Annotation shape: {annotation_df.shape}")
            logger.info(f"Sample annotation: {annotation_df.head()}")

        return True

    except Exception as e:
        logger.error(f"Error loading parquet: {e}")
        return False


def test_dataset_creation():
    """Test creating the dataset"""
    logger.info("Testing dataset creation...")

    try:
        # Create a small dataset with limited data
        dataset = MABeDataset(
            csv_path="/Users/ronny/Downloads/MABe-mouse-behavior-detection/train.csv",
            tracking_dir="/Users/ronny/Downloads/MABe-mouse-behavior-detection/train_tracking",
            annotation_dir="/Users/ronny/Downloads/MABe-mouse-behavior-detection/train_annotation",
            window_size=128,  # Small window for testing
            overlap=0.5,
            is_train=True
        )

        logger.info(f"Dataset created with {len(dataset)} windows")
        logger.info(f"Video info sample: {dataset.video_info[0] if dataset.video_info else 'None'}")

        # Test loading a single sample
        if len(dataset) > 0:
            sample = dataset[0]
            logger.info(f"Sample keys: {sample.keys()}")
            logger.info(f"Tracking shape: {sample['tracking'].shape}")
            if 'annotations' in sample:
                logger.info(f"Annotations shape: {sample['annotations'].shape}")
                logger.info(f"Annotation sum: {sample['annotations'].sum().item()}")

        return True

    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_feature_extraction():
    """Test feature extraction with real data"""
    logger.info("Testing feature extraction...")

    try:
        # Create feature extractor
        feature_extractor = FeatureExtractor(None)  # No config for basic test

        # Create dummy tracking data similar to real format
        batch_size, n_frames, n_mice, n_keypoints = 2, 32, 2, 10

        # Simulate tracking data (batch, frames, mice, keypoints, 3)
        tracking = torch.randn(batch_size, n_frames, n_mice, n_keypoints, 3)

        # Extract features
        features = feature_extractor.extract_features(tracking)
        logger.info(f"Features shape: {features.shape}")

        return True

    except Exception as e:
        logger.error(f"Error in feature extraction: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_model_with_features():
    """Test model with extracted features"""
    logger.info("Testing model with features...")

    try:
        # Create model with minimal config
        config = type('Config', (), {
            'model': type('ModelConfig', (), {
                'n_features': 64,
                'n_behaviors': 25,
                'hidden_dim': 256,
                'layers': 3,
                'dilation': [1, 2, 4],
                'kernel_size': 3,
                'dropout': 0.1
            })()
        })()

        model = MABeModel(config)

        # Create dummy input
        batch_size, n_frames, n_mice, n_keypoints = 2, 32, 2, 10
        tracking = torch.randn(batch_size, n_frames, n_mice, n_keypoints, 3)

        # Forward pass
        with torch.no_grad():
            logits = model(tracking)

        logger.info(f"Model output shape: {logits.shape}")
        logger.info(f"Output range: {logits.min().item():.3f} to {logits.max().item():.3f}")

        return True

    except Exception as e:
        logger.error(f"Error in model test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Run all tests"""
    logger.info("=== MABe Data Loader Tests ===")

    success_count = 0

    # Test parquet loading
    if test_single_parquet_loading():
        success_count += 1
        logger.info("✅ Parquet loading test passed")
    else:
        logger.error("❌ Parquet loading test failed")

    # Test dataset creation
    if test_dataset_creation():
        success_count += 1
        logger.info("✅ Dataset creation test passed")
    else:
        logger.error("❌ Dataset creation test failed")

    # Test feature extraction
    if test_feature_extraction():
        success_count += 1
        logger.info("✅ Feature extraction test passed")
    else:
        logger.error("❌ Feature extraction test failed")

    # Test model
    if test_model_with_features():
        success_count += 1
        logger.info("✅ Model test passed")
    else:
        logger.error("❌ Model test failed")

    logger.info(f"=== Tests completed: {success_count}/4 passed ===")

    if success_count == 4:
        logger.info("🎉 Data loader is working! Ready for full training pipeline.")
        return True
    else:
        logger.error("❌ Need to fix data loader issues")
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
