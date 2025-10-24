#!/usr/bin/env python3
"""
Inspect actual parquet files to understand their structure
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def inspect_single_file():
    """Inspect a single parquet file in detail"""
    logger.info("Inspecting single parquet file...")

    # Use a small file
    tracking_path = "/Users/ronny/Downloads/MABe-mouse-behavior-detection/train_tracking/AdaptableSnail/44566106.parquet"

    if not Path(tracking_path).exists():
        logger.error(f"File not found: {tracking_path}")
        return False

    try:
        # Read with pyarrow
        table = pq.read_table(tracking_path)
        df = table.to_pandas()

        logger.info(f"Table schema: {table.schema}")
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")

        # Inspect first few rows
        logger.info("First few rows:")
        logger.info(df.head())

        # Check data types
        logger.info("Data types:")
        for col in df.columns:
            logger.info(f"  {col}: {df[col].dtype}")

        # Inspect mouse1 column specifically
        if 'mouse1' in df.columns:
            mouse1_data = df['mouse1']
            logger.info(f"Mouse1 column type: {type(mouse1_data)}")
            logger.info(f"Mouse1 length: {len(mouse1_data)}")

            # Look at first element
            first_element = mouse1_data.iloc[0]
            logger.info(f"First mouse1 element type: {type(first_element)}")

            if hasattr(first_element, 'shape'):
                logger.info(f"First mouse1 element shape: {first_element.shape}")
                logger.info(f"First mouse1 element:\n{first_element}")

        return True

    except Exception as e:
        logger.error(f"Error inspecting file: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def inspect_annotation_file():
    """Inspect annotation file structure"""
    logger.info("Inspecting annotation file...")

    annotation_path = "/Users/ronny/Downloads/MABe-mouse-behavior-detection/train_annotation/AdaptableSnail/44566106.parquet"

    if not Path(annotation_path).exists():
        logger.error(f"Annotation file not found: {annotation_path}")
        return False

    try:
        table = pq.read_table(annotation_path)
        df = table.to_pandas()

        logger.info(f"Annotation schema: {table.schema}")
        logger.info(f"Annotation shape: {df.shape}")
        logger.info(f"Annotation columns: {df.columns.tolist()}")
        logger.info("Annotation sample:")
        logger.info(df.head())

        # Count unique behaviors
        if 'action' in df.columns:
            unique_actions = df['action'].unique()
            logger.info(f"Unique actions: {unique_actions}")

        return True

    except Exception as e:
        logger.error(f"Error inspecting annotation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_numpy_conversion():
    """Test converting parquet data to numpy arrays"""
    logger.info("Testing numpy conversion...")

    tracking_path = "/Users/ronny/Downloads/MABe-mouse-behavior-detection/train_tracking/AdaptableSnail/44566106.parquet"

    try:
        df = pq.read_table(tracking_path).to_pandas()

        # Try to convert mouse1 data to numpy
        if 'mouse1' in df.columns:
            mouse1_data = df['mouse1'].iloc[0]

            if hasattr(mouse1_data, 'shape'):
                # Already a numpy array
                logger.info(f"Mouse1 data is already numpy array with shape {mouse1_data.shape}")
                logger.info(f"Mouse1 data type: {mouse1_data.dtype}")

                # Test reshaping
                reshaped = mouse1_data.reshape(-1, 3)
                logger.info(f"Reshaped to: {reshaped.shape}")

            else:
                logger.info(f"Mouse1 data type: {type(mouse1_data)}")
                logger.info(f"Mouse1 data: {mouse1_data}")

        return True

    except Exception as e:
        logger.error(f"Error in numpy conversion: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Run inspection"""
    logger.info("=== Parquet File Inspection ===")

    success_count = 0

    # Inspect tracking file
    if inspect_single_file():
        success_count += 1
        logger.info("✅ Tracking file inspection passed")
    else:
        logger.error("❌ Tracking file inspection failed")

    # Inspect annotation file
    if inspect_annotation_file():
        success_count += 1
        logger.info("✅ Annotation file inspection passed")
    else:
        logger.error("❌ Annotation file inspection failed")

    # Test numpy conversion
    if test_numpy_conversion():
        success_count += 1
        logger.info("✅ Numpy conversion test passed")
    else:
        logger.error("❌ Numpy conversion test failed")

    logger.info(f"=== Inspection completed: {success_count}/3 passed ===")

    if success_count >= 2:
        logger.info("🎉 Understanding parquet structure!")
        return True
    else:
        logger.error("❌ Need to understand data format better")
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
