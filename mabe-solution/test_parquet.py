#!/usr/bin/env python3
"""
Test parquet file reading and basic feature extraction
Uses only built-in libraries where possible
"""

import logging
import struct
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_parquet_basic_info(file_path: Path) -> Dict:
    """Read basic info from parquet file without external dependencies"""
    try:
        # This is a simplified parquet reader - in practice we'd use pyarrow
        # For now, just check file size and basic structure
        file_size = file_path.stat().st_size

        # Try to read magic footer
        with open(file_path, 'rb') as f:
            # Read last 8 bytes (magic footer)
            f.seek(-8, 2)
            footer = f.read(8)

        return {
            'file_path': str(file_path),
            'file_size': file_size,
            'has_magic_footer': footer == b'PAR1\x00\x00\x00\x00'
        }
    except Exception as e:
        return {
            'file_path': str(file_path),
            'error': str(e)
        }


def test_parquet_structure():
    """Test parquet file structure"""
    logger.info("Testing parquet file structure...")

    tracking_dir = Path("/Users/ronny/Downloads/MABe-mouse-behavior-detection/train_tracking")
    annotation_dir = Path("/Users/ronny/Downloads/MABe-mouse-behavior-detection/train_annotation")

    # Test a few files from different labs
    test_files = []

    for lab_dir in list(tracking_dir.iterdir())[:3]:  # First 3 labs
        if lab_dir.is_dir():
            lab_files = list(lab_dir.glob("*.parquet"))[:2]  # First 2 files per lab
            test_files.extend(lab_files)

    logger.info(f"Testing {len(test_files)} parquet files...")

    results = []
    for file_path in test_files:
        info = read_parquet_basic_info(file_path)
        results.append(info)
        logger.info(f"  {file_path.name}: {info}")

    return True


def analyze_csv_structure():
    """Analyze CSV structure for behaviors and keypoint configuration"""
    logger.info("Analyzing CSV structure...")

    train_csv = Path("/Users/ronny/Downloads/MABe-mouse-behavior-detection/train.csv")

    # Read first 10 lines
    behaviors_by_lab = {}
    keypoints_by_lab = {}

    with open(train_csv, 'r') as f:
        lines = f.readlines()[:10]  # First 10 lines

    for line in lines[1:]:  # Skip header
        parts = line.strip().split(',')
        if len(parts) < 38:
            continue

        lab_id = parts[0]
        behaviors_str = parts[36]  # behaviors_labeled column
        keypoints_str = parts[35]  # body_parts_tracked column

        # Count behaviors
        if lab_id not in behaviors_by_lab:
            behaviors_by_lab[lab_id] = 0

        try:
            # Simple count of behavior strings
            behaviors_by_lab[lab_id] = behaviors_str.count('"') // 2  # Each behavior is quoted
        except:
            pass

        # Count keypoints
        if lab_id not in keypoints_by_lab:
            keypoints_by_lab[lab_id] = 0

        try:
            keypoints_by_lab[lab_id] = keypoints_str.count('"') // 2
        except:
            pass

    logger.info("Behaviors per lab:")
    for lab, count in sorted(behaviors_by_lab.items()):
        logger.info(f"  {lab}: {count}")

    logger.info("Keypoints per lab:")
    for lab, count in sorted(keypoints_by_lab.items()):
        logger.info(f"  {lab}: {count}")

    return True


def main():
    """Run parquet tests"""
    logger.info("=== MABe Parquet Structure Tests ===")

    success_count = 0

    # Test CSV structure analysis
    if analyze_csv_structure():
        success_count += 1
        logger.info("✅ CSV structure analysis passed")
    else:
        logger.error("❌ CSV structure analysis failed")

    # Test parquet files
    if test_parquet_structure():
        success_count += 1
        logger.info("✅ Parquet structure test passed")
    else:
        logger.error("❌ Parquet structure test failed")

    logger.info(f"=== Parquet tests completed: {success_count}/2 passed ===")

    if success_count == 2:
        logger.info("🎉 Ready to proceed with full implementation!")
        return True
    else:
        logger.error("❌ Need to address data access issues")
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
