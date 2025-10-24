#!/usr/bin/env python3
"""
Basic test for MABe data structure understanding
Tests CSV parsing and basic data structure
"""

import csv
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_csv_structure():
    """Test CSV file structure understanding"""
    logger.info("Testing CSV structure...")

    train_csv = Path("/Users/ronny/Downloads/MABe-mouse-behavior-detection/train.csv")

    if not train_csv.exists():
        logger.error(f"Train CSV not found: {train_csv}")
        return False

    # Read first few lines
    with open(train_csv, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [next(reader) for _ in range(3)]

    logger.info(f"CSV Header ({len(header)} columns):")
    for i, col in enumerate(header):
        logger.info(f"  {i}: {col}")

    # Parse behaviors from first row
    behaviors_col = header.index('behaviors_labeled')
    behaviors_str = rows[0][behaviors_col]
    logger.info(f"Sample behaviors string: {behaviors_str[:100]}...")

    # Parse behaviors
    try:
        behaviors = json.loads(behaviors_str.replace("'", '"'))
        logger.info(f"Parsed {len(behaviors)} behaviors")
        logger.info(f"Sample behaviors: {behaviors[:5]}")
    except:
        logger.info("Behaviors format: list of strings")

    # Count unique labs
    labs = set()
    with open(train_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            labs.add(row[0])  # lab_id is first column

    logger.info(f"Found {len(labs)} unique labs: {sorted(list(labs))[:10]}...")

    return True


def test_tracking_files():
    """Test tracking file structure"""
    logger.info("Testing tracking files...")

    tracking_dir = Path("/Users/ronny/Downloads/MABe-mouse-behavior-detection/train_tracking")

    if not tracking_dir.exists():
        logger.error(f"Tracking dir not found: {tracking_dir}")
        return False

    # Count files per lab
    lab_counts = {}
    for lab_dir in tracking_dir.iterdir():
        if lab_dir.is_dir():
            lab = lab_dir.name
            count = len(list(lab_dir.glob("*.parquet")))
            lab_counts[lab] = count

    logger.info(f"Tracking files per lab: {dict(list(lab_counts.items())[:10])}")

    # Test reading a parquet file (basic info)
    first_lab = list(lab_counts.keys())[0]
    first_file = list((tracking_dir / first_lab).glob("*.parquet"))[0]

    logger.info(f"Sample file: {first_file}")
    logger.info(f"File exists: {first_file.exists()}")

    return True


def test_annotation_files():
    """Test annotation file structure"""
    logger.info("Testing annotation files...")

    annotation_dir = Path("/Users/ronny/Downloads/MABe-mouse-behavior-detection/train_annotation")

    if not annotation_dir.exists():
        logger.error(f"Annotation dir not found: {annotation_dir}")
        return False

    # Count annotation files per lab
    lab_counts = {}
    for lab_dir in annotation_dir.iterdir():
        if lab_dir.is_dir():
            lab = lab_dir.name
            count = len(list(lab_dir.glob("*.parquet")))
            lab_counts[lab] = count

    logger.info(f"Annotation files per lab: {dict(list(lab_counts.items())[:10])}")

    # Check if annotation files match tracking files
    tracking_dir = Path("/Users/ronny/Downloads/MABe-mouse-behavior-detection/train_tracking")
    matching_labs = set(lab_counts.keys()) & set([d.name for d in tracking_dir.iterdir() if d.is_dir()])
    logger.info(f"Matching labs: {len(matching_labs)}/{len(lab_counts)}")

    return True


def main():
    """Run basic tests"""
    logger.info("=== MABe Basic Data Structure Tests ===")

    success_count = 0

    # Test CSV structure
    if test_csv_structure():
        success_count += 1
        logger.info("✅ CSV structure test passed")
    else:
        logger.error("❌ CSV structure test failed")

    # Test tracking files
    if test_tracking_files():
        success_count += 1
        logger.info("✅ Tracking files test passed")
    else:
        logger.error("❌ Tracking files test failed")

    # Test annotation files
    if test_annotation_files():
        success_count += 1
        logger.info("✅ Annotation files test passed")
    else:
        logger.error("❌ Annotation files test failed")

    logger.info(f"=== Basic tests completed: {success_count}/3 passed ===")

    if success_count == 3:
        logger.info("🎉 Basic data structure understanding is solid!")
        return True
    else:
        logger.error("❌ Need to fix data access issues")
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
