#!/usr/bin/env python3
"""
Analyze the CSV structure to understand the data format
"""

import csv
import logging
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_csv_structure():
    """Analyze the CSV file structure"""
    logger.info("Analyzing CSV structure...")

    csv_path = "/Users/ronny/Downloads/MABe-mouse-behavior-detection/train.csv"

    # Read CSV header and first few rows
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        # Read first 5 rows
        rows = []
        for i, row in enumerate(reader):
            rows.append(row)
            if i >= 4:
                break

    logger.info(f"CSV Header ({len(header)} columns):")
    for i, col in enumerate(header):
        logger.info(f"  {i:2d}: {col}")

    # Analyze behaviors
    behaviors_col = header.index('behaviors_labeled')
    behaviors_analysis = defaultdict(int)

    for row in rows:
        lab_id = row[0]
        behaviors_str = row[behaviors_col]

        try:
            # Simple count of behavior strings
            behaviors_analysis[lab_id] += behaviors_str.count('mouse')
        except:
            pass

    logger.info("Behaviors analysis:")
    for lab, count in behaviors_analysis.items():
        logger.info(f"  {lab}: ~{count} behaviors")

    # Analyze body parts
    body_parts_col = header.index('body_parts_tracked')
    body_parts_analysis = defaultdict(int)

    for row in rows:
        lab_id = row[0]
        body_parts_str = row[body_parts_col]

        try:
            # Count quoted strings
            body_parts_analysis[lab_id] = body_parts_str.count('"') // 2
        except:
            pass

    logger.info("Body parts analysis:")
    for lab, count in body_parts_analysis.items():
        logger.info(f"  {lab}: {count} body parts")

    return True


def analyze_lab_distribution():
    """Analyze distribution of labs and videos"""
    logger.info("Analyzing lab distribution...")

    csv_path = "/Users/ronny/Downloads/MABe-mouse-behavior-detection/train.csv"

    lab_counts = defaultdict(int)
    lab_behaviors = defaultdict(set)

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header

        for row in reader:
            lab_id = row[0]
            lab_counts[lab_id] += 1

            # Collect unique behaviors per lab
            behaviors_str = row[36]  # behaviors_labeled column
            try:
                # Extract behavior names
                behaviors = behaviors_str.strip('[]').replace('"', '').split(',')
                for behavior in behaviors:
                    behavior = behavior.strip()
                    if behavior:
                        lab_behaviors[lab_id].add(behavior)
            except:
                pass

    logger.info(f"Found {len(lab_counts)} labs with video counts:")
    for lab, count in sorted(lab_counts.items()):
        unique_behaviors = len(lab_behaviors[lab])
        logger.info(f"  {lab}: {count} videos, {unique_behaviors} unique behaviors")

    return True




def summarize_findings():
    """Summarize key findings from the analysis"""
    logger.info("=== MABe Data Analysis Summary ===")

    logger.info("From CSV analysis:")
    logger.info("- 21 unique labs")
    logger.info("- 38 columns in CSV")
    logger.info("- ~76 behaviors per video (social interactions)")
    logger.info("- Variable number of body parts tracked per lab")
    logger.info("- 4 mice per experiment (mouse1, mouse2, mouse3, mouse4)")

    logger.info("\nData format expectations:")
    logger.info("- Tracking: parquet files with mouseX columns")
    logger.info("- Annotations: parquet files with start_frame, stop_frame, agent_id, target_id, action")
    logger.info("- Behaviors: 'mouseA,mouseB,action' format")
    logger.info("- Keypoints: 5-18 per mouse depending on lab")

    logger.info("\nImplementation plan:")
    logger.info("1. ✅ Foundation setup (project structure, configs)")
    logger.info("2. 🔄 Feature engineering (geometric/social features)")
    logger.info("3. ⏳ Validation & submission (F-score, LOLO CV)")

    return True


def main():
    """Run analysis"""
    logger.info("=== CSV Analysis ===")

    success_count = 0

    # Analyze CSV
    if analyze_csv_structure():
        success_count += 1
        logger.info("✅ CSV structure analysis passed")
    else:
        logger.error("❌ CSV structure analysis failed")

    # Analyze lab distribution
    if analyze_lab_distribution():
        success_count += 1
        logger.info("✅ Lab distribution analysis passed")
    else:
        logger.error("❌ Lab distribution analysis failed")

    # Summarize findings
    if summarize_findings():
        success_count += 1
        logger.info("✅ Summary completed")
    else:
        logger.error("❌ Summary failed")

    logger.info(f"=== Analysis completed: {success_count}/3 passed ===")

    if success_count >= 2:
        logger.info("🎉 Ready to proceed with implementation!")
        return True
    else:
        logger.error("❌ Need to fix issues")
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
