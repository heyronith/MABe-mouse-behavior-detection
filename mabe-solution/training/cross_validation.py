import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Callable
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class CrossValidator:
    """Cross-validation framework for MABe"""

    def __init__(self, f_score_calculator, n_splits: int = 5):
        self.f_score_calculator = f_score_calculator
        self.n_splits = n_splits

    def k_fold_cross_validation(self,
                              predictions: List[Dict],
                              ground_truth: List[Dict],
                              fold_generator: Callable = None) -> Dict[str, float]:
        """
        Perform k-fold cross-validation

        Args:
            predictions: All predictions
            ground_truth: All ground truth
            fold_generator: Function to generate train/val splits

        Returns:
            Cross-validation results
        """
        if fold_generator is None:
            # Default: random splits
            fold_generator = self._random_folds

        cv_results = []

        for fold_idx in range(self.n_splits):
            logger.info(f"Cross-validation fold {fold_idx + 1}/{self.n_splits}")

            # Split data
            train_pred, val_pred, train_gt, val_gt = fold_generator(
                predictions, ground_truth, fold_idx
            )

            # Calculate scores
            scores = self.f_score_calculator.calculate_f_score(val_pred, val_gt)
            cv_results.append(scores['global_scores'])

        # Aggregate results
        aggregated = self._aggregate_cv_results(cv_results)

        return aggregated

    def _random_folds(self,
                     predictions: List[Dict],
                     ground_truth: List[Dict],
                     fold_idx: int) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
        """Generate random train/val splits"""
        # Group by video
        pred_by_video = self._group_by_video(predictions)
        gt_by_video = self._group_by_video(ground_truth)

        all_videos = list(set(pred_by_video.keys()) | set(gt_by_video.keys()))
        np.random.shuffle(all_videos)

        # Split videos into train/val
        n_val = len(all_videos) // self.n_splits
        val_start = fold_idx * n_val
        val_end = val_start + n_val if fold_idx < self.n_splits - 1 else len(all_videos)

        val_videos = all_videos[val_start:val_end]
        train_videos = [v for v in all_videos if v not in val_videos]

        # Create splits
        train_pred = [p for p in predictions if p['video_id'] in train_videos]
        val_pred = [p for p in predictions if p['video_id'] in val_videos]
        train_gt = [g for g in ground_truth if g['video_id'] in train_videos]
        val_gt = [g for g in ground_truth if g['video_id'] in val_videos]

        return train_pred, val_pred, train_gt, val_gt

    def _group_by_video(self, segments: List[Dict]) -> Dict[str, List[Dict]]:
        """Group segments by video_id"""
        by_video = {}
        for segment in segments:
            video_id = segment['video_id']
            if video_id not in by_video:
                by_video[video_id] = []
            by_video[video_id].append(segment)
        return by_video

    def _aggregate_cv_results(self, cv_results: List[Dict]) -> Dict[str, float]:
        """Aggregate cross-validation results"""
        if not cv_results:
            return {'f_score': 0.0, 'precision': 0.0, 'recall': 0.0}

        # Extract metrics
        f_scores = [r['f_score'] for r in cv_results]
        precisions = [r['precision'] for r in cv_results]
        recalls = [r['recall'] for r in cv_results]

        return {
            'f_score_mean': np.mean(f_scores),
            'f_score_std': np.std(f_scores),
            'precision_mean': np.mean(precisions),
            'recall_mean': np.mean(recalls),
            'f_score_min': np.min(f_scores),
            'f_score_max': np.max(f_scores)
        }


class LeaveOneLabOutValidator:
    """Leave-One-Lab-Out cross-validation for domain generalization"""

    def __init__(self, f_score_calculator):
        self.f_score_calculator = f_score_calculator

    def validate(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict[str, any]:
        """
        Perform Leave-One-Lab-Out validation

        Args:
            predictions: All predictions
            ground_truth: All ground truth

        Returns:
            Validation results
        """
        # Group by lab
        pred_by_lab = self._group_by_lab(predictions)
        gt_by_lab = self._group_by_lab(ground_truth)

        all_labs = sorted(set(pred_by_lab.keys()) | set(gt_by_lab.keys()))
        lab_results = {}

        for test_lab in all_labs:
            logger.info(f"Testing on lab: {test_lab}")

            # Training labs (all except test_lab)
            train_labs = [lab for lab in all_labs if lab != test_lab]

            # Get predictions and ground truth for test lab
            test_pred = pred_by_lab.get(test_lab, [])
            test_gt = gt_by_lab.get(test_lab, [])

            if not test_pred or not test_gt:
                logger.warning(f"No data for lab {test_lab}")
                continue

            # Calculate scores for test lab
            scores = self.f_score_calculator.calculate_f_score(test_pred, test_gt)
            lab_results[test_lab] = scores['global_scores']

        # Calculate summary statistics
        if lab_results:
            lab_f_scores = [scores['f_score'] for scores in lab_results.values()]

            results = {
                'global_f_score': np.mean(lab_f_scores),
                'worst_lab_f_score': np.min(lab_f_scores),
                'best_lab_f_score': np.max(lab_f_scores),
                'lab_f_scores': lab_results,
                'n_labs': len(all_labs),
                'n_tested_labs': len(lab_results)
            }
        else:
            results = {
                'global_f_score': 0.0,
                'worst_lab_f_score': 0.0,
                'best_lab_f_score': 0.0,
                'lab_f_scores': {},
                'n_labs': len(all_labs),
                'n_tested_labs': 0
            }

        return results

    def _group_by_lab(self, segments: List[Dict]) -> Dict[str, List[Dict]]:
        """Group segments by lab_id"""
        by_lab = {}
        for segment in segments:
            lab_id = segment['lab_id']
            if lab_id not in by_lab:
                by_lab[lab_id] = []
            by_lab[lab_id].append(segment)
        return by_lab


def load_annotations_from_csv(csv_path: str, tracking_dir: str) -> List[Dict]:
    """
    Load annotations from CSV and parquet files

    Args:
        csv_path: Path to train.csv
        tracking_dir: Path to train_tracking directory

    Returns:
        List of annotation dictionaries
    """
    annotations = []

    # Read CSV
    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        lab_id = row['lab_id']
        video_id = row['video_id']

        # Load annotation file
        annotation_path = Path(tracking_dir) / lab_id / f"{video_id}.parquet"
        if not annotation_path.exists():
            continue

        try:
            # Read annotation parquet
            annotation_df = pd.read_parquet(annotation_path)

            # Convert to annotation format
            for _, ann_row in annotation_df.iterrows():
                annotation = {
                    'video_id': video_id,
                    'lab_id': lab_id,
                    'agent_id': ann_row['agent_id'],
                    'target_id': ann_row['target_id'],
                    'action': ann_row['action'],
                    'start_frame': int(ann_row['start_frame']),
                    'stop_frame': int(ann_row['stop_frame'])
                }
                annotations.append(annotation)

        except Exception as e:
            logger.warning(f"Error loading annotations for {video_id}: {e}")
            continue

    logger.info(f"Loaded {len(annotations)} annotations from {len(df)} videos")
    return annotations


def create_mock_predictions(annotations: List[Dict],
                          noise_factor: float = 0.1) -> List[Dict]:
    """
    Create mock predictions for testing (with some noise)

    Args:
        annotations: Ground truth annotations
        noise_factor: Amount of noise to add (0.0 = perfect, 1.0 = random)

    Returns:
        Mock predictions
    """
    predictions = []

    for ann in annotations:
        # Copy ground truth
        pred = ann.copy()

        # Add noise to boundaries
        duration = ann['stop_frame'] - ann['start_frame']
        noise_frames = int(duration * noise_factor)

        if noise_frames > 0:
            # Random boundary adjustments
            start_noise = np.random.randint(-noise_frames, noise_frames + 1)
            stop_noise = np.random.randint(-noise_frames, noise_frames + 1)

            pred['start_frame'] = max(0, ann['start_frame'] + start_noise)
            pred['stop_frame'] = ann['stop_frame'] + stop_noise

        # Occasionally miss some annotations
        if np.random.random() > (1 - noise_factor):
            continue

        predictions.append(pred)

    logger.info(f"Created {len(predictions)} mock predictions from {len(annotations)} annotations")
    return predictions
