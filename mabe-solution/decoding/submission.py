import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SubmissionWriter:
    """Write predictions in Kaggle submission format"""

    def __init__(self, output_path: str = "submission.csv"):
        self.output_path = output_path

    def write_submission(self,
                        predictions: List[Dict],
                        test_videos: Optional[List[str]] = None) -> str:
        """
        Write predictions to submission CSV

        Args:
            predictions: List of prediction dictionaries
            test_videos: List of test video IDs (for validation)

        Returns:
            Path to written submission file
        """
        # Ensure required fields
        required_fields = ['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']
        for pred in predictions:
            for field in required_fields:
                if field not in pred:
                    logger.warning(f"Missing field {field} in prediction: {pred}")

        # Create DataFrame
        df = pd.DataFrame(predictions)

        # Add row_id (required by submission format)
        df['row_id'] = range(len(df))

        # Reorder columns to match submission format
        submission_columns = ['row_id', 'video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']
        df = df[submission_columns]

        # Ensure correct data types
        df['start_frame'] = df['start_frame'].astype(int)
        df['stop_frame'] = df['stop_frame'].astype(int)
        df['row_id'] = df['row_id'].astype(int)

        # Write to CSV
        df.to_csv(self.output_path, index=False)

        logger.info(f"Written submission to {self.output_path}")
        logger.info(f"Submission shape: {df.shape}")
        logger.info(f"Unique videos: {df['video_id'].nunique()}")
        logger.info(f"Unique behaviors: {df['action'].nunique()}")

        return self.output_path

    def validate_submission(self, submission_path: str) -> Dict[str, any]:
        """Validate submission file format"""
        try:
            df = pd.read_csv(submission_path)

            # Check required columns
            required_columns = ['row_id', 'video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']
            missing_columns = set(required_columns) - set(df.columns)

            if missing_columns:
                return {
                    'valid': False,
                    'error': f"Missing columns: {missing_columns}",
                    'shape': df.shape
                }

            # Check data types
            type_errors = []
            if not df['start_frame'].dtype in ['int64', 'int32']:
                type_errors.append('start_frame should be integer')
            if not df['stop_frame'].dtype in ['int64', 'int32']:
                type_errors.append('stop_frame should be integer')
            if not df['row_id'].dtype in ['int64', 'int32']:
                type_errors.append('row_id should be integer')

            # Check logical constraints
            logic_errors = []
            if (df['start_frame'] >= df['stop_frame']).any():
                logic_errors.append('start_frame >= stop_frame')
            if (df['start_frame'] < 0).any():
                logic_errors.append('start_frame < 0')
            if (df['stop_frame'] < 0).any():
                logic_errors.append('stop_frame < 0')

            return {
                'valid': len(type_errors) == 0 and len(logic_errors) == 0,
                'shape': df.shape,
                'type_errors': type_errors,
                'logic_errors': logic_errors,
                'unique_videos': df['video_id'].nunique(),
                'unique_behaviors': df['action'].nunique(),
                'total_segments': len(df)
            }

        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'shape': None
            }


class ProbabilityCalibrator:
    """Calibrate model probabilities for better F-score"""

    def __init__(self, method: str = 'temperature'):
        self.method = method
        self.calibration_params = {}

    def fit(self, probs: np.ndarray, targets: np.ndarray, behavior_indices: Optional[List[int]] = None):
        """
        Fit calibration parameters

        Args:
            probs: Predicted probabilities (n_samples, n_behaviors)
            targets: Ground truth labels (n_samples, n_behaviors)
            behavior_indices: Which behaviors to calibrate (default: all)
        """
        if behavior_indices is None:
            behavior_indices = list(range(probs.shape[1]))

        for behavior_idx in behavior_indices:
            behavior_probs = probs[:, behavior_idx]
            behavior_targets = targets[:, behavior_idx]

            if self.method == 'temperature':
                self._fit_temperature_scaling(behavior_probs, behavior_targets, behavior_idx)
            elif self.method == 'isotonic':
                self._fit_isotonic_regression(behavior_probs, behavior_targets, behavior_idx)

    def _fit_temperature_scaling(self, probs: np.ndarray, targets: np.ndarray, behavior_idx: int):
        """Fit temperature scaling parameters"""
        # Simple temperature scaling: T = mean(probs) / mean(targets)
        positive_mask = targets > 0
        if positive_mask.sum() > 0:
            mean_prob = probs[positive_mask].mean()
            mean_target = targets[positive_mask].mean()

            if mean_target > 0:
                temperature = mean_prob / mean_target
                temperature = np.clip(temperature, 0.1, 10.0)  # Reasonable range
            else:
                temperature = 1.0
        else:
            temperature = 1.0

        self.calibration_params[behavior_idx] = {'temperature': temperature}

    def _fit_isotonic_regression(self, probs: np.ndarray, targets: np.ndarray, behavior_idx: int):
        """Fit isotonic regression (simplified)"""
        # For simplicity, use same method as temperature scaling
        self._fit_temperature_scaling(probs, targets, behavior_idx)

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Calibrate probabilities"""
        calibrated = probs.copy()

        for behavior_idx in range(probs.shape[1]):
            if behavior_idx in self.calibration_params:
                params = self.calibration_params[behavior_idx]

                if self.method == 'temperature':
                    temperature = params['temperature']
                    calibrated[:, behavior_idx] = self._apply_temperature_scaling(
                        probs[:, behavior_idx], temperature
                    )
                elif self.method == 'isotonic':
                    # Simplified isotonic - same as temperature for now
                    temperature = params['temperature']
                    calibrated[:, behavior_idx] = self._apply_temperature_scaling(
                        probs[:, behavior_idx], temperature
                    )

        return calibrated

    def _apply_temperature_scaling(self, probs: np.ndarray, temperature: float) -> np.ndarray:
        """Apply temperature scaling"""
        # Convert to logits, scale, convert back to probabilities
        eps = 1e-8
        probs = np.clip(probs, eps, 1 - eps)
        logits = np.log(probs / (1 - probs))
        scaled_logits = logits / temperature
        calibrated_probs = 1 / (1 + np.exp(-scaled_logits))
        return np.clip(calibrated_probs, eps, 1 - eps)


class ThresholdOptimizer:
    """Optimize thresholds for maximum F-score"""

    def __init__(self, f_score_calculator: 'FScoreCalculator'):
        self.f_score_calculator = f_score_calculator

    def optimize_thresholds(self,
                           predictions: List[Dict],
                           ground_truth: List[Dict],
                           threshold_range: Tuple[float, float] = (0.1, 0.9),
                           n_thresholds: int = 20) -> Dict[str, float]:
        """
        Find optimal thresholds for each behavior

        Args:
            predictions: Model predictions
            ground_truth: Ground truth annotations
            threshold_range: Range of thresholds to search
            n_thresholds: Number of threshold values to try

        Returns:
            Dictionary mapping behavior names to optimal thresholds
        """
        # Group by behavior
        behaviors = set()
        for pred in predictions:
            behaviors.add(pred['action'])
        for gt in ground_truth:
            behaviors.add(gt['action'])

        behaviors = sorted(list(behaviors))
        optimal_thresholds = {}

        # Optimize threshold for each behavior
        for behavior in behaviors:
            best_threshold = self._optimize_single_behavior(
                behavior, predictions, ground_truth, threshold_range, n_thresholds
            )
            optimal_thresholds[behavior] = best_threshold

        return optimal_thresholds

    def _optimize_single_behavior(self,
                                behavior: str,
                                predictions: List[Dict],
                                ground_truth: List[Dict],
                                threshold_range: Tuple[float, float],
                                n_thresholds: int) -> float:
        """Optimize threshold for a single behavior"""
        thresholds = np.linspace(threshold_range[0], threshold_range[1], n_thresholds)
        best_f_score = 0.0
        best_threshold = 0.5

        for threshold in thresholds:
            # Apply threshold to predictions
            thresholded_pred = self._apply_threshold(predictions, behavior, threshold)

            # Calculate F-score
            scores = self.f_score_calculator.calculate_f_score(thresholded_pred, ground_truth)
            f_score = scores['global_scores']['f_score']

            if f_score > best_f_score:
                best_f_score = f_score
                best_threshold = threshold

        return best_threshold

    def _apply_threshold(self, predictions: List[Dict], behavior: str, threshold: float) -> List[Dict]:
        """Apply threshold to predictions for a specific behavior"""
        thresholded = []

        for pred in predictions:
            if pred['action'] == behavior:
                # Check if this prediction should be kept (would need confidence score)
                # For now, assume all predictions above threshold are kept
                thresholded.append(pred)
            else:
                thresholded.append(pred)

        return thresholded


def create_submission_from_predictions(predictions: List[Dict],
                                     output_path: str = "submission.csv") -> str:
    """
    Create Kaggle submission from model predictions

    Args:
        predictions: List of prediction dictionaries with required fields
        output_path: Path to output submission file

    Returns:
        Path to created submission file
    """
    writer = SubmissionWriter(output_path)
    submission_path = writer.write_submission(predictions)

    # Validate submission
    validation = writer.validate_submission(submission_path)

    if validation['valid']:
        logger.info("✅ Submission validation passed")
        logger.info(f"   Shape: {validation['shape']}")
        logger.info(f"   Videos: {validation['unique_videos']}")
        logger.info(f"   Behaviors: {validation['unique_behaviors']}")
    else:
        logger.error("❌ Submission validation failed")
        logger.error(f"   Errors: {validation.get('error', 'Unknown error')}")
        if validation.get('type_errors'):
            logger.error(f"   Type errors: {validation['type_errors']}")
        if validation.get('logic_errors'):
            logger.error(f"   Logic errors: {validation['logic_errors']}")

    return submission_path
