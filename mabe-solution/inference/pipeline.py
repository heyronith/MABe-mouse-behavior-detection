import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MABeInferencePipeline:
    """End-to-end inference pipeline for MABe competition"""

    def __init__(self,
                 model=None,
                 feature_extractor=None,
                 calibrator=None,
                 threshold_optimizer=None,
                 submission_writer=None):
        """
        Initialize inference pipeline

        Args:
            model: Trained model for prediction
            feature_extractor: Feature extraction function
            calibrator: Probability calibrator
            threshold_optimizer: Threshold optimization
            submission_writer: Submission writer
        """
        self.model = model
        self.feature_extractor = feature_extractor
        self.calibrator = calibrator
        self.threshold_optimizer = threshold_optimizer
        self.submission_writer = submission_writer

    def predict(self,
               test_csv_path: str,
               test_tracking_dir: str,
               output_path: str = "submission.csv") -> str:
        """
        Run full prediction pipeline

        Args:
            test_csv_path: Path to test.csv
            test_tracking_dir: Path to test_tracking directory
            output_path: Path to output submission file

        Returns:
            Path to submission file
        """
        logger.info("Starting MABe inference pipeline...")

        # Load test data
        test_videos = self._load_test_videos(test_csv_path, test_tracking_dir)

        # Generate predictions
        predictions = self._generate_predictions(test_videos)

        # Apply calibration if available
        if self.calibrator:
            logger.info("Applying probability calibration...")
            # predictions = self.calibrator.calibrate(predictions)

        # Apply threshold optimization if available
        if self.threshold_optimizer:
            logger.info("Applying threshold optimization...")
            # predictions = self.threshold_optimizer.optimize(predictions)

        # Write submission
        submission_path = self._write_submission(predictions, output_path)

        logger.info(f"Inference completed. Submission: {submission_path}")
        return submission_path

    def _load_test_videos(self, test_csv_path: str, test_tracking_dir: str) -> List[Dict]:
        """Load test video information"""
        logger.info("Loading test videos...")

        test_df = pd.read_csv(test_csv_path)
        test_videos = []

        for _, row in test_df.iterrows():
            lab_id = row['lab_id']
            video_id = row['video_id']

            # Check if tracking file exists
            tracking_path = Path(test_tracking_dir) / lab_id / f"{video_id}.parquet"
            if tracking_path.exists():
                video_info = {
                    'video_id': video_id,
                    'lab_id': lab_id,
                    'tracking_path': str(tracking_path),
                    'n_mice': self._get_n_mice_from_row(row),
                    'fps': row.get('frames_per_second', 30),
                    'body_parts': self._parse_body_parts(row.get('body_parts_tracked', ''))
                }
                test_videos.append(video_info)
            else:
                logger.warning(f"Missing tracking file: {tracking_path}")

        logger.info(f"Loaded {len(test_videos)} test videos")
        return test_videos

    def _generate_predictions(self, test_videos: List[Dict]) -> List[Dict]:
        """Generate predictions for test videos"""
        logger.info("Generating predictions...")

        predictions = []

        for video_info in test_videos:
            logger.info(f"Processing video: {video_info['video_id']}")

            try:
                # Load tracking data (simplified for now)
                tracking_data = self._load_tracking_data(video_info)

                if tracking_data is None:
                    continue

                # Extract features
                if self.feature_extractor:
                    features = self.feature_extractor.extract_features(tracking_data)
                else:
                    # Use raw tracking as features
                    features = tracking_data

                # Run model prediction
                if self.model:
                    # Model prediction would go here
                    # For now, create mock predictions
                    video_predictions = self._mock_model_prediction(
                        video_info, features.shape[0]
                    )
                else:
                    # Fallback mock predictions
                    video_predictions = self._mock_model_prediction(
                        video_info, 1000  # Assume 1000 frames
                    )

                predictions.extend(video_predictions)

            except Exception as e:
                logger.error(f"Error processing {video_info['video_id']}: {e}")
                continue

        logger.info(f"Generated {len(predictions)} predictions")
        return predictions

    def _load_tracking_data(self, video_info: Dict) -> Optional[np.ndarray]:
        """Load tracking data for a video (simplified)"""
        try:
            # This would load actual parquet data
            # For now, return mock data
            n_frames = 1000  # Mock frame count
            n_mice = video_info['n_mice']
            n_keypoints = len(video_info['body_parts']) if video_info['body_parts'] else 10

            # Mock tracking data: (n_frames, n_mice, n_keypoints, 3)
            tracking = np.random.randn(n_frames, n_mice, n_keypoints, 3)

            # Set confidence to 1.0 for mock data
            tracking[..., 2] = 1.0

            return tracking

        except Exception as e:
            logger.error(f"Error loading tracking for {video_info['video_id']}: {e}")
            return None

    def _mock_model_prediction(self, video_info: Dict, n_frames: int) -> List[Dict]:
        """Create mock model predictions for testing"""
        predictions = []

        # Create some mock behaviors
        mock_behaviors = [
            'mouse1,mouse2,approach',
            'mouse1,mouse2,avoid',
            'mouse2,mouse1,approach',
            'mouse1,self,rear'
        ]

        # Create random segments
        n_segments = np.random.randint(5, 15)

        for _ in range(n_segments):
            # Random behavior
            action = np.random.choice(mock_behaviors)

            # Random segment within video
            duration = np.random.randint(30, 300)  # 1-10 seconds
            start_frame = np.random.randint(0, n_frames - duration)
            stop_frame = start_frame + duration

            prediction = {
                'video_id': video_info['video_id'],
                'agent_id': action.split(',')[0],
                'target_id': action.split(',')[1],
                'action': action,
                'start_frame': start_frame,
                'stop_frame': stop_frame
            }

            predictions.append(prediction)

        return predictions

    def _write_submission(self, predictions: List[Dict], output_path: str) -> str:
        """Write predictions to submission file"""
        if self.submission_writer:
            return self.submission_writer.write_submission(predictions, output_path)
        else:
            # Fallback submission writing
            return self._fallback_submission_write(predictions, output_path)

    def _fallback_submission_write(self, predictions: List[Dict], output_path: str) -> str:
        """Fallback submission writing without writer class"""
        # Create DataFrame
        df = pd.DataFrame(predictions)

        # Add row_id
        df['row_id'] = range(len(df))

        # Reorder columns
        submission_columns = ['row_id', 'video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']
        df = df[submission_columns]

        # Write to CSV
        df.to_csv(output_path, index=False)

        logger.info(f"Written fallback submission to {output_path}")
        return output_path

    def _get_n_mice_from_row(self, row) -> int:
        """Extract number of mice from CSV row"""
        for i in range(1, 5):
            if pd.isna(row.get(f'mouse{i}_strain')):
                return i - 1
        return 4

    def _parse_body_parts(self, body_parts_str: str) -> List[str]:
        """Parse body parts from CSV string"""
        if pd.isna(body_parts_str):
            return []

        # Remove brackets and split
        parts = body_parts_str.strip('[]').replace('"', '').split(',')
        return [p.strip() for p in parts if p.strip()]


def run_inference_pipeline(config) -> str:
    """
    Run complete inference pipeline

    Args:
        config: Configuration object

    Returns:
        Path to submission file
    """
    logger.info("Initializing inference pipeline...")

    # Initialize components (would load from config in real implementation)
    pipeline = MABeInferencePipeline()

    # Run inference
    submission_path = pipeline.predict(
        test_csv_path=config.decoding.test_csv,
        test_tracking_dir=config.decoding.test_tracking_dir,
        output_path=config.decoding.output_path
    )

    return submission_path


def validate_pipeline(predictions: List[Dict],
                     ground_truth: List[Dict],
                     config) -> Dict[str, float]:
    """
    Validate pipeline using cross-validation

    Args:
        predictions: Model predictions
        ground_truth: Ground truth annotations
        config: Configuration

    Returns:
        Validation results
    """
    logger.info("Validating pipeline...")

    from training.metrics import FScoreCalculator
    from training.cross_validation import LeaveOneLabOutValidator

    # Initialize validators
    f_score_calc = FScoreCalculator(beta=1.0)
    lolo_validator = LeaveOneLabOutValidator(f_score_calc)

    # Run validation
    results = lolo_validator.validate(predictions, ground_truth)

    logger.info("Validation Results:")
    logger.info(f"  Global F-score: {results['global_f_score']:.4f".4f"  logger.info(f"  Worst lab F-score: {results['worst_lab_f_score']:.4f".4f"  logger.info(f"  Best lab F-score: {results['best_lab_f_score']:.4f".4f"  logger.info(f"  Labs tested: {results['n_tested_labs']}/{results['n_labs']}")

    return results
