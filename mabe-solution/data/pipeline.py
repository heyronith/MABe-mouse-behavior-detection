import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MABeDataPipeline:
    """Integrated data pipeline for MABe competition"""

    def __init__(self, config):
        """
        Initialize complete data pipeline

        Args:
            config: Configuration object with all settings
        """
        self.config = config

        # Initialize components
        from .loader import MABeDataset, create_data_loaders
        from .features import FeatureExtractor
        from .preprocessing import TrajectorySmoother, ArenaNormalizer, EgocentricTransformer
        from .sampling import PositiveAwareSampler, DataAugmentation, AdvancedWindowing

        # Core components
        self.feature_extractor = FeatureExtractor(config)
        self.smoother = TrajectorySmoother(method='adaptive', window=7, adaptive=True)
        self.normalizer = ArenaNormalizer(center=True, scale=True)
        self.transformer = EgocentricTransformer()

        # Advanced components
        self.augmentation = DataAugmentation(
            rotation_range=config.data.augmentation.rotation_range,
            scale_range=config.data.augmentation.scale_range,
            flip_probability=config.data.augmentation.flip_probability,
            noise_std=config.data.augmentation.noise_std
        )

        self.sampler = PositiveAwareSampler(
            positive_ratio=config.data.positive_sampling_ratio,
            rare_behavior_threshold=config.data.rare_behavior_threshold
        )

        self.windowing = AdvancedWindowing(
            window_sizes=config.data.window_sizes,
            overlap=config.data.overlap
        )

        logger.info("MABe data pipeline initialized")

    def create_datasets(self, train_csv: str, val_csv: Optional[str] = None):
        """Create training and validation datasets"""

        # Training dataset
        train_dataset = MABeDataset(
            csv_path=train_csv,
            tracking_dir=self.config.data.train_tracking_dir,
            annotation_dir=self.config.data.train_annotation_dir,
            window_sizes=self.config.data.window_sizes,
            overlap=self.config.data.overlap,
            is_train=True
        )

        # Validation dataset
        if val_csv:
            val_dataset = MABeDataset(
                csv_path=val_csv,
                tracking_dir=self.config.data.train_tracking_dir,
                annotation_dir=self.config.data.train_annotation_dir,
                window_sizes=self.config.data.window_sizes,
                overlap=self.config.data.overlap,
                is_train=False
            )
        else:
            # Use subset of training data for validation
            val_dataset = MABeDataset(
                csv_path=train_csv,
                tracking_dir=self.config.data.train_tracking_dir,
                annotation_dir=self.config.data.train_annotation_dir,
                window_sizes=self.config.data.window_sizes,
                overlap=self.config.data.overlap,
                is_train=True
            )

        logger.info(f"Created datasets:")
        logger.info(f"  Train: {len(train_dataset)} windows from {len(train_dataset.video_info)} videos")
        logger.info(f"  Val: {len(val_dataset)} windows from {len(val_dataset.video_info)} videos")

        return train_dataset, val_dataset

    def create_data_loaders(self, train_dataset, val_dataset):
        """Create optimized data loaders"""

        def collate_fn(batch):
            """Custom collate function with augmentation and sampling"""
            augmented_batch = []

            for sample in batch:
                # Apply augmentation during training
                if self.config.data.augmentation.enabled and torch.rand(1).item() < 0.5:
                    sample['tracking'] = self.augmentation.augment_window(sample['tracking'])

                # Apply preprocessing
                confidence = sample['tracking'][..., 2] if sample['tracking'].shape[-1] == 3 else None
                processed_tracking = self.smoother.smooth(sample['tracking'], confidence)
                processed_tracking = self.normalizer.normalize(processed_tracking)
                processed_tracking = self.transformer.transform(processed_tracking)

                sample['tracking'] = processed_tracking
                augmented_batch.append(sample)

            return augmented_batch

        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.training.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.training.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )

        return train_loader, val_loader

    def extract_features_batch(self, tracking_batch: torch.Tensor) -> torch.Tensor:
        """Extract features for a batch of tracking data"""
        batch_size = tracking_batch.shape[0]

        # Preprocess each sample in batch
        processed_batch = []
        for i in range(batch_size):
            tracking = tracking_batch[i]

            # Apply preprocessing
            confidence = tracking[..., 2] if tracking.shape[-1] == 3 else None
            processed = self.smoother.smooth(tracking, confidence)
            processed = self.normalizer.normalize(processed)
            processed = self.transformer.transform(processed)

            processed_batch.append(processed)

        # Stack processed samples
        processed_batch = torch.stack(processed_batch)

        # Extract features
        features = self.feature_extractor.extract_features(processed_batch)

        return features

    def validate_pipeline(self, sample_batch: List[Dict]) -> Dict[str, float]:
        """Validate pipeline on a sample batch"""
        logger.info("Validating data pipeline...")

        validation_results = {
            'tracking_shape': None,
            'features_shape': None,
            'annotation_shape': None,
            'preprocessing_success': True,
            'feature_extraction_success': True,
            'augmentation_success': True
        }

        try:
            # Test preprocessing
            if sample_batch and 'tracking' in sample_batch[0]:
                tracking = sample_batch[0]['tracking']
                validation_results['tracking_shape'] = tracking.shape

                # Apply preprocessing
                confidence = tracking[..., 2] if tracking.shape[-1] == 3 else None
                processed = self.smoother.smooth(tracking, confidence)
                processed = self.normalizer.normalize(processed)
                processed = self.transformer.transform(processed)

                if torch.isnan(processed).any():
                    validation_results['preprocessing_success'] = False

            # Test feature extraction
            if sample_batch and 'tracking' in sample_batch[0]:
                try:
                    features = self.extract_features_batch(tracking_batch.unsqueeze(0))
                    validation_results['features_shape'] = features.shape

                    if torch.isnan(features).any():
                        validation_results['feature_extraction_success'] = False

                except Exception as e:
                    logger.error(f"Feature extraction failed: {e}")
                    validation_results['feature_extraction_success'] = False

            # Test augmentation
            if sample_batch and 'tracking' in sample_batch[0]:
                try:
                    tracking = sample_batch[0]['tracking']
                    augmented = self.augmentation.augment_window(tracking)

                    if torch.isnan(augmented).any():
                        validation_results['augmentation_success'] = False

                except Exception as e:
                    logger.error(f"Augmentation failed: {e}")
                    validation_results['augmentation_success'] = False

        except Exception as e:
            logger.error(f"Pipeline validation failed: {e}")
            validation_results['preprocessing_success'] = False

        # Log results
        logger.info("Pipeline validation results:")
        for key, value in validation_results.items():
            logger.info(f"  {key}: {value}")

        return validation_results


class DataPipelineOptimizer:
    """Optimize data pipeline for efficiency"""

    def __init__(self, config):
        self.config = config

    def optimize_memory_usage(self, dataset_size: int) -> Dict[str, any]:
        """Calculate optimal memory usage and batch sizes"""
        # Estimate memory per sample
        avg_window_size = np.mean(self.config.data.window_sizes)
        memory_per_sample = avg_window_size * 2 * 10 * 3 * 4  # frames * mice * keypoints * coords * bytes

        # Calculate optimal batch size
        available_memory_gb = 8  # Assume 8GB available for data
        available_memory_bytes = available_memory_gb * 1024**3

        optimal_batch_size = int(available_memory_bytes / (memory_per_sample * 4))  # 4x safety margin
        optimal_batch_size = min(optimal_batch_size, 64)  # Cap at 64
        optimal_batch_size = max(optimal_batch_size, 4)   # Min 4

        # Estimate total memory usage
        total_memory_gb = (dataset_size * memory_per_sample) / (1024**3)

        recommendations = {
            'estimated_memory_per_sample': memory_per_sample,
            'optimal_batch_size': optimal_batch_size,
            'total_dataset_memory': total_memory_gb,
            'recommendations': []
        }

        if total_memory_gb > 16:
            recommendations['recommendations'].append("Consider reducing window sizes or max_windows_per_video")
        if optimal_batch_size < 8:
            recommendations['recommendations'].append("Batch size is small, consider reducing feature dimensions")

        logger.info(f"Memory optimization:")
        logger.info(f"  Per sample: {memory_per_sample / 1024:.1f} KB")
        logger.info(f"  Optimal batch: {optimal_batch_size}")
        logger.info(f"  Total dataset: {total_memory_gb:.1f} GB")

        return recommendations

    def validate_configuration(self) -> Dict[str, any]:
        """Validate configuration for potential issues"""
        issues = []
        warnings = []

        # Check window sizes
        if max(self.config.data.window_sizes) > 5000:
            warnings.append("Large window sizes may cause memory issues")

        # Check overlap
        if self.config.data.overlap > 0.8:
            warnings.append("High overlap may create redundant windows")

        # Check batch size vs window size
        avg_window = np.mean(self.config.data.window_sizes)
        if self.config.training.batch_size * avg_window > 50000:
            warnings.append("Large batch size * window size may cause memory issues")

        # Check augmentation settings
        if (self.config.data.augmentation.scale_range[1] - self.config.data.augmentation.scale_range[0]) > 0.5:
            warnings.append("Large scale range may distort behavior patterns")

        # Check sampling ratios
        if self.config.data.positive_sampling_ratio > 0.7:
            warnings.append("Very high positive sampling ratio may reduce diversity")

        validation = {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'recommendations': []
        }

        if validation['valid']:
            logger.info("✅ Configuration validation passed")
        else:
            logger.error("❌ Configuration issues found:")
            for issue in issues:
                logger.error(f"  - {issue}")

        for warning in warnings:
            logger.warning(f"  Warning: {warning}")

        return validation
