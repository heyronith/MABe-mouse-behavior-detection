import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PositiveAwareSampler:
    """Positive-aware sampling for rare behaviors"""

    def __init__(self,
                 positive_ratio: float = 0.3,
                 rare_behavior_threshold: float = 0.05,
                 max_samples_per_epoch: int = 10000):
        """
        Args:
            positive_ratio: Target ratio of positive windows in batch
            rare_behavior_threshold: Behaviors with less than this ratio are considered rare
            max_samples_per_epoch: Maximum samples to generate per epoch
        """
        self.positive_ratio = positive_ratio
        self.rare_behavior_threshold = rare_behavior_threshold
        self.max_samples_per_epoch = max_samples_per_epoch

    def create_balanced_batch(self,
                            windows: List[Dict],
                            annotations: List[torch.Tensor],
                            rare_behaviors: Optional[List[str]] = None) -> List[int]:
        """
        Create balanced batch with positive-aware sampling

        Args:
            windows: List of window dictionaries
            annotations: List of annotation tensors for each window
            rare_behaviors: List of rare behavior names to prioritize

        Returns:
            List of window indices for the batch
        """
        if rare_behaviors is None:
            rare_behaviors = self._identify_rare_behaviors(annotations)

        # Separate positive and negative windows
        positive_windows = []
        negative_windows = []

        for i, (window, annotation) in enumerate(zip(windows, annotations)):
            is_positive = self._window_has_positive(annotation, rare_behaviors)

            if is_positive:
                positive_windows.append(i)
            else:
                negative_windows.append(i)

        # Calculate target counts
        total_windows = len(windows)
        target_positive = int(total_windows * self.positive_ratio)
        target_negative = total_windows - target_positive

        # Sample with priority for rare behaviors
        selected_indices = []

        # Prioritize positive windows with rare behaviors
        rare_positive_windows = []
        common_positive_windows = []

        for idx in positive_windows:
            annotation = annotations[idx]
            if self._has_rare_behavior(annotation, rare_behaviors):
                rare_positive_windows.append(idx)
            else:
                common_positive_windows.append(idx)

        # Sample rare positive windows (all of them if not too many)
        n_rare_positive = min(len(rare_positive_windows), target_positive // 2)
        selected_rare = np.random.choice(rare_positive_windows, n_rare_positive, replace=False)
        selected_indices.extend(selected_rare)

        # Sample remaining positive windows
        remaining_positive = target_positive - len(selected_rare)
        if remaining_positive > 0 and common_positive_windows:
            n_common_positive = min(remaining_positive, len(common_positive_windows))
            selected_common = np.random.choice(common_positive_windows, n_common_positive, replace=False)
            selected_indices.extend(selected_common)

        # Sample negative windows
        remaining_slots = total_windows - len(selected_indices)
        n_negative = min(remaining_slots, len(negative_windows), target_negative)
        if n_negative > 0:
            selected_negative = np.random.choice(negative_windows, n_negative, replace=False)
            selected_indices.extend(selected_negative)

        # Fill remaining slots randomly
        remaining_slots = total_windows - len(selected_indices)
        if remaining_slots > 0:
            all_windows = list(range(total_windows))
            remaining_candidates = [i for i in all_windows if i not in selected_indices]
            n_fill = min(remaining_slots, len(remaining_candidates))
            if n_fill > 0:
                selected_fill = np.random.choice(remaining_candidates, n_fill, replace=False)
                selected_indices.extend(selected_fill)

        logger.info(f"Created balanced batch: {len(selected_indices)}/{total_windows} windows")
        logger.info(f"  Positive windows: {len([i for i in selected_indices if i in positive_windows])}")
        logger.info(f"  Rare positive windows: {len([i for i in selected_indices if i in rare_positive_windows])}")

        return selected_indices

    def _identify_rare_behaviors(self, annotations: List[torch.Tensor]) -> List[str]:
        """Identify rare behaviors based on frequency"""
        if not annotations:
            return []

        # Count positive occurrences for each behavior
        n_windows = len(annotations)
        n_behaviors = annotations[0].shape[1] if len(annotations[0].shape) > 1 else 1

        behavior_frequencies = np.zeros(n_behaviors)

        for annotation in annotations:
            if len(annotation.shape) > 1:
                behavior_frequencies += (annotation.sum(dim=0) > 0).float().numpy()
            else:
                behavior_frequencies += (annotation > 0).float().numpy()

        # Calculate ratios
        behavior_ratios = behavior_frequencies / n_windows

        # Identify rare behaviors
        rare_behaviors = []
        for i, ratio in enumerate(behavior_ratios):
            if ratio < self.rare_behavior_threshold:
                rare_behaviors.append(f"behavior_{i}")

        logger.info(f"Identified {len(rare_behaviors)} rare behaviors: {rare_behaviors[:5]}...")
        return rare_behaviors

    def _window_has_positive(self, annotation: torch.Tensor, rare_behaviors: List[str]) -> bool:
        """Check if window has any positive annotations"""
        return (annotation.sum() > 0).item()

    def _has_rare_behavior(self, annotation: torch.Tensor, rare_behaviors: List[str]) -> bool:
        """Check if window has rare behaviors"""
        # This would need mapping from behavior indices to names
        # For now, assume first few behaviors are common, later ones are rare
        if len(annotation.shape) > 1:
            # Check if any annotation is positive (simplified)
            return (annotation.sum() > 0).item()
        else:
            return (annotation > 0).item()


class AdvancedWindowing:
    """Advanced windowing strategies for behavior detection"""

    def __init__(self,
                 window_sizes: List[int] = None,
                 overlap: float = 0.5,
                 max_windows_per_video: int = 100):
        """
        Args:
            window_sizes: List of window sizes to use (frames)
            overlap: Overlap between windows
            max_windows_per_video: Maximum windows per video
        """
        if window_sizes is None:
            window_sizes = [256, 512, 1024, 2048]  # Multi-scale windows

        self.window_sizes = window_sizes
        self.overlap = overlap
        self.max_windows_per_video = max_windows_per_video

    def create_multi_scale_windows(self, video_length: int) -> List[Dict]:
        """Create multi-scale windows for a video"""
        windows = []

        for window_size in self.window_sizes:
            stride = int(window_size * (1 - self.overlap))

            # Create sliding windows
            for start_frame in range(0, video_length - window_size + 1, stride):
                window_info = {
                    'window_size': window_size,
                    'start_frame': start_frame,
                    'end_frame': start_frame + window_size,
                    'scale': self._get_scale_name(window_size)
                }
                windows.append(window_info)

                # Limit windows per video
                if len(windows) >= self.max_windows_per_video:
                    break

            if len(windows) >= self.max_windows_per_video:
                break

        logger.info(f"Created {len(windows)} multi-scale windows for video of length {video_length}")
        return windows

    def _get_scale_name(self, window_size: int) -> str:
        """Get descriptive name for window size"""
        if window_size <= 256:
            return 'fast'
        elif window_size <= 512:
            return 'medium'
        elif window_size <= 1024:
            return 'slow'
        else:
            return 'very_slow'


class DataAugmentation:
    """Pose-safe data augmentation for mouse tracking"""

    def __init__(self,
                 rotation_range: float = 0.2,
                 scale_range: Tuple[float, float] = (0.9, 1.1),
                 flip_probability: float = 0.5,
                 noise_std: float = 2.0):
        """
        Args:
            rotation_range: Rotation range in radians
            scale_range: Scale range (min, max)
            flip_probability: Probability of horizontal flip
            noise_std: Standard deviation of coordinate noise
        """
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.flip_probability = flip_probability
        self.noise_std = noise_std

    def augment_window(self, tracking: torch.Tensor) -> torch.Tensor:
        """
        Apply pose-safe augmentations to tracking window

        Args:
            tracking: (n_frames, n_mice, n_keypoints, 3)

        Returns:
            augmented: Same shape as input
        """
        augmented = tracking.clone()

        # Apply augmentations that preserve relative geometry
        augmented = self._apply_rotation(augmented)
        augmented = self._apply_scaling(augmented)
        augmented = self._apply_flipping(augmented)
        augmented = self._apply_noise(augmented)

        return augmented

    def _apply_rotation(self, tracking: torch.Tensor) -> torch.Tensor:
        """Apply random rotation to entire window"""
        if np.random.random() < 0.5:  # 50% chance
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)

            # Rotate around center of arena
            center_x = tracking[..., 0].mean()
            center_y = tracking[..., 1].mean()

            # Translate to origin
            tracking[..., 0] -= center_x
            tracking[..., 1] -= center_y

            # Apply rotation
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            x_rot = tracking[..., 0] * cos_a - tracking[..., 1] * sin_a
            y_rot = tracking[..., 0] * sin_a + tracking[..., 1] * cos_a

            tracking[..., 0] = x_rot
            tracking[..., 1] = y_rot

            # Translate back
            tracking[..., 0] += center_x
            tracking[..., 1] += center_y

        return tracking

    def _apply_scaling(self, tracking: torch.Tensor) -> torch.Tensor:
        """Apply random scaling"""
        if np.random.random() < 0.5:  # 50% chance
            scale = np.random.uniform(self.scale_range[0], self.scale_range[1])

            # Scale around center
            center_x = tracking[..., 0].mean()
            center_y = tracking[..., 1].mean()

            tracking[..., 0] -= center_x
            tracking[..., 1] -= center_y

            tracking[..., 0] *= scale
            tracking[..., 1] *= scale

            tracking[..., 0] += center_x
            tracking[..., 1] += center_y

        return tracking

    def _apply_flipping(self, tracking: torch.Tensor) -> torch.Tensor:
        """Apply horizontal flipping (pose-safe)"""
        if np.random.random() < self.flip_probability:
            # Flip x-coordinates around center
            center_x = tracking[..., 0].mean()
            tracking[..., 0] = 2 * center_x - tracking[..., 0]

        return tracking

    def _apply_noise(self, tracking: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian noise to coordinates"""
        if self.noise_std > 0:
            # Add noise only to x, y coordinates (not confidence)
            noise = torch.normal(0, self.noise_std, tracking[..., :2].shape)
            tracking[..., :2] += noise

        return tracking


def create_positive_aware_dataloader(dataset, batch_size: int = 32, config=None):
    """Create data loader with positive-aware sampling"""

    if config is None:
        # Default configuration
        sampler = PositiveAwareSampler(positive_ratio=0.3)
        augmentation = DataAugmentation()
        windowing = AdvancedWindowing()
    else:
        sampler = PositiveAwareSampler(
            positive_ratio=config.data.positive_sampling_ratio,
            rare_behavior_threshold=0.05
        )
        augmentation = DataAugmentation(
            rotation_range=0.2,
            scale_range=(0.9, 1.1),
            flip_probability=0.5,
            noise_std=config.data.augmentation.noise
        )
        windowing = AdvancedWindowing(
            window_sizes=config.data.window_sizes,
            overlap=config.data.overlap
        )

    # Custom collate function with augmentation
    def collate_fn(batch):
        # Apply augmentation to each sample
        augmented_batch = []
        for sample in batch:
            if np.random.random() < 0.5:  # 50% augmentation probability
                sample['tracking'] = augmentation.augment_window(sample['tracking'])
            augmented_batch.append(sample)

        return augmented_batch

    # Create sampler for positive-aware sampling
    def create_sampler(dataset):
        # This would create a custom sampler that uses positive-aware sampling
        # For now, return None (use default sampling)
        return None

    sampler_obj = create_sampler(dataset)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler_obj,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
