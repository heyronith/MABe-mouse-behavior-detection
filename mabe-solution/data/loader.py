import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pyarrow.parquet as pq
import pyarrow as pa
import os
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MABeDataset(Dataset):
    """Dataset for MABe mouse behavior detection"""

    def __init__(self,
                 csv_path: str,
                 tracking_dir: str,
                 annotation_dir: Optional[str] = None,
                 window_sizes: List[int] = None,
                 overlap: float = 0.5,
                 transform=None,
                 is_train: bool = True):
        """
        Args:
            csv_path: Path to CSV file with video metadata
            tracking_dir: Directory containing parquet tracking files
            annotation_dir: Directory containing parquet annotation files (for training)
            window_size: Size of sliding window in frames
            overlap: Overlap between windows (0.0 to 1.0)
            transform: Optional transforms to apply
            is_train: Whether this is training data
        """
        self.csv_path = csv_path
        self.tracking_dir = Path(tracking_dir)
        self.annotation_dir = Path(annotation_dir) if annotation_dir else None
        self.window_sizes = window_sizes if window_sizes else [512]
        self.overlap = overlap
        self.transform = transform
        self.is_train = is_train

        # Load metadata
        self.metadata = pd.read_csv(csv_path)

        # Create video index
        self.video_info = self._create_video_index()

        # Create multi-scale window index for efficient sampling
        self.windows = self._create_multi_scale_window_index()

        logger.info(f"Loaded {len(self.video_info)} videos with {len(self.windows)} windows")

    def _create_video_index(self) -> List[Dict]:
        """Create index of all videos with their properties"""
        video_info = []

        for _, row in self.metadata.iterrows():
            video_id = row['video_id']
            lab_id = row['lab_id']

            # Check if tracking file exists
            tracking_path = self.tracking_dir / lab_id / f"{video_id}.parquet"
            if not tracking_path.exists():
                logger.warning(f"Missing tracking file: {tracking_path}")
                continue

            # Get video duration from parquet
            try:
                tracking_data = pq.read_table(tracking_path)
                n_frames = len(tracking_data)

                info = {
                    'video_id': video_id,
                    'lab_id': lab_id,
                    'tracking_path': str(tracking_path),
                    'n_frames': n_frames,
                    'n_mice': self._get_n_mice_from_csv(row),
                    'behaviors': self._parse_behaviors(row['behaviors_labeled']),
                    'fps': row['frames_per_second'],
                    'body_parts': self._parse_body_parts(row['body_parts_tracked'])
                }

                if self.is_train and self.annotation_dir:
                    annotation_path = self.annotation_dir / lab_id / f"{video_id}.parquet"
                    if annotation_path.exists():
                        info['annotation_path'] = str(annotation_path)
                    else:
                        logger.warning(f"Missing annotation file: {annotation_path}")
                        continue

                video_info.append(info)

            except Exception as e:
                logger.error(f"Error loading {video_id}: {e}")
                continue

        return video_info

    def _create_multi_scale_window_index(self) -> List[Dict]:
        """Create index of multi-scale sliding windows"""
        windows = []

        for video_idx, video in enumerate(self.video_info):
            n_frames = video['n_frames']

            # Create windows for each scale
            for window_size in self.window_sizes:
                stride = int(window_size * (1 - self.overlap))

                # Create sliding windows for this scale
                for start_frame in range(0, n_frames - window_size + 1, stride):
                    window_info = {
                        'video_idx': video_idx,
                        'start_frame': start_frame,
                        'end_frame': start_frame + window_size,
                        'window_size': window_size,
                        'scale': self._get_scale_name(window_size),
                        'window_idx': len(windows)
                    }
                    windows.append(window_info)

                    # Limit windows per video to avoid memory issues
                    if len([w for w in windows if w['video_idx'] == video_idx]) >= 50:
                        break

                if len([w for w in windows if w['video_idx'] == video_idx]) >= 50:
                    break

        logger.info(f"Created {len(windows)} multi-scale windows across {len(self.video_info)} videos")
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

    def _get_n_mice_from_csv(self, row) -> int:
        """Extract number of mice from CSV row"""
        n_mice = 0
        for i in range(1, 5):  # mouse1 to mouse4
            if pd.notna(row.get(f'mouse{i}_strain')):
                n_mice = i
        return n_mice

    def _parse_behaviors(self, behaviors_str: str) -> List[str]:
        """Parse behaviors from CSV string"""
        if pd.isna(behaviors_str):
            return []

        # Remove brackets and split
        behaviors = behaviors_str.strip('[]').replace('"', '').split(',')
        return [b.strip() for b in behaviors if b.strip()]

    def _parse_body_parts(self, body_parts_str: str) -> List[str]:
        """Parse body parts from CSV string"""
        if pd.isna(body_parts_str):
            return []

        # Remove brackets and split
        parts = body_parts_str.strip('[]').replace('"', '').split(',')
        return [p.strip() for p in parts]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single window of data"""
        window_info = self.windows[idx]
        video_info = self.video_info[window_info['video_idx']]

        # Load tracking data for this window
        tracking_data = self._load_tracking_window(video_info, window_info)

        item = {
            'video_id': video_info['video_id'],
            'lab_id': video_info['lab_id'],
            'tracking': tracking_data,
            'window_start': window_info['start_frame'],
            'window_end': window_info['end_frame'],
        }

        # Load annotations if training
        if self.is_train and 'annotation_path' in video_info:
            annotations = self._load_annotations_window(video_info, window_info)
            item['annotations'] = annotations

        # Apply transforms
        if self.transform:
            item = self.transform(item)

        return item

    def _load_tracking_window(self, video_info: Dict, window_info: Dict) -> torch.Tensor:
        """Load tracking data for a specific window"""
        try:
            # Read parquet file
            table = pq.read_table(video_info['tracking_path'])

            # Get frame range
            start_frame = window_info['start_frame']
            end_frame = window_info['end_frame']

            # Convert to pandas for easier indexing
            df = table.to_pandas()

            # Filter to window
            window_df = df.iloc[start_frame:end_frame].copy()

            # Get shape: (n_frames, n_mice, n_keypoints, 3) where 3 = (x, y, confidence)
            n_frames = len(window_df)
            n_mice = video_info['n_mice']
            n_keypoints = len(video_info['body_parts'])

            # Initialize tensor
            tracking = torch.full((n_frames, n_mice, n_keypoints, 3), torch.nan)

            # Fill tracking data - handle different column structures
            for mouse_idx in range(n_mice):
                mouse_col = f"mouse{mouse_idx+1}"
                if mouse_col in window_df.columns:
                    mouse_data = window_df[mouse_col].values

                    # Handle different data formats
                    if hasattr(mouse_data, 'shape') and len(mouse_data.shape) == 3:
                        # Direct numpy array (n_frames, n_keypoints, 3)
                        tracking[:, mouse_idx, :, :] = torch.tensor(mouse_data, dtype=torch.float32)
                    elif hasattr(mouse_data, '__len__') and len(mouse_data) > 0:
                        # List of arrays or other structure
                        if hasattr(mouse_data[0], 'shape') and len(mouse_data[0].shape) == 2:
                            # List of (n_keypoints, 3) arrays
                            for frame_idx in range(min(n_frames, len(mouse_data))):
                                if mouse_data[frame_idx] is not None:
                                    tracking[frame_idx, mouse_idx, :, :] = torch.tensor(
                                        mouse_data[frame_idx], dtype=torch.float32
                                    )
                        else:
                            # Single frame data repeated or other format
                            tracking[:, mouse_idx, :, :] = torch.nan
                    else:
                        tracking[:, mouse_idx, :, :] = torch.nan

            return tracking

        except Exception as e:
            logger.error(f"Error loading tracking for {video_info['video_id']}: {e}")
            # Return empty tensor
            n_frames = window_info['end_frame'] - window_info['start_frame']
            n_mice = video_info['n_mice']
            n_keypoints = len(video_info['body_parts'])
            return torch.full((n_frames, n_mice, n_keypoints, 3), torch.nan)

    def _load_annotations_window(self, video_info: Dict, window_info: Dict) -> torch.Tensor:
        """Load annotations for a specific window"""
        try:
            table = pq.read_table(video_info['annotation_path'])
            df = table.to_pandas()

            # Filter to window frames
            start_frame = window_info['start_frame']
            end_frame = window_info['end_frame']

            # Check if columns exist (handle different annotation formats)
            required_cols = ['start_frame', 'stop_frame', 'agent_id', 'target_id', 'action']
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"Missing required columns in {video_info['video_id']} annotation")
                n_frames = end_frame - start_frame
                n_behaviors = len(video_info['behaviors'])
                return torch.zeros(n_frames, n_behaviors)

            window_annotations = df[
                (df['start_frame'] >= start_frame) &
                (df['stop_frame'] <= end_frame)
            ].copy()

            # Convert to tensor format: (n_frames, n_behaviors)
            n_frames = end_frame - start_frame
            n_behaviors = len(video_info['behaviors'])

            annotations = torch.zeros(n_frames, n_behaviors)

            for _, row in window_annotations.iterrows():
                start = max(0, int(row['start_frame']) - start_frame)
                stop = min(n_frames, int(row['stop_frame']) - start_frame)

                if start < stop:
                    # Find behavior index
                    behavior_name = f"{row['agent_id']},{row['target_id']},{row['action']}"
                    if behavior_name in video_info['behaviors']:
                        behavior_idx = video_info['behaviors'].index(behavior_name)
                        annotations[start:stop, behavior_idx] = 1.0

            return annotations

        except Exception as e:
            logger.error(f"Error loading annotations for {video_info['video_id']}: {e}")
            n_frames = window_info['end_frame'] - window_info['start_frame']
            n_behaviors = len(video_info['behaviors'])
            return torch.zeros(n_frames, n_behaviors)


def create_data_loaders(config):
    """Create train and validation data loaders with advanced sampling"""

    # Training dataset with multi-scale windows
    train_dataset = MABeDataset(
        csv_path=config.data.train_csv,
        tracking_dir=config.data.train_tracking_dir,
        annotation_dir=config.data.train_annotation_dir,
        window_sizes=config.data.window_sizes,
        overlap=config.data.overlap,
        is_train=True
    )

    # Validation dataset (for now, use a subset of training data)
    # TODO: Implement proper train/val split in Phase 0 Layer 3
    val_dataset = MABeDataset(
        csv_path=config.data.train_csv,
        tracking_dir=config.data.train_tracking_dir,
        annotation_dir=config.data.train_annotation_dir,
        window_sizes=config.data.window_sizes,
        overlap=config.data.overlap,
        is_train=True
    )

    # Import sampling module
    from .sampling import PositiveAwareSampler, DataAugmentation

    # Create samplers
    train_sampler = PositiveAwareSampler(
        positive_ratio=config.data.positive_sampling_ratio,
        rare_behavior_threshold=config.data.rare_behavior_threshold
    )

    # Custom collate function with augmentation
    def collate_fn(batch):
        augmented_batch = []
        for sample in batch:
            # Apply random augmentation
            if hasattr(config.data.augmentation, 'enabled') and config.data.augmentation.enabled:
                augmentation = DataAugmentation(
                    rotation_range=config.data.augmentation.rotation_range,
                    scale_range=config.data.augmentation.scale_range,
                    flip_probability=config.data.augmentation.flip_probability,
                    noise_std=config.data.augmentation.noise_std
                )
                if torch.rand(1).item() < 0.5:  # 50% augmentation probability
                    sample['tracking'] = augmentation.augment_window(sample['tracking'])
            augmented_batch.append(sample)
        return augmented_batch

    # Create data loaders with advanced features
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True
    )

    logger.info(f"Created data loaders:")
    logger.info(f"  Train: {len(train_dataset)} windows, {len(train_loader)} batches")
    logger.info(f"  Val: {len(val_dataset)} windows, {len(val_loader)} batches")
    logger.info(f"  Window scales: {config.data.window_sizes}")

    return train_loader, val_loader
