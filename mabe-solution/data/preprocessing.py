import torch
import numpy as np
from scipy import ndimage
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TrajectorySmoother:
    """Advanced trajectory smoothing with multiple algorithms and adaptive selection"""

    def __init__(self, method: str = 'adaptive', window: int = 7, adaptive: bool = True):
        """
        Args:
            method: 'savitzky_golay', 'ema', 'median', 'butterworth', 'kalman', 'adaptive'
            window: Smoothing window size
            adaptive: Whether to adaptively select best method per trajectory
        """
        self.method = method
        self.window = window
        self.adaptive = adaptive

    def smooth(self, trajectories: torch.Tensor, confidence: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Smooth trajectories with quality-based method selection

        Args:
            trajectories: (n_frames, n_mice, n_keypoints, 3) where 3 = (x, y, confidence)
            confidence: Optional confidence scores for adaptive smoothing

        Returns:
            smoothed: Same shape as input
        """
        if self.adaptive and confidence is not None:
            return self._adaptive_smooth(trajectories, confidence)
        elif self.method == 'savitzky_golay':
            return self._savitzky_golay_smooth(trajectories)
        elif self.method == 'ema':
            return self._ema_smooth(trajectories)
        elif self.method == 'median':
            return self._median_smooth(trajectories)
        elif self.method == 'butterworth':
            return self._butterworth_smooth(trajectories)
        elif self.method == 'kalman':
            return self._kalman_smooth(trajectories)
        else:
            return trajectories

    def _adaptive_smooth(self, trajectories: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        """Adaptively select best smoothing method based on data quality"""
        smoothed = trajectories.clone()

        for mouse in range(trajectories.shape[1]):
            for kp in range(trajectories.shape[2]):
                # Assess data quality
                quality_score = self._assess_trajectory_quality(
                    trajectories[:, mouse, kp, :2], confidence[:, mouse, kp]
                )

                # Select method based on quality
                if quality_score > 0.8:
                    method = 'savitzky_golay'  # High quality - preserve details
                elif quality_score > 0.5:
                    method = 'ema'  # Medium quality - smooth noise
                else:
                    method = 'median'  # Low quality - robust to outliers

                # Apply selected method
                for coord in range(2):  # x, y coordinates
                    signal = trajectories[:, mouse, kp, coord]
                    if method == 'savitzky_golay':
                        smoothed_signal = self._savitzky_golay_1d(signal)
                    elif method == 'ema':
                        smoothed_signal = self._ema_1d(signal, alpha=0.3)
                    elif method == 'median':
                        smoothed_signal = self._median_1d(signal)
                    else:
                        smoothed_signal = signal

                    smoothed[:, mouse, kp, coord] = smoothed_signal

        return smoothed

    def _assess_trajectory_quality(self, positions: torch.Tensor, conf: torch.Tensor) -> float:
        """Assess trajectory quality (0-1, higher is better)"""
        # Factor 1: Confidence score
        conf_score = conf.mean().item() if len(conf) > 0 else 1.0

        # Factor 2: Smoothness (low variance in velocities indicates high quality)
        if len(positions) > 3:
            velocities = positions[1:] - positions[:-1]
            velocity_magnitude = torch.norm(velocities, dim=1)
            smoothness_score = 1.0 / (1.0 + velocity_magnitude.std().item())
        else:
            smoothness_score = 1.0

        # Factor 3: Missing data ratio
        missing_ratio = (torch.isnan(positions).any(dim=1).float().mean().item())

        # Combine factors
        quality_score = 0.4 * conf_score + 0.4 * smoothness_score + 0.2 * (1.0 - missing_ratio)
        return min(1.0, max(0.0, quality_score))

    def _savitzky_golay_smooth(self, trajectories: torch.Tensor) -> torch.Tensor:
        """Apply Savitzky-Golay filter"""
        n_frames, n_mice, n_keypoints, n_coords = trajectories.shape
        smoothed = trajectories.clone()

        # Apply to each mouse and keypoint separately
        for mouse in range(n_mice):
            for kp in range(n_keypoints):
                for coord in range(n_coords - 1):  # Skip confidence
                    signal = trajectories[:, mouse, kp, coord]
                    smoothed[:, mouse, kp, coord] = self._savitzky_golay_1d(signal)

        return smoothed

    def _savitzky_golay_1d(self, signal: torch.Tensor) -> torch.Tensor:
        """1D Savitzky-Golay filter"""
        try:
            from scipy.signal import savgol_filter
            window_length = min(self.window, len(signal), 15)  # Cap at 15
            if window_length >= 3 and window_length % 2 == 1:  # Must be odd
                smoothed_signal = savgol_filter(
                    signal.numpy(),
                    window_length=window_length,
                    polyorder=min(2, window_length - 1)
                )
                return torch.tensor(smoothed_signal, dtype=signal.dtype)
            else:
                return signal
        except Exception:
            return signal

    def _ema_smooth(self, trajectories: torch.Tensor, alpha: Optional[float] = None) -> torch.Tensor:
        """Apply exponential moving average"""
        if alpha is None:
            alpha = 2.0 / (self.window + 1)  # Standard EMA alpha

        smoothed = trajectories.clone()

        for mouse in range(trajectories.shape[1]):
            for kp in range(trajectories.shape[2]):
                for coord in range(2):  # Only x, y coordinates
                    signal = trajectories[:, mouse, kp, coord]
                    smoothed_signal = self._ema_1d(signal, alpha)
                    smoothed[:, mouse, kp, coord] = smoothed_signal

        return smoothed

    def _ema_1d(self, signal: torch.Tensor, alpha: float) -> torch.Tensor:
        """1D exponential moving average"""
        smoothed = torch.zeros_like(signal)
        smoothed[0] = signal[0]

        for i in range(1, len(signal)):
            smoothed[i] = alpha * signal[i] + (1 - alpha) * smoothed[i-1]

        return smoothed

    def _median_smooth(self, trajectories: torch.Tensor) -> torch.Tensor:
        """Apply median filter"""
        smoothed = trajectories.clone()

        for mouse in range(trajectories.shape[1]):
            for kp in range(trajectories.shape[2]):
                for coord in range(2):  # Only x, y coordinates
                    signal = trajectories[:, mouse, kp, coord]
                    smoothed_signal = self._median_1d(signal)
                    smoothed[:, mouse, kp, coord] = smoothed_signal

        return smoothed

    def _median_1d(self, signal: torch.Tensor) -> torch.Tensor:
        """1D median filter"""
        try:
            from scipy.ndimage import median_filter
            return torch.tensor(median_filter(signal.numpy(), size=self.window))
        except Exception:
            return signal

    def _butterworth_smooth(self, trajectories: torch.Tensor) -> torch.Tensor:
        """Apply Butterworth low-pass filter"""
        try:
            from scipy.signal import butter, filtfilt

            # Design filter (4th order Butterworth, cutoff at 0.3 normalized frequency)
            cutoff = 0.3  # Normalized frequency
            b, a = butter(4, cutoff, btype='low')

            smoothed = trajectories.clone()

            for mouse in range(trajectories.shape[1]):
                for kp in range(trajectories.shape[2]):
                    for coord in range(2):  # Only x, y coordinates
                        signal = trajectories[:, mouse, kp, coord]
                        smoothed_signal = filtfilt(b, a, signal.numpy())
                        smoothed[:, mouse, kp, coord] = torch.tensor(smoothed_signal)

            return smoothed

        except Exception:
            return trajectories

    def _kalman_smooth(self, trajectories: torch.Tensor) -> torch.Tensor:
        """Apply Kalman filter (simplified 1D version)"""
        # For now, use EMA with optimized alpha (would need full Kalman implementation)
        # Kalman filter would be ideal for handling missing data and uncertainty
        return self._ema_smooth(trajectories, alpha=0.5)


class ArenaNormalizer:
    """Normalize arena coordinates and scale"""

    def __init__(self, center: bool = True, scale: bool = True, target_size: float = 100.0):
        self.center = center
        self.scale = scale
        self.target_size = target_size

    def normalize(self, tracking: torch.Tensor, metadata: Optional[Dict] = None) -> torch.Tensor:
        """
        Normalize tracking data

        Args:
            tracking: (n_frames, n_mice, n_keypoints, 3)
            metadata: Optional video metadata (arena dimensions, etc.)

        Returns:
            normalized: Same shape as input
        """
        normalized = tracking.clone()

        # Center on arena center
        if self.center:
            normalized = self._center_coordinates(normalized)

        # Scale to consistent size
        if self.scale:
            normalized = self._scale_coordinates(normalized, metadata)

        return normalized

    def _center_coordinates(self, tracking: torch.Tensor) -> torch.Tensor:
        """Center coordinates on arena center"""
        # Use median position across all valid points as center
        valid_positions = tracking[..., :2]  # x, y coordinates
        valid_mask = ~torch.isnan(valid_positions) & (valid_positions != 0)

        if valid_mask.any():
            # Calculate mean position per frame or overall
            mean_x = torch.mean(valid_positions[valid_mask][..., 0])
            mean_y = torch.mean(valid_positions[valid_mask][..., 1])

            # Center coordinates
            tracking[..., 0] = tracking[..., 0] - mean_x
            tracking[..., 1] = tracking[..., 1] - mean_y

        return tracking

    def _scale_coordinates(self, tracking: torch.Tensor, metadata: Optional[Dict] = None) -> torch.Tensor:
        """Scale coordinates to consistent size"""
        # Calculate body length scale factor
        body_lengths = self._calculate_body_lengths(tracking)

        if len(body_lengths) > 0:
            median_length = torch.median(torch.tensor(body_lengths))
            scale_factor = self.target_size / median_length if median_length > 0 else 1.0

            # Scale coordinates
            tracking[..., :2] *= scale_factor

        return tracking

    def _calculate_body_lengths(self, tracking: torch.Tensor) -> List[float]:
        """Calculate body lengths for scaling"""
        lengths = []

        # Assume nose (index 6) to tail_base (index 7) distance
        nose_idx, tail_idx = 6, 7

        if tracking.shape[2] > max(nose_idx, tail_idx):
            for mouse in range(tracking.shape[1]):
                mouse_tracking = tracking[:, mouse, :, :]

                # Calculate distances between nose and tail_base
                nose_pos = mouse_tracking[:, nose_idx, :2]
                tail_pos = mouse_tracking[:, tail_idx, :2]

                # Filter out invalid positions
                valid_mask = (~torch.isnan(nose_pos) & ~torch.isnan(tail_pos) &
                             (nose_pos != 0) & (tail_pos != 0)).all(dim=1)

                if valid_mask.any():
                    distances = torch.norm(tail_pos[valid_mask] - nose_pos[valid_mask], dim=1)
                    lengths.extend(distances.tolist())

        return lengths


class EgocentricTransformer:
    """Transform coordinates to egocentric frame per mouse"""

    def __init__(self):
        pass

    def transform(self, tracking: torch.Tensor) -> torch.Tensor:
        """
        Transform to egocentric coordinates (nose pointing +x)

        Args:
            tracking: (n_frames, n_mice, n_keypoints, 3)

        Returns:
            transformed: Same shape as input
        """
        transformed = tracking.clone()

        for mouse in range(tracking.shape[1]):
            transformed[:, mouse, :, :] = self._transform_mouse(
                tracking[:, mouse, :, :]
            )

        return transformed

    def _transform_mouse(self, mouse_tracking: torch.Tensor) -> torch.Tensor:
        """Transform single mouse to egocentric frame"""
        n_frames, n_keypoints, n_coords = mouse_tracking.shape

        # Assume neck (index 6) to nose (index 6) as head direction
        neck_idx, nose_idx = 6, 6  # Use nose position as reference

        if n_keypoints > nose_idx:
            for frame in range(n_frames):
                # Get head vector (nose relative to neck)
                if neck_idx < n_keypoints and nose_idx < n_keypoints:
                    neck_pos = mouse_tracking[frame, neck_idx, :2]
                    nose_pos = mouse_tracking[frame, nose_idx, :2]

                    # Check if positions are valid
                    if not (torch.isnan(neck_pos).any() or torch.isnan(nose_pos).any() or
                            (neck_pos == 0).all() or (nose_pos == 0).all()):

                        # Calculate rotation angle
                        head_vector = nose_pos - neck_pos
                        angle = torch.atan2(head_vector[1], head_vector[0])

                        # Rotate all keypoints around neck position
                        mouse_tracking[frame, :, :] = self._rotate_around_point(
                            mouse_tracking[frame, :, :], neck_pos, -angle
                        )

        return mouse_tracking

    def _rotate_around_point(self, keypoints: torch.Tensor, center: torch.Tensor, angle: float) -> torch.Tensor:
        """Rotate keypoints around a center point"""
        # Translate to origin
        translated = keypoints.clone()
        translated[:, :2] = keypoints[:, :2] - center

        # Rotation matrix
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        rotation_matrix = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]])

        # Apply rotation
        rotated = torch.matmul(translated[:, :2], rotation_matrix.t())
        translated[:, :2] = rotated

        # Translate back
        translated[:, :2] = translated[:, :2] + center

        return translated


def preprocess_window(tracking: torch.Tensor, config) -> torch.Tensor:
    """Complete preprocessing pipeline for a tracking window"""

    # Extract confidence scores
    confidence = tracking[..., 2]  # Confidence is last coordinate

    # Initialize preprocessors with enhanced settings
    smoother = TrajectorySmoother(method='adaptive', window=7, adaptive=True)
    normalizer = ArenaNormalizer(center=True, scale=True)
    transformer = EgocentricTransformer()

    # Apply preprocessing steps
    processed = smoother.smooth(tracking, confidence)
    processed = normalizer.normalize(processed)
    processed = transformer.transform(processed)

    return processed


def compare_smoothing_methods(tracking: torch.Tensor, methods: List[str] = None) -> Dict[str, torch.Tensor]:
    """Compare different smoothing methods and return results"""
    if methods is None:
        methods = ['savitzky_golay', 'ema', 'median', 'butterworth']

    results = {}
    confidence = tracking[..., 2] if tracking.shape[-1] == 3 else None

    for method in methods:
        smoother = TrajectorySmoother(method=method, window=7)
        smoothed = smoother.smooth(tracking, confidence)
        results[method] = smoothed

    return results
