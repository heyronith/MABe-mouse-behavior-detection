import torch
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract geometric and social features from mouse trajectories"""

    def __init__(self, config):
        self.config = config
        # Based on CSV analysis: variable keypoints per lab
        self.keypoint_names = [
            'body_center', 'ear_left', 'ear_right', 'lateral_left', 'lateral_right',
            'neck', 'nose', 'tail_base', 'tail_midpoint', 'tail_tip'
        ]
        self.keypoint_indices = {name: i for i, name in enumerate(self.keypoint_names)}

    def extract_features(self, tracking: torch.Tensor) -> torch.Tensor:
        """
        Extract features from tracking data

        Args:
            tracking: (batch, n_frames, n_mice, n_keypoints, 3) where 3 = (x, y, confidence)

        Returns:
            features: (batch, n_frames, n_features)
        """
        batch_size, n_frames, n_mice, n_keypoints, _ = tracking.shape

        features_list = []

        # Per-mouse kinematic features (speed, acceleration, angular velocity)
        kinematic_features = self._extract_kinematic_features(tracking)
        features_list.append(kinematic_features)

        # Per-mouse geometric features (body elongation, head orientation)
        geometric_features = self._extract_geometric_features(tracking)
        features_list.append(geometric_features)

        # Social interaction features (distances, angles, facing)
        social_features = self._extract_social_features(tracking)
        features_list.append(social_features)

        # Concatenate all features
        features = torch.cat(features_list, dim=-1)

        logger.info(f"Extracted features shape: {features.shape}")
        return features

    def _extract_kinematic_features(self, tracking: torch.Tensor) -> torch.Tensor:
        """Extract kinematic features: speed, acceleration, angular velocity"""
        batch_size, n_frames, n_mice, n_keypoints, _ = tracking.shape

        # Use body_center as reference (most reliable keypoint)
        body_center_idx = self.keypoint_indices.get('body_center', 0)

        kinematic_features = []

        for mouse_idx in range(n_mice):
            mouse_tracking = tracking[:, :, mouse_idx, :, :]

            # Extract body center positions (x, y)
            if body_center_idx < n_keypoints:
                positions = mouse_tracking[:, body_center_idx, :2]
                positions = torch.nan_to_num(positions, nan=0.0)

                # Calculate velocities
                velocities = positions[:, 1:] - positions[:, :-1]  # (batch, n_frames-1, 2)
                velocities = torch.cat([velocities[:, :1], velocities], dim=1)  # Pad first frame

                # Speed (magnitude of velocity)
                speed = torch.norm(velocities, dim=-1)  # (batch, n_frames)

                # Acceleration
                acceleration = velocities[:, 1:] - velocities[:, :-1]
                acceleration = torch.cat([acceleration[:, :1], acceleration], dim=1)
                acceleration = torch.norm(acceleration, dim=-1)

                # Angular velocity (change in direction)
                vel_angle = torch.atan2(velocities[..., 1], velocities[..., 0])
                angular_vel = vel_angle[:, 1:] - vel_angle[:, :-1]
                angular_vel = torch.cat([angular_vel[:, :1], angular_vel], dim=1)

                # Normalize to [-pi, pi]
                angular_vel = torch.atan2(torch.sin(angular_vel), torch.cos(angular_vel))

                # Stack features for this mouse
                mouse_kinematic = torch.stack([speed, acceleration, angular_vel], dim=-1)
                kinematic_features.append(mouse_kinematic)
            else:
                # Fallback: use first available keypoint
                positions = mouse_tracking[:, 0, :2]
                positions = torch.nan_to_num(positions, nan=0.0)

                velocities = positions[:, 1:] - positions[:, :-1]
                velocities = torch.cat([velocities[:, :1], velocities], dim=1)

                speed = torch.norm(velocities, dim=-1)
                acceleration = torch.norm(velocities[:, 1:] - velocities[:, :-1], dim=-1)
                acceleration = torch.cat([acceleration[:, :1], acceleration], dim=1)

                # No angular velocity fallback
                angular_vel = torch.zeros_like(speed)

                mouse_kinematic = torch.stack([speed, acceleration, angular_vel], dim=-1)
                kinematic_features.append(mouse_kinematic)

        # Concatenate all mice kinematic features
        if kinematic_features:
            kinematic = torch.cat(kinematic_features, dim=-1)
        else:
            kinematic = torch.zeros(batch_size, n_frames, 1)

        return kinematic

    def _extract_geometric_features(self, tracking: torch.Tensor) -> torch.Tensor:
        """Extract geometric features: body elongation, head orientation"""
        batch_size, n_frames, n_mice, n_keypoints, _ = tracking.shape

        geometric_features = []

        for mouse_idx in range(n_mice):
            mouse_tracking = tracking[:, :, mouse_idx, :, :]
            mouse_features = []

            # Body elongation (neck to tail_base distance)
            neck_idx = self.keypoint_indices.get('neck', -1)
            tail_idx = self.keypoint_indices.get('tail_base', -1)

            if neck_idx >= 0 and tail_idx >= 0 and max(neck_idx, tail_idx) < n_keypoints:
                neck_pos = mouse_tracking[:, neck_idx, :2]
                tail_pos = mouse_tracking[:, tail_idx, :2]

                neck_pos = torch.nan_to_num(neck_pos, nan=0.0)
                tail_pos = torch.nan_to_num(tail_pos, nan=0.0)

                body_length = torch.norm(tail_pos - neck_pos, dim=-1)
                mouse_features.append(body_length.unsqueeze(-1))
            else:
                # Fallback: use nose to body_center
                nose_idx = self.keypoint_indices.get('nose', 0)
                body_idx = self.keypoint_indices.get('body_center', 0)

                if max(nose_idx, body_idx) < n_keypoints:
                    nose_pos = mouse_tracking[:, nose_idx, :2]
                    body_pos = mouse_tracking[:, body_idx, :2]

                    nose_pos = torch.nan_to_num(nose_pos, nan=0.0)
                    body_pos = torch.nan_to_num(body_pos, nan=0.0)

                    body_length = torch.norm(nose_pos - body_pos, dim=-1)
                    mouse_features.append(body_length.unsqueeze(-1))

            # Head orientation (angle of head vector)
            head_angle = self._calculate_head_orientation(mouse_tracking, n_keypoints)
            mouse_features.append(head_angle.unsqueeze(-1))

            # Concatenate features for this mouse
            if mouse_features:
                mouse_geometric = torch.cat(mouse_features, dim=-1)
                geometric_features.append(mouse_geometric)

        # Concatenate all mice
        if geometric_features:
            geometric = torch.cat(geometric_features, dim=-1)
        else:
            geometric = torch.zeros(batch_size, n_frames, 1)

        return geometric

    def _calculate_head_orientation(self, mouse_tracking: torch.Tensor, n_keypoints: int) -> torch.Tensor:
        """Calculate head orientation angle"""
        neck_idx = self.keypoint_indices.get('neck', 6)
        nose_idx = self.keypoint_indices.get('nose', 6)

        if max(neck_idx, nose_idx) < n_keypoints:
            neck_pos = mouse_tracking[:, neck_idx, :2]
            nose_pos = mouse_tracking[:, nose_idx, :2]

            neck_pos = torch.nan_to_num(neck_pos, nan=0.0)
            nose_pos = torch.nan_to_num(nose_pos, nan=0.0)

            head_vector = nose_pos - neck_pos
            head_angle = torch.atan2(head_vector[..., 1], head_vector[..., 0])
        else:
            # Fallback: use first two keypoints
            pos1 = mouse_tracking[:, 0, :2]
            pos2 = mouse_tracking[:, 1, :2]
            pos1 = torch.nan_to_num(pos1, nan=0.0)
            pos2 = torch.nan_to_num(pos2, nan=0.0)

            head_vector = pos2 - pos1
            head_angle = torch.atan2(head_vector[..., 1], head_vector[..., 0])

        return head_angle

    def _extract_social_features(self, tracking: torch.Tensor) -> torch.Tensor:
        """Extract social interaction features between mice"""
        batch_size, n_frames, n_mice, n_keypoints, _ = tracking.shape

        if n_mice < 2:
            return torch.zeros(batch_size, n_frames, 1)

        social_features = []

        # Pairwise distances between key body parts
        keypoint_pairs = [
            ('nose', 'nose'),        # Face to face
            ('body_center', 'body_center'),  # Center to center
            ('tail_base', 'tail_base'),      # Tail to tail
        ]

        for kp1_name, kp2_name in keypoint_pairs:
            kp1_idx = self.keypoint_indices.get(kp1_name, 0)
            kp2_idx = self.keypoint_indices.get(kp2_name, 0)

            if kp1_idx < n_keypoints and kp2_idx < n_keypoints:
                for mouse1_idx in range(n_mice):
                    for mouse2_idx in range(mouse1_idx + 1, n_mice):
                        # Get positions for both mice
                        pos1 = tracking[:, :, mouse1_idx, kp1_idx, :2]
                        pos2 = tracking[:, :, mouse2_idx, kp2_idx, :2]

                        pos1 = torch.nan_to_num(pos1, nan=0.0)
                        pos2 = torch.nan_to_num(pos2, nan=0.0)

                        # Calculate distance
                        distance = torch.norm(pos2 - pos1, dim=-1)
                        social_features.append(distance.unsqueeze(-1))

        # Relative orientations and facing
        for mouse1_idx in range(n_mice):
            for mouse2_idx in range(mouse1_idx + 1, n_mice):
                # Vector from mouse1 to mouse2
                body1_idx = self.keypoint_indices.get('body_center', 0)
                body2_idx = self.keypoint_indices.get('body_center', 0)

                if body1_idx < n_keypoints and body2_idx < n_keypoints:
                    pos1 = tracking[:, :, mouse1_idx, body1_idx, :2]
                    pos2 = tracking[:, :, mouse2_idx, body2_idx, :2]

                    pos1 = torch.nan_to_num(pos1, nan=0.0)
                    pos2 = torch.nan_to_num(pos2, nan=0.0)

                    # Vector between mice
                    vec12 = pos2 - pos1
                    distance = torch.norm(vec12, dim=-1)

                    # Angle of this vector
                    angle = torch.atan2(vec12[..., 1], vec12[..., 0])
                    social_features.append(angle.unsqueeze(-1))

                    # Is mouse1 facing mouse2?
                    head1_angle = self._calculate_head_orientation(
                        tracking[:, :, mouse1_idx, :, :], n_keypoints
                    )

                    # Angle difference
                    angle_diff = angle - head1_angle
                    angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))

                    # Facing score (1 = perfectly facing, 0 = facing away)
                    facing_score = 1.0 - torch.abs(angle_diff) / torch.pi
                    social_features.append(facing_score.unsqueeze(-1))

        # Contact heuristics (within body length)
        contact_features = self._calculate_contact_features(tracking)
        social_features.append(contact_features)

        # Concatenate social features
        if social_features:
            social = torch.cat(social_features, dim=-1)
        else:
            social = torch.zeros(batch_size, n_frames, 1)

        return social

    def _calculate_contact_features(self, tracking: torch.Tensor) -> torch.Tensor:
        """Calculate advanced contact heuristics (mice within interaction distance)"""
        batch_size, n_frames, n_mice, n_keypoints, _ = tracking.shape

        contact_features = []

        for mouse1_idx in range(n_mice):
            for mouse2_idx in range(mouse1_idx + 1, n_mice):
                # Multiple contact heuristics for robustness

                # 1. Body center distance (most reliable)
                body1_idx = self.keypoint_indices.get('body_center', 0)
                body2_idx = self.keypoint_indices.get('body_center', 0)

                body_contact = self._calculate_body_contact(tracking, mouse1_idx, mouse2_idx, body1_idx, body2_idx, n_keypoints)

                # 2. Nose-to-body contact (for sniffing, investigation)
                nose1_idx = self.keypoint_indices.get('nose', 6)
                nose2_idx = self.keypoint_indices.get('nose', 6)
                body2_idx = self.keypoint_indices.get('body_center', 0)

                nose_body_contact = self._calculate_nose_body_contact(tracking, mouse1_idx, mouse2_idx,
                                                                   nose1_idx, body2_idx, n_keypoints)

                # 3. Head-to-head contact (for social investigation)
                head_contact = self._calculate_head_contact(tracking, mouse1_idx, mouse2_idx, n_keypoints)

                # 4. Tail contact (for following behaviors)
                tail1_idx = self.keypoint_indices.get('tail_base', 7)
                tail2_idx = self.keypoint_indices.get('tail_base', 7)
                tail_contact = self._calculate_tail_contact(tracking, mouse1_idx, mouse2_idx,
                                                         tail1_idx, tail2_idx, n_keypoints)

                # Combine contact features
                combined_contact = torch.stack([body_contact, nose_body_contact, head_contact, tail_contact], dim=-1)
                contact_features.append(combined_contact.mean(dim=-1, keepdim=True))

        if contact_features:
            return torch.cat(contact_features, dim=-1).mean(dim=-1, keepdim=True)
        else:
            return torch.zeros(batch_size, n_frames, 1)

    def _calculate_body_contact(self, tracking: torch.Tensor, mouse1_idx: int, mouse2_idx: int,
                              body1_idx: int, body2_idx: int, n_keypoints: int) -> torch.Tensor:
        """Calculate body-to-body contact"""
        if body1_idx < n_keypoints and body2_idx < n_keypoints:
            pos1 = tracking[:, :, mouse1_idx, body1_idx, :2]
            pos2 = tracking[:, :, mouse2_idx, body2_idx, :2]

            pos1 = torch.nan_to_num(pos1, nan=0.0)
            pos2 = torch.nan_to_num(pos2, nan=0.0)

            distance = torch.norm(pos2 - pos1, dim=-1)

            # Adaptive body length estimation
            body_length = self._estimate_body_length(tracking, mouse1_idx, mouse2_idx, n_keypoints)

            # Contact score: 1 when very close, 0 when far
            contact_score = torch.sigmoid(-distance + body_length * 1.2)
            return contact_score
        else:
            return torch.zeros(tracking.shape[0], tracking.shape[1])

    def _calculate_nose_body_contact(self, tracking: torch.Tensor, mouse1_idx: int, mouse2_idx: int,
                                   nose1_idx: int, body2_idx: int, n_keypoints: int) -> torch.Tensor:
        """Calculate nose-to-body contact (sniffing behavior)"""
        if nose1_idx < n_keypoints and body2_idx < n_keypoints:
            nose1_pos = tracking[:, :, mouse1_idx, nose1_idx, :2]
            body2_pos = tracking[:, :, mouse2_idx, body2_idx, :2]

            nose1_pos = torch.nan_to_num(nose1_pos, nan=0.0)
            body2_pos = torch.nan_to_num(body2_pos, nan=0.0)

            distance = torch.norm(body2_pos - nose1_pos, dim=-1)

            # Sniffing distance is smaller than body contact
            sniff_distance = 15.0  # pixels
            contact_score = torch.sigmoid(-distance + sniff_distance)
            return contact_score
        else:
            return torch.zeros(tracking.shape[0], tracking.shape[1])

    def _calculate_head_contact(self, tracking: torch.Tensor, mouse1_idx: int, mouse2_idx: int, n_keypoints: int) -> torch.Tensor:
        """Calculate head-to-head contact"""
        nose1_idx = self.keypoint_indices.get('nose', 6)
        nose2_idx = self.keypoint_indices.get('nose', 6)

        if nose1_idx < n_keypoints and nose2_idx < n_keypoints:
            nose1_pos = tracking[:, :, mouse1_idx, nose1_idx, :2]
            nose2_pos = tracking[:, :, mouse2_idx, nose2_idx, :2]

            nose1_pos = torch.nan_to_num(nose1_pos, nan=0.0)
            nose2_pos = torch.nan_to_num(nose2_pos, nan=0.0)

            distance = torch.norm(nose2_pos - nose1_pos, dim=-1)

            # Head contact distance
            head_contact_distance = 25.0  # pixels
            contact_score = torch.sigmoid(-distance + head_contact_distance)
            return contact_score
        else:
            return torch.zeros(tracking.shape[0], tracking.shape[1])

    def _calculate_tail_contact(self, tracking: torch.Tensor, mouse1_idx: int, mouse2_idx: int,
                              tail1_idx: int, tail2_idx: int, n_keypoints: int) -> torch.Tensor:
        """Calculate tail contact (following behavior)"""
        if tail1_idx < n_keypoints and tail2_idx < n_keypoints:
            tail1_pos = tracking[:, :, mouse1_idx, tail1_idx, :2]
            tail2_pos = tracking[:, :, mouse2_idx, tail2_idx, :2]

            tail1_pos = torch.nan_to_num(tail1_pos, nan=0.0)
            tail2_pos = torch.nan_to_num(tail2_pos, nan=0.0)

            distance = torch.norm(tail2_pos - tail1_pos, dim=-1)

            # Tail contact for following
            tail_contact_distance = 20.0  # pixels
            contact_score = torch.sigmoid(-distance + tail_contact_distance)
            return contact_score
        else:
            return torch.zeros(tracking.shape[0], tracking.shape[1])

    def _estimate_body_length(self, tracking: torch.Tensor, mouse1_idx: int, mouse2_idx: int, n_keypoints: int) -> float:
        """Estimate average body length for contact threshold"""
        body_lengths = []

        # Use nose to tail_base distance
        nose_idx = self.keypoint_indices.get('nose', 6)
        tail_idx = self.keypoint_indices.get('tail_base', 7)

        for mouse_idx in [mouse1_idx, mouse2_idx]:
            if nose_idx < n_keypoints and tail_idx < n_keypoints:
                mouse_tracking = tracking[:, :, mouse_idx, :, :]

                nose_pos = mouse_tracking[:, nose_idx, :2]
                tail_pos = mouse_tracking[:, tail_idx, :2]

                nose_pos = torch.nan_to_num(nose_pos, nan=0.0)
                tail_pos = torch.nan_to_num(tail_pos, nan=0.0)

                distances = torch.norm(tail_pos - nose_pos, dim=-1)
                valid_distances = distances[distances > 0]

                if len(valid_distances) > 0:
                    body_lengths.append(valid_distances.median().item())

        return np.median(body_lengths) if body_lengths else 30.0  # Default body length


def preprocess_tracking(tracking: torch.Tensor, config) -> torch.Tensor:
    """Preprocess tracking data: center, scale, rotate"""
    batch_size, n_frames, n_mice, n_keypoints, _ = tracking.shape

    # Arena centering and scaling
    if hasattr(config, 'arena_center') and config.arena_center:
        # Center on arena center (assume 0,0 is center)
        # For now, center on mean position across all mice and frames
        valid_positions = tracking[..., :2]  # x, y coordinates
        valid_mask = ~torch.isnan(valid_positions)

        if valid_mask.any():
            mean_x = torch.mean(valid_positions[valid_mask][..., 0])
            mean_y = torch.mean(valid_positions[valid_mask][..., 1])
            tracking[..., 0] -= mean_x  # Subtract mean x
            tracking[..., 1] -= mean_y  # Subtract mean y

    # Body length scaling
    if hasattr(config, 'arena_scale') and config.arena_scale:
        # Scale by median body length
        body_center_idx = 0  # Assume body_center is first keypoint
        neck_idx = 6  # Assume neck is 6th keypoint
        tail_idx = 7  # Assume tail_base is 7th keypoint

        if neck_idx < n_keypoints and tail_idx < n_keypoints:
            neck_pos = tracking[:, :, :, neck_idx, :2]
            tail_pos = tracking[:, :, :, tail_idx, :2]

            neck_pos = torch.nan_to_num(neck_pos, nan=0.0)
            tail_pos = torch.nan_to_num(tail_pos, nan=0.0)

            body_lengths = torch.norm(tail_pos - neck_pos, dim=-1)
            valid_lengths = body_lengths[~torch.isnan(body_lengths)]

            if len(valid_lengths) > 0:
                median_length = torch.median(valid_lengths)
                scale_factor = 100.0 / median_length  # Scale to 100 pixel body length
                tracking[..., :2] *= scale_factor

    return tracking
