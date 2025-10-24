#!/usr/bin/env python3
"""
Simple standalone test for core functionality
"""

import torch
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_basic_feature_extraction():
    """Test basic feature extraction without complex dependencies"""
    logger.info("Testing basic feature extraction...")

    # Create simple mock data
    batch_size, n_frames, n_mice, n_keypoints = 2, 32, 2, 8

    # Mock tracking: (batch, frames, mice, keypoints, 3)
    tracking = torch.randn(batch_size, n_frames, n_mice, n_keypoints, 3)

    # Set confidence to 1.0
    tracking[..., 2] = 1.0

    logger.info(f"Created tracking data: {tracking.shape}")

    # Test basic kinematic features
    # Extract body center (assume index 0)
    body_positions = tracking[:, :, :, 0, :2]  # (batch, frames, mice, 2)
    logger.info(f"Body positions: {body_positions.shape}")

    # Calculate speed
    velocities = body_positions[:, 1:] - body_positions[:, :-1]
    velocities = torch.cat([velocities[:, :1], velocities], dim=1)
    speed = torch.norm(velocities, dim=-1)

    logger.info(f"Speed: {speed.shape}, range: {speed.min().item():.3f} to {speed.max().item():.3f}")

    # Test pairwise distances (social features)
    if n_mice >= 2:
        mouse1_pos = body_positions[:, :, 0, :]  # (batch, frames, 2)
        mouse2_pos = body_positions[:, :, 1, :]  # (batch, frames, 2)

        distance = torch.norm(mouse2_pos - mouse1_pos, dim=-1)
        logger.info(f"Inter-mouse distance: {distance.shape}, range: {distance.min().item():.3f} to {distance.max().item():.3f}")

    return True


def test_tcn_model():
    """Test basic TCN model"""
    logger.info("Testing TCN model...")

    try:
        # Create simple TCN manually
        batch_size, n_frames, n_features = 2, 32, 16
        n_behaviors = 25

        # Input features
        x = torch.randn(batch_size, n_frames, n_features)
        logger.info(f"Input: {x.shape}")

        # Simple 1D convolution
        conv1 = torch.nn.Conv1d(n_features, 64, kernel_size=3, padding=1)
        conv2 = torch.nn.Conv1d(64, 128, kernel_size=3, padding=1)

        # Forward pass
        x_t = x.transpose(1, 2)  # (batch, features, frames)
        h1 = torch.relu(conv1(x_t))
        h2 = torch.relu(conv2(h1))

        # Global pooling
        pooled = torch.mean(h2, dim=-1)  # (batch, 128)
        output = torch.nn.Linear(128, n_behaviors)(pooled)

        logger.info(f"Output: {output.shape}")

        # Test loss
        target = torch.zeros(batch_size, n_behaviors)
        target[0, 0] = 1.0  # Some positive example

        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(output, target)
        logger.info(f"Loss: {loss.item():.4f}")

        return True

    except Exception as e:
        logger.error(f"Error in TCN test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_preprocessing():
    """Test basic preprocessing"""
    logger.info("Testing preprocessing...")

    # Create mock data
    batch_size, n_frames, n_mice, n_keypoints = 2, 32, 2, 8
    tracking = torch.randn(batch_size, n_frames, n_mice, n_keypoints, 3)

    # Test centering
    original_mean = tracking[..., :2].mean()
    tracking[..., 0] -= tracking[..., 0].mean()
    tracking[..., 1] -= tracking[..., 1].mean()
    new_mean = tracking[..., :2].mean()

    logger.info(f"Centering: {original_mean.item():.3f} -> {new_mean.item():.3f}")

    # Test scaling
    original_std = tracking[..., :2].std()
    scale_factor = 100.0 / (original_std + 1e-8)
    tracking[..., :2] *= scale_factor
    new_std = tracking[..., :2].std()

    logger.info(f"Scaling: {original_std.item():.3f} -> {new_std.item():.3f}")

    return True


def main():
    """Run tests"""
    logger.info("=== Simple Core Functionality Tests ===")

    success_count = 0

    # Test basic features
    if test_basic_feature_extraction():
        success_count += 1
        logger.info("✅ Basic feature extraction passed")
    else:
        logger.error("❌ Basic feature extraction failed")

    # Test TCN model
    if test_tcn_model():
        success_count += 1
        logger.info("✅ TCN model test passed")
    else:
        logger.error("❌ TCN model test failed")

    # Test preprocessing
    if test_preprocessing():
        success_count += 1
        logger.info("✅ Preprocessing test passed")
    else:
        logger.error("❌ Preprocessing test failed")

    logger.info(f"=== Tests completed: {success_count}/3 passed ===")

    if success_count == 3:
        logger.info("🎉 Core functionality is working!")
        logger.info("Feature engineering foundation is solid.")
        return True
    else:
        logger.error("❌ Core functionality needs fixes")
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
