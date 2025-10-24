import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class DilatedTCN(nn.Module):
    """Dilated Temporal Convolutional Network for behavior detection"""

    def __init__(self,
                 n_features: int,
                 n_behaviors: int,
                 hidden_dim: int = 512,
                 layers: int = 4,
                 dilation: List[int] = None,
                 kernel_size: int = 3,
                 dropout: float = 0.1):
        """
        Args:
            n_features: Number of input features
            n_behaviors: Number of behavior classes
            hidden_dim: Hidden dimension
            layers: Number of TCN layers
            dilation: Dilation rates for each layer
            kernel_size: Kernel size for convolutions
            dropout: Dropout probability
        """
        super().__init__()

        if dilation is None:
            dilation = [2 ** i for i in range(layers)]

        self.n_features = n_features
        self.n_behaviors = n_behaviors
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.dilation = dilation
        self.kernel_size = kernel_size

        # Calculate receptive field
        self.receptive_field = self._calculate_receptive_field()

        # TCN layers
        self.tcn_layers = nn.ModuleList()

        for i in range(layers):
            layer = TCNBlock(
                in_channels=n_features if i == 0 else hidden_dim,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                dilation=dilation[i],
                dropout=dropout
            )
            self.tcn_layers.append(layer)

        # Output head
        self.output_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_behaviors)
        )

        # Initialize weights
        self._initialize_weights()

        logger.info(f"TCN receptive field: {self.receptive_field} frames")

    def _calculate_receptive_field(self) -> int:
        """Calculate the receptive field of the TCN"""
        rf = 1
        for i in range(self.layers):
            rf += (self.kernel_size - 1) * self.dilation[i]
        return rf

    def _initialize_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input features (batch_size, n_frames, n_features)

        Returns:
            logits: Behavior predictions (batch_size, n_frames, n_behaviors)
        """
        batch_size, n_frames, n_features = x.shape

        # Ensure input dimensions match
        if n_features != self.n_features:
            logger.warning(f"Feature dimension mismatch: {n_features} vs {self.n_features}")

        # Pass through TCN layers
        for layer in self.tcn_layers:
            x = layer(x)

        # Global average pooling and output head
        logits = self.output_head(x.transpose(1, 2))  # (batch, n_behaviors, n_frames)
        logits = logits.transpose(1, 2)  # (batch, n_frames, n_behaviors)

        return logits


class TCNBlock(nn.Module):
    """Single TCN block with dilated convolution"""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 dilation: int = 1,
                 dropout: float = 0.1):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation
        )

        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation
        )

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, n_frames, channels)

        Returns:
            output: (batch_size, n_frames, channels)
        """
        # TCN expects (batch, channels, frames)
        residual = self.residual(x.transpose(1, 2))
        x = x.transpose(1, 2)

        # Two convolutional layers with residual
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)

        # Add residual and transpose back
        out = out + residual
        out = out.transpose(1, 2)

        return out


class MABeModel(nn.Module):
    """Complete MABe model with feature extraction and TCN"""

    def __init__(self, config):
        super().__init__()

        self.feature_extractor = FeatureExtractor(config)
        self.tcn = DilatedTCN(
            n_features=config.model.n_features,
            n_behaviors=config.model.n_behaviors,
            hidden_dim=config.model.hidden_dim,
            layers=config.model.layers,
            dilation=config.model.dilation,
            kernel_size=config.model.kernel_size,
            dropout=config.model.dropout
        )

    def forward(self, tracking: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tracking: Raw tracking data (batch, n_frames, n_mice, n_keypoints, 3)

        Returns:
            logits: Behavior predictions (batch, n_frames, n_behaviors)
        """
        # Extract features
        features = self.feature_extractor.extract_features(tracking)

        # Pass through TCN
        logits = self.tcn(features)

        return logits
