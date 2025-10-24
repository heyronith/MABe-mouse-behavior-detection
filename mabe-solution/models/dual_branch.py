import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences"""

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input"""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TemporalTransformer(nn.Module):
    """Temporal Transformer with cross-agent attention for behavior modeling"""

    def __init__(self,
                 d_model: int = 256,
                 n_heads: int = 8,
                 n_layers: int = 6,
                 max_len: int = 2048,
                 dropout: float = 0.1,
                 cross_agent_attention: bool = True):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            max_len: Maximum sequence length
            dropout: Dropout probability
            cross_agent_attention: Whether to use cross-agent attention
        """
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.cross_agent_attention = cross_agent_attention

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # Input projection (if needed)
        self.input_proj = nn.Linear(d_model, d_model) if d_model != d_model else nn.Identity()

        # Transformer layers
        self.layers = nn.ModuleList([
            TemporalTransformerLayer(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                cross_agent_attention=cross_agent_attention
            ) for _ in range(n_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)

        # Layer normalization
        self.norm = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        logger.info(f"Temporal Transformer: {n_layers} layers, {d_model} dim, {n_heads} heads")

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through temporal transformer

        Args:
            x: Input features (batch, seq_len, d_model) or (batch, seq_len, n_mice, d_model)
            mask: Optional attention mask

        Returns:
            output: Transformed features with same shape as input
        """
        batch_size, seq_len = x.shape[:2]

        # Handle multi-agent input
        if len(x.shape) == 4:  # (batch, seq_len, n_mice, d_model)
            n_mice = x.shape[2]
            # Reshape to (batch*n_mice, seq_len, d_model) for processing
            x_flat = x.view(-1, seq_len, self.d_model)
        else:
            x_flat = x

        # Apply input projection and positional encoding
        x_flat = self.input_proj(x_flat)
        x_flat = self.pos_encoding(x_flat)
        x_flat = self.dropout(x_flat)

        # Pass through transformer layers
        for layer in self.layers:
            x_flat = layer(x_flat, mask)

        # Apply output projection and normalization
        x_flat = self.output_proj(x_flat)
        x_flat = self.norm(x_flat)

        # Reshape back if multi-agent
        if len(x.shape) == 4:
            output = x_flat.view(batch_size, seq_len, n_mice, self.d_model)
        else:
            output = x_flat

        return output


class TemporalTransformerLayer(nn.Module):
    """Single temporal transformer layer with cross-agent attention"""

    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 dropout: float = 0.1,
                 cross_agent_attention: bool = True):
        super().__init__()

        self.cross_agent_attention = cross_agent_attention
        self.d_model = d_model
        self.n_heads = n_heads

        # Self-attention for temporal modeling
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Cross-agent attention (if enabled)
        if cross_agent_attention:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True
            )

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        if cross_agent_attention:
            self.norm3 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through transformer layer

        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            output: Layer output
        """
        # Self-attention (temporal modeling)
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Cross-agent attention (if enabled)
        if self.cross_agent_attention and len(x.shape) == 3:
            # For cross-agent attention, treat different mice as different sequences
            # This is a simplified approach - full implementation would be more complex
            batch_size, seq_len, d_model = x.shape

            # Create cross-agent queries (using different mice as keys/values)
            # This is a placeholder - real implementation would need proper agent separation
            cross_output, _ = self.cross_attn(x, x, x, attn_mask=mask)
            x = self.norm2(x + self.dropout(cross_output))
        elif self.cross_agent_attention:
            x = self.norm2(x)

        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output)) if self.cross_agent_attention else self.norm2(x + self.dropout(ff_output))

        return x


class EnhancedTCN(nn.Module):
    """Enhanced Dilated TCN for local temporal patterns"""

    def __init__(self,
                 n_features: int,
                 hidden_dim: int = 512,
                 layers: int = 4,
                 dilation: list = None,
                 kernel_size: int = 3,
                 dropout: float = 0.1):
        """
        Args:
            n_features: Input feature dimension
            hidden_dim: Hidden dimension
            layers: Number of TCN layers
            dilation: Dilation rates
            kernel_size: Convolution kernel size
            dropout: Dropout probability
        """
        super().__init__()

        if dilation is None:
            dilation = [2 ** i for i in range(layers)]

        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.dilation = dilation
        self.kernel_size = kernel_size

        # Calculate receptive field
        self.receptive_field = self._calculate_receptive_field()

        # Input projection
        self.input_proj = nn.Linear(n_features, hidden_dim)

        # TCN layers with residual connections
        self.tcn_layers = nn.ModuleList()
        for i in range(layers):
            layer = EnhancedTCNBlock(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                dilation=dilation[i],
                dropout=dropout
            )
            self.tcn_layers.append(layer)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)

        logger.info(f"Enhanced TCN: {layers} layers, {hidden_dim} dim, receptive field: {self.receptive_field}")

    def _calculate_receptive_field(self) -> int:
        """Calculate the receptive field of the TCN"""
        rf = 1
        for i in range(self.layers):
            rf += (self.kernel_size - 1) * self.dilation[i]
        return rf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through enhanced TCN

        Args:
            x: Input features (batch, seq_len, n_features)

        Returns:
            output: TCN features (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, n_features = x.shape

        # Input projection
        x = self.input_proj(x)

        # Pass through TCN layers
        for layer in self.tcn_layers:
            x = layer(x)

        # Output projection and normalization
        x = self.output_proj(x)
        x = self.norm(x)

        return x


class EnhancedTCNBlock(nn.Module):
    """Enhanced TCN block with better temporal modeling"""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 dilation: int = 1,
                 dropout: float = 0.1):
        super().__init__()

        # Two convolutional layers with dilation
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

        # Temporal attention mechanism
        self.temporal_attn = nn.Sequential(
            nn.Conv1d(out_channels, out_channels // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(out_channels // 8, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Activation and dropout
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through enhanced TCN block

        Args:
            x: Input (batch, seq_len, channels)

        Returns:
            output: Block output
        """
        residual = self.residual(x.transpose(1, 2))
        x_t = x.transpose(1, 2)  # (batch, channels, seq_len)

        # First convolution
        out = self.conv1(x_t)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)

        # Second convolution
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.dropout(out)

        # Temporal attention
        attn_weights = self.temporal_attn(out)  # (batch, 1, seq_len)
        out = out * attn_weights

        # Gating
        gate_weights = self.gate(out)
        out = out * gate_weights

        # Add residual
        out = out + residual

        return out.transpose(1, 2)  # Back to (batch, seq_len, channels)


class DualBranchModel(nn.Module):
    """Dual-branch model for MABe behavior detection"""

    def __init__(self, config):
        """
        Args:
            config: Model configuration
        """
        super().__init__()

        self.n_features = config.model.n_features
        self.n_behaviors = config.model.n_behaviors

        # Local branch (fast dynamics)
        self.local_branch = EnhancedTCN(
            n_features=self.n_features,
            hidden_dim=config.model.local_branch.hidden_dim,
            layers=config.model.local_branch.layers,
            dilation=config.model.local_branch.dilation,
            kernel_size=config.model.local_branch.kernel_size,
            dropout=config.model.local_branch.dropout
        )

        # Global branch (long-range interactions)
        self.global_branch = TemporalTransformer(
            d_model=config.model.global_branch.d_model,
            n_heads=config.model.global_branch.n_heads,
            n_layers=config.model.global_branch.n_layers,
            max_len=config.model.global_branch.max_len,
            dropout=config.model.global_branch.dropout,
            cross_agent_attention=config.model.global_branch.cross_agent_attention
        )

        # Feature fusion
        fusion_dim = config.model.local_branch.hidden_dim + config.model.global_branch.d_model
        self.fusion = self._create_fusion_layer(config.model.fusion, fusion_dim)

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.model.fusion.dropout),
            nn.Linear(fusion_dim // 2, self.n_behaviors)
        )

        logger.info(f"Dual-branch model initialized:")
        logger.info(f"  Local branch: TCN with {self.local_branch.receptive_field} receptive field")
        logger.info(f"  Global branch: Transformer with {config.model.global_branch.n_layers} layers")
        logger.info(f"  Fusion: {config.model.fusion.type}")

    def _create_fusion_layer(self, fusion_config, input_dim: int):
        """Create fusion layer based on configuration"""
        fusion_type = fusion_config.type

        if fusion_type == 'linear':
            return nn.Sequential(
                nn.Linear(input_dim, fusion_config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(fusion_config.dropout)
            )
        elif fusion_type == 'attention':
            return nn.MultiheadAttention(
                embed_dim=input_dim,
                num_heads=8,
                dropout=fusion_config.dropout,
                batch_first=True
            )
        elif fusion_type == 'adaptive':
            # Adaptive fusion with learned weights
            return nn.Sequential(
                nn.Linear(input_dim * 2, fusion_config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(fusion_config.dropout),
                nn.Linear(fusion_config.hidden_dim, input_dim)
            )
        else:
            return nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through dual-branch model

        Args:
            x: Input features (batch, seq_len, n_features)

        Returns:
            logits: Behavior predictions (batch, seq_len, n_behaviors)
        """
        batch_size, seq_len, n_features = x.shape

        # Local branch (fast temporal patterns)
        local_features = self.local_branch(x)  # (batch, seq_len, hidden_dim)

        # Global branch (long-range dependencies)
        global_features = self.global_branch(x)  # (batch, seq_len, d_model)

        # Handle different output dimensions
        if global_features.shape[-1] != local_features.shape[-1]:
            # Project global features to match local dimensions
            proj = nn.Linear(global_features.shape[-1], local_features.shape[-1])
            global_features = proj(global_features)

        # Concatenate features
        if len(global_features.shape) == 3:  # Standard case
            combined_features = torch.cat([local_features, global_features], dim=-1)
        else:  # Multi-agent case
            # For multi-agent, concatenate along feature dimension
            combined_features = torch.cat([local_features, global_features], dim=-1)

        # Apply fusion
        if isinstance(self.fusion, nn.MultiheadAttention):
            # Attention-based fusion
            fused_features, _ = self.fusion(combined_features, combined_features, combined_features)
        else:
            # Linear fusion
            fused_features = self.fusion(combined_features)

        # Output head
        logits = self.output_head(fused_features)

        return logits
