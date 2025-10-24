import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predicted logits (batch, n_frames, n_behaviors)
            targets: Ground truth labels (batch, n_frames, n_behaviors)
        """
        # BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )

        # Focal modulation
        pt = torch.exp(-bce_loss)  # Probability of correct prediction
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class EffectiveNumLoss(FocalLoss):
    """Class-balanced focal loss using effective number of samples"""

    def __init__(self, beta: float = 0.999, gamma: float = 2.0):
        super().__init__(alpha=1.0, gamma=gamma)
        self.beta = beta

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Calculate class frequencies
        n_samples = targets.numel()
        n_positive = targets.sum().item()
        n_negative = n_samples - n_positive

        # Effective number of samples
        effective_pos = (1 - self.beta ** n_positive) / (1 - self.beta)
        effective_neg = (1 - self.beta ** n_negative) / (1 - self.beta)

        # Class weights
        total_effective = effective_pos + effective_neg
        pos_weight = total_effective / effective_pos
        neg_weight = total_effective / effective_neg

        # Apply class weights
        weights = targets * pos_weight + (1 - targets) * neg_weight
        weights = weights.unsqueeze(-1).expand_as(inputs)

        # BCE loss with weights
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )

        # Focal modulation
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss * weights

        return focal_loss.mean()


class TemporalConsistencyLoss(nn.Module):
    """Temporal consistency loss to discourage flickering predictions"""

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate temporal consistency loss

        Args:
            logits: Model predictions (batch, seq_len, n_behaviors)
            targets: Ground truth (batch, seq_len, n_behaviors)

        Returns:
            consistency_loss: Scalar loss value
        """
        batch_size, seq_len, n_behaviors = logits.shape

        if seq_len < 2:
            return torch.tensor(0.0, device=logits.device)

        # Calculate frame-to-frame differences
        logits_diff = logits[:, 1:] - logits[:, :-1]  # (batch, seq_len-1, n_behaviors)
        targets_diff = targets[:, 1:] - targets[:, :-1]

        # Consistency loss: penalize when predictions change but targets don't (and vice versa)
        consistency_loss = torch.mean(
            torch.abs(logits_diff - targets_diff)
        )

        return self.weight * consistency_loss


class MABeLightningModule(pl.LightningModule):
    """PyTorch Lightning module for MABe training with dual-branch support"""

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        # Model selection based on configuration
        if config.model.name == 'dual_branch':
            from models.dual_branch import DualBranchModel
            self.model = DualBranchModel(config)
        elif config.model.name == 'tcn':
            from models.tcn import MABeModel
            self.model = MABeModel(config)
        else:
            raise ValueError(f"Unknown model type: {config.model.name}")

        # Loss function with Phase 2 enhancements
        if config.training.loss == 'focal':
            if config.training.focal_alpha == 'effective_num':
                self.loss_fn = EffectiveNumLoss(gamma=config.training.focal_gamma)
            else:
                self.loss_fn = FocalLoss(
                    alpha=config.training.focal_alpha,
                    gamma=config.training.focal_gamma
                )
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()

        # Temporal consistency loss for Phase 2
        self.temporal_consistency_weight = getattr(config.training, 'temporal_consistency_weight', 0.1)
        self.temporal_consistency_loss = TemporalConsistencyLoss()

        # Metrics
        self.val_f1_scores = []

    def forward(self, tracking: torch.Tensor) -> torch.Tensor:
        return self.model(tracking)

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Training step with Phase 2 enhancements"""
        tracking = batch['tracking']
        annotations = batch['annotations']

        # Forward pass
        logits = self(tracking)

        # Main loss
        loss = self.loss_fn(logits, annotations)

        # Temporal consistency loss (Phase 2 enhancement)
        if self.temporal_consistency_weight > 0:
            consistency_loss = self.temporal_consistency_loss(logits, annotations)
            loss = loss + self.temporal_consistency_weight * consistency_loss
            self.log('train_consistency_loss', consistency_loss, prog_bar=False)

        # Log metrics
        self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> Dict:
        """Validation step"""
        tracking = batch['tracking']
        annotations = batch['annotations']

        # Forward pass
        logits = self(tracking)

        # Loss
        val_loss = self.loss_fn(logits, annotations)

        # Convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Calculate F1 score (simplified)
        # TODO: Implement proper F-score calculation in Phase 0 Layer 3
        threshold = 0.5
        predictions = (probs > threshold).float()

        # True positives, false positives, false negatives
        tp = (predictions * annotations).sum()
        fp = (predictions * (1 - annotations)).sum()
        fn = ((1 - predictions) * annotations).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        # Log metrics
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_f1', f1, prog_bar=True)
        self.log('val_precision', precision)
        self.log('val_recall', recall)

        return {
            'val_loss': val_loss,
            'val_f1': f1,
            'predictions': predictions,
            'annotations': annotations,
            'probs': probs
        }

    def validation_epoch_end(self, outputs: List[Dict]) -> None:
        """Aggregate validation metrics"""
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_f1 = torch.stack([x['val_f1'] for x in outputs]).mean()

        self.val_f1_scores.append(avg_f1.item())

        self.log('val_loss_epoch', avg_loss)
        self.log('val_f1_epoch', avg_f1)

        # Log worst-case performance (for lab-wise validation)
        if len(self.val_f1_scores) > 5:  # After some epochs
            worst_f1 = min(self.val_f1_scores[-5:])
            self.log('val_f1_worst_recent', worst_f1)

    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.config.training.lr,
            weight_decay=self.hparams.config.training.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.config.training.max_epochs,
            eta_min=self.hparams.config.training.min_lr
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
