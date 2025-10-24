#!/usr/bin/env python3
"""
MABe Training Script - Phase 0 Layer 1
Basic foundation: data loading, feature extraction, TCN model, training loop
"""

import os
import sys
import logging
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.loader import create_data_loaders
from models.tcn import MABeModel
from training.trainer import MABeLightningModule
from utils import load_config


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('training.log')
        ]
    )


def main():
    """Main training function with Phase 1 enhancements"""
    setup_logging()
    logger = logging.getLogger(__name__)

    # Load configuration
    config = load_config()

    logger.info("Starting MABe training - Phase 2 Dual-Branch")
    logger.info(f"Config: {config}")

    # Set random seed
    pl.seed_everything(config.seed)

    # Create enhanced data pipeline
    logger.info("Creating enhanced data pipeline...")
    from data.pipeline import MABeDataPipeline, DataPipelineOptimizer

    # Initialize pipeline
    pipeline = MABeDataPipeline(config)
    optimizer = DataPipelineOptimizer(config)

    # Validate configuration
    validation = optimizer.validate_configuration()
    if not validation['valid']:
        logger.error("Configuration validation failed!")
        return

    # Optimize memory usage
    memory_opt = optimizer.optimize_memory_usage(dataset_size=10000)
    logger.info(f"Memory optimization: batch_size={memory_opt['optimal_batch_size']}")

    # Create datasets with multi-scale windows
    train_dataset, val_dataset = pipeline.create_datasets(
        train_csv=config.data.train_csv,
        val_csv=None  # Use subset of training for validation
    )

    # Create data loaders with advanced sampling
    train_loader, val_loader = pipeline.create_data_loaders(train_dataset, val_dataset)

    # Validate pipeline on sample data
    sample_batch = [train_dataset[i] for i in range(min(10, len(train_dataset)))]
    pipeline_validation = pipeline.validate_pipeline(sample_batch)

    if not all([pipeline_validation['preprocessing_success'],
                pipeline_validation['feature_extraction_success'],
                pipeline_validation['augmentation_success']]):
        logger.warning("Pipeline validation had issues, but continuing...")

    # Create model
    logger.info("Creating enhanced model...")
    model = MABeLightningModule(config)

    # Setup callbacks with Phase 1 enhancements
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints',
        filename='mabe-phase1-{epoch:02d}-{val_f1:.3f}',
        monitor='val_f1',
        mode='max',
        save_top_k=3,
        save_last=True
    )

    early_stopping = EarlyStopping(
        monitor='val_f1_worst_recent',
        patience=15,  # Increased patience for Phase 1
        mode='max'
    )

    # Setup logger
    wandb_logger = WandbLogger(
        project=config.wandb.project,
        name=f"{config.experiment_name}_phase2",
        config=config,
        tags=['phase2', 'dual_branch', 'temporal_transformer', 'cross_agent_attention']
    ) if config.wandb.enabled else None

    # Create trainer with Phase 2 optimizations
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator='auto',
        devices='auto',
        precision=config.training.precision,
        callbacks=[checkpoint_callback, early_stopping],
        logger=wandb_logger,
        log_every_n_steps=20,
        val_check_interval=0.25,
        gradient_clip_val=config.model.gradient_clip,
        # Phase 2 enhancements
        accumulate_grad_batches=getattr(config.training, 'accumulate_grad_batches', 2),
        deterministic=getattr(config.training, 'deterministic_training', True),
        gradient_checkpointing=getattr(config.training, 'gradient_checkpointing', False)
    )

    # Train model with Phase 2 enhancements
    logger.info("Starting Phase 2 training...")
    logger.info("Phase 2 features:")
    logger.info("  - Dual-branch architecture (TCN + Transformer)")
    logger.info("  - Cross-agent attention mechanisms")
    logger.info("  - Temporal consistency loss")
    logger.info("  - Enhanced receptive fields (0.5-4s)")
    logger.info("  - Multi-scale feature fusion")

    trainer.fit(model, train_loader, val_loader)

    logger.info("Phase 2 training completed!")
    logger.info(f"Best checkpoint: {checkpoint_callback.best_model_path}")

    # Generate Phase 2 summary
    logger.info("=== PHASE 2 TRAINING SUMMARY ===")
    logger.info(f"Final validation F1: {trainer.callback_metrics.get('val_f1', 0):.4f}")
    logger.info(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    logger.info("Phase 2 dual-branch architecture successfully trained!")


if __name__ == '__main__':
    main()
