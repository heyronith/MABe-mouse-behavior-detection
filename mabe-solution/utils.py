import hydra
from omegaconf import DictConfig, OmegaConf
import yaml
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> DictConfig:
    """Load configuration from Hydra config files"""
    with hydra.initialize(config_path="../configs"):
        config = hydra.compose(config_name="config")
    return config


def save_config(config: DictConfig, output_path: str):
    """Save configuration to file"""
    with open(output_path, 'w') as f:
        yaml.dump(OmegaConf.to_yaml(config), f)


def setup_experiment(config: DictConfig) -> str:
    """Setup experiment directory and save config"""
    import os
    from datetime import datetime

    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{config.experiment_name}_{timestamp}"
    experiment_dir = f"./experiments/{experiment_name}"

    os.makedirs(experiment_dir, exist_ok=True)

    # Save config
    save_config(config, f"{experiment_dir}/config.yaml")

    return experiment_dir


class AverageMeter:
    """Compute and store the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_progress(epoch: int, total_epochs: int, metrics: dict):
    """Print training progress"""
    progress = f"Epoch [{epoch}/{total_epochs}]"
    for key, value in metrics.items():
        progress += f" {key}: {value".4f"}"
    print(progress)
