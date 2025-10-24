#!/usr/bin/env python3
"""
Setup script for MABe solution in Google Colab
Run this to prepare the environment for GPU training
"""

import os
import sys
from pathlib import Path

def setup_environment():
    """Setup Colab environment for MABe training"""
    print("🐭 Setting up MABe Colab environment...")

    # Check if we're in Colab
    if 'google.colab' in sys.modules:
        print("✅ Running in Google Colab")

        # Install dependencies
        print("📦 Installing dependencies...")
        !pip install --upgrade pip
        !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        !pip install pytorch-lightning hydra-core wandb
        !pip install pandas numpy scipy scikit-learn pyarrow
        !pip install matplotlib seaborn tqdm joblib torchmetrics

        print("✅ Dependencies installed")

        # Verify GPU
        import torch
        if torch.cuda.is_available():
            print(f"🔥 GPU available: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("❌ No GPU detected! Enable GPU in Runtime settings.")

        return True

    else:
        print("❌ Not running in Google Colab")
        print("This script is designed for Google Colab environment")
        return False

def setup_data_paths():
    """Setup data paths for Colab"""
    print("📁 Setting up data paths...")

    # Create data directory
    data_dir = "/content/MABe-data"
    os.makedirs(data_dir, exist_ok=True)

    print(f"✅ Data directory created: {data_dir}")
    print("\n📤 Upload Instructions:")
    print("1. Go to Files tab (left sidebar)")
    print("2. Upload these files to /content/MABe-data/:")
    print("   - train.csv")
    print("   - train_tracking/ (folder)")
    print("   - train_annotation/ (folder)")
    print("   - test.csv")
    print("   - test_tracking/ (folder)")

    return data_dir

def test_implementation():
    """Test the implementation"""
    print("🧪 Testing implementation...")

    try:
        # Test architecture
        print("Testing dual-branch architecture...")
        import subprocess
        result = subprocess.run([sys.executable, "test_architecture_simple.py"],
                              capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            print("✅ Architecture tests passed")
        else:
            print(f"⚠️ Architecture tests had issues: {result.stderr}")

        # Test basic functionality
        print("Testing basic functionality...")
        result = subprocess.run([sys.executable, "test_basic.py"],
                              capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            print("✅ Basic tests passed")
        else:
            print(f"⚠️ Basic tests had issues: {result.stderr}")

        return True

    except Exception as e:
        print(f"❌ Testing failed: {e}")
        return False

def create_training_config():
    """Create optimized training configuration for Colab"""
    print("⚙️ Creating training configuration...")

    colab_config = """
defaults:
  - data: default
  - model: dual_branch
  - training: phase2

experiment_name: colab_gpu_training
seed: 42

# GPU-optimized settings
training:
  batch_size: 8  # Optimized for Colab GPU memory
  max_epochs: 20
  val_check_interval: 0.5
  accumulate_grad_batches: 4  # Larger effective batch
  mixed_precision: true
  gradient_checkpointing: true

# Model optimizations for Colab
model:
  global_branch:
    n_layers: 4  # Reduced for Colab memory constraints
    d_model: 256
  local_branch:
    layers: 3    # Reduced for Colab
    hidden_dim: 512

# Data optimizations
data:
  window_sizes: [256, 512, 1024]
  overlap: 0.5
  positive_sampling_ratio: 0.3
  max_windows_per_video: 30  # Reduced for memory

wandb:
  enabled: false  # Disable for Colab unless you want to use it
"""

    with open('configs/colab_config.yaml', 'w') as f:
        f.write(colab_config)

    print("✅ Colab configuration created")
    return True

def main():
    """Main setup function"""
    print("🚀 MABe Colab Setup")
    print("=" * 50)

    # Setup environment
    if not setup_environment():
        return False

    # Setup data paths
    data_dir = setup_data_paths()

    # Test implementation
    test_implementation()

    # Create training config
    create_training_config()

    print("\n🎉 Setup Complete!")
    print("=" * 50)
    print("📋 Next Steps:")
    print("1. Upload MABe data files to /content/MABe-data/")
    print("2. Run: !python train.py --config-path configs --config-name colab_config")
    print("3. Monitor training with GPU acceleration")
    print("\n💡 Tips:")
    print("  - Use smaller batch sizes if you run out of GPU memory")
    print("  - Monitor validation F1 scores for early stopping")
    print("  - Save checkpoints regularly")

    return True

if __name__ == '__main__':
    main()
