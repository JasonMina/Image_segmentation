"""
Configuration for sparse segmentation pipeline
"""

import argparse
from dataclasses import dataclass
from typing import Tuple, Optional
import torch
@dataclass
class Config:
    """
    Central configuration class that stores all hyperparameters and settings
    Single source of truth for all training/inference parameters
    Allows easy switching between different model architectures, loss functions,
         and training strategies without changing code
    """
    # Data paths
    data_dir: str = "Dataset-trial-difficult"
    easy_data_dir: str = "Dataset-trial-easy"
    
    # Model architecture - easy switching 
    model_type: str = "unet"  # unet, deeplabv3plus, unetplusplus
    encoder: str = "resnet34"  # resnet34, resnet50, efficientnet-b4
    pretrained: bool = True
    
    # Training strategy - patch-based for sparse labels
    use_patches: bool = True
    patch_size: int = 256
    num_background_patches: int = 3  # per positive patch
    max_total_patches: int=30000 
    min_mask_area: int=5
    label_sampling_stride: int=1
    
    # Training params - optimized for sparse segmentation
    batch_size: int = 8
    epochs: int = 50
    learning_rate: float = 1e-4  # Lower LR for stability with sparse labels
    weight_decay: float = 1e-5
    
    # Loss function
    loss_type: str = "focal_dice"  # focal_dice, dice, focal, bce
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    dice_weight: float = 1.0
    focal_weight: float = 1.0
    
    # Image processing
    input_size: Tuple[int, int] = (256, 256)  # For patches
    full_image_inference: bool = True  # Sliding window for full images
    
    # Hardware
    device: str = "auto"  # auto, cuda, cpu
    num_workers: int = 4
    
    # Logging - use tensorboard 
    log_dir: str = "runs"
    save_dir: str = "checkpoints"
    log_images_every: int = 5  # Log prediction images every N epochs
    
    # Inference
    threshold: float = 0.5
    use_tta: bool = False  # Test time augmentation
    
    @classmethod
    def from_args(cls):
        """
        Creates Config object from command line arguments
        Allows easy switching of settings from terminal without code changes
        Essential for test day - quickly try different models/losses/settings
        Parses command line args and overrides default config values
        """
        parser = argparse.ArgumentParser()
        

        parser.add_argument('--model', choices=['unet', 'deeplabv3plus', 'unetplusplus'], 
                          default='unet', help='Model architecture')
        parser.add_argument('--encoder', choices=['resnet34', 'resnet50', 'efficientnet-b4'],
                          default='resnet34', help='Encoder backbone')
        parser.add_argument('--loss', choices=['focal_dice', 'dice', 'focal', 'bce'],
                          default='focal_dice', help='Loss function')
        parser.add_argument('--patch-size', type=int, default=256, help='Patch size for training')
        parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
        parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
        parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
        parser.add_argument('--data-dir', default='Dataset-trial-difficult', help='Data directory')
        parser.add_argument('--no-patches', action='store_true', help='Train on full images instead of patches')
        
        args = parser.parse_args()
        
        # Create config with overrides
        config = cls()
        config.model_type = args.model
        config.encoder = args.encoder
        config.loss_type = args.loss
        config.patch_size = args.patch_size
        config.learning_rate = args.lr
        config.epochs = args.epochs
        config.batch_size = args.batch_size
        config.data_dir = args.data_dir
        config.use_patches = not args.no_patches
        config.input_size = (args.patch_size, args.patch_size)
        
        return config
    
    def get_device(self):
        """
        WHAT: Auto-detects the best available computing device
        PURPOSE: Automatically uses GPU if available, falls back to CPU
        WHY: Ensures code works on any machine without manual device setting
        HOW: Checks for CUDA availability and returns appropriate torch.device
        """
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        return torch.device(self.device)
    
    def print_config(self):
        """
        WHAT: Prints current configuration settings in readable format
        PURPOSE: Shows user exactly what settings are being used for training
        WHY: Essential for debugging and ensuring correct hyperparameters
        HOW: Formats and displays key configuration values
        """
        print("=" * 50)
        print("CONFIGURATION")
        print("=" * 50)
        print(f"Model: {self.model_type} + {self.encoder}")
        print(f"Loss: {self.loss_type}")
        print(f"Training: {'Patch-based' if self.use_patches else 'Full image'}")
        print(f"Patch size: {self.patch_size}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Batch size: {self.batch_size}")
        print(f"Epochs: {self.epochs}")
        print("=" * 50)
