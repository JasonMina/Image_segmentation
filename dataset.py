"""
Patch-based dataset for sparse segmentation labels
Based on conversation about 1-100 labeled pixels out of 700x2000 images
"""

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import glob
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SparseSegmentationDataset(Dataset):
    """
    WHAT: Dataset class that handles sparse segmentation labels with patch sampling
    PURPOSE: Convert sparse labels (1-100 pixels) into balanced training patches
    WHY: Direct training on full images with sparse labels leads to class imbalance
         and poor learning - patches focus model attention on relevant areas
    HOW: Extracts patches around labeled pixels + adds background patches for balance
    """
    
    def __init__(self, data_dir, config, mode='train'):
        """
        WHAT: Initialize dataset with image/mask pairs and generate patch coordinates
        PURPOSE: Set up all data needed for patch-based training
        WHY: Pre-computing patch locations makes training faster and more balanced
        """
        self.data_dir = Path(data_dir)
        self.config = config
        self.mode = mode
        
        # Find image and mask pairs
        self.image_paths = sorted(glob.glob(str(self.data_dir / "*.jpg")) + 
                                 glob.glob(str(self.data_dir / "*.png")))
        self.mask_paths = []
        
        for img_path in self.image_paths:
            # Assume masks have same name but different suffix/extension
            mask_path = img_path.replace('image', 'label').replace('.jpg', '.png')
            if not os.path.exists(mask_path):
                mask_path = img_path.replace('.jpg', '_mask.png').replace('.png', '_mask.png')
            self.mask_paths.append(mask_path)
        
        # Verify pairs exist
        valid_pairs = []
        for img_path, mask_path in zip(self.image_paths, self.mask_paths):
            if os.path.exists(img_path) and os.path.exists(mask_path):
                valid_pairs.append((img_path, mask_path))
        
        self.image_paths, self.mask_paths = zip(*valid_pairs) if valid_pairs else ([], [])
        print(f"Found {len(self.image_paths)} valid image-mask pairs in {data_dir}")
        
        # Generate patches if using patch-based training
        if config.use_patches:
            self.patches = self._generate_patches()
        
        # Augmentations
        self.transform = self._get_transforms()
    
    def _generate_patches(self):
        """
        WHAT: Pre-compute all patch coordinates for positive and background regions
        PURPOSE: Creates balanced dataset of patches around sparse labels
        WHY: Avoids class imbalance by ensuring equal positive/negative patch sampling
        HOW: Find labeled pixels, create patches around them, then sample background patches
        """
        patches = []
        
        for img_path, mask_path in zip(self.image_paths, self.mask_paths):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
                
            # Find labeled pixels (conversation mentions very sparse labels)
            labeled_coords = np.argwhere(mask > 0)
            
            if len(labeled_coords) == 0:
                continue
            
            # Patches around labeled pixels
            for y, x in labeled_coords:
                patches.append({
                    'image_path': img_path,
                    'mask_path': mask_path,
                    'center': (y, x),
                    'type': 'positive'
                })
            
            # Background patches (conversation suggests 3x more background)
            h, w = mask.shape
            num_bg = len(labeled_coords) * self.config.num_background_patches
            
            for _ in range(num_bg):
                attempts = 0
                while attempts < 50:  # Avoid infinite loop
                    y = np.random.randint(self.config.patch_size//2, h - self.config.patch_size//2)
                    x = np.random.randint(self.config.patch_size//2, w - self.config.patch_size//2)
                    
                    # Check if this area has no labels
                    patch_mask = mask[y-self.config.patch_size//2:y+self.config.patch_size//2,
                                    x-self.config.patch_size//2:x+self.config.patch_size//2]
                    
                    if patch_mask.sum() == 0:  # Pure background
                        patches.append({
                            'image_path': img_path,
                            'mask_path': mask_path,
                            'center': (y, x),
                            'type': 'background'
                        })
                        break
                    attempts += 1
        
        print(f"Generated {len(patches)} patches")
        return patches
    
    def _get_transforms(self):
        """
        WHAT: Define data augmentation transforms for training vs validation
        PURPOSE: Increase data diversity and prevent overfitting
        WHY: Sparse labels need heavy augmentation to generalize well
        HOW: Applies geometric and color transforms consistently to image and mask
        """
        if self.mode == 'train':
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussNoise(var_limit=(10, 50), p=0.3),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __len__(self):
        """
        WHAT: Return dataset size for DataLoader
        PURPOSE: Tell PyTorch how many samples exist
        WHY: Required by Dataset interface
        """
        if self.config.use_patches:
            return len(self.patches)
        else:
            return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        WHAT: Get single training sample (image, mask) by index
        PURPOSE: Core Dataset method that DataLoader calls to get batches
        WHY: Required by PyTorch Dataset interface
        HOW: Routes to patch-based or full-image loading based on config
        """
        if self.config.use_patches:
            return self._get_patch(idx)
        else:
            return self._get_full_image(idx)
    
    def _get_patch(self, idx):
        """
        WHAT: Extract and return a single patch sample with augmentations
        PURPOSE: Get focused training sample around sparse labels
        WHY: Patches provide balanced signal-to-noise ratio for sparse segmentation
        HOW: Load full image/mask, extract patch around center point, apply transforms
        """
        patch_info = self.patches[idx]
        
        # Load full image and mask
        image = cv2.imread(patch_info['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(patch_info['mask_path'], cv2.IMREAD_GRAYSCALE)
        
        # Extract patch
        y, x = patch_info['center']
        half_size = self.config.patch_size // 2
        
        y1, y2 = max(0, y - half_size), min(image.shape[0], y + half_size)
        x1, x2 = max(0, x - half_size), min(image.shape[1], x + half_size)
        
        patch_img = image[y1:y2, x1:x2]
        patch_mask = mask[y1:y2, x1:x2]
        
        # Pad if needed
        target_size = self.config.patch_size
        if patch_img.shape[0] != target_size or patch_img.shape[1] != target_size:
            patch_img = cv2.resize(patch_img, (target_size, target_size))
            patch_mask = cv2.resize(patch_mask, (target_size, target_size))
        
        # Convert mask to binary
        patch_mask = (patch_mask > 127).astype(np.uint8)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=patch_img, mask=patch_mask)
            patch_img = transformed['image']
            patch_mask = transformed['mask']
        
        return patch_img, patch_mask.float()
    
    def _get_full_image(self, idx):
        """
        WHAT: Load and return full image resized to input size
        PURPOSE: Alternative to patch-based training for comparison
        WHY: Sometimes useful for testing different training strategies
        HOW: Load full image/mask, resize to config size, apply transforms
        """
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # Resize to input size
        image = cv2.resize(image, self.config.input_size)
        mask = cv2.resize(mask, self.config.input_size)
        mask = (mask > 127).astype(np.uint8)
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return image, mask.float()

def create_dataloaders(config):
    """
    WHAT: Create PyTorch DataLoaders for training and validation
    PURPOSE: Set up efficient batch loading with proper train/val split
    WHY: DataLoaders handle batching, shuffling, and parallel loading
    HOW: Create dataset, split it, wrap in DataLoaders with proper settings
    """
    
    # Create datasets
    train_dataset = SparseSegmentationDataset(config.data_dir, config, mode='train')
    
    # Simple train/val split (conversation mentions this is for test day scenario)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True if config.get_device().type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if config.get_device().type == 'cuda' else False
    )
    
    return train_loader, val_loaderdataset 
