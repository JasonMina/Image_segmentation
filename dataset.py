"""
Patch-based dataset for sparse segmentation labels
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
import random

class SparseSegmentationDataset(Dataset):
    """
    Dataset class that handles sparse segmentation labels with patch sampling.
    
    Converts sparse labels (1-100 pixels) into balanced training patches. Direct training 
    on full images with sparse labels leads to class imbalance and poor learning - patches 
    focus model attention on relevant areas by extracting patches around labeled pixels 
    and adding background patches for balance.
    """
    
    def __init__(self, data_dir, config, mode='train'):
        """
        Initialize dataset with image/mask pairs and generate patch coordinates.
        
        Args:
            data_dir (str): Path to directory containing image and mask files
            config (object): Configuration object with dataset parameters
            mode (str): Dataset mode, either 'train' or 'val'
        """
        self.data_dir = Path(data_dir)
        self.config = config
        self.mode = mode
        
        # Find image and mask pairs
        self.image_paths = sorted(glob.glob(str(self.data_dir / "*image.jpg")) + 
                                 glob.glob(str(self.data_dir / "*image.png")))
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
        Pre-compute all patch coordinates for positive and background regions.
        
        Creates balanced dataset of patches around sparse labels to avoid class imbalance 
        by ensuring equal positive/negative patch sampling. Finds labeled pixels, creates 
        patches around them, then samples background patches.
        
        Returns:
            list: List of dicts containing patch information with keys:
                - image_path (str): Path to source image
                - mask_path (str): Path to source mask
                - center (tuple): (y, x) coordinates of patch center
                - type (str): 'positive' or 'background'
        """
        patches = []

        for img_path, mask_path in zip(self.image_paths, self.mask_paths):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            h, w = mask.shape
            ps = self.config.patch_size // 2

            # Connected components instead of every labeled pixel
            mask_bin = (mask > 0).astype(np.uint8)
            num_labels, labeled_mask, stats, centroids = cv2.connectedComponentsWithStats(mask_bin)
            positive_patches = []
            for i in range(1, num_labels):  # skip background (label 0)
                x, y = centroids[i]
                y, x = int(y), int(x)
                area = stats[i, cv2.CC_STAT_AREA]
                
                if area >= self.config.min_mask_area:
                    if ps <= y < h - ps and ps <= x < w - ps:
                        positive_patches.append({
                            'image_path': img_path,
                            'mask_path': mask_path,
                            'center': (y, x),
                            'type': 'positive'
                        })

                # Sample background patches
                background_patches = []
                bg_needed = len(positive_patches) * self.config.num_background_patches

                attempts = 0
                while len(background_patches) < bg_needed and attempts < 10000:
                    y = np.random.randint(ps, h - ps)
                    x = np.random.randint(ps, w - ps)
                    patch_mask = mask[y - ps:y + ps, x - ps:x + ps]

                    if patch_mask.sum() == 0:
                        background_patches.append({
                            'image_path': img_path,
                            'mask_path': mask_path,
                            'center': (y, x),
                            'type': 'background'
                        })
                    attempts += 1

                patches.extend(positive_patches + background_patches)

        # Shuffle and optionally cap
        random.shuffle(patches)

        if len(patches) > self.config.max_total_patches:
            patches = random.sample(patches, self.config.max_total_patches)

        print(f"✅ Generated {len(patches)} patches "
            f"({sum(p['type']=='positive' for p in patches)} positive, "
            f"{sum(p['type']=='background' for p in patches)} background)")
        
        return patches
    
    def _get_transforms(self):
        """
        Define data augmentation transforms for training vs validation.
        
        Increases data diversity and prevents overfitting. Sparse labels need heavy 
        augmentation to generalize well by applying geometric and color transforms 
        consistently to image and mask.
        
        Returns:
            albumentations.Compose: Composed augmentation pipeline
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
        Return dataset size for DataLoader.
        
        Returns:
            int: Number of samples in the dataset
        """
        if self.config.use_patches:
            return len(self.patches)
        else:
            return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Get single training sample (image, mask) by index.
        
        Core Dataset method that DataLoader calls to get batches. Routes to patch-based 
        or full-image loading based on config.
        
        Args:
            idx (int): Index of sample to retrieve
            
        Returns:
            tuple: (image, mask) where image is tensor and mask is float tensor
        """
        if self.config.use_patches:
            return self._get_patch(idx)
        else:
            return self._get_full_image(idx)
    
    def _get_patch(self, idx):
        """
        Extract and return a single patch sample with augmentations.
        
        Gets focused training sample around sparse labels. Patches provide balanced 
        signal-to-noise ratio for sparse segmentation by loading full image/mask, 
        extracting patch around center point, and applying transforms.
        
        Args:
            idx (int): Index of patch to retrieve
            
        Returns:
            tuple: (patch_image, patch_mask) as tensors
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
        Load and return full image resized to input size.
        
        Alternative to patch-based training for comparison. Sometimes useful for testing 
        different training strategies by loading full image/mask, resizing to config size, 
        and applying transforms.
        
        Args:
            idx (int): Index of image to retrieve
            
        Returns:
            tuple: (image, mask) as tensors
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
    Create PyTorch DataLoaders for training and validation.
    
    Sets up efficient batch loading with proper train/val split. DataLoaders handle 
    batching, shuffling, and parallel loading by creating dataset, splitting it, and 
    wrapping in DataLoaders with proper settings.
    
    Args:
        config (object): Configuration object with training parameters including:
            - data_dir (str): Path to data directory
            - batch_size (int): Batch size for training
            - num_workers (int): Number of worker processes for data loading
            - get_device() method: Returns device (cuda/cpu)
            
    Returns:
        tuple: (train_loader, val_loader) PyTorch DataLoader objects
    """
    
    # Create datasets
    train_dataset = SparseSegmentationDataset(config.data_dir, config, mode='train')
    
    # Simple train/val split 
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
    
    return train_loader, val_loader