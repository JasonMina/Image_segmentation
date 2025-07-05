"""
Loss functions optimized for sparse segmentation
Based on conversation emphasis on Focal + Dice for class imbalance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    WHAT: Dice loss implementation for segmentation tasks
    PURPOSE: Measures overlap between predicted and true masks
    WHY: Particularly good for small objects and class imbalance (mentioned in conversation)
    HOW: Calculates 2*intersection / (prediction + target), returns 1-dice as loss
    """
    
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # Flatten tensors
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        return 1 - dice

class FocalLoss(nn.Module):
    """
    WHAT: Focal Loss implementation for addressing extreme class imbalance
    PURPOSE: Down-weights easy examples, focuses training on hard cases
    WHY: Essential for sparse segmentation where background dominates (conversation emphasis)
    HOW: Modifies BCE with (1-pt)^gamma weighting term and alpha class balancing
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred, target):
        # Convert to probabilities
        pred_sigmoid = torch.sigmoid(pred)
        
        # Calculate focal loss
        pt = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CombinedLoss(nn.Module):
    """
    WHAT: Combination of Focal Loss and Dice Loss
    PURPOSE: Gets benefits of both losses - focal handles class imbalance, dice handles small objects
    WHY: Conversation emphasized this combination for sparse segmentation
    HOW: Weighted sum of focal and dice losses with configurable weights
    """
    
    def __init__(self, focal_weight=1.0, dice_weight=1.0, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss()
    
    def forward(self, pred, target):
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        
        total_loss = self.focal_weight * focal + self.dice_weight * dice
        
        return total_loss, focal, dice

class BoundaryLoss(nn.Module):
    """
    WHAT: Boundary-aware loss that emphasizes pixels near object edges
    PURPOSE: Improves segmentation quality at object boundaries
    WHY: Mentioned in conversation for pixel-level precision tasks
    HOW: Uses Sobel filters to detect edges, weights loss higher near boundaries
    """
    
    def __init__(self, theta=3):
        super().__init__()
        self.theta = theta
    
    def forward(self, pred, target):
        # Calculate boundary map using Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=target.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=target.device)
        
        sobel_x = sobel_x.view(1, 1, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3)
        
        target_expanded = target.unsqueeze(1) if target.dim() == 3 else target
        
        edge_x = F.conv2d(target_expanded, sobel_x, padding=1)
        edge_y = F.conv2d(target_expanded, sobel_y, padding=1)
        
        boundary_map = torch.sqrt(edge_x**2 + edge_y**2)
        boundary_map = (boundary_map > 0.5).float()
        
        # Weight loss by boundary proximity
        weights = 1 + self.theta * boundary_map.squeeze(1)
        
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        weighted_loss = (bce_loss * weights).mean()
        
        return weighted_loss

def create_loss_function(config):
    """
    WHAT: Factory function that creates loss function based on configuration
    PURPOSE: Allows easy switching between different loss functions
    WHY: Test day requires quick experimentation with different loss strategies
    HOW: Returns appropriate loss class instance based on config.loss_type
    """
    
    if config.loss_type == 'focal_dice':
        return CombinedLoss(
            focal_weight=config.focal_weight,
            dice_weight=config.dice_weight,
            focal_alpha=config.focal_alpha,
            focal_gamma=config.focal_gamma
        )
    elif config.loss_type == 'dice':
        return DiceLoss()
    elif config.loss_type == 'focal':
        return FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma)
    elif config.loss_type == 'bce':
        return nn.BCEWithLogitsLoss()
    elif config.loss_type == 'boundary':
        return BoundaryLoss()
    else:
        raise ValueError(f"Unknown loss type: {config.loss_type}")

def calculate_metrics(pred, target, threshold=0.5):
    """
    WHAT: Calculate standard segmentation evaluation metrics
    PURPOSE: Provides quantitative assessment of model performance
    WHY: Need to track IoU, Dice, precision, recall for model comparison
    HOW: Converts predictions to binary, calculates intersection-based metrics
    """
    
    # Convert predictions to binary
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    
    # Flatten for calculations
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)
    
    # Calculate metrics
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    
    # IoU
    iou = intersection / (union + 1e-7)
    
    # Dice
    dice = (2 * intersection) / (pred_flat.sum() + target_flat.sum() + 1e-7)
    
    # Accuracy
    correct = (pred_flat == target_flat).sum()
    accuracy = correct / target_flat.numel()
    
    # Precision and Recall
    true_positive = intersection
    false_positive = pred_flat.sum() - intersection
    false_negative = target_flat.sum() - intersection
    
    precision = true_positive / (true_positive + false_positive + 1e-7)
    recall = true_positive / (true_positive + false_negative + 1e-7)
    
    return {
        'iou': iou.item(),
        'dice': dice.item(),
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item()
    }
