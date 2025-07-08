"""
Model creation with easy switching between architectures
Based on UNet, DeepLabV3+, UNet++ options
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import numpy as np

def create_model(config):
    """
    Factory function that creates segmentation models based on config.
    
    Allows easy switching between different model architectures for test day 
    experimentation using segmentation_models_pytorch library with consistent interface.
    
    Args:
        config (object): Configuration object with model parameters including:
            - model_type (str): Type of model ('unet', 'deeplabv3plus', 'unetplusplus')
            - encoder (str): Encoder backbone name (e.g., 'resnet34', 'efficientnet-b0')
            - pretrained (bool): Whether to use ImageNet pretrained weights
            
    Returns:
        nn.Module: Configured segmentation model instance
        
    Raises:
        ValueError: If config.model_type is not recognized
    """
    
    model_params = {
        'encoder_name': config.encoder,
        'encoder_weights': 'imagenet' if config.pretrained else None,
        'in_channels': 3,
        'classes': 1,  # Binary segmentation
        'activation': None  # We'll use sigmoid in loss/inference
    }
    
    if config.model_type == 'unet':
        model = smp.Unet(**model_params)
    elif config.model_type == 'deeplabv3plus':
        model = smp.DeepLabV3Plus(**model_params)
    elif config.model_type == 'unetplusplus':
        model = smp.UnetPlusPlus(**model_params)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
    
    print(f"Created {config.model_type} with {config.encoder} encoder")
    return model

class ModelWrapper(nn.Module):
    """
    Wrapper class that adds inference utilities to base segmentation models.
    
    Provides convenient methods for prediction and large image handling. Separates 
    training logic from inference logic and adds sliding window capability by wrapping 
    the base model and adding predict methods with different options.
    """
    
    def __init__(self, model, config):
        """
        Initialize model wrapper.
        
        Args:
            model (nn.Module): Base segmentation model to wrap
            config (object): Configuration object with inference parameters
        """
        super().__init__()
        self.model = model
        self.config = config
        
    def forward(self, x):
        """
        Standard forward pass for training.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W]
            
        Returns:
            torch.Tensor: Raw logits of shape [B, 1, H, W]
        """
        return self.model(x)
    
    def predict(self, x, use_sigmoid=True, threshold=None):
        """
        Inference method with optional sigmoid activation and thresholding.
        
        Clean interface for getting predictions during evaluation. Separates inference 
        logic from training and allows flexible output formats by applying sigmoid and 
        threshold based on parameters.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W]
            use_sigmoid (bool): Whether to apply sigmoid activation
            threshold (float, optional): Threshold for binary prediction. If None, 
                                       returns probabilities
                                       
        Returns:
            torch.Tensor: Predictions of shape [B, 1, H, W]. Either logits, 
                         probabilities, or binary masks depending on parameters
        """
        with torch.no_grad():
            logits = self.model(x)
            if use_sigmoid:
                probs = torch.sigmoid(logits)
                if threshold is not None:
                    return (probs > threshold).float()
                return probs
            return logits
    
    def predict_full_image(self, image, stride_ratio=0.5):
        """
        Sliding window inference for large images
        
        Handle images larger than model's input size by processing patches to overcome 
        GPU memory limitations. Slides overlapping windows across image and averages 
        overlapping predictions.
        
        Args:
            image (torch.Tensor or np.ndarray): Input image. If numpy array, should be 
                                               of shape [H, W, C]. If tensor, should be 
                                               of shape [1, C, H, W]
            stride_ratio (float): Ratio of stride to patch_size for overlap control
            
        Returns:
            torch.Tensor: Full image prediction of shape [1, 1, H, W] on CPU
        """
        self.eval()
        device = next(self.parameters()).device
        
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        
        image = image.to(device)
        _, _, H, W = image.shape
        patch_size = self.config.patch_size
        stride = int(patch_size * stride_ratio)
        
        # Output mask
        prediction = torch.zeros((1, 1, H, W), device=device)
        count_map = torch.zeros((1, 1, H, W), device=device)
        
        # Sliding window
        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                patch = image[:, :, y:y+patch_size, x:x+patch_size]
                
                with torch.no_grad():
                    pred_patch = torch.sigmoid(self.model(patch))
                
                prediction[:, :, y:y+patch_size, x:x+patch_size] += pred_patch
                count_map[:, :, y:y+patch_size, x:x+patch_size] += 1
        
        # Handle borders
        if H % stride != 0:
            y = H - patch_size
            for x in range(0, W - patch_size + 1, stride):
                patch = image[:, :, y:y+patch_size, x:x+patch_size]
                with torch.no_grad():
                    pred_patch = torch.sigmoid(self.model(patch))
                prediction[:, :, y:y+patch_size, x:x+patch_size] += pred_patch
                count_map[:, :, y:y+patch_size, x:x+patch_size] += 1
        
        if W % stride != 0:
            x = W - patch_size
            for y in range(0, H - patch_size + 1, stride):
                patch = image[:, :, y:y+patch_size, x:x+patch_size]
                with torch.no_grad():
                    pred_patch = torch.sigmoid(self.model(patch))
                prediction[:, :, y:y+patch_size, x:x+patch_size] += pred_patch
                count_map[:, :, y:y+patch_size, x:x+patch_size] += 1
        
        # Corner patch
        if H % stride != 0 and W % stride != 0:
            patch = image[:, :, H-patch_size:H, W-patch_size:W]
            with torch.no_grad():
                pred_patch = torch.sigmoid(self.model(patch))
            prediction[:, :, H-patch_size:H, W-patch_size:W] += pred_patch
            count_map[:, :, H-patch_size:H, W-patch_size:W] += 1
        
        # Average overlapping predictions
        prediction = prediction / count_map.clamp(min=1)
        
        return prediction.cpu()
    
    def test_time_augmentation(self, x):
        """
        Test-time augmentation that applies multiple transforms and averages results.
        
        Improve prediction quality by ensemble of augmented inputs. Often gives 1-2% 
        performance boost with minimal computational cost by applying different flips, 
        running inference, reversing transforms, and averaging predictions.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W]
            
        Returns:
            torch.Tensor: Averaged predictions of shape [B, 1, H, W] with sigmoid applied
        """
        predictions = []
        
        # Original
        pred = torch.sigmoid(self.model(x))
        predictions.append(pred)
        
        # Horizontal flip
        pred_hflip = torch.sigmoid(self.model(torch.flip(x, dims=[3])))
        predictions.append(torch.flip(pred_hflip, dims=[3]))
        
        # Vertical flip
        pred_vflip = torch.sigmoid(self.model(torch.flip(x, dims=[2])))
        predictions.append(torch.flip(pred_vflip, dims=[2]))
        
        # Both flips
        pred_bothflip = torch.sigmoid(self.model(torch.flip(x, dims=[2, 3])))
        predictions.append(torch.flip(pred_bothflip, dims=[2, 3]))
        
        # Average all predictions
        final_pred = torch.stack(predictions).mean(dim=0)
        return final_pred