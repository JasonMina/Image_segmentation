"""
Model creation with easy switching between architectures
Based on conversation about UNet, DeepLabV3+, UNet++ options
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

def create_model(config):
    """
    WHAT: Factory function that creates segmentation models based on config
    PURPOSE: Allows easy switching between different model architectures
    WHY: Test day requires quick experimentation with different models
    HOW: Uses segmentation_models_pytorch library with consistent interface
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
    WHAT: Wrapper class that adds inference utilities to base segmentation models
    PURPOSE: Provides convenient methods for prediction and large image handling
    WHY: Separates training logic from inference logic, adds sliding window capability
    HOW: Wraps the base model and adds predict methods with different options
    """
    
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        
    def forward(self, x):
        """
        WHAT: Standard forward pass for training
        PURPOSE: Required for PyTorch nn.Module interface
        WHY: Ensures compatibility with PyTorch training loops
        """
        return self.model(x)
    
    def predict(self, x, use_sigmoid=True, threshold=None):
        """
        WHAT: Inference method with optional sigmoid activation and thresholding
        PURPOSE: Clean interface for getting predictions during evaluation
        WHY: Separates inference logic from training, allows flexible output formats
        HOW: Applies sigmoid and threshold based on parameters
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
        WHAT: Sliding window inference for large images (like 700x2000 from conversation)
        PURPOSE: Handle images larger than model's input size by processing patches
        WHY: GPU memory limitations prevent processing very large images directly
        HOW: Slides overlapping windows across image, averages overlapping predictions
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
        WHAT: Test-time augmentation that applies multiple transforms and averages results
        PURPOSE: Improve prediction quality by ensemble of augmented inputs
        WHY: Often gives 1-2% performance boost with minimal computational cost
        HOW: Apply different flips, run inference, reverse transforms, average predictions
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
