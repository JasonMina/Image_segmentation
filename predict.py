"""
Inference script for sparse segmentation
Handles both patch-based and full image prediction
"""

import torch
import cv2
import numpy as np
import argparse
import os
import time
from pathlib import Path

from config import Config
from models import create_model, ModelWrapper
from train import load_checkpoint

def predict_image(image_path, model_path, config, output_dir=None, use_tta=False):
    """
    Predict segmentation mask for a single image using sliding window.
    Main inference function for generating masks on large images.
    Handles images larger than model input size
    Loads model, applies sliding window inference, saves multiple output formats.
    
    Args:
        image_path (str): Path to input image file
        model_path (str): Path to trained model checkpoint
        config (Config): Configuration object containing model parameters
        output_dir (str, optional): Directory to save output files. Defaults to None.
        use_tta (bool, optional): Whether to use test-time augmentation. Defaults to False.
    
    Returns:
        tuple: (mask, prediction_np) where mask is binary segmentation mask (numpy array)
               and prediction_np is probability map (numpy array)
    """
    
    # Setup
    device = config.get_device()
    
    # Load model
    model = create_model(config)
    model = ModelWrapper(model, config).to(device)
    
    print(f"Loading checkpoint from {model_path}")
    epoch, metric = load_checkpoint(model_path, model)
    print(f"Loaded model from epoch {epoch} with metric {metric:.4f}")
    
    model.eval()
    
    # Load image
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_shape = image.shape[:2]
    print(f"Image shape: {original_shape}")
    
    # Normalize image
    image_norm = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_norm = (image_norm - mean) / std
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image_norm.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    
    print("Running inference...")
    start_time = time.time()
    
    if use_tta:
        # Test-time augmentation
        prediction = model.test_time_augmentation(image_tensor.to(device))
    else:
        # Regular sliding window inference
        prediction = model.predict_full_image(image_tensor, stride_ratio=0.4)
    
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time:.2f} seconds")
    
    # Convert to numpy and threshold
    prediction_np = prediction.squeeze().cpu().numpy()
    mask = (prediction_np > config.threshold).astype(np.uint8) * 255
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save binary mask
        mask_path = os.path.join(output_dir, f"{Path(image_path).stem}_mask.png")
        cv2.imwrite(mask_path, mask)
        print(f"Saved mask: {mask_path}")
        
        # Save probability map
        prob_path = os.path.join(output_dir, f"{Path(image_path).stem}_prob.png")
        prob_map = (prediction_np * 255).astype(np.uint8)
        cv2.imwrite(prob_path, prob_map)
        print(f"Saved probability map: {prob_path}")
        
        # Save overlay
        overlay = create_overlay(image, mask)
        overlay_path = os.path.join(output_dir, f"{Path(image_path).stem}_overlay.png")
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"Saved overlay: {overlay_path}")
    
    return mask, prediction_np

def create_overlay(image, mask, alpha=0.5):
    """
    Create visual overlay of predicted mask on original image.
    Generate intuitive visualization for human evaluation.
    Essential for visual inspection of model performance.
    Blends colored mask with original image using alpha transparency.
    
    Args:
        image (numpy.ndarray): Original RGB image
        mask (numpy.ndarray): Binary segmentation mask
        alpha (float, optional): Transparency level for overlay. Defaults to 0.5.
    
    Returns:
        numpy.ndarray: RGB image with mask overlay
    """
    overlay = image.copy()
    
    # Create colored mask (red for positive predictions)
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = [255, 0, 0]  # Red
    
    # Blend
    overlay = cv2.addWeighted(overlay, 1-alpha, colored_mask, alpha, 0)
    
    return overlay

def predict_batch(image_dir, model_path, config, output_dir, extensions=('.jpg', '.png', '.jpeg')):
    """
    Run inference on all images in a directory.
    Batch processing for multiple images efficiently.
    Test day scenario requires processing multiple unseen images.
    Iterates through directory, processes each image, tracks statistics.
    
    Args:
        image_dir (str): Directory containing input images
        model_path (str): Path to trained model checkpoint
        config (Config): Configuration object containing model parameters
        output_dir (str): Directory to save output files
        extensions (tuple, optional): Valid image file extensions. Defaults to ('.jpg', '.png', '.jpeg').
    
    Returns:
        list: List of dictionaries containing processing results for each image
    """
    
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_paths = []
    for ext in extensions:
        image_paths.extend(list(image_dir.glob(f"*{ext}")))
        image_paths.extend(list(image_dir.glob(f"*{ext.upper()}")))
    
    print(f"Found {len(image_paths)} images")
    
    if not image_paths:
        print("No images found!")
        return
    
    # Process each image
    results = []
    total_time = 0
    
    for i, image_path in enumerate(image_paths):
        print(f"\nProcessing {i+1}/{len(image_paths)}: {image_path.name}")
        
        try:
            if "image" in str(image_path):
                start_time = time.time()
            
                mask, prob = predict_image(str(image_path), model_path, config, str(output_dir))
                process_time = time.time() - start_time
                total_time += process_time
            
                # Calculate some basic stats
                positive_pixels = (mask > 0).sum()
                total_pixels = mask.size
                positive_ratio = positive_pixels / total_pixels
                
                results.append({
                    'image': image_path.name,
                    'positive_pixels': positive_pixels,
                    'positive_ratio': positive_ratio,
                    'process_time': process_time
                })
                
                print(f"  Positive pixels: {positive_pixels} ({positive_ratio*100:.2f}%)")
                print(f"  Processing time: {process_time:.2f}s")
            
        except Exception as e:
            print(f"  Error processing {image_path.name}: {e}")
            continue
    
    print(f"\nBatch processing completed!")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per image: {total_time/len(image_paths):.2f}s")
    
    # Save summary
    summary_path = output_dir / "prediction_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Prediction Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Total images: {len(image_paths)}\n")
        f.write(f"Total time: {total_time:.2f}s\n")
        f.write(f"Average time: {total_time/len(image_paths):.2f}s\n\n")
        
        f.write("Per-image results:\n")
        f.write("-" * 50 + "\n")
        for result in results:
            f.write(f"{result['image']}: {result['positive_pixels']} pixels "
                   f"({result['positive_ratio']*100:.2f}%) in {result['process_time']:.2f}s\n")
    
    print(f"Summary saved: {summary_path}")
    
    return results

def main():
    """
    Command-line interface for running predictions.
    Provides easy-to-use interface for single or batch inference.
    Essential for test day - quick prediction on unseen data.
    Parses command line arguments and calls appropriate prediction function.
    
    Args:
        None (uses command line arguments)
    
    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Predict segmentation masks")
    parser.add_argument('--image', type=str, help='Single image path')
    parser.add_argument('--image-dir', type=str, help='Directory of images')
    parser.add_argument('--model', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--output', type=str, default='predictions', help='Output directory')
    parser.add_argument('--config', type=str, help='Config file (optional)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Prediction threshold')
    parser.add_argument('--tta', action='store_true', help='Use test-time augmentation')
    
    # Model config arguments
    parser.add_argument('--model-type', default='unet', choices=['unet', 'deeplabv3plus', 'unetplusplus'])
    parser.add_argument('--encoder', default='resnet34', choices=['resnet34', 'resnet50', 'efficientnet-b4'])
    parser.add_argument('--patch-size', type=int, default=256)
    
    args = parser.parse_args()
    
    # Create config
    config = Config()
    config.model_type = args.model_type
    config.encoder = args.encoder
    config.patch_size = args.patch_size
    config.threshold = args.threshold
    
    print("Prediction Configuration:")
    print(f"  Model: {config.model_type} + {config.encoder}")
    print(f"  Patch size: {config.patch_size}")
    print(f"  Threshold: {config.threshold}")
    print(f"  TTA: {args.tta}")
    
    if args.image:
        # Single image prediction
        print(f"\nPredicting single image...")
        mask, prob = predict_image(args.image, args.model, config, args.output, args.tta)
        print("Done!")
        
    elif args.image_dir:
        # Batch prediction
        print(f"\nPredicting batch...")
        results = predict_batch(args.image_dir, args.model, config, args.output)
        print("Done!")
        
    else:
        print("Please specify either --image or --image-dir")

if __name__ == "__main__":
    main()