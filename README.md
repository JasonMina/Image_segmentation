# Sparse Segmentation Pipeline

Optimized for datasets with extremely sparse labels (1-100 labeled pixels out of large images).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train with default settings (UNet + ResNet34 + Focal+Dice loss)
python train.py --data-dir Dataset-trial-difficult

# Train with different model
python train.py --model deeplabv3plus --encoder efficientnet-b4 --loss focal_dice

# Predict on single image
python predict.py --image test_image.jpg --model checkpoints/best_iou_model.pth

# Predict on directory
python predict.py --image-dir test_images/ --model checkpoints/best_iou_model.pth --output results/
```

## Easy Configuration Switching

As discussed in the conversation, easily switch between different setups:

### Model Architecture
```bash
# UNet (default, lightweight)
python train.py --model unet --encoder resnet34

# DeepLabV3+ (better for fine details)  
python train.py --model deeplabv3plus --encoder efficientnet-b4

# UNet++ (better feature fusion)
python train.py --model unetplusplus --encoder resnet50
```

### Loss Functions
```bash
# Focal + Dice (recommended for sparse labels)
python train.py --loss focal_dice

# Just Dice loss
python train.py --loss dice

# Just Focal loss  
python train.py --loss focal

# Basic BCE
python train.py --loss bce
```

### Training Strategy
```bash
# Patch-based training (default for sparse labels)
python train.py --patch-size 256

# Full image training
python train.py --no-patches

# Different patch sizes
python train.py --patch-size 512  # Larger context
python train.py --patch-size 128  # Smaller, faster
```

## Key Features

### ✅ Patch-Based Training
- Extracts patches around sparse labeled pixels
- Adds background patches for class balance
- Configurable patch size and background ratio

### ✅ Optimized Loss Functions
- **Focal + Dice**: Handles extreme class imbalance (recommended)
- **Boundary Loss**: For pixel-level precision
- Easy switching via command line

### ✅ Smart Inference
- Sliding window for large images (700×2000+)
- Test-time augmentation option
- Efficient overlapping patch handling

### ✅ TensorBoard Integration
- Real-time loss and metric monitoring
- Prediction visualization every few epochs
- Learning rate and gradient tracking

## File Structure

```
project/
├── train.py              # Training script with TensorBoard
├── predict.py             # Inference on single/batch images
├── config.py              # Flexible configuration system
├── dataset.py             # Patch-based dataset for sparse labels
├── models.py              # Model creation with easy switching
├── losses.py              # Focal+Dice and other loss functions
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Usage Examples

### Training Examples
```bash
# Quick training for test day (few epochs)
python train.py --epochs 10 --batch-size 4

# Full training with best settings for sparse data
python train.py --model deeplabv3plus --encoder efficientnet-b4 --loss focal_dice --lr 5e-5 --epochs 50

# Train on easy dataset first
python train.py --data-dir Dataset-trial-easy --epochs 20
```

### Prediction Examples
```bash
# Basic prediction
python predict.py --image unseen_image.jpg --model checkpoints/best_iou_model.pth

# With test-time augmentation
python predict.py --image unseen_image.jpg --model checkpoints/best_iou_model.pth --tta

# Batch prediction
python predict.py --image-dir unseen_images/ --model checkpoints/best_iou_model.pth --output results/

# Custom threshold
python predict.py --image test.jpg --model checkpoints/best_dice_model.pth --threshold 0.3
```

## Monitoring Training

```bash
# Start TensorBoard
tensorboard --logdir runs

# View at http://localhost:6006
```

TensorBoard shows:
- Training/validation loss curves
- IoU, Dice, accuracy metrics
- Learning rate schedule
- Prediction visualizations
- Loss component breakdown (Focal + Dice)

## For Test Day

### Recommended Workflow
1. **Quick setup**: `pip install -r requirements.txt`
2. **Train baseline**: `python train.py --epochs 20 --data-dir Dataset-trial-difficult`
3. **Monitor**: Start TensorBoard to watch training
4. **Predict**: `python predict.py --image-dir unseen_data/ --model checkpoints/best_iou_model.pth`
5. **Visualize**: Check output overlays and TensorBoard

### Key Switches for Adaptation
```bash
# If overfitting
python train.py --lr 2e-5 --batch-size 4

# If underfitting  
python train.py --lr 1e-4 --model deeplabv3plus

# If too slow
python train.py --patch-size 128 --batch-size 16

# If poor small object detection
python train.py --loss focal_dice --model unetplusplus
```

## Outputs

### Training Outputs
- `checkpoints/best_iou_model.pth` - Best IoU model
- `checkpoints/best_dice_model.pth` - Best Dice model  
- `runs/` - TensorBoard logs

### Prediction Outputs
- `*_mask.png` - Binary segmentation mask
- `*_prob.png` - Probability map
- `*_overlay.png` - Mask overlaid on original image
- `prediction_summary.txt` - Batch processing summary

## Tips for Test Day

1. **Start simple**: UNet + ResNet34 + Focal+Dice
2. **Monitor early**: Use TensorBoard from epoch 1
3. **Quick iterations**: 10-20 epoch experiments first
4. **Visual check**: Always look at overlay results
5. **Have fallbacks**: If one model fails, switch architecture quickly

This pipeline is designed for the sparse segmentation challenge with easy switching between configurations as discussed in the conversation.
