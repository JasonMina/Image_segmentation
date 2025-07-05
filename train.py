"""
Training loop for sparse segmentation
Uses PyTorch capabilities and TensorBoard as conversation suggests
"""

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import numpy as np
import os
from tqdm import tqdm
import time

from config import Config
from dataset import create_dataloaders
from models import create_model, ModelWrapper
from losses import create_loss_function, calculate_metrics

def train_model(config):
    """
    WHAT: Main training function that orchestrates the entire training process
    PURPOSE: Runs complete training loop with logging, checkpointing, and validation
    WHY: Central function that coordinates all training components
    HOW: Sets up model/data/loss, runs epoch loop, logs to TensorBoard, saves best models
    """
    
    # Setup
    device = config.get_device()
    print(f"Using device: {device}")
    config.print_config()
    
    # Create model
    model = create_model(config)
    model = ModelWrapper(model, config).to(device)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Loss and optimizer
    criterion = create_loss_function(config)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # TensorBoard setup
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.save_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_name = f"{config.model_type}_{config.encoder}_{config.loss_type}_{timestamp}"
    writer = SummaryWriter(os.path.join(config.log_dir, log_name))
    
    # Training state
    best_iou = 0.0
    best_dice = 0.0
    global_step = 0
    
    print("Starting training...")
    
    for epoch in range(config.epochs):
        # Training phase
        model.train()
        train_losses = []
        train_metrics = {'iou': [], 'dice': [], 'accuracy': []}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        for batch_idx, (images, masks) in enumerate(pbar):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Calculate loss
            if config.loss_type == 'focal_dice':
                loss, focal_loss, dice_loss = criterion(outputs, masks)
                # Log individual loss components
                writer.add_scalar('Loss/focal', focal_loss.item(), global_step)
                writer.add_scalar('Loss/dice', dice_loss.item(), global_step)
            else:
                loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Calculate metrics
            metrics = calculate_metrics(outputs, masks, config.threshold)
            for key, value in metrics.items():
                train_metrics[key].append(value)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'iou': f"{metrics['iou']:.3f}",
                'dice': f"{metrics['dice']:.3f}"
            })
            
            # Log to tensorboard every 10 steps
            if global_step % 10 == 0:
                writer.add_scalar('Loss/train', loss.item(), global_step)
                writer.add_scalar('Metrics/train_iou', metrics['iou'], global_step)
                writer.add_scalar('Metrics/train_dice', metrics['dice'], global_step)
                writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
            
            global_step += 1
        
        # Calculate epoch averages
        avg_train_loss = np.mean(train_losses)
        avg_train_iou = np.mean(train_metrics['iou'])
        avg_train_dice = np.mean(train_metrics['dice'])
        avg_train_acc = np.mean(train_metrics['accuracy'])
        
        # Validation phase
        model.eval()
        val_losses = []
        val_metrics = {'iou': [], 'dice': [], 'accuracy': []}
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validation"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                
                if config.loss_type == 'focal_dice':
                    loss, _, _ = criterion(outputs, masks)
                else:
                    loss = criterion(outputs, masks)
                
                val_losses.append(loss.item())
                
                metrics = calculate_metrics(outputs, masks, config.threshold)
                for key, value in metrics.items():
                    val_metrics[key].append(value)
        
        # Calculate validation averages
        avg_val_loss = np.mean(val_losses)
        avg_val_iou = np.mean(val_metrics['iou'])
        avg_val_dice = np.mean(val_metrics['dice'])
        avg_val_acc = np.mean(val_metrics['accuracy'])
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Log epoch metrics to tensorboard
        writer.add_scalar('Epoch/train_loss', avg_train_loss, epoch)
        writer.add_scalar('Epoch/val_loss', avg_val_loss, epoch)
        writer.add_scalar('Epoch/train_iou', avg_train_iou, epoch)
        writer.add_scalar('Epoch/val_iou', avg_val_iou, epoch)
        writer.add_scalar('Epoch/train_dice', avg_train_dice, epoch)
        writer.add_scalar('Epoch/val_dice', avg_val_dice, epoch)
        
        # Log images every few epochs
        if epoch % config.log_images_every == 0:
            log_predictions(writer, model, val_loader, device, epoch, config)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{config.epochs}:")
        print(f"  Train - Loss: {avg_train_loss:.4f}, IoU: {avg_train_iou:.4f}, Dice: {avg_train_dice:.4f}")
        print(f"  Val   - Loss: {avg_val_loss:.4f}, IoU: {avg_val_iou:.4f}, Dice: {avg_val_dice:.4f}")
        
        # Save best models
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            save_checkpoint(model, optimizer, epoch, avg_val_iou, 
                          os.path.join(config.save_dir, "best_iou_model.pth"))
            print(f"  ✓ New best IoU: {best_iou:.4f}")
        
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            save_checkpoint(model, optimizer, epoch, avg_val_dice,
                          os.path.join(config.save_dir, "best_dice_model.pth"))
            print(f"  ✓ New best Dice: {best_dice:.4f}")
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch, avg_val_iou,
                          os.path.join(config.save_dir, f"checkpoint_epoch_{epoch+1}.pth"))
    
    writer.close()
    print(f"\nTraining completed!")
    print(f"Best IoU: {best_iou:.4f}")
    print(f"Best Dice: {best_dice:.4f}")
    print(f"TensorBoard logs: {os.path.join(config.log_dir, log_name)}")
    
    return model

def log_predictions(writer, model, val_loader, device, epoch, config):
    """
    WHAT: Log prediction visualizations to TensorBoard
    PURPOSE: Visual monitoring of model predictions during training
    WHY: Essential for debugging and understanding model behavior
    HOW: Takes samples from validation set, runs inference, logs image comparisons
    """
    model.eval()
    
    # Get one batch for visualization
    images, masks = next(iter(val_loader))
    images, masks = images.to(device), masks.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        predictions = torch.sigmoid(outputs) > config.threshold
    
    # Take first few samples from batch
    num_samples = min(4, images.shape[0])
    
    for i in range(num_samples):
        # Original image (denormalize)
        img = images[i].clone()
        img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
        img = img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        img = torch.clamp(img, 0, 1)
        
        # Ground truth mask (convert to 3-channel for visualization)
        gt_mask = masks[i].unsqueeze(0).repeat(3, 1, 1)
        
        # Prediction mask (convert to 3-channel)
        pred_mask = predictions[i].float().unsqueeze(0).repeat(3, 1, 1)
        
        # Create comparison grid
        comparison = torch.cat([img, gt_mask, pred_mask], dim=2)  # Concatenate horizontally
        
        writer.add_image(f'Predictions/sample_{i}', comparison, epoch)

def save_checkpoint(model, optimizer, epoch, metric, filepath):
    """
    WHAT: Save model checkpoint with training state
    PURPOSE: Preserve model weights and training state for later use
    WHY: Allows resuming training and loading best models for inference
    HOW: Saves model state_dict, optimizer state, epoch, and performance metric
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metric': metric,
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(filepath, model, optimizer=None):
    """
    WHAT: Load model checkpoint and restore training state
    PURPOSE: Resume training or load trained model for inference
    WHY: Essential for model deployment and continuing interrupted training
    HOW: Loads state dicts into model and optimizer, returns epoch and metric info
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['metric']

if __name__ == "__main__":
    """
    WHAT: Main entry point when script is run directly
    PURPOSE: Parse command line arguments and start training
    WHY: Allows easy command-line usage for different configurations
    HOW: Creates config from args and calls train_model function
    """
    # Create config from command line args
    config = Config.from_args()
    
    # Train model
    trained_model = train_model(config)
