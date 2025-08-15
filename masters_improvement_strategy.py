#!/usr/bin/env python3
"""
Master's Degree Performance Improvement Strategy
Advanced training techniques to achieve Whole DSC ≥ 0.91 and Classification F1 ≥ 0.70
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
import time
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt

class AdvancedMultiTaskLoss(nn.Module):
    """Enhanced loss function for better performance"""
    
    def __init__(self, seg_weight: float = 1.0, cls_weight: float = 0.5, 
                 focal_alpha: float = 0.25, focal_gamma: float = 2.0):
        super().__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.ce_loss = nn.CrossEntropyLoss()
        
    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Focal loss for handling hard examples"""
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1-pt)**self.focal_gamma * ce_loss
        return focal_loss.mean()
    
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Enhanced Dice loss with per-class weighting"""
        smooth = 1e-5
        num_classes = pred.shape[1]
        
        target_one_hot = F.one_hot(target, num_classes).permute(0, 4, 1, 2, 3).float()
        pred_soft = F.softmax(pred, dim=1)
        
        dice_scores = []
        class_weights = [0.1, 1.0, 1.2]  # Background, Pancreas, Lesion weights
        
        for c in range(num_classes):
            pred_c = pred_soft[:, c]
            target_c = target_one_hot[:, c]
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            dice = (2 * intersection + smooth) / (union + smooth)
            weighted_dice = dice * class_weights[c] if c < len(class_weights) else dice
            dice_scores.append(weighted_dice)
        
        return 1 - torch.stack(dice_scores).mean()
    
    def boundary_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Boundary loss for better edge detection"""
        def compute_gradient(tensor):
            dx = tensor[:, :, 1:, :, :] - tensor[:, :, :-1, :, :]
            dy = tensor[:, :, :, 1:, :] - tensor[:, :, :, :-1, :]
            dz = tensor[:, :, :, :, 1:] - tensor[:, :, :, :, :-1]
            return dx, dy, dz
        
        pred_soft = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, pred.shape[1]).permute(0, 4, 1, 2, 3).float()
        
        pred_grad = compute_gradient(pred_soft)
        target_grad = compute_gradient(target_one_hot)
        
        boundary_loss = 0
        for pg, tg in zip(pred_grad, target_grad):
            if pg.shape == tg.shape:
                boundary_loss += F.mse_loss(pg, tg)
        
        return boundary_loss / len(pred_grad)
    
    def forward(self, predictions: dict, seg_target: torch.Tensor, cls_target: torch.Tensor):
        seg_pred = predictions['segmentation']
        cls_pred = predictions['classification']
        
        # Enhanced segmentation loss
        ce_loss = self.ce_loss(seg_pred, seg_target)
        dice_loss = self.dice_loss(seg_pred, seg_target)
        boundary_loss = self.boundary_loss(seg_pred, seg_target)
        seg_loss = 0.4 * ce_loss + 0.4 * dice_loss + 0.2 * boundary_loss
        
        # Enhanced classification loss with focal loss
        cls_loss = 0.7 * self.focal_loss(cls_pred, cls_target) + 0.3 * self.ce_loss(cls_pred, cls_target)
        
        total_loss = self.seg_weight * seg_loss + self.cls_weight * cls_loss
        
        return {
            'total_loss': total_loss,
            'seg_loss': seg_loss,
            'cls_loss': cls_loss,
            'ce_loss': ce_loss,
            'dice_loss': dice_loss,
            'boundary_loss': boundary_loss
        }

class EnhancedDataAugmentation:
    """Advanced data augmentation for better generalization"""
    
    def __init__(self, intensity_range: tuple = (0.8, 1.2), 
                 noise_std: float = 0.05, rotation_range: int = 15):
        self.intensity_range = intensity_range
        self.noise_std = noise_std
        self.rotation_range = rotation_range
    
    def apply_augmentation(self, image: torch.Tensor, mask: torch.Tensor) -> tuple:
        """Apply random augmentations"""
        
        # Intensity scaling
        if np.random.random() > 0.5:
            scale = np.random.uniform(*self.intensity_range)
            image = image * scale
        
        # Gaussian noise
        if np.random.random() > 0.5:
            noise = torch.randn_like(image) * self.noise_std
            image = image + noise
        
        # Random flip
        if np.random.random() > 0.5:
            axis = np.random.choice([2, 3, 4])  # Random spatial axis
            image = torch.flip(image, [axis])
            mask = torch.flip(mask, [axis-1])  # Adjust for mask dimensions
        
        # Gamma correction
        if np.random.random() > 0.5:
            gamma = np.random.uniform(0.8, 1.2)
            image = torch.pow(image.clamp(min=1e-7), gamma)
        
        return image, mask

def create_balanced_sampler(dataset):
    """Create weighted sampler for balanced training"""
    
    # Count samples per class
    class_counts = {}
    for sample in dataset.samples:
        subtype = sample['subtype']
        class_counts[subtype] = class_counts.get(subtype, 0) + 1
    
    # Calculate weights
    total_samples = len(dataset)
    num_classes = len(class_counts)
    
    weights = []
    for sample in dataset.samples:
        subtype = sample['subtype']
        if subtype >= 0:  # Valid class
            class_weight = total_samples / (num_classes * class_counts[subtype])
            weights.append(class_weight)
        else:
            weights.append(1.0)
    
    return WeightedRandomSampler(weights, len(weights), replacement=True)

def advanced_training(data_root: str = ".", max_epochs: int = 200, 
                     target_whole_dsc: float = 0.91, target_cls_f1: float = 0.70):
    """Advanced training strategy for Master's targets"""
    
    print("🎓 MASTER'S DEGREE ADVANCED TRAINING")
    print("=" * 50)
    print(f"🎯 Targets:")
    print(f"   • Whole Pancreas DSC ≥ {target_whole_dsc}")
    print(f"   • Classification F1 ≥ {target_cls_f1}")
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Using device: {device}")
    
    # Import required classes
    from working_standalone import PancreasCancerDataset, MultiTaskUNet3D, calculate_dice_scores
    
    # Enhanced datasets with augmentation
    train_dataset = PancreasCancerDataset(data_root, 'train', patch_size=(64, 64, 64))
    val_dataset = PancreasCancerDataset(data_root, 'validation', patch_size=(64, 64, 64))
    
    # Balanced sampling
    balanced_sampler = create_balanced_sampler(train_dataset)
    
    # Data loaders with balanced sampling
    train_loader = DataLoader(
        train_dataset, 
        batch_size=3,  # Slightly larger batch
        sampler=balanced_sampler,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    # Enhanced model
    model = MultiTaskUNet3D(base_filters=48).to(device)  # Larger base filters
    print(f"🎓 Enhanced model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Advanced loss function
    criterion = AdvancedMultiTaskLoss(seg_weight=1.2, cls_weight=0.8)
    
    # Advanced optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=0.0008,  # Slightly lower LR
        weight_decay=2e-5,
        betas=(0.9, 0.999)
    )
    
    # Cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=1e-6
    )
    
    scaler = torch.amp.GradScaler('cuda')
    
    # Data augmentation
    augmentor = EnhancedDataAugmentation()
    
    # Training tracking
    best_combined_score = 0.0
    best_whole_dsc = 0.0
    best_cls_f1 = 0.0
    patience = 25
    patience_counter = 0
    
    results_path = Path(os.environ['nnUNet_results']) / "Dataset500_PancreasCancer"
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Training history for plots
    history = {
        'train_loss': [],
        'val_loss': [],
        'whole_dsc': [],
        'lesion_dsc': [],
        'cls_f1': [],
        'lr': []
    }
    
    for epoch in range(max_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}")
        
        for batch in progress_bar:
            images = batch['image'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)
            subtypes = batch['subtype'].to(device, non_blocking=True)
            
            # Filter valid samples
            valid_mask = subtypes >= 0
            if not valid_mask.any():
                continue
            
            images = images[valid_mask]
            masks = masks[valid_mask]
            subtypes = subtypes[valid_mask]
            
            # Apply augmentation
            if np.random.random() > 0.3:  # 70% chance of augmentation
                images, masks = augmentor.apply_augmentation(images, masks)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss_dict = criterion(outputs, masks, subtypes)
            
            scaler.scale(loss_dict['total_loss']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss_dict['total_loss'].item()
            train_batches += 1
            
            progress_bar.set_postfix({
                'Loss': f"{loss_dict['total_loss'].item():.4f}",
                'LR': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
        
        scheduler.step()
        
        if train_batches > 0:
            train_loss /= train_batches
            history['train_loss'].append(train_loss)
            history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Validation phase (every 2 epochs for efficiency)
        if epoch % 2 == 0:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            val_predictions = []
            val_targets = []
            val_cls_preds = []
            val_cls_true = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    images = batch['image'].to(device, non_blocking=True)
                    masks = batch['mask'].to(device, non_blocking=True)
                    subtypes = batch['subtype'].to(device, non_blocking=True)
                    
                    valid_mask = subtypes >= 0
                    if not valid_mask.any():
                        continue
                    
                    images = images[valid_mask]
                    masks = masks[valid_mask]
                    subtypes = subtypes[valid_mask]
                    
                    with torch.amp.autocast('cuda'):
                        outputs = model(images)
                        loss_dict = criterion(outputs, masks, subtypes)
                    
                    val_loss += loss_dict['total_loss'].item()
                    val_batches += 1
                    
                    # Collect predictions
                    seg_pred = torch.argmax(outputs['segmentation'], dim=1)
                    cls_pred = torch.argmax(outputs['classification'], dim=1)
                    
                    val_predictions.extend(seg_pred.cpu().numpy())
                    val_targets.extend(masks.cpu().numpy())
                    val_cls_preds.extend(cls_pred.cpu().numpy())
                    val_cls_true.extend(subtypes.cpu().numpy())
            
            if val_batches > 0:
                val_loss /= val_batches
                history['val_loss'].append(val_loss)
                
                # Calculate metrics
                dice_scores = calculate_dice_scores(val_predictions, val_targets)
                whole_dsc = dice_scores['whole_dice']
                lesion_dsc = dice_scores['lesion_dice']
                
                history['whole_dsc'].append(whole_dsc)
                history['lesion_dsc'].append(lesion_dsc)
                
                # Classification metrics
                if val_cls_preds and val_cls_true:
                    _, _, cls_f1, _ = precision_recall_fscore_support(
                        val_cls_true, val_cls_preds, average='macro', zero_division=0
                    )
                    history['cls_f1'].append(cls_f1)
                else:
                    cls_f1 = 0.0
                    history['cls_f1'].append(0.0)
                
                # Target achievement check
                whole_target_met = whole_dsc >= target_whole_dsc
                cls_target_met = cls_f1 >= target_cls_f1
                
                # Combined score with emphasis on target metrics
                if whole_target_met and cls_target_met:
                    combined_score = whole_dsc * 0.5 + cls_f1 * 0.5  # Both targets met
                else:
                    combined_score = whole_dsc * 0.4 + lesion_dsc * 0.2 + cls_f1 * 0.4
                
                print(f"\nEpoch {epoch:3d}: "
                      f"Loss={train_loss:.4f}, "
                      f"Val Loss={val_loss:.4f}")
                print(f"         "
                      f"Whole DSC={whole_dsc:.3f} ({'✅' if whole_target_met else '❌'}), "
                      f"Lesion DSC={lesion_dsc:.3f}, "
                      f"Cls F1={cls_f1:.3f} ({'✅' if cls_target_met else '❌'})")
                
                # Track best performance
                if whole_dsc > best_whole_dsc:
                    best_whole_dsc = whole_dsc
                if cls_f1 > best_cls_f1:
                    best_cls_f1 = cls_f1
                
                # Save best model
                if combined_score > best_combined_score:
                    best_combined_score = combined_score
                    patience_counter = 0
                    
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'metrics': dice_scores,
                        'cls_f1': cls_f1,
                        'combined_score': combined_score,
                        'best_whole_dsc': best_whole_dsc,
                        'best_cls_f1': best_cls_f1
                    }, results_path / "masters_best_model.pth")
                    
                    print(f"         💾 Best model saved (score: {combined_score:.3f})")
                    
                    # Check if Master's targets achieved
                    if whole_target_met and cls_target_met:
                        print(f"         🎯 🎉 MASTER'S TARGETS ACHIEVED! 🎉")
                        print(f"         Whole DSC: {whole_dsc:.3f} ≥ {target_whole_dsc}")
                        print(f"         Classification F1: {cls_f1:.3f} ≥ {target_cls_f1}")
                        
                        # Continue training for a few more epochs to solidify performance
                        if patience < 10:
                            patience = 10
                
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= patience:
                    print(f"\n⏹️ Early stopping at epoch {epoch}")
                    break
        
        # Progress update every 10 epochs
        if epoch % 10 == 0 and epoch > 0:
            print(f"\n📊 Progress Update (Epoch {epoch}):")
            print(f"   Best Whole DSC so far: {best_whole_dsc:.3f} (Target: {target_whole_dsc})")
            print(f"   Best Classification F1: {best_cls_f1:.3f} (Target: {target_cls_f1})")
            gap_whole = ((target_whole_dsc / best_whole_dsc - 1) * 100) if best_whole_dsc < target_whole_dsc else 0
            gap_cls = ((target_cls_f1 / best_cls_f1 - 1) * 100) if best_cls_f1 < target_cls_f1 else 0
            if gap_whole > 0:
                print(f"   Whole DSC gap: {gap_whole:.1f}% improvement needed")
            if gap_cls > 0:
                print(f"   Classification gap: {gap_cls:.1f}% improvement needed")
    
    # Generate training plots
    create_training_plots(history, results_path)
    
    print("\n✅ Advanced training completed!")
    print(f"🎯 Final Best Performance:")
    print(f"   Whole DSC: {best_whole_dsc:.3f} (Target: {target_whole_dsc}) {'✅' if best_whole_dsc >= target_whole_dsc else '❌'}")
    print(f"   Classification F1: {best_cls_f1:.3f} (Target: {target_cls_f1}) {'✅' if best_cls_f1 >= target_cls_f1 else '❌'}")
    
    return best_whole_dsc >= target_whole_dsc and best_cls_f1 >= target_cls_f1

def create_training_plots(history, save_path):
    """Create comprehensive training plots"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Master\'s Degree Advanced Training Results', fontsize=16)
    
    epochs = range(len(history['train_loss']))
    val_epochs = range(0, len(history['train_loss']), 2)  # Validation every 2 epochs
    
    # Loss curves
    axes[0,0].plot(epochs, history['train_loss'], label='Training Loss', color='blue')
    if history['val_loss']:
        axes[0,0].plot(val_epochs[:len(history['val_loss'])], history['val_loss'], 
                      label='Validation Loss', color='red')
    axes[0,0].set_title('Loss Curves')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Whole DSC with target
    if history['whole_dsc']:
        axes[0,1].plot(val_epochs[:len(history['whole_dsc'])], history['whole_dsc'], 
                      label='Whole DSC', color='green', linewidth=2)
        axes[0,1].axhline(y=0.91, color='green', linestyle='--', alpha=0.8, 
                         label='Master Target', linewidth=2)
        axes[0,1].set_title('Whole Pancreas DSC vs Master Target')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Dice Score')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
    
    # Classification F1 with target
    if history['cls_f1']:
        axes[0,2].plot(val_epochs[:len(history['cls_f1'])], history['cls_f1'], 
                      label='Classification F1', color='purple', linewidth=2)
        axes[0,2].axhline(y=0.7, color='purple', linestyle='--', alpha=0.8, 
                         label='Master Target', linewidth=2)
        axes[0,2].set_title('Classification F1 vs Master Target')
        axes[0,2].set_xlabel('Epoch')
        axes[0,2].set_ylabel('F1 Score')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
    
    # Learning rate
    if history['lr']:
        axes[1,0].plot(epochs, history['lr'], label='Learning Rate', color='orange')
        axes[1,0].set_title('Learning Rate Schedule')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('Learning Rate')
        axes[1,0].set_yscale('log')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
    
    # Lesion DSC
    if history['lesion_dsc']:
        axes[1,1].plot(val_epochs[:len(history['lesion_dsc'])], history['lesion_dsc'], 
                      label='Lesion DSC', color='orange', linewidth=2)
        axes[1,1].axhline(y=0.31, color='orange', linestyle='--', alpha=0.8, 
                         label='Master Target', linewidth=2)
        axes[1,1].set_title('Lesion DSC (Already Achieved)')
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_ylabel('Dice Score')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
    
    # Target achievement timeline
    if history['whole_dsc'] and history['cls_f1']:
        whole_target_met = [1 if x >= 0.91 else 0 for x in history['whole_dsc']]
        cls_target_met = [1 if x >= 0.7 else 0 for x in history['cls_f1']]
        both_met = [w * c for w, c in zip(whole_target_met, cls_target_met)]
        
        axes[1,2].plot(val_epochs[:len(whole_target_met)], whole_target_met, 
                      label='Whole DSC Target Met', color='green', alpha=0.7)
        axes[1,2].plot(val_epochs[:len(cls_target_met)], cls_target_met, 
                      label='Classification Target Met', color='purple', alpha=0.7)
        axes[1,2].plot(val_epochs[:len(both_met)], both_met, 
                      label='Both Targets Met', color='black', linewidth=3)
        axes[1,2].set_title('Master\'s Target Achievement')
        axes[1,2].set_xlabel('Epoch')
        axes[1,2].set_ylabel('Target Met (1=Yes, 0=No)')
        axes[1,2].legend()
        axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = save_path / "masters_advanced_training.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Advanced training plots saved to: {plot_path}")

def main():
    """Main function for Master's degree improvement"""
    
    print("🎓 MASTER'S DEGREE PERFORMANCE IMPROVEMENT")
    print("=" * 55)
    print("This script implements advanced techniques to achieve:")
    print("• Whole Pancreas DSC ≥ 0.91 (Currently: 0.767)")
    print("• Classification F1 ≥ 0.70 (Currently: 0.599)")
    print()
    
    success = advanced_training(
        data_root=".",
        max_epochs=200,
        target_whole_dsc=0.91,
        target_cls_f1=0.70
    )
    
    if success:
        print("\n🎉 CONGRATULATIONS! MASTER'S TARGETS ACHIEVED!")
        print("✅ Whole Pancreas DSC ≥ 0.91")
        print("✅ Classification F1 ≥ 0.70")
        print("\nNext step: Run inference with speed optimization")
    else:
        print("\n⚠️ Targets not fully achieved. Consider:")
        print("1. Running for more epochs (300+)")
        print("2. Further hyperparameter tuning")
        print("3. Ensemble methods")
        print("4. Advanced architectures (Transformer components)")

if __name__ == "__main__":
    main()