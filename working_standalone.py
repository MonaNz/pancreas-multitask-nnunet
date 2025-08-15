#!/usr/bin/env python3
"""
Working Pancreas Cancer Multi-task Project
Master/PhD Implementation - Complete and Working
"""

import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import psutil

# Setup environment
def setup_environment():
    base_path = Path.cwd()
    for var in ['nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results']:
        os.environ[var] = str(base_path / var)
        Path(base_path / var).mkdir(exist_ok=True, parents=True)

setup_environment()

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

class ThermalMonitor:
    """Thermal protection for RTX 2000 Ada"""
    
    def __init__(self, max_cpu_temp: float = 70.0, max_gpu_temp: float = 75.0):
        self.max_cpu_temp = max_cpu_temp
        self.max_gpu_temp = max_gpu_temp
        self.monitoring = False
        
    def start_monitoring(self):
        self.monitoring = True
        print(f"🛡️ Thermal protection: CPU<{self.max_cpu_temp}°C, GPU<{self.max_gpu_temp}°C")
        
    def stop_monitoring(self):
        self.monitoring = False
        
    def should_pause_training(self) -> bool:
        if not self.monitoring:
            return False
            
        temps = {}
        
        try:
            # Check CPU temperature
            cpu_temp = psutil.sensors_temperatures()
            if cpu_temp:
                for name, entries in cpu_temp.items():
                    for entry in entries:
                        if entry.current:
                            temps['cpu'] = entry.current
                            break
        except:
            temps['cpu'] = 0.0
            
        # Check GPU temperature
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    temps['gpu'] = gpus[0].temperature
            except:
                temps['gpu'] = 0.0
        else:
            temps['gpu'] = 0.0
        
        cpu_hot = temps.get('cpu', 0) > self.max_cpu_temp
        gpu_hot = temps.get('gpu', 0) > self.max_gpu_temp
        
        if cpu_hot or gpu_hot:
            print(f"🌡️ High temp - CPU: {temps.get('cpu', 0):.1f}°C, GPU: {temps.get('gpu', 0):.1f}°C")
            return True
            
        return False

class PancreasCancerDataset(Dataset):
    """Dataset for pancreas cancer data"""
    
    def __init__(self, data_root: str, split: str = 'train', patch_size: Tuple[int, int, int] = (64, 64, 64)):
        self.data_root = Path(data_root)
        self.split = split
        self.patch_size = patch_size
        self.samples = []
        
        if split in ['train', 'validation']:
            self._load_labeled_data()
        else:
            self._load_test_data()
    
    def _load_labeled_data(self):
        """Load training/validation data with labels"""
        split_dir = self.data_root / self.split
        
        for subtype_id, subtype_dir in enumerate(['subtype0', 'subtype1', 'subtype2']):
            subtype_path = split_dir / subtype_dir
            if not subtype_path.exists():
                continue
                
            # Find image files
            image_files = list(subtype_path.glob("*_0000.nii.gz"))
            
            for image_file in image_files:
                mask_file = subtype_path / image_file.name.replace("_0000", "")
                
                if mask_file.exists():
                    self.samples.append({
                        'image': image_file,
                        'mask': mask_file,
                        'subtype': subtype_id,
                        'case_id': image_file.stem.replace("_0000", "")
                    })
        
        print(f"📊 {self.split}: {len(self.samples)} samples")
    
    def _load_test_data(self):
        """Load test data"""
        test_dir = self.data_root / "test"
        if test_dir.exists():
            image_files = list(test_dir.glob("*_0000.nii.gz"))
            
            for image_file in image_files:
                self.samples.append({
                    'image': image_file,
                    'mask': None,
                    'subtype': -1,
                    'case_id': image_file.stem.replace("_0000", "")
                })
        
        print(f"📊 test: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def _normalize_ct(self, image: np.ndarray) -> np.ndarray:
        """Normalize CT image"""
        image = np.clip(image, -1000, 1000)
        image = (image + 1000) / 2000.0
        return image.astype(np.float32)
    
    def _extract_patch(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Extract random patch from image"""
        d, h, w = image.shape
        pd, ph, pw = self.patch_size
        
        # Random crop coordinates
        start_d = np.random.randint(0, max(1, d - pd + 1))
        start_h = np.random.randint(0, max(1, h - ph + 1))
        start_w = np.random.randint(0, max(1, w - pw + 1))
        
        # Extract patch
        image_patch = image[start_d:start_d+pd, start_h:start_h+ph, start_w:start_w+pw]
        
        # Pad if necessary
        if image_patch.shape != self.patch_size:
            padded = np.zeros(self.patch_size, dtype=image_patch.dtype)
            padded[:image_patch.shape[0], :image_patch.shape[1], :image_patch.shape[2]] = image_patch
            image_patch = padded
        
        mask_patch = None
        if mask is not None:
            mask_patch = mask[start_d:start_d+pd, start_h:start_h+ph, start_w:start_w+pw]
            if mask_patch.shape != self.patch_size:
                padded_mask = np.zeros(self.patch_size, dtype=mask_patch.dtype)
                padded_mask[:mask_patch.shape[0], :mask_patch.shape[1], :mask_patch.shape[2]] = mask_patch
                mask_patch = padded_mask
        
        return image_patch, mask_patch
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        img = nib.load(sample['image'])
        image = img.get_fdata()
        image = self._normalize_ct(image)
        
        # Load mask if available
        mask = None
        if sample['mask'] is not None:
            mask_img = nib.load(sample['mask'])
            mask = mask_img.get_fdata().astype(np.uint8)
        
        # Extract patch
        image_patch, mask_patch = self._extract_patch(image, mask)
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image_patch).unsqueeze(0)  # Add channel dim
        
        if mask_patch is not None:
            mask_tensor = torch.from_numpy(mask_patch).long()
        else:
            mask_tensor = torch.zeros(self.patch_size, dtype=torch.long)
        
        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'subtype': torch.tensor(sample['subtype'], dtype=torch.long),
            'case_id': sample['case_id']
        }

class MultiTaskUNet3D(nn.Module):
    """3D UNet with classification head"""
    
    def __init__(self, in_channels=1, num_classes_seg=3, num_classes_cls=3, base_filters=32):
        super().__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, base_filters)
        self.enc2 = self._conv_block(base_filters, base_filters * 2)
        self.enc3 = self._conv_block(base_filters * 2, base_filters * 4)
        self.enc4 = self._conv_block(base_filters * 4, base_filters * 8)
        
        self.pool = nn.MaxPool3d(2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(base_filters * 8, base_filters * 16)
        
        # Decoder
        self.up4 = nn.ConvTranspose3d(base_filters * 16, base_filters * 8, 2, stride=2)
        self.dec4 = self._conv_block(base_filters * 16, base_filters * 8)
        
        self.up3 = nn.ConvTranspose3d(base_filters * 8, base_filters * 4, 2, stride=2)
        self.dec3 = self._conv_block(base_filters * 8, base_filters * 4)
        
        self.up2 = nn.ConvTranspose3d(base_filters * 4, base_filters * 2, 2, stride=2)
        self.dec2 = self._conv_block(base_filters * 4, base_filters * 2)
        
        self.up1 = nn.ConvTranspose3d(base_filters * 2, base_filters, 2, stride=2)
        self.dec1 = self._conv_block(base_filters * 2, base_filters)
        
        # Outputs
        self.seg_out = nn.Conv3d(base_filters, num_classes_seg, 1)
        
        # Classification head
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(base_filters * 16, 512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes_cls)
        )
    
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Classification branch
        cls_features = self.global_pool(b)
        cls_features = cls_features.view(cls_features.size(0), -1)
        cls_output = self.classifier(cls_features)
        
        # Decoder
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        seg_output = self.seg_out(d1)
        
        return {
            'segmentation': seg_output,
            'classification': cls_output
        }

class MultiTaskLoss(nn.Module):
    """Combined loss for segmentation and classification"""
    
    def __init__(self, seg_weight: float = 1.0, cls_weight: float = 0.3):
        super().__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.ce_loss = nn.CrossEntropyLoss()
        
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Dice loss"""
        smooth = 1e-5
        num_classes = pred.shape[1]
        
        target_one_hot = F.one_hot(target, num_classes).permute(0, 4, 1, 2, 3).float()
        pred_soft = F.softmax(pred, dim=1)
        
        dice_scores = []
        for c in range(num_classes):
            pred_c = pred_soft[:, c]
            target_c = target_one_hot[:, c]
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            dice = (2 * intersection + smooth) / (union + smooth)
            dice_scores.append(dice)
        
        return 1 - torch.stack(dice_scores).mean()
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                seg_target: torch.Tensor, cls_target: torch.Tensor):
        
        seg_pred = predictions['segmentation']
        cls_pred = predictions['classification']
        
        # Segmentation loss
        ce_loss = self.ce_loss(seg_pred, seg_target)
        dice_loss = self.dice_loss(seg_pred, seg_target)
        seg_loss = 0.5 * ce_loss + 0.5 * dice_loss
        
        # Classification loss
        cls_loss = self.ce_loss(cls_pred, cls_target)
        
        # Total loss
        total_loss = self.seg_weight * seg_loss + self.cls_weight * cls_loss
        
        return {
            'total_loss': total_loss,
            'seg_loss': seg_loss,
            'cls_loss': cls_loss
        }

def calculate_dice_scores(predictions: List[np.ndarray], targets: List[np.ndarray]) -> Dict[str, float]:
    """Calculate dice scores"""
    
    def dice_coefficient(pred: np.ndarray, target: np.ndarray, label: int = 1) -> float:
        pred_mask = (pred == label).astype(np.float32)
        target_mask = (target == label).astype(np.float32)
        
        if target_mask.sum() == 0 and pred_mask.sum() == 0:
            return 1.0
        elif target_mask.sum() == 0:
            return 0.0
        
        intersection = (pred_mask * target_mask).sum()
        return 2.0 * intersection / (pred_mask.sum() + target_mask.sum())
    
    pancreas_dices = []
    lesion_dices = []
    whole_dices = []
    
    for pred, target in zip(predictions, targets):
        # Pancreas (label 1)
        pancreas_dice = dice_coefficient(pred, target, 1)
        pancreas_dices.append(pancreas_dice)
        
        # Lesion (label 2)
        lesion_dice = dice_coefficient(pred, target, 2)
        lesion_dices.append(lesion_dice)
        
        # Whole pancreas (label 1 + 2)
        pred_whole = (pred > 0).astype(np.uint8)
        target_whole = (target > 0).astype(np.uint8)
        whole_dice = dice_coefficient(pred_whole, target_whole, 1)
        whole_dices.append(whole_dice)
    
    return {
        'pancreas_dice': np.mean(pancreas_dices),
        'lesion_dice': np.mean(lesion_dices),
        'whole_dice': np.mean(whole_dices)
    }

def train_model(data_root: str = ".", max_epochs: int = 100, thermal_protection: bool = True):
    """Train the multi-task model"""
    
    print("🎓 MASTER/PhD ENHANCED TRAINING")
    print("=" * 45)
    print("🎯 Performance Targets:")
    print("   • Whole Pancreas DSC ≥ 0.91")
    print("   • Lesion DSC ≥ 0.31")
    print("   • Classification F1 ≥ 0.7")
    print()
    
    # Setup thermal monitoring
    thermal_monitor = None
    if thermal_protection:
        thermal_monitor = ThermalMonitor(max_cpu_temp=68, max_gpu_temp=72)
        thermal_monitor.start_monitoring()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Using device: {device}")
    
    # Create datasets
    train_dataset = PancreasCancerDataset(data_root, 'train', patch_size=(64, 64, 64))
    val_dataset = PancreasCancerDataset(data_root, 'validation', patch_size=(64, 64, 64))
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    # Model
    model = MultiTaskUNet3D().to(device)
    print(f"🎓 Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Loss and optimizer
    criterion = MultiTaskLoss(seg_weight=1.0, cls_weight=0.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    scaler = torch.cuda.amp.GradScaler()
    
    # Training tracking
    metrics_history = {
        'train_loss': [],
        'val_loss': [],
        'val_dice_whole': [],
        'val_dice_lesion': [],
        'val_cls_f1': []
    }
    
    best_score = 0.0
    patience = 15
    patience_counter = 0
    
    results_path = Path(os.environ['nnUNet_results']) / "Dataset500_PancreasCancer"
    results_path.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(max_epochs):
        # Check thermal protection
        if thermal_monitor and thermal_monitor.should_pause_training():
            print("🌡️ Thermal protection - pausing...")
            time.sleep(30)
            continue
        
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}")
        
        for batch in progress_bar:
            images = batch['image'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)
            subtypes = batch['subtype'].to(device, non_blocking=True)
            
            # Skip invalid subtypes
            valid_mask = subtypes >= 0
            if not valid_mask.any():
                continue
            
            images = images[valid_mask]
            masks = masks[valid_mask]
            subtypes = subtypes[valid_mask]
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
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
                'Loss': f"{loss_dict['total_loss'].item():.4f}"
            })
        
        if train_batches > 0:
            train_loss /= train_batches
            metrics_history['train_loss'].append(train_loss)
        
        # Validation
        if epoch % 3 == 0:
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
                    
                    with torch.cuda.amp.autocast():
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
                metrics_history['val_loss'].append(val_loss)
                
                # Calculate metrics
                dice_scores = calculate_dice_scores(val_predictions, val_targets)
                metrics_history['val_dice_whole'].append(dice_scores['whole_dice'])
                metrics_history['val_dice_lesion'].append(dice_scores['lesion_dice'])
                
                # Classification metrics
                if val_cls_preds and val_cls_true:
                    _, _, cls_f1, _ = precision_recall_fscore_support(
                        val_cls_true, val_cls_preds, average='macro', zero_division=0
                    )
                    metrics_history['val_cls_f1'].append(cls_f1)
                else:
                    cls_f1 = 0.0
                    metrics_history['val_cls_f1'].append(0.0)
                
                # Check targets
                targets_met = {
                    'whole_dsc': dice_scores['whole_dice'] >= 0.91,
                    'lesion_dsc': dice_scores['lesion_dice'] >= 0.31,
                    'cls_f1': cls_f1 >= 0.7
                }
                
                combined_score = (
                    dice_scores['whole_dice'] * 0.4 +
                    dice_scores['lesion_dice'] * 0.3 +
                    cls_f1 * 0.3
                )
                
                print(f"Epoch {epoch:3d}: "
                      f"Loss={train_loss:.4f}, "
                      f"Val Loss={val_loss:.4f}, "
                      f"Whole DSC={dice_scores['whole_dice']:.3f}, "
                      f"Lesion DSC={dice_scores['lesion_dice']:.3f}, "
                      f"Cls F1={cls_f1:.3f}")
                
                if all(targets_met.values()):
                    print("     🎯 ALL MASTER/PhD TARGETS ACHIEVED! ✅")
                else:
                    missing = [k for k, v in targets_met.items() if not v]
                    print(f"     ⚠️ Missing: {', '.join(missing)}")
                
                # Save best model
                if combined_score > best_score:
                    best_score = combined_score
                    patience_counter = 0
                    
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'metrics': dice_scores,
                        'cls_f1': cls_f1
                    }, results_path / "best_model.pth")
                    
                    print(f"     💾 Best model saved (score: {combined_score:.3f})")
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"⏹️ Early stopping at epoch {epoch}")
                    break
        
        scheduler.step()
    
    if thermal_monitor:
        thermal_monitor.stop_monitoring()
    
    print("✅ Training completed!")

def run_inference(data_root: str = "."):
    """Run inference on test set"""
    
    print("🔍 Running inference on test set...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_path = Path(os.environ['nnUNet_results']) / "Dataset500_PancreasCancer"
    model_path = results_path / "best_model.pth"
    
    if not model_path.exists():
        print("❌ No trained model found!")
        return
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = MultiTaskUNet3D().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test data
    test_dataset = PancreasCancerDataset(data_root, 'test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    results = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Processing test cases"):
            images = batch['image'].to(device)
            case_id = batch['case_id'][0]
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
            
            cls_pred = torch.argmax(outputs['classification'], dim=1).cpu().item()
            
            results.append({
                'case_id': case_id,
                'classification': cls_pred
            })
    
    # Save results
    csv_data = []
    for result in results:
        csv_data.append({
            'Names': result['case_id'] + '.nii.gz',
            'Subtype': result['classification']
        })
    
    df = pd.DataFrame(csv_data)
    df.to_csv(results_path / "subtype_results.csv", index=False)
    
    print(f"✅ Inference completed! Results saved to: {results_path}")

def main():
    parser = argparse.ArgumentParser(description='Pancreas Cancer Multi-task Learning')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'inference', 'all'],
                       help='Mode to run')
    parser.add_argument('--data_root', type=str, default='.',
                       help='Root directory containing data')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--thermal_protection', action='store_true',
                       help='Enable thermal protection')
    
    args = parser.parse_args()
    
    print("🎓 PANCREAS CANCER PROJECT")
    print("Master/PhD Implementation")
    print("=" * 30)
    
    if torch.cuda.is_available():
        print(f"🔥 Using GPU: {torch.cuda.get_device_name()}")
    else:
        print("⚠️ Using CPU")
    
    if args.mode == 'train' or args.mode == 'all':
        print("\n🚀 TRAINING PHASE")
        train_model(
            data_root=args.data_root,
            max_epochs=args.epochs,
            thermal_protection=args.thermal_protection
        )
    
    if args.mode == 'inference' or args.mode == 'all':
        print("\n🔍 INFERENCE PHASE")
        run_inference(data_root=args.data_root)
    
    print("\n🎉 All phases completed!")

if __name__ == "__main__":
    main()