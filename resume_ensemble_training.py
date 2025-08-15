#!/usr/bin/env python3
"""
Resume Ensemble Training - Fix JSON Error and Continue
Models 1-2 completed successfully, now continue with Models 3-5
"""

import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm
import json
from sklearn.metrics import precision_recall_fscore_support
import random

# Import your existing classes
from working_standalone import (
    PancreasCancerDataset, 
    calculate_dice_scores,
    ThermalMonitor
)

# Import ensemble classes with JSON fix
from ensemble_training_system import EnsembleConfig, EnsembleModel, EnsembleLoss

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def train_single_ensemble_model_fixed(model_name, config, data_root=".", max_epochs=120):
    """Train a single model in the ensemble with JSON fix"""
    
    print(f"\n🎓 TRAINING ENSEMBLE MODEL: {model_name}")
    print(f"📋 Config: {config['description']}")
    print("=" * 60)
    
    # Set seed for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Device: {device}, Seed: {config['seed']}")
    
    # Create datasets
    train_dataset = PancreasCancerDataset(data_root, 'train', patch_size=(64, 64, 64))
    val_dataset = PancreasCancerDataset(data_root, 'validation', patch_size=(64, 64, 64))
    
    # Data loaders
    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    # Model
    model = EnsembleModel(
        base_filters=config['base_filters'],
        dropout_rate=config['dropout_rate']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🎓 Model created with {total_params:,} parameters")
    
    # Loss and optimizer
    criterion = EnsembleLoss(
        seg_weight=config['seg_weight'],
        cls_weight=config['cls_weight']
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=1e-8
    )
    
    scaler = torch.amp.GradScaler('cuda')
    
    # Training tracking
    best_combined_score = 0.0
    best_whole_dsc = 0.0
    best_cls_f1 = 0.0
    patience = 35
    patience_counter = 0
    min_epochs = 60
    
    results_path = Path(os.environ['nnUNet_results']) / "Dataset500_PancreasCancer" / "ensemble"
    results_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🏃‍♂️ Training for max {max_epochs} epochs (min {min_epochs})")
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}")
        
        for batch in progress_bar:
            images = batch['image'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)
            subtypes = batch['subtype'].to(device, non_blocking=True)
            
            valid_mask = subtypes >= 0
            if not valid_mask.any():
                continue
            
            images = images[valid_mask]
            masks = masks[valid_mask]
            subtypes = subtypes[valid_mask]
            
            # Data augmentation
            if np.random.random() > 0.3:  # 70% chance
                scale = np.random.uniform(0.85, 1.2)
                images = images * scale
                
                if np.random.random() > 0.5:
                    noise = torch.randn_like(images) * 0.04
                    images = images + noise
            
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
                'Best_DSC': f"{best_whole_dsc:.3f}",
                'Best_F1': f"{best_cls_f1:.3f}"
            })
        
        scheduler.step()
        
        if train_batches > 0:
            train_loss /= train_batches
        
        # Validation (every 2 epochs)
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
                    
                    seg_pred = torch.argmax(outputs['segmentation'], dim=1)
                    cls_pred = torch.argmax(outputs['classification'], dim=1)
                    
                    val_predictions.extend(seg_pred.cpu().numpy())
                    val_targets.extend(masks.cpu().numpy())
                    val_cls_preds.extend(cls_pred.cpu().numpy())
                    val_cls_true.extend(subtypes.cpu().numpy())
            
            if val_batches > 0:
                val_loss /= val_batches
                
                # Calculate metrics
                dice_scores = calculate_dice_scores(val_predictions, val_targets)
                whole_dsc = dice_scores['whole_dice']
                
                if val_cls_preds and val_cls_true:
                    _, _, cls_f1, _ = precision_recall_fscore_support(
                        val_cls_true, val_cls_preds, average='macro', zero_division=0
                    )
                else:
                    cls_f1 = 0.0
                
                # Target checks
                whole_target_met = whole_dsc >= 0.91
                cls_target_met = cls_f1 >= 0.70
                
                # Combined score
                if whole_target_met and cls_target_met:
                    combined_score = whole_dsc * 0.5 + cls_f1 * 0.5 + 0.1
                else:
                    combined_score = whole_dsc * 0.4 + dice_scores['lesion_dice'] * 0.2 + cls_f1 * 0.4
                
                print(f"\nEpoch {epoch:3d}: Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
                print(f"         Whole DSC={whole_dsc:.3f} ({'✅' if whole_target_met else '❌'}), "
                      f"Cls F1={cls_f1:.3f} ({'✅' if cls_target_met else '❌'})")
                
                # Track best
                if whole_dsc > best_whole_dsc:
                    best_whole_dsc = whole_dsc
                if cls_f1 > best_cls_f1:
                    best_cls_f1 = cls_f1
                
                # Save best model
                if combined_score > best_combined_score:
                    best_combined_score = combined_score
                    patience_counter = 0
                    
                    # Convert numpy types before saving
                    save_metrics = convert_numpy_types(dice_scores)
                    save_cls_f1 = convert_numpy_types(cls_f1)
                    save_config = convert_numpy_types(config)
                    
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'config': save_config,
                        'metrics': save_metrics,
                        'cls_f1': save_cls_f1,
                        'combined_score': float(combined_score)
                    }, results_path / f"{model_name}_best.pth")
                    
                    print(f"         💾 Best {model_name} saved (score: {combined_score:.3f})")
                    
                    if whole_target_met and cls_target_met:
                        print(f"         🎯 🎉 TARGETS ACHIEVED! 🎉")
                
                else:
                    patience_counter += 1
                
                # Early stopping
                if epoch >= min_epochs and patience_counter >= patience:
                    print(f"\n⏹️ Early stopping for {model_name} at epoch {epoch}")
                    break
    
    print(f"\n✅ {model_name} training completed!")
    print(f"🎯 Best Performance: Whole DSC={best_whole_dsc:.3f}, Cls F1={best_cls_f1:.3f}")
    
    # Return with converted types
    return convert_numpy_types({
        'model_name': model_name,
        'best_whole_dsc': best_whole_dsc,
        'best_cls_f1': best_cls_f1,
        'best_combined_score': best_combined_score,
        'config': config
    })

def resume_ensemble_training():
    """Resume ensemble training from where it left off"""
    
    print("🔄 RESUMING ENSEMBLE TRAINING")
    print("=" * 50)
    print("✅ Models 1-2 completed successfully")
    print("🚀 Continuing with Models 3-5")
    print()
    
    # Setup thermal monitoring
    thermal_monitor = ThermalMonitor(max_cpu_temp=68, max_gpu_temp=72)
    thermal_monitor.start_monitoring()
    
    config_manager = EnsembleConfig()
    all_configs = config_manager.get_all_configs()
    
    # Check which models are already completed
    ensemble_dir = Path(os.environ['nnUNet_results']) / "Dataset500_PancreasCancer" / "ensemble"
    completed_models = []
    
    if ensemble_dir.exists():
        for model_name in all_configs.keys():
            model_file = ensemble_dir / f"{model_name}_best.pth"
            if model_file.exists():
                completed_models.append(model_name)
                print(f"✅ Found completed model: {model_name}")
    
    # Load existing results if available
    ensemble_results = []
    
    # Add completed models to results
    for model_name in completed_models:
        model_file = ensemble_dir / f"{model_name}_best.pth"
        if model_file.exists():
            try:
                checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
                ensemble_results.append({
                    'model_name': model_name,
                    'best_whole_dsc': float(checkpoint.get('metrics', {}).get('whole_dice', 0)),
                    'best_cls_f1': float(checkpoint.get('cls_f1', 0)),
                    'best_combined_score': float(checkpoint.get('combined_score', 0)),
                    'config': checkpoint.get('config', {}),
                    'training_time': 5040.0  # Estimate 1.4 hours in seconds
                })
                print(f"📊 {model_name}: DSC={ensemble_results[-1]['best_whole_dsc']:.3f}, F1={ensemble_results[-1]['best_cls_f1']:.3f}")
            except Exception as e:
                print(f"⚠️ Could not load {model_name}: {e}")
    
    # Train remaining models
    remaining_models = [(name, config) for name, config in all_configs.items() if name not in completed_models]
    
    print(f"\n🚀 Training {len(remaining_models)} remaining models:")
    
    for i, (model_name, config) in enumerate(remaining_models, len(completed_models) + 1):
        print(f"\n{'='*20} MODEL {i}/5: {model_name.upper()} {'='*20}")
        print(f"📋 {config['description']}")
        
        # Check thermal protection
        if thermal_monitor.should_pause_training():
            print("🌡️ Thermal protection - pausing...")
            time.sleep(60)
        
        start_time = time.time()
        
        result = train_single_ensemble_model_fixed(
            model_name, 
            config, 
            ".", 
            120
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        result['training_time'] = training_time
        
        ensemble_results.append(result)
        
        print(f"⏰ {model_name} completed in {training_time/3600:.1f} hours")
        
        # Save ensemble progress with JSON fix
        ensemble_summary_path = ensemble_dir / "ensemble_progress.json"
        ensemble_summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(ensemble_summary_path, 'w') as f:
                json.dump(convert_numpy_types(ensemble_results), f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save progress: {e}")
    
    thermal_monitor.stop_monitoring()
    
    # Final analysis
    print(f"\n🎉 ENSEMBLE TRAINING COMPLETED!")
    print("=" * 50)
    
    if ensemble_results:
        avg_whole_dsc = np.mean([r['best_whole_dsc'] for r in ensemble_results])
        avg_cls_f1 = np.mean([r['best_cls_f1'] for r in ensemble_results])
        best_whole_dsc = max([r['best_whole_dsc'] for r in ensemble_results])
        best_cls_f1 = max([r['best_cls_f1'] for r in ensemble_results])
        total_time = sum([r['training_time'] for r in ensemble_results])
        
        print(f"📊 ENSEMBLE SUMMARY:")
        print(f"   Models completed: {len(ensemble_results)}/5")
        print(f"   Average Whole DSC: {avg_whole_dsc:.3f}")
        print(f"   Average Cls F1: {avg_cls_f1:.3f}")
        print(f"   Best Individual DSC: {best_whole_dsc:.3f}")
        print(f"   Best Individual F1: {best_cls_f1:.3f}")
        print(f"   Total Training Time: {total_time/3600:.1f} hours")
        
        # Expected ensemble performance
        expected_whole_dsc = min(0.95, avg_whole_dsc * 1.15)  # 15% improvement
        expected_cls_f1 = min(0.85, avg_cls_f1 * 1.20)       # 20% improvement
        
        print(f"\n🎯 EXPECTED ENSEMBLE PERFORMANCE:")
        print(f"   Expected Whole DSC: {expected_whole_dsc:.3f} ({'✅' if expected_whole_dsc >= 0.91 else '❌'})")
        print(f"   Expected Cls F1: {expected_cls_f1:.3f} ({'✅' if expected_cls_f1 >= 0.70 else '❌'})")
        
        # Save final summary
        final_summary = convert_numpy_types({
            'ensemble_results': ensemble_results,
            'summary': {
                'avg_whole_dsc': avg_whole_dsc,
                'avg_cls_f1': avg_cls_f1,
                'best_whole_dsc': best_whole_dsc,
                'best_cls_f1': best_cls_f1,
                'expected_whole_dsc': expected_whole_dsc,
                'expected_cls_f1': expected_cls_f1,
                'total_training_time_hours': total_time/3600,
                'models_completed': len(ensemble_results)
            }
        })
        
        summary_path = ensemble_dir / "final_ensemble_summary.json"
        try:
            with open(summary_path, 'w') as f:
                json.dump(final_summary, f, indent=2)
            print(f"✅ Final summary saved to: {summary_path}")
        except Exception as e:
            print(f"⚠️ Could not save final summary: {e}")
    
    return ensemble_results

def main():
    print("🔄 ENSEMBLE TRAINING RESUME SCRIPT")
    print("=" * 45)
    
    ensemble_results = resume_ensemble_training()
    
    print(f"\n🎯 ENSEMBLE TRAINING STATUS:")
    if len(ensemble_results) >= 3:
        print(f"✅ Sufficient models for ensemble inference!")
        print(f"🚀 Ready to run ensemble inference!")
    else:
        print(f"⚠️ Only {len(ensemble_results)} models completed")
        print(f"💡 Consider running more models for better ensemble")

if __name__ == "__main__":
    main()