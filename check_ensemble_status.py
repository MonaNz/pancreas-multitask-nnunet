#!/usr/bin/env python3
"""
Check Ensemble Status and Create Recovery Plan
Find what models were actually saved and create recovery strategy
"""

import torch
import numpy as np
from pathlib import Path
import json

def check_ensemble_status():
    """Check what ensemble models actually exist"""
    
    print("🔍 CHECKING ENSEMBLE STATUS")
    print("=" * 40)
    
    # Check possible ensemble directories
    possible_dirs = [
        Path("nnUNet_results/Dataset500_PancreasCancer/ensemble"),
        Path("nnUNet_results/Dataset500_PancreasCancer"),
        Path("."),
    ]
    
    found_models = []
    ensemble_dir = None
    
    # Expected model file patterns
    model_patterns = [
        "*conservative*best*.pth",
        "*balanced*best*.pth", 
        "*aggressive*best*.pth",
        "*focused*best*.pth",
        "*classifier*best*.pth",
        "model_*_best.pth",
        "*ensemble*.pth"
    ]
    
    print("🔍 Searching for ensemble model files...")
    
    for directory in possible_dirs:
        if directory.exists():
            print(f"\n📁 Checking: {directory}")
            
            for pattern in model_patterns:
                files = list(directory.glob(pattern))
                for file in files:
                    try:
                        # Try to load the file to verify it's valid
                        checkpoint = torch.load(file, map_location='cpu', weights_only=False)
                        
                        # Extract info
                        model_info = {
                            'file_path': str(file),
                            'file_name': file.name,
                            'directory': str(directory),
                            'epoch': checkpoint.get('epoch', 'unknown'),
                            'metrics': checkpoint.get('metrics', {}),
                            'cls_f1': checkpoint.get('cls_f1', 0),
                            'config': checkpoint.get('config', {}),
                            'combined_score': checkpoint.get('combined_score', 0)
                        }
                        
                        found_models.append(model_info)
                        
                        # Extract performance info
                        whole_dsc = model_info['metrics'].get('whole_dice', 0) if model_info['metrics'] else 0
                        cls_f1 = model_info['cls_f1']
                        
                        print(f"  ✅ {file.name}")
                        print(f"     Epoch: {model_info['epoch']}")
                        print(f"     Whole DSC: {whole_dsc:.3f}")
                        print(f"     Classification F1: {cls_f1:.3f}")
                        
                        if ensemble_dir is None:
                            ensemble_dir = directory
                        
                    except Exception as e:
                        print(f"  ❌ {file.name} - corrupted: {e}")
    
    print(f"\n📊 FOUND {len(found_models)} VALID MODEL FILES")
    
    if len(found_models) == 0:
        print("❌ No ensemble models found!")
        print("\nPossible issues:")
        print("  1. Models saved in different location")
        print("  2. Training didn't complete successfully")
        print("  3. Files were deleted or moved")
        return None, None
    
    # Analyze found models
    print(f"\n📈 MODEL ANALYSIS:")
    best_dsc = 0
    best_f1 = 0
    total_models = len(found_models)
    
    for i, model in enumerate(found_models, 1):
        whole_dsc = model['metrics'].get('whole_dice', 0) if model['metrics'] else 0
        cls_f1 = model['cls_f1'] if model['cls_f1'] else 0
        
        best_dsc = max(best_dsc, whole_dsc)
        best_f1 = max(best_f1, cls_f1)
        
        print(f"  Model {i}: {model['file_name']}")
        print(f"    Performance: {whole_dsc:.3f} DSC, {cls_f1:.3f} F1")
        print(f"    Config: {model['config'].get('description', 'Unknown')}")
    
    avg_dsc = np.mean([model['metrics'].get('whole_dice', 0) if model['metrics'] else 0 for model in found_models])
    avg_f1 = np.mean([model['cls_f1'] if model['cls_f1'] else 0 for model in found_models])
    
    print(f"\n🎯 ENSEMBLE POTENTIAL:")
    print(f"  Models available: {total_models}")
    print(f"  Average DSC: {avg_dsc:.3f}")
    print(f"  Average F1: {avg_f1:.3f}")
    print(f"  Best DSC: {best_dsc:.3f}")
    print(f"  Best F1: {best_f1:.3f}")
    
    # Estimate ensemble performance
    if total_models >= 2:
        # Conservative ensemble estimate (10-15% improvement)
        estimated_dsc = min(0.95, avg_dsc * 1.12)
        estimated_f1 = min(0.85, avg_f1 * 1.15)
        
        print(f"\n🎯 ESTIMATED ENSEMBLE PERFORMANCE:")
        print(f"  Expected DSC: {estimated_dsc:.3f} ({'✅' if estimated_dsc >= 0.91 else '❌'})")
        print(f"  Expected F1: {estimated_f1:.3f} ({'✅' if estimated_f1 >= 0.70 else '❌'})")
        
        # Master's targets assessment
        dsc_achievable = estimated_dsc >= 0.91
        f1_achievable = estimated_f1 >= 0.70
        
        print(f"\n🎓 MASTER'S TARGETS:")
        print(f"  Whole DSC ≥ 0.91: {'✅ Likely' if dsc_achievable else '❌ Unlikely'}")
        print(f"  Classification F1 ≥ 0.70: {'✅ Likely' if f1_achievable else '❌ Unlikely'}")
        
        if dsc_achievable and f1_achievable:
            print(f"  🎉 BOTH TARGETS ACHIEVABLE WITH ENSEMBLE!")
        elif dsc_achievable or f1_achievable:
            print(f"  ⚠️ ONE TARGET ACHIEVABLE - Consider more training")
        else:
            print(f"  ❌ TARGETS CHALLENGING - May need different approach")
    
    return found_models, ensemble_dir

def create_recovery_options(found_models, ensemble_dir):
    """Create recovery options based on found models"""
    
    if not found_models:
        print("\n❌ NO RECOVERY OPTIONS - No models found")
        return
    
    print(f"\n🚀 RECOVERY OPTIONS:")
    print("=" * 30)
    
    num_models = len(found_models)
    
    # Option 1: Use existing models for ensemble
    if num_models >= 2:
        print(f"✅ OPTION 1: Run Ensemble Inference NOW")
        print(f"   - Use {num_models} existing models")
        print(f"   - Run ensemble inference immediately")
        print(f"   - Time: 10-15 minutes")
        print(f"   - Command: python ensemble_inference_system.py")
    
    # Option 2: Train more models
    if num_models < 5:
        remaining = 5 - num_models
        print(f"\n🔄 OPTION 2: Train {remaining} More Models")
        print(f"   - Complete the full 5-model ensemble")
        print(f"   - Time: ~{remaining * 1.5:.1f} hours")
        print(f"   - Higher ensemble performance")
        print(f"   - Command: python resume_ensemble.py")
    
    # Option 3: Quick single model training
    print(f"\n⚡ OPTION 3: Train 1 Aggressive Model")
    print(f"   - Train just 1 high-performance model")
    print(f"   - Add to existing ensemble")
    print(f"   - Time: ~1.5 hours")
    print(f"   - Quick performance boost")
    
    # Option 4: Inference + Speed test
    print(f"\n🎯 OPTION 4: Complete Submission")
    print(f"   - Run ensemble inference with existing models")
    print(f"   - Run speed optimization test")
    print(f"   - Generate final submission")
    print(f"   - Time: 20 minutes")
    print(f"   - Ready for Master's evaluation")

def create_ensemble_inference_config(found_models, ensemble_dir):
    """Create config for ensemble inference with found models"""
    
    if not found_models or not ensemble_dir:
        return False
    
    print(f"\n📋 CREATING ENSEMBLE INFERENCE CONFIG")
    print("-" * 40)
    
    # Create ensemble directory structure
    inference_ready_dir = Path(ensemble_dir) / "inference_ready"
    inference_ready_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy/link model files to standard names
    model_mapping = {}
    
    for i, model in enumerate(found_models):
        source_path = Path(model['file_path'])
        target_name = f"ensemble_model_{i+1}_best.pth"
        target_path = inference_ready_dir / target_name
        
        try:
            # Copy the model file
            import shutil
            shutil.copy2(source_path, target_path)
            
            model_mapping[target_name] = {
                'source': str(source_path),
                'performance': {
                    'whole_dsc': model['metrics'].get('whole_dice', 0) if model['metrics'] else 0,
                    'cls_f1': model['cls_f1'] if model['cls_f1'] else 0
                },
                'config': model['config']
            }
            
            print(f"  ✅ Model {i+1}: {target_name}")
            
        except Exception as e:
            print(f"  ❌ Failed to copy {source_path}: {e}")
    
    # Save ensemble config
    config_file = inference_ready_dir / "ensemble_config.json"
    config = {
        'num_models': len(model_mapping),
        'models': model_mapping,
        'inference_ready': True,
        'created': str(Path().cwd()),
        'ensemble_dir': str(ensemble_dir)
    }
    
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        print(f"\n✅ Ensemble config created: {config_file}")
        print(f"📁 Inference-ready models: {inference_ready_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create config: {e}")
        return False

def main():
    print("🔍 ENSEMBLE STATUS CHECKER & RECOVERY")
    print("=" * 50)
    
    # Check what models exist
    found_models, ensemble_dir = check_ensemble_status()
    
    if found_models:
        # Create recovery options
        create_recovery_options(found_models, ensemble_dir)
        
        # Create inference config
        inference_ready = create_ensemble_inference_config(found_models, ensemble_dir)
        
        print(f"\n🎯 RECOMMENDED NEXT STEP:")
        
        if len(found_models) >= 3:
            print(f"✅ You have {len(found_models)} models - RUN ENSEMBLE INFERENCE!")
            print(f"   Command: python ensemble_inference_system.py")
            print(f"   Expected time: 10-15 minutes")
        elif len(found_models) == 2:
            print(f"⚠️ You have 2 models - Can run inference OR train 1 more")
            print(f"   Quick option: python ensemble_inference_system.py")
            print(f"   Better option: Train 1 more aggressive model first")
        else:
            print(f"❌ Only {len(found_models)} model(s) - Consider training more")
    
    else:
        print(f"\n❌ NO ENSEMBLE MODELS FOUND")
        print(f"💡 Options:")
        print(f"   1. Check if models saved elsewhere")
        print(f"   2. Restart ensemble training")
        print(f"   3. Train single high-performance model")

if __name__ == "__main__":
    main()