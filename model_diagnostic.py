#!/usr/bin/env python3
"""
Diagnostic script to check which models can be loaded
"""

import torch
from pathlib import Path

def diagnose_models():
    """Check which models can be loaded for ensemble"""
    
    print("🔍 MODEL LOADING DIAGNOSTIC")
    print("=" * 50)
    
    # Check available models
    ensemble_dir = Path("nnUNet_results/Dataset500_PancreasCancer/ensemble")
    
    if not ensemble_dir.exists():
        print("❌ Ensemble directory not found!")
        return
    
    model_files = list(ensemble_dir.glob("*_best.pth"))
    print(f"📁 Found {len(model_files)} model files:")
    
    loadable_models = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for i, model_file in enumerate(model_files, 1):
        print(f"\n{i}. Testing {model_file.name}:")
        
        try:
            # Try to load checkpoint
            checkpoint = torch.load(model_file, map_location=device)
            config = checkpoint.get('config', {})
            
            # Check if we can create the model
            from ensemble_training_system import EnsembleModel
            
            model = EnsembleModel(
                base_filters=config.get('base_filters', 64),
                dropout_rate=config.get('dropout_rate', 0.5)
            )
            
            # Try to load weights
            model.load_state_dict(checkpoint['model_state_dict'])
            
            print(f"   ✅ Successfully loaded")
            print(f"   📊 Config: {config.get('base_filters', 64)} filters, {config.get('dropout_rate', 0.5)} dropout")
            
            if 'metrics' in checkpoint:
                metrics = checkpoint['metrics']
                whole_dsc = metrics.get('whole_dice', 'N/A')
                print(f"   🎯 Performance: DSC={whole_dsc}")
            
            loadable_models.append(model_file.name)
            
        except Exception as e:
            print(f"   ❌ Failed to load: {e}")
    
    print(f"\n📊 SUMMARY:")
    print(f"   Total files found: {len(model_files)}")
    print(f"   Successfully loadable: {len(loadable_models)}")
    
    if len(loadable_models) > 0:
        print(f"\n✅ READY FOR ENSEMBLE:")
        for model in loadable_models:
            print(f"   • {model}")
        
        print(f"\n🚀 RECOMMENDATION:")
        print(f"   Run: python validation_ensemble_runner.py")
        print(f"   Expected models in ensemble: {len(loadable_models)}")
    else:
        print(f"\n❌ NO MODELS CAN BE LOADED!")
        print(f"   Check your model files and configurations")

if __name__ == "__main__":
    diagnose_models()