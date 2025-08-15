#!/usr/bin/env python3
"""
Check which models were actually used in validation
"""

import json
from pathlib import Path

def check_model_usage():
    """Check which models were used in validation"""
    
    print("🔍 CHECKING MODEL USAGE IN VALIDATION")
    print("=" * 50)
    
    # Check validation results
    results_file = Path("validation_results/validation_ensemble_results.json")
    
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
        
        model_count = results['ensemble_info']['model_count']
        print(f"📊 Models used in validation: {model_count}")
        
        if model_count < 5:
            print(f"⚠️ Only {model_count}/5 models were used!")
            print("💡 You could get better results with all 5 models")
        else:
            print(f"✅ All {model_count} models were used")
    
    # Check available models
    ensemble_dir = Path("nnUNet_results/Dataset500_PancreasCancer/ensemble")
    if ensemble_dir.exists():
        model_files = list(ensemble_dir.glob("*_best.pth"))
        print(f"\n📁 Available model files: {len(model_files)}")
        
        for i, model_file in enumerate(model_files, 1):
            print(f"  {i}. {model_file.name}")
    
    print(f"\n🚀 RECOMMENDATION:")
    print(f"Run validation again to potentially use all 5 models:")
    print(f"python validation_ensemble_runner.py")

if __name__ == "__main__":
    check_model_usage()