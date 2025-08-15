#!/usr/bin/env python3
"""
Quick Inference Runner
Runs inference on test data and generates submission files
"""

import os
import subprocess
import sys
from pathlib import Path

def check_requirements():
    """Check if everything is ready for inference"""
    
    print("🔍 Checking inference requirements...")
    
    issues = []
    
    # Check for trained model
    model_path = Path("nnUNet_results/Dataset500_PancreasCancer/best_model.pth")
    if not model_path.exists():
        issues.append("❌ No trained model found")
    else:
        print("✅ Trained model found")
    
    # Check for test data
    test_dir = Path("test")
    if not test_dir.exists():
        issues.append("❌ No test directory found")
    else:
        test_files = list(test_dir.glob("*_0000.nii.gz"))
        if not test_files:
            issues.append("❌ No test files found in test directory")
        else:
            print(f"✅ Found {len(test_files)} test files")
    
    # Check for main script
    if not Path("working_standalone.py").exists():
        issues.append("❌ working_standalone.py not found")
    else:
        print("✅ Main script found")
    
    if issues:
        print("\n⚠️ Issues found:")
        for issue in issues:
            print(f"  {issue}")
        return False
    
    return True

def run_inference():
    """Run the inference"""
    
    print("\n🚀 Running inference...")
    
    try:
        # Run inference
        cmd = [sys.executable, "working_standalone.py", "--mode", "inference"]
        
        print(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("✅ Inference completed successfully!")
            print("Output:", result.stdout[-500:])  # Last 500 chars
        else:
            print("❌ Inference failed!")
            print("Error:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Inference timed out (>10 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error running inference: {e}")
        return False
    
    return True

def check_results():
    """Check what results were generated"""
    
    print("\n📋 Checking generated results...")
    
    results_dir = Path("nnUNet_results/Dataset500_PancreasCancer")
    
    # Check for segmentation results
    seg_results = list(results_dir.glob("quiz_*.nii.gz"))
    if seg_results:
        print(f"✅ Found {len(seg_results)} segmentation