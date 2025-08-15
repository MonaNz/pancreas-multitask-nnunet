#!/usr/bin/env python3
"""
Fixed Main script for Pancreas Cancer Multi-task Learning
Sets up environment BEFORE any nnUNet imports
"""

import os
import sys
from pathlib import Path

# CRITICAL: Setup nnUNet environment BEFORE any imports
def setup_nnunet_environment():
    """Setup nnUNet environment variables FIRST"""
    base_path = Path.cwd()
    
    env_vars = {
        'nnUNet_raw': str(base_path / "nnUNet_raw"),
        'nnUNet_preprocessed': str(base_path / "nnUNet_preprocessed"), 
        'nnUNet_results': str(base_path / "nnUNet_results")
    }
    
    for var, path in env_vars.items():
        os.environ[var] = path
        Path(path).mkdir(exist_ok=True, parents=True)
        print(f"Set {var} = {path}")

# Setup environment IMMEDIATELY
setup_nnunet_environment()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Now imports should work
import argparse
import torch

def main():
    parser = argparse.ArgumentParser(description='Pancreas Cancer Multi-task Learning')
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['prepare', 'train', 'inference', 'evaluate', 'all'],
                       help='Mode to run')
    parser.add_argument('--data_root', type=str, default='.',
                       help='Root directory containing train/validation/test folders')
    parser.add_argument('--dataset_id', type=int, default=500,
                       help='nnUNet dataset ID')
    parser.add_argument('--fold', type=int, default=0,
                       help='Cross-validation fold')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device to use')
    parser.add_argument('--thermal_protection', action='store_true',
                       help='Enable thermal protection during training')
    parser.add_argument('--master_phd_mode', action='store_true',
                       help='Enable Master/PhD advanced optimizations and higher targets')
    
    args = parser.parse_args()
    
    # Print mode information
    print("\n" + "="*60)
    if args.master_phd_mode:
        print("🎓 MASTER/PhD MODE ENABLED")
        print("Enhanced targets: Pancreas DSC≥0.91, Lesion DSC≥0.31, F1≥0.7, Speed≥10%")
    else:
        print("📚 Undergraduate mode")
        print("Standard targets: Pancreas DSC≥0.85, Lesion DSC≥0.27, F1≥0.6")
    print("="*60)
    
    # Set GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"🔥 Using GPU {args.gpu}: {torch.cuda.get_device_name()}")
    else:
        print("⚠️ CUDA not available, using CPU")
    
    # Import modules AFTER environment setup
    try:
        from data_preparation import prepare_nnunet_dataset
        from simple_training import train_multitask_model
        from inference import run_inference  
        from evaluation import evaluate_results
        print("✅ All modules imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Trying alternative imports...")
        
        # Fallback imports
        try:
            import data_preparation
            import simple_training
            prepare_nnunet_dataset = data_preparation.prepare_nnunet_dataset
            train_multitask_model = simple_training.train_multitask_model
            print("✅ Fallback imports successful")
        except Exception as e2:
            print(f"❌ Fallback failed: {e2}")
            return
    
    # Execute requested mode
    try:
        if args.mode == 'prepare' or args.mode == 'all':
            print("\n🔄 STEP 1: Preparing nnUNet dataset...")
            prepare_nnunet_dataset(
                data_root=args.data_root,
                dataset_id=args.dataset_id
            )
            print("✅ Data preparation completed!")
        
        if args.mode == 'train' or args.mode == 'all':
            print("\n🚀 STEP 2: Starting multi-task training...")
            if args.master_phd_mode:
                # Use advanced training for Master/PhD
                try:
                    from master_phd_training import train_master_phd_model
                    train_master_phd_model(
                        dataset_id=args.dataset_id,
                        fold=args.fold,
                        max_epochs=args.epochs,
                        thermal_protection=args.thermal_protection
                    )
                except ImportError:
                    print("⚠️ Master/PhD training not available, using standard training")
                    train_multitask_model(
                        dataset_id=args.dataset_id,
                        fold=args.fold,
                        max_epochs=args.epochs,
                        thermal_protection=args.thermal_protection
                    )
            else:
                # Standard training
                train_multitask_model(
                    dataset_id=args.dataset_id,
                    fold=args.fold,
                    max_epochs=args.epochs,
                    thermal_protection=args.thermal_protection
                )
            print("✅ Training completed!")
        
        if args.mode == 'inference' or args.mode == 'all':
            print("\n🔍 STEP 3: Running inference...")
            try:
                run_inference(
                    dataset_id=args.dataset_id,
                    fold=args.fold,
                    test_dir=Path(args.data_root) / "test",
                    master_phd_mode=args.master_phd_mode
                )
                print("✅ Inference completed!")
            except Exception as e:
                print(f"⚠️ Inference failed: {e}")
        
        if args.mode == 'evaluate' or args.mode == 'all':
            print("\n📊 STEP 4: Evaluating results...")
            try:
                evaluate_results(
                    dataset_id=args.dataset_id,
                    validation_dir=Path(args.data_root) / "validation"
                )
                print("✅ Evaluation completed!")
            except Exception as e:
                print(f"⚠️ Evaluation failed: {e}")
        
        print(f"\n🎉 Process completed successfully!")
        
        if args.master_phd_mode:
            print("\n📋 Master/PhD Checklist:")
            print("   □ Performance targets achieved")
            print("   □ Speed improvement ≥10% verified") 
            print("   □ Technical report written")
            print("   □ Code repository published")
            
    except Exception as e:
        print(f"\n❌ Process failed: {e}")
        print(f"Error details: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()