#!/usr/bin/env python3
"""
Results Extraction and Analysis Script
Analyzes your pancreas cancer training results and generates reports
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

def analyze_training_results():
    """Extract and analyze training results"""
    
    print("🔍 TRAINING RESULTS ANALYSIS")
    print("=" * 40)
    
    # Check for results directory
    results_path = Path("nnUNet_results/Dataset500_PancreasCancer")
    
    if not results_path.exists():
        print("❌ Results directory not found!")
        print(f"Expected: {results_path}")
        return None
    
    # Load best model checkpoint
    model_path = results_path / "best_model.pth"
    
    if model_path.exists():
        print("✅ Found best model checkpoint")
        
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            
            print(f"\n📊 BEST MODEL PERFORMANCE (Epoch {checkpoint['epoch']}):")
            print("-" * 50)
            
            metrics = checkpoint.get('metrics', {})
            cls_f1 = checkpoint.get('cls_f1', 0.0)
            
            print(f"Whole Pancreas DSC: {metrics.get('whole_dice', 0.0):.3f}")
            print(f"Pancreas DSC: {metrics.get('pancreas_dice', 0.0):.3f}")
            print(f"Lesion DSC: {metrics.get('lesion_dice', 0.0):.3f}")
            print(f"Classification F1: {cls_f1:.3f}")
            
            # Performance vs targets
            print(f"\n🎯 MASTER'S DEGREE TARGET COMPARISON:")
            print("-" * 40)
            
            # Master's targets
            whole_target = 0.91
            lesion_target = 0.31
            cls_target = 0.7
            
            whole_dsc = metrics.get('whole_dice', 0.0)
            lesion_dsc = metrics.get('lesion_dice', 0.0)
            
            print("Master's Degree Requirements:")
            print(f"  Whole DSC: {whole_dsc:.3f} / {whole_target:.2f} {'✅' if whole_dsc >= whole_target else '❌'} {f'({((whole_dsc/whole_target-1)*100):+.1f}%)' if whole_dsc < whole_target else ''}")
            print(f"  Lesion DSC: {lesion_dsc:.3f} / {lesion_target:.2f} {'✅' if lesion_dsc >= lesion_target else '❌'} {f'({((lesion_dsc/lesion_target-1)*100):+.1f}%)' if lesion_dsc >= lesion_target else ''}")
            print(f"  Class F1: {cls_f1:.3f} / {cls_target:.2f} {'✅' if cls_f1 >= cls_target else '❌'} {f'({((cls_f1/cls_target-1)*100):+.1f}%)' if cls_f1 < cls_target else ''}")
            print(f"  Speed Improvement: TBD / 10%+ ⏳ (Mandatory - not yet tested)")
            
            # Calculate gap analysis
            gaps = []
            if whole_dsc < whole_target:
                gaps.append(f"Whole DSC needs +{((whole_target/whole_dsc-1)*100):.1f}% improvement")
            if lesion_dsc < lesion_target:
                gaps.append(f"Lesion DSC needs +{((lesion_target/lesion_dsc-1)*100):.1f}% improvement")
            if cls_f1 < cls_target:
                gaps.append(f"Classification F1 needs +{((cls_target/cls_f1-1)*100):.1f}% improvement")
            
            if gaps:
                print(f"\n⚠️ Performance Gaps:")
                for gap in gaps:
                    print(f"  • {gap}")
            else:
                print(f"\n🎉 All performance targets achieved! Missing only speed optimization.")
            
            return checkpoint
            
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            return None
    else:
        print("❌ No best model checkpoint found")
        return None

def parse_training_log():
    """Parse training log from console output"""
    
    print(f"\n📈 TRAINING PROGRESSION ANALYSIS")
    print("-" * 40)
    
    # This would typically parse a log file, but since you have console output,
    # we'll provide a manual analysis based on your log
    
    training_data = [
        # (epoch, train_loss, val_loss, whole_dsc, lesion_dsc, cls_f1)
        (0, 0.9387, 0.8061, 0.056, 0.500, 0.167),
        (3, 0.7762, 0.7768, 0.331, 0.472, 0.196),
        (6, 0.7536, 0.7473, 0.400, 0.333, 0.196),
        (9, 0.7371, 0.7390, 0.505, 0.472, 0.196),
        (12, 0.7260, 0.7175, 0.442, 0.417, 0.196),
        (15, 0.7350, 0.6997, 0.493, 0.286, 0.196),
        (18, 0.7095, 0.7007, 0.511, 0.249, 0.196),
        (21, 0.6933, 0.7111, 0.604, 0.248, 0.196),
        (24, 0.6806, 0.6782, 0.577, 0.198, 0.251),
        (27, 0.6612, 0.6645, 0.609, 0.253, 0.361),
        (30, 0.6562, 0.6632, 0.613, 0.240, 0.381),
        (33, 0.6328, 0.6511, 0.640, 0.209, 0.318),
        (36, 0.6354, 0.6901, 0.651, 0.338, 0.381),
        (39, 0.6078, 0.5697, 0.652, 0.364, 0.414),
        (42, 0.5841, 0.6231, 0.694, 0.360, 0.364),
        (45, 0.5904, 0.5982, 0.593, 0.298, 0.429),
        (48, 0.5920, 0.5893, 0.633, 0.294, 0.404),
        (51, 0.5612, 0.5629, 0.630, 0.520, 0.438),
        (54, 0.5709, 0.5924, 0.680, 0.375, 0.382),
        (57, 0.5619, 0.5654, 0.675, 0.402, 0.475),
        (60, 0.5519, 0.5876, 0.631, 0.396, 0.403),
        (63, 0.5311, 0.6106, 0.689, 0.386, 0.296),
        (66, 0.5107, 0.6258, 0.664, 0.327, 0.453),
        (69, 0.5061, 0.6073, 0.651, 0.419, 0.411),
        (72, 0.5053, 0.6195, 0.690, 0.416, 0.422),
        (75, 0.4838, 0.5991, 0.691, 0.508, 0.462),
        (78, 0.4872, 0.5784, 0.731, 0.374, 0.529),
        (81, 0.4627, 0.5660, 0.709, 0.319, 0.561),
        (84, 0.4889, 0.6967, 0.713, 0.390, 0.451),
        (87, 0.4602, 0.6936, 0.681, 0.327, 0.432),
        (90, 0.4738, 0.6281, 0.672, 0.299, 0.507),
        (93, 0.4758, 0.6346, 0.744, 0.460, 0.514),
        (96, 0.4705, 0.5857, 0.741, 0.361, 0.599),
        (99, 0.4427, 0.6330, 0.767, 0.395, 0.382),
    ]
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(training_data, columns=[
        'epoch', 'train_loss', 'val_loss', 'whole_dsc', 'lesion_dsc', 'cls_f1'
    ])
    
    # Find best performance
    best_whole = df.loc[df['whole_dsc'].idxmax()]
    best_lesion = df.loc[df['lesion_dsc'].idxmax()]
    best_cls = df.loc[df['cls_f1'].idxmax()]
    
    print(f"Best Whole DSC: {best_whole['whole_dsc']:.3f} at epoch {best_whole['epoch']}")
    print(f"Best Lesion DSC: {best_lesion['lesion_dsc']:.3f} at epoch {best_lesion['epoch']}")
    print(f"Best Classification F1: {best_cls['cls_f1']:.3f} at epoch {best_cls['epoch']}")
    
    # Training trends
    print(f"\n📊 Training Trends:")
    initial_loss = df.iloc[0]['train_loss']
    final_loss = df.iloc[-1]['train_loss']
    print(f"Loss Reduction: {initial_loss:.3f} → {final_loss:.3f} (-{((initial_loss-final_loss)/initial_loss)*100:.1f}%)")
    
    return df

def create_training_plots(df):
    """Create training visualization plots"""
    
    print(f"\n📊 Creating training plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Pancreas Cancer Multi-task Training Results', fontsize=16)
    
    # Loss curves
    axes[0,0].plot(df['epoch'], df['train_loss'], label='Training Loss', color='blue')
    axes[0,0].plot(df['epoch'], df['val_loss'], label='Validation Loss', color='red')
    axes[0,0].set_title('Training and Validation Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Segmentation metrics
    axes[0,1].plot(df['epoch'], df['whole_dsc'], label='Whole Pancreas DSC', color='green')
    axes[0,1].plot(df['epoch'], df['lesion_dsc'], label='Lesion DSC', color='orange')
    axes[0,1].axhline(y=0.91, color='green', linestyle='--', alpha=0.8, label='Master Target (Whole)', linewidth=2)
    axes[0,1].axhline(y=0.31, color='orange', linestyle='--', alpha=0.8, label='Master Target (Lesion)', linewidth=2)
    axes[0,1].set_title('Segmentation Performance (DSC)')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Dice Score')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # Classification metrics
    axes[1,0].plot(df['epoch'], df['cls_f1'], label='Classification F1', color='purple')
    axes[1,0].axhline(y=0.7, color='purple', linestyle='--', alpha=0.8, label='Master Target', linewidth=2)
    axes[1,0].set_title('Classification Performance (F1)')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('F1 Score')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # Combined score
    combined_score = (df['whole_dsc'] * 0.4 + df['lesion_dsc'] * 0.3 + df['cls_f1'] * 0.3)
    axes[1,1].plot(df['epoch'], combined_score, label='Combined Score', color='black')
    axes[1,1].set_title('Combined Performance Score')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('Combined Score')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path("training_results.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Training plots saved to: {plot_path}")
    
    return plot_path

def generate_summary_report(checkpoint, df):
    """Generate a summary report"""
    
    report_path = Path("training_summary_report.md")
    
    with open(report_path, 'w') as f:
        f.write("# Pancreas Cancer Multi-task Learning - Training Summary\n\n")
        
        f.write("## Overview\n")
        f.write(f"- **Training Epochs**: {len(df)} validation checkpoints\n")
        f.write(f"- **Model Parameters**: 22,970,150\n")
        f.write(f"- **Training Samples**: 252\n")
        f.write(f"- **Validation Samples**: 36\n\n")
        
        if checkpoint:
            metrics = checkpoint.get('metrics', {})
            cls_f1 = checkpoint.get('cls_f1', 0.0)
            
            f.write("## Best Model Performance\n")
            f.write(f"- **Epoch**: {checkpoint['epoch']}\n")
            f.write(f"- **Whole Pancreas DSC**: {metrics.get('whole_dice', 0.0):.3f}\n")
            f.write(f"- **Lesion DSC**: {metrics.get('lesion_dice', 0.0):.3f}\n")
            f.write(f"- **Classification F1**: {cls_f1:.3f}\n\n")
        
        # Final performance
        final_row = df.iloc[-1]
        f.write("## Final Training Performance\n")
        f.write(f"- **Training Loss**: {final_row['train_loss']:.4f}\n")
        f.write(f"- **Validation Loss**: {final_row['val_loss']:.4f}\n")
        f.write(f"- **Whole Pancreas DSC**: {final_row['whole_dsc']:.3f}\n")
        f.write(f"- **Lesion DSC**: {final_row['lesion_dsc']:.3f}\n")
        f.write(f"- **Classification F1**: {final_row['cls_f1']:.3f}\n\n")
        
        # Target comparison
        f.write("## Target Achievement Status - Master's Degree\n\n")
        f.write(f"### Current Performance vs Master's Requirements\n")
        f.write(f"- **Whole Pancreas DSC**: {final_row['whole_dsc']:.3f} / 0.91 {'✅' if final_row['whole_dsc'] >= 0.91 else '❌'}\n")
        f.write(f"- **Lesion DSC**: {final_row['lesion_dsc']:.3f} / 0.31 {'✅' if final_row['lesion_dsc'] >= 0.31 else '❌'}\n")
        f.write(f"- **Classification F1**: {final_row['cls_f1']:.3f} / 0.70 {'✅' if final_row['cls_f1'] >= 0.7 else '❌'}\n")
        f.write(f"- **Speed Improvement**: TBD / 10%+ ⏳ (Mandatory requirement)\n\n")
        
        # Performance gaps
        gaps = []
        if final_row['whole_dsc'] < 0.91:
            improvement_needed = ((0.91/final_row['whole_dsc'] - 1) * 100)
            gaps.append(f"- Whole Pancreas DSC needs **{improvement_needed:.1f}% improvement**")
        if final_row['lesion_dsc'] < 0.31:
            improvement_needed = ((0.31/final_row['lesion_dsc'] - 1) * 100)
            gaps.append(f"- Lesion DSC needs **{improvement_needed:.1f}% improvement**")
        if final_row['cls_f1'] < 0.7:
            improvement_needed = ((0.7/final_row['cls_f1'] - 1) * 100)
            gaps.append(f"- Classification F1 needs **{improvement_needed:.1f}% improvement**")
        
        if gaps:
            f.write("### Performance Gaps to Address\n")
            for gap in gaps:
                f.write(f"{gap}\n")
            f.write("\n")
        else:
            f.write("### ✅ All Performance Targets Met!\n")
            f.write("Only speed optimization remains to complete Master's requirements.\n\n")
        
        f.write("## Master's Degree Action Plan\n")
        f.write("### Priority 1: Whole Pancreas Segmentation Improvement\n")
        f.write("- **Current**: 0.767 DSC\n")
        f.write("- **Target**: 0.91 DSC\n")
        f.write("- **Gap**: 14.4% improvement needed\n")
        f.write("- **Strategies**: Extended training (150+ epochs), advanced loss functions, data augmentation\n\n")
        
        f.write("### Priority 2: Classification Performance Enhancement\n")
        f.write("- **Current**: 0.514 F1\n")
        f.write("- **Target**: 0.70 F1\n")
        f.write("- **Gap**: 18.6% improvement needed\n")
        f.write("- **Strategies**: Class balancing, improved classification head, ensemble methods\n\n")
        
        f.write("### Priority 3: Speed Optimization (Mandatory)\n")
        f.write("- **Requirement**: 10%+ inference speed improvement\n")
        f.write("- **Status**: Not yet implemented\n")
        f.write("- **Strategies**: TensorRT optimization, mixed precision, ROI cropping, model quantization\n\n")
        
        f.write("### Immediate Next Steps\n")
        f.write("1. Run inference to establish baseline speed\n")
        f.write("2. Implement speed optimizations from FLARE challenge solutions\n")
        f.write("3. Extended training with hyperparameter tuning\n")
        f.write("4. Advanced data augmentation and loss function optimization\n")
    
    print(f"✅ Summary report saved to: {report_path}")
    return report_path

def run_inference_check():
    """Check if inference can be run"""
    
    print(f"\n🔍 INFERENCE READINESS CHECK")
    print("-" * 40)
    
    # Check for test data
    test_dir = Path("test")
    if test_dir.exists():
        test_files = list(test_dir.glob("*_0000.nii.gz"))
        print(f"✅ Found {len(test_files)} test files")
        
        # Show some examples
        if test_files:
            print("Sample test files:")
            for f in test_files[:5]:
                print(f"  - {f.name}")
            if len(test_files) > 5:
                print(f"  ... and {len(test_files) - 5} more")
    else:
        print("❌ No test directory found")
    
    # Check for model
    model_path = Path("nnUNet_results/Dataset500_PancreasCancer/best_model.pth")
    if model_path.exists():
        print("✅ Best model checkpoint available for inference")
        return True
    else:
        print("❌ No model checkpoint found for inference")
        return False

def main():
    """Main analysis function"""
    
    print("🎓 PANCREAS CANCER PROJECT - RESULTS ANALYSIS")
    print("=" * 60)
    
    # Analyze training results
    checkpoint = analyze_training_results()
    
    # Parse training progression
    df = parse_training_log()
    
    # Create visualizations
    if df is not None:
        plot_path = create_training_plots(df)
        
        # Generate report
        report_path = generate_summary_report(checkpoint, df)
    
    # Check inference readiness
    inference_ready = run_inference_check()
    
    print(f"\n🎉 ANALYSIS COMPLETE!")
    print("-" * 30)
    print("Generated files:")
    print("  - training_results.png (training plots)")
    print("  - training_summary_report.md (summary report)")
    
    if inference_ready:
        print(f"\n✅ Ready for inference! Run:")
        print("python working_standalone.py --mode inference")
    else:
        print(f"\n⚠️ Set up test data and model before inference")

if __name__ == "__main__":
    main()