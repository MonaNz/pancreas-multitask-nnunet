#!/usr/bin/env python3
"""
Proper Speed Optimization for Master's 10%+ Requirement
Multiple optimization strategies to achieve the speed target
"""

import os
import time
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Fix PyTorch loading
torch.serialization.add_safe_globals([
    'numpy.core.multiarray.scalar',
    'numpy._core.multiarray.scalar'
])

class OptimizedInferenceRunner:
    """Multiple speed optimization strategies"""
    
    def __init__(self, model_path, test_dir, device):
        self.model_path = model_path
        self.test_dir = test_dir
        self.device = device
        self.model = None
        self.test_files = list(Path(test_dir).glob("*_0000.nii.gz"))
        
    def load_model(self):
        """Load model with optimizations"""
        print("Loading model...")
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        try:
            from working_standalone import MegaEnhancedUNet3D as ModelClass
            print("✅ Using MegaEnhancedUNet3D")
        except ImportError:
            from working_standalone import MultiTaskUNet3D as ModelClass
            print("✅ Using MultiTaskUNet3D")
        
        self.model = ModelClass().to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Optimization 1: Set to inference mode
        for module in self.model.modules():
            if hasattr(module, 'training'):
                module.training = False
        
        print("✅ Model loaded and optimized")
        return True
    
    def create_optimized_dataset(self, batch_size=4):
        """Create optimized dataset and dataloader"""
        
        class FastTestDataset(Dataset):
            def __init__(self, test_files):
                self.test_files = test_files
                # Pre-compute patch size
                self.patch_size = (64, 64, 64)
            
            def __len__(self):
                return len(self.test_files)
            
            def _fast_normalize(self, image):
                """Faster normalization"""
                return ((np.clip(image, -1000, 1000) + 1000) / 2000.0).astype(np.float32)
            
            def _fast_extract_patch(self, image):
                """Faster patch extraction"""
                d, h, w = image.shape
                pd, ph, pw = self.patch_size
                
                # Center crop
                start_d, start_h, start_w = (d-pd)//2, (h-ph)//2, (w-pw)//2
                start_d, start_h, start_w = max(0, start_d), max(0, start_h), max(0, start_w)
                
                patch = image[start_d:start_d+pd, start_h:start_h+ph, start_w:start_w+pw]
                
                # Fast padding if needed
                if patch.shape != self.patch_size:
                    padded = np.zeros(self.patch_size, dtype=np.float32)
                    padded[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
                    return padded
                
                return patch
            
            def __getitem__(self, idx):
                file_path = self.test_files[idx]
                case_id = file_path.stem.replace("_0000", "")
                
                # Fast loading
                img = nib.load(file_path)
                image = img.get_fdata()
                
                # Fast processing
                image = self._fast_normalize(image)
                patch = self._fast_extract_patch(image)
                tensor = torch.from_numpy(patch).unsqueeze(0)
                
                return {
                    'image': tensor,
                    'case_id': case_id
                }
        
        dataset = FastTestDataset(self.test_files)
        
        # Optimized dataloader
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size,  # Larger batch for efficiency
            shuffle=False, 
            num_workers=0,  # Avoid multiprocessing overhead
            pin_memory=True,  # Faster GPU transfer
            persistent_workers=False
        )
        
        return dataset, dataloader
    
    def baseline_inference(self):
        """Standard baseline inference"""
        print("\n🕐 BASELINE INFERENCE")
        
        dataset, dataloader = self.create_optimized_dataset(batch_size=1)
        times = []
        results = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Baseline"):
                start_time = time.perf_counter()  # More precise timing
                
                images = batch['image'].to(self.device)
                case_ids = batch['case_id']
                
                # Standard inference
                outputs = self.model(images)
                
                seg_preds = torch.argmax(outputs['segmentation'], dim=1)
                cls_preds = torch.argmax(outputs['classification'], dim=1)
                
                # Ensure GPU sync for accurate timing
                torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                batch_time = end_time - start_time
                times.append(batch_time / len(case_ids))  # Per case time
                
                for i, case_id in enumerate(case_ids):
                    results.append({
                        'case_id': case_id,
                        'segmentation': seg_preds[i].cpu().numpy(),
                        'classification': cls_preds[i].cpu().item()
                    })
        
        avg_time = np.mean(times)
        print(f"✅ Baseline: {avg_time:.4f}s per case")
        return avg_time, results
    
    def optimized_inference_v1(self):
        """Optimization v1: Batching + GPU optimizations"""
        print("\n⚡ OPTIMIZATION V1: Batching + GPU")
        
        dataset, dataloader = self.create_optimized_dataset(batch_size=4)  # Larger batch
        times = []
        results = []
        
        # GPU optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Optimized v1"):
                start_time = time.perf_counter()
                
                images = batch['image'].to(self.device, non_blocking=True)
                case_ids = batch['case_id']
                
                # Batched inference
                outputs = self.model(images)
                
                seg_preds = torch.argmax(outputs['segmentation'], dim=1)
                cls_preds = torch.argmax(outputs['classification'], dim=1)
                
                torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                batch_time = end_time - start_time
                times.append(batch_time / len(case_ids))
                
                for i, case_id in enumerate(case_ids):
                    results.append({
                        'case_id': case_id,
                        'segmentation': seg_preds[i].cpu().numpy(),
                        'classification': cls_preds[i].cpu().item()
                    })
        
        avg_time = np.mean(times)
        print(f"✅ Optimized v1: {avg_time:.4f}s per case")
        return avg_time, results
    
    def optimized_inference_v2(self):
        """Optimization v2: Model optimizations"""
        print("\n⚡ OPTIMIZATION V2: Model Optimizations")
        
        # Compile model for optimization (PyTorch 2.0+)
        try:
            if hasattr(torch, 'compile'):
                print("🔧 Compiling model with torch.compile...")
                compiled_model = torch.compile(self.model, mode='reduce-overhead')
            else:
                compiled_model = self.model
        except:
            compiled_model = self.model
        
        dataset, dataloader = self.create_optimized_dataset(batch_size=4)
        times = []
        results = []
        
        # Advanced GPU settings
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Optimized v2"):
                start_time = time.perf_counter()
                
                images = batch['image'].to(self.device, non_blocking=True)
                case_ids = batch['case_id']
                
                # Use compiled model
                outputs = compiled_model(images)
                
                seg_preds = torch.argmax(outputs['segmentation'], dim=1)
                cls_preds = torch.argmax(outputs['classification'], dim=1)
                
                torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                batch_time = end_time - start_time
                times.append(batch_time / len(case_ids))
                
                for i, case_id in enumerate(case_ids):
                    results.append({
                        'case_id': case_id,
                        'segmentation': seg_preds[i].cpu().numpy(),
                        'classification': cls_preds[i].cpu().item()
                    })
        
        avg_time = np.mean(times)
        print(f"✅ Optimized v2: {avg_time:.4f}s per case")
        return avg_time, results
    
    def optimized_inference_v3(self):
        """Optimization v3: Data pipeline + reduced precision"""
        print("\n⚡ OPTIMIZATION V3: Pipeline + Precision")
        
        dataset, dataloader = self.create_optimized_dataset(batch_size=6)
        times = []
        results = []
        
        # Maximum optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        with torch.no_grad():
            # Use autocast for potential speedup on newer GPUs
            with torch.amp.autocast('cuda', enabled=True):
                for batch in tqdm(dataloader, desc="Optimized v3"):
                    start_time = time.perf_counter()
                    
                    # Async GPU transfer
                    images = batch['image'].to(self.device, non_blocking=True)
                    case_ids = batch['case_id']
                    
                    # Wait for transfer to complete
                    torch.cuda.current_stream().synchronize()
                    
                    # Inference
                    outputs = self.model(images)
                    
                    # Fast argmax
                    seg_preds = outputs['segmentation'].argmax(dim=1)
                    cls_preds = outputs['classification'].argmax(dim=1)
                    
                    torch.cuda.synchronize()
                    
                    end_time = time.perf_counter()
                    batch_time = end_time - start_time
                    times.append(batch_time / len(case_ids))
                    
                    for i, case_id in enumerate(case_ids):
                        results.append({
                            'case_id': case_id,
                            'segmentation': seg_preds[i].cpu().numpy(),
                            'classification': cls_preds[i].cpu().item()
                        })
        
        avg_time = np.mean(times)
        print(f"✅ Optimized v3: {avg_time:.4f}s per case")
        return avg_time, results

def run_comprehensive_speed_test():
    """Run comprehensive speed optimization test"""
    
    print("🎯 COMPREHENSIVE SPEED OPTIMIZATION TEST")
    print("=" * 60)
    print("Goal: Achieve 10%+ speed improvement for Master's degree")
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Device: {device}")
    
    # Find model
    results_path = Path("nnUNet_results/Dataset500_PancreasCancer")
    model_files = ["best_model.pth", "mega_best_model.pth", "enhanced_best_model.pth"]
    
    model_path = None
    for model_file in model_files:
        test_path = results_path / model_file
        if test_path.exists():
            model_path = test_path
            break
    
    if not model_path:
        print("❌ No model found!")
        return False
    
    test_dir = "test"
    if not Path(test_dir).exists():
        print("❌ No test directory!")
        return False
    
    # Initialize optimizer
    optimizer = OptimizedInferenceRunner(model_path, test_dir, device)
    
    if not optimizer.load_model():
        return False
    
    # Run all optimization strategies
    baseline_time, baseline_results = optimizer.baseline_inference()
    opt1_time, opt1_results = optimizer.optimized_inference_v1()
    opt2_time, opt2_results = optimizer.optimized_inference_v2()
    opt3_time, opt3_results = optimizer.optimized_inference_v3()
    
    # Calculate improvements
    improvements = {
        'Optimization v1 (Batching)': ((baseline_time - opt1_time) / baseline_time) * 100,
        'Optimization v2 (Model)': ((baseline_time - opt2_time) / baseline_time) * 100,
        'Optimization v3 (Pipeline)': ((baseline_time - opt3_time) / baseline_time) * 100
    }
    
    print(f"\n🎯 SPEED OPTIMIZATION RESULTS:")
    print("=" * 50)
    print(f"Baseline time:           {baseline_time:.4f}s")
    print(f"Optimized v1 time:       {opt1_time:.4f}s")
    print(f"Optimized v2 time:       {opt2_time:.4f}s") 
    print(f"Optimized v3 time:       {opt3_time:.4f}s")
    print()
    
    # Find best optimization
    best_improvement = max(improvements.values())
    best_method = max(improvements, key=improvements.get)
    best_time = min(opt1_time, opt2_time, opt3_time)
    
    print(f"🏆 BEST OPTIMIZATION: {best_method}")
    print(f"📊 Best time:            {best_time:.4f}s")
    print(f"⚡ Speed improvement:     {best_improvement:.1f}%")
    print(f"💾 Time saved:           {(baseline_time - best_time):.4f}s per case")
    
    # Master's requirement check
    master_target = 10.0
    if best_improvement >= master_target:
        print(f"\n🎉 MASTER'S SPEED TARGET ACHIEVED! 🎉")
        print(f"✅ {best_improvement:.1f}% ≥ {master_target}% (requirement)")
        success = True
    else:
        print(f"\n⚠️ Master's target not met with current optimizations")
        print(f"❌ {best_improvement:.1f}% < {master_target}% (requirement)")
        print(f"📈 Need additional {master_target - best_improvement:.1f}% improvement")
        success = False
    
    # Save best results
    print(f"\n💾 Saving optimized results...")
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Use best optimization results
    best_results = opt3_results if opt3_time == best_time else (opt2_results if opt2_time == best_time else opt1_results)
    
    # Save segmentation files
    for result in best_results:
        case_id = result['case_id']
        seg_mask = result['segmentation']
        
        seg_img = nib.Nifti1Image(seg_mask.astype(np.uint8), affine=np.eye(4))
        seg_path = results_path / f"{case_id}.nii.gz"
        nib.save(seg_img, seg_path)
    
    # Save classification results
    csv_data = [{
        'Names': f"{r['case_id']}.nii.gz",
        'Subtype': r['classification']
    } for r in best_results]
    
    df = pd.DataFrame(csv_data)
    csv_path = results_path / "subtype_results.csv"
    df.to_csv(csv_path, index=False)
    
    # Save comprehensive speed report
    speed_report = results_path / "comprehensive_speed_report.txt"
    with open(speed_report, 'w') as f:
        f.write("MASTER'S DEGREE COMPREHENSIVE SPEED OPTIMIZATION REPORT\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Master's Requirement: 10%+ speed improvement\n")
        f.write(f"Best Achievement: {best_improvement:.1f}%\n")
        f.write(f"Status: {'PASSED' if success else 'NEEDS MORE WORK'}\n\n")
        f.write(f"Timing Results:\n")
        f.write(f"  Baseline:          {baseline_time:.4f}s per case\n")
        f.write(f"  Optimization v1:   {opt1_time:.4f}s per case ({improvements['Optimization v1 (Batching)']:+.1f}%)\n")
        f.write(f"  Optimization v2:   {opt2_time:.4f}s per case ({improvements['Optimization v2 (Model)']:+.1f}%)\n")
        f.write(f"  Optimization v3:   {opt3_time:.4f}s per case ({improvements['Optimization v3 (Pipeline)']:+.1f}%)\n\n")
        f.write(f"Best Method: {best_method}\n")
        f.write(f"Time Saved: {(baseline_time - best_time):.4f}s per case\n\n")
        f.write("Optimizations Applied:\n")
        f.write("- Increased batch size (1→4→6)\n")
        f.write("- CUDNN benchmark optimization\n")
        f.write("- TF32 acceleration\n")
        f.write("- Model compilation (PyTorch 2.0+)\n")
        f.write("- Optimized data pipeline\n")
        f.write("- Async GPU transfers\n")
        f.write("- Precision timing with perf_counter\n")
    
    print(f"✅ Results saved to: {results_path}")
    print(f"📊 Generated files:")
    print(f"  - {len(best_results)} segmentation files")
    print(f"  - 1 classification file (subtype_results.csv)")
    print(f"  - 1 comprehensive speed report")
    
    return success

def main():
    print("🎓 MASTER'S DEGREE SPEED OPTIMIZATION")
    print("=" * 45)
    
    success = run_comprehensive_speed_test()
    
    if success:
        print(f"\n🎉 MASTER'S SPEED REQUIREMENT ACHIEVED!")
        print(f"✅ 1/3 Master's requirements completed!")
    else:
        print(f"\n💡 Additional optimization strategies:")
        print(f"  - Model quantization (INT8)")
        print(f"  - TensorRT optimization")
        print(f"  - ONNX export and optimization")
        print(f"  - Custom CUDA kernels")

if __name__ == "__main__":
    main()