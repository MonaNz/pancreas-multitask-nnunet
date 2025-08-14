# main.py

import os
import sys
import torch
import argparse
from pathlib import Path
import json
import shutil

def setup_environment():
    """Setup nnUNet environment variables"""
    project_root = Path.cwd()
    
    # Create nnUNet directories
    nnunet_raw = project_root / "nnUNet_raw"
    nnunet_preprocessed = project_root / "nnUNet_preprocessed" 
    nnunet_results = project_root / "nnUNet_results"
    
    nnunet_raw.mkdir(exist_ok=True)
    nnunet_preprocessed.mkdir(exist_ok=True)
    nnunet_results.mkdir(exist_ok=True)
    
    # Set environment variables
    os.environ['nnUNet_raw'] = str(nnunet_raw)
    os.environ['nnUNet_preprocessed'] = str(nnunet_preprocessed)
    os.environ['nnUNet_results'] = str(nnunet_results)
    
    print(f"nnUNet environment setup complete:")
    print(f"  nnUNet_raw: {nnunet_raw}")
    print(f"  nnUNet_preprocessed: {nnunet_preprocessed}")
    print(f"  nnUNet_results: {nnunet_results}")
    
    return nnunet_raw, nnunet_preprocessed, nnunet_results

class PancreasCancerPipeline:
    """Complete pipeline for pancreatic cancer segmentation and classification"""
    
    def __init__(self):
        self.dataset_id = 500
        self.dataset_name = "PancreasCancer"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup paths
        self.project_root = Path.cwd()
        self.data_path = self.project_root
        
        # Setup nnUNet environment
        self.nnunet_raw, self.nnunet_preprocessed, self.nnunet_results = setup_environment()
        self.dataset_path = self.nnunet_raw / f"Dataset{self.dataset_id:03d}_{self.dataset_name}"
        
        print(f"Using device: {self.device}")
    
    def step1_prepare_data(self):
        """Step 1: Convert data to nnUNet format"""
        print("\n" + "="*60)
        print("STEP 1: DATA PREPARATION")
        print("="*60)
        
        from src.data_preparation import DataPreparation
        
        data_prep = DataPreparation(
            raw_data_path=str(self.data_path),
            nnunet_data_path=str(self.nnunet_raw)
        )
        
        dataset_path, train_cases, test_cases = data_prep.convert_to_nnunet_format()
        data_prep.analyze_dataset(dataset_path)
        
        print(f"✓ Data preparation completed!")
        print(f"  Dataset: {dataset_path}")
        print(f"  Training cases: {len(train_cases)}")
        print(f"  Test cases: {len(test_cases)}")
        
        return dataset_path, train_cases, test_cases
    
    def step2_preprocess_data(self):
        """Step 2: Run nnUNet preprocessing"""
        print("\n" + "="*60)
        print("STEP 2: DATA PREPROCESSING")
        print("="*60)
        
        # Run nnUNet preprocessing
        cmd = f"nnUNetv2_plan_and_preprocess -d {self.dataset_id} --verify_dataset_integrity"
        print(f"Running: {cmd}")
        
        os.system(cmd)
        
        print("✓ Data preprocessing completed!")
    
    def step3_train_model(self, num_epochs=50):
        """Step 3: Train the multi-task model"""
        print("\n" + "="*60)
        print("STEP 3: MODEL TRAINING")
        print("="*60)
        
        from src.simple_training import SimpleTrainer
        
        # High-performance training configuration for 64GB RAM
        config = {
            'batch_size': 8,              # Much larger batch size!
            'learning_rate': 3e-4,        # Higher LR for larger batches
            'num_epochs': num_epochs,
            'seg_weight': 1.0,
            'cls_weight': 0.5,
            'early_stopping_patience': 12,
            'save_best_only': True,
            'target_size': (96, 96, 96)   # Much larger images for better quality!
        }
        
        # Initialize training manager
        output_dir = self.project_root / "training_output"
        trainer = SimpleTrainer(
            dataset_path=str(self.dataset_path),
            output_dir=str(output_dir),
            config=config
        )
        
        # Train the model
        trainer.train()
        
        print("✓ Model training completed!")
        
        return output_dir / "best_model.pth"
    
    def step4_inference(self, model_path):
        """Step 4: Run inference on test data"""
        print("\n" + "="*60)
        print("STEP 4: MODEL INFERENCE")
        print("="*60)
        
        from src.inference import MultiTaskInference
        
        # Initialize inference
        inference_engine = MultiTaskInference(
            model_path=str(model_path),
            dataset_path=str(self.dataset_path),
            output_dir=str(self.project_root / "results")
        )
        
        # Run inference
        seg_results, cls_results = inference_engine.predict_test_cases()
        
        print("✓ Inference completed!")
        print(f"  Segmentation results: {len(seg_results)} cases")
        print(f"  Classification results: {len(cls_results)} cases")
        
        return seg_results, cls_results
    
    def step5_evaluate(self, model_path):
        """Step 5: Evaluate on validation set"""
        print("\n" + "="*60)
        print("STEP 5: MODEL EVALUATION")
        print("="*60)
        
        from src.evaluation import ModelEvaluator
        
        evaluator = ModelEvaluator(
            model_path=str(model_path),
            dataset_path=str(self.dataset_path),
            output_dir=str(self.project_root / "evaluation")
        )
        
        # Evaluate on validation set
        results = evaluator.evaluate_validation_set()
        
        print("✓ Evaluation completed!")
        print(f"  Whole pancreas DSC: {results['whole_pancreas_dsc']:.3f}")
        print(f"  Lesion DSC: {results['lesion_dsc']:.3f}")
        print(f"  Classification F1: {results['classification_f1']:.3f}")
        
        return results
    
    def run_complete_pipeline(self, num_epochs=50):
        """Run the complete pipeline"""
        print("Starting Pancreatic Cancer Analysis Pipeline")
        print("=" * 80)
        
        try:
            # Step 1: Data preparation
            dataset_path, train_cases, test_cases = self.step1_prepare_data()
            
            # Step 2: Train model (using simple training)
            model_path = self.step3_train_model(num_epochs)
            
            # Step 3: Evaluate model
            evaluation_results = self.step5_evaluate(model_path)
            
            # Step 4: Run inference on test set
            seg_results, cls_results = self.step4_inference(model_path)
            
            print("\n" + "="*80)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*80)
            print(f"✓ Data prepared: {len(train_cases)} training, {len(test_cases)} test cases")
            print(f"✓ Model trained and saved: {model_path}")
            print(f"✓ Evaluation completed - DSC: {evaluation_results.get('whole_pancreas_dsc', 0):.3f}")
            print(f"✓ Test inference completed: {len(seg_results)} cases")
            
            return {
                'dataset_path': dataset_path,
                'model_path': model_path,
                'evaluation_results': evaluation_results,
                'test_results': {'segmentation': seg_results, 'classification': cls_results}
            }
            
        except Exception as e:
            print(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Pancreatic Cancer Analysis Pipeline')
    parser.add_argument('--mode', type=str, choices=['full', 'prepare', 'train', 'evaluate', 'inference'], 
                       default='full', help='Pipeline mode')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--dataset_path', type=str, help='Path to prepared dataset (for specific modes)')
    parser.add_argument('--model_path', type=str, help='Path to trained model (for evaluation/inference)')
    
    args = parser.parse_args()
    
    pipeline = PancreasCancerPipeline()
    
    if args.mode == 'full':
        # Run complete pipeline
        results = pipeline.run_complete_pipeline(num_epochs=args.epochs)
        
    elif args.mode == 'prepare':
        # Data preparation only
        dataset_path, train_cases, test_cases = pipeline.step1_prepare_data()
        print(f"Data preparation completed: {dataset_path}")
        
    elif args.mode == 'train':
        # Training only
        if not args.dataset_path:
            print("Error: --dataset_path required for training mode")
            return
        
        from src.simple_training import SimpleTrainer
        
        config = {
            'batch_size': 1,
            'learning_rate': 1e-4,
            'num_epochs': args.epochs,
            'seg_weight': 1.0,
            'cls_weight': 0.5,
            'target_size': (32, 32, 32)  # Much smaller = much faster
        }
        
        trainer = SimpleTrainer(
            dataset_path=args.dataset_path,
            output_dir="training_output",
            config=config
        )
        trainer.train()
        
    elif args.mode == 'evaluate':
        # Evaluation only
        if not args.model_path or not args.dataset_path:
            print("Error: --model_path and --dataset_path required for evaluation mode")
            return
            
        from src.evaluation import ModelEvaluator
        
        evaluator = ModelEvaluator(
            model_path=args.model_path,
            dataset_path=args.dataset_path,
            output_dir="evaluation"
        )
        evaluator.evaluate_validation_set()
        
    elif args.mode == 'inference':
        # Inference only
        if not args.model_path or not args.dataset_path:
            print("Error: --model_path and --dataset_path required for inference mode")
            return
            
        from src.inference import MultiTaskInference
        
        inference_engine = MultiTaskInference(
            model_path=args.model_path,
            dataset_path=args.dataset_path,
            output_dir="results"
        )
        inference_engine.predict_test_cases()

if __name__ == "__main__":
    main()