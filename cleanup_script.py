#!/usr/bin/env python3
"""
Project Cleanup Script
Organizes your project by removing unnecessary files and keeping essential ones
"""

import os
import shutil
from pathlib import Path

def cleanup_project():
    """Clean up the project directory"""
    
    print("🧹 PROJECT CLEANUP SCRIPT")
    print("=" * 50)
    print("This will help organize your project for final submission")
    print()
    
    # Files to delete (safe to remove)
    files_to_delete = [
        "fixed_results_extractor.py",
        "immediate_fix_script.py", 
        "main.py",
        "proper_speed_optimization.py",
        "quick_inference_fix.py",
        "quick_validation_fix.py",
        "results_extractor.py",
        "run_inference.py",
        "setup_and_run.py"
    ]
    
    # Essential files to keep
    essential_files = [
        "validation_ensemble_runner.py",
        "ensemble_finetuning_system.py",
        "visualization.py",
        "working_standalone.py",
        "ensemble_training_system.py",
        "ensemble_inference_system.py",
        "resume_ensemble_training.py",
        "check_ensemble_status.py",
        "masters_requirements_test.py",
        "README.md",
        "requirements.txt",
        "master_phd_guide.txt"
    ]
    
    # Optional files (can delete if not needed)
    optional_files = [
        "realistic_masters_assessment.json",
        "masters_requirements_evaluation.json", 
        "training_results.png",
        "training_summary_report.md"
    ]
    
    # Directories to keep
    keep_directories = [
        "data",
        "nnUNet_results", 
        "ensemble_visualizations",
        "validation_results"
    ]
    
    print("📋 CLEANUP PLAN:")
    print(f"  Files to delete: {len(files_to_delete)}")
    print(f"  Essential files: {len(essential_files)}")
    print(f"  Optional files: {len(optional_files)}")
    print()
    
    # Show what will be deleted
    print("🗑️ FILES TO DELETE:")
    for file in files_to_delete:
        if Path(file).exists():
            print(f"  ❌ {file}")
        else:
            print(f"  ⚪ {file} (not found)")
    
    # Show essential files status
    print(f"\n✅ ESSENTIAL FILES STATUS:")
    for file in essential_files:
        if Path(file).exists():
            print(f"  ✅ {file}")
        else:
            print(f"  ⚠️ {file} (missing)")
    
    # Ask for confirmation
    print(f"\n❓ Do you want to proceed with cleanup? (y/n): ", end="")
    response = input().strip().lower()
    
    if response != 'y':
        print("❌ Cleanup cancelled")
        return
    
    print(f"\n🧹 Starting cleanup...")
    
    deleted_count = 0
    
    # Delete unnecessary files
    for file in files_to_delete:
        file_path = Path(file)
        if file_path.exists():
            try:
                file_path.unlink()
                print(f"  🗑️ Deleted: {file}")
                deleted_count += 1
            except Exception as e:
                print(f"  ⚠️ Could not delete {file}: {e}")
        
    # Clean up cache directories
    cache_dirs = ["__pycache__", ".pytest_cache"]
    for cache_dir in cache_dirs:
        cache_path = Path(cache_dir)
        if cache_path.exists():
            try:
                shutil.rmtree(cache_path)
                print(f"  🗑️ Deleted cache: {cache_dir}")
                deleted_count += 1
            except Exception as e:
                print(f"  ⚠️ Could not delete {cache_dir}: {e}")
    
    # Clean up .pyc files
    pyc_count = 0
    for pyc_file in Path(".").rglob("*.pyc"):
        try:
            pyc_file.unlink()
            pyc_count += 1
        except:
            pass
    
    if pyc_count > 0:
        print(f"  🗑️ Deleted {pyc_count} .pyc files")
        deleted_count += pyc_count
    
    print(f"\n✅ CLEANUP COMPLETED!")
    print(f"  Files deleted: {deleted_count}")
    
    # Show final project structure
    print(f"\n📁 FINAL PROJECT STRUCTURE:")
    
    all_files = sorted([f for f in Path(".").iterdir() if f.is_file() and not f.name.startswith(".")])
    all_dirs = sorted([d for d in Path(".").iterdir() if d.is_dir() and not d.name.startswith(".")])
    
    for directory in all_dirs:
        print(f"  📁 {directory.name}/")
    
    for file in all_files:
        if file.name in essential_files:
            print(f"  ✅ {file.name}")
        elif file.name in optional_files:
            print(f"  📄 {file.name} (optional)")
        else:
            print(f"  📄 {file.name}")
    
    print(f"\n🎯 PROJECT READY FOR:")
    print(f"  • Master's degree submission")
    print(f"  • Performance testing")
    print(f"  • Academic presentation")
    print(f"  • Further development")

def show_recommended_workflow():
    """Show recommended workflow after cleanup"""
    
    print(f"\n🚀 RECOMMENDED NEXT STEPS:")
    print("=" * 40)
    
    workflow = [
        ("1. Test Performance", "python validation_ensemble_runner.py"),
        ("2. Generate Plots", "python visualization.py"), 
        ("3. Improve if Needed", "python ensemble_finetuning_system.py"),
        ("4. Check Status", "python check_ensemble_status.py"),
        ("5. Final Report", "Review generated visualizations and metrics")
    ]
    
    for step, command in workflow:
        print(f"  {step}:")
        print(f"    {command}")
        print()
    
    print("🎓 Your project is now organized and ready for Master's evaluation!")

def main():
    """Main cleanup function"""
    
    # Run cleanup
    cleanup_project()
    
    # Show workflow
    show_recommended_workflow()

if __name__ == "__main__":
    main()