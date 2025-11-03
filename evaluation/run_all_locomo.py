#!/usr/bin/env python3
"""
Script to run pipeline experiments for multiple datasets (locomo10_0 to locomo10_9).
Modifies the JSON config file for each dataset and runs the pipeline command.
"""

import json
import subprocess
import sys
import os
from pathlib import Path

def modify_json_config(dataset_name, config_path="config/mem0.json"):
    """Modify the JSON configuration file with the new dataset name."""
    try:
        # Read the current config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Update the dataset name
        config["experiment_setup"]["dataset_name"] = dataset_name
        
        # Write back to file
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Updated config with dataset: {dataset_name}")
        sys.stdout.flush()
        return True
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"✗ Error modifying config file: {e}")
        print(f"Full error traceback: {error_details}")
        return False

def run_pipeline():
    """Run the pipeline command."""
    try:
        print("🚀 Starting pipeline execution...")
        # 使用Popen来捕获所有输出，包括错误
        process = subprocess.Popen([
            "python", "-u", "run_pipeline.py", 
            "--config", "config/mem0.json"
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
           universal_newlines=True, bufsize=1)
        
        # 实时输出所有内容
        for line in process.stdout:
            print(line.rstrip())
            sys.stdout.flush()  # 确保在nohup下也能正常输出
        
        # 等待进程完成
        return_code = process.wait()
        
        if return_code == 0:
            print("✓ Pipeline completed successfully")
            sys.stdout.flush()
            return True
        else:
            print(f"✗ Pipeline failed with return code: {return_code}")
            sys.stdout.flush()
            return False
        
    except Exception as e:
        print(f"✗ Error running pipeline: {e}")
        return False

def main():
    """Main function to run experiments for all datasets."""
    # 确保输出缓冲区立即刷新，适合nohup环境
    sys.stdout.flush()
    sys.stderr.flush()
    
    print("=" * 60)
    print("🧪 Starting batch experiments for locomo10_0 to locomo10_9")
    print("=" * 60)
    sys.stdout.flush()
    
    # Ensure we're in the correct directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # List of datasets to process
    datasets = [f"locomo10_{i}" for i in range(7, 10)]
    
    successful_runs = 0
    failed_runs = 0
    
    for i, dataset_name in enumerate(datasets, 1):
        print(f"\n📊 Processing dataset {i}/10: {dataset_name}")
        print("-" * 40)
        
        # Modify the JSON config
        if not modify_json_config(dataset_name):
            print(f"❌ Failed to modify config for {dataset_name}")
            failed_runs += 1
            continue
        
        # Run the pipeline
        if run_pipeline():
            successful_runs += 1
            print(f"✅ Successfully completed experiment for {dataset_name}")
        else:
            failed_runs += 1
            print(f"❌ Failed experiment for {dataset_name}")
        
        print(f"📈 Progress: {i}/10 datasets processed")
        sys.stdout.flush()
    
    # Final summary
    print("\n" + "=" * 60)
    print("📋 FINAL SUMMARY")
    print("=" * 60)
    print(f"✅ Successful runs: {successful_runs}")
    print(f"❌ Failed runs: {failed_runs}")
    print(f"📊 Total datasets: {len(datasets)}")
    
    if failed_runs > 0:
        print(f"\n⚠️  {failed_runs} experiments failed. Check the logs above for details.")
        sys.exit(1)
    else:
        print("\n🎉 All experiments completed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()
