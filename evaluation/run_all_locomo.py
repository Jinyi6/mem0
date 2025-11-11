#!/usr/bin/env python3
"""
Script to run pipeline experiments for multiple datasets (locomo10_0 to locomo10_9).
Modifies the JSON config file for each dataset and runs the pipeline command.
"""

import json
import subprocess
import sys
import os
from copy import deepcopy
from pathlib import Path

# -----------------------------------------------------------------------------
# Configure everything up-front so there are no runtime args to manage.
# Update CONFIG_PATH if you want to point to another mem config.
# Populate DATASET_RUNS with the dataset names (full strings) you want to run
# and the matching existing workspaces (leave empty for fresh runs).
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
CONFIG_PATH = SCRIPT_DIR / "config" / "mem0_qwen.json"
DATASET_RUNS = [
    # {"dataset_name": f"locomo10_{i}", "existing_workspace_dir": ""} for i in range(10)
    {"dataset_name": "locomo10_0", "existing_workspace_dir": ""},
    {"dataset_name": "locomo10_1", "existing_workspace_dir": ""},
    {"dataset_name": "locomo10_2", "existing_workspace_dir": ""},
    {"dataset_name": "locomo10_3", "existing_workspace_dir": ""},
    {"dataset_name": "locomo10_4", "existing_workspace_dir": ""},
    {"dataset_name": "locomo10_5", "existing_workspace_dir": ""},
    {"dataset_name": "locomo10_6", "existing_workspace_dir": ""},
    {"dataset_name": "locomo10_7", "existing_workspace_dir": ""},
    {"dataset_name": "locomo10_8", "existing_workspace_dir": ""},
    {"dataset_name": "locomo10_9", "existing_workspace_dir": ""},
]


def load_base_config(config_path: Path):
    """Load the base config once so later mutations of the file won't matter."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"✗ Config file not found: {config_path}")
        raise
    except json.JSONDecodeError as exc:
        print(f"✗ Invalid JSON config ({config_path}): {exc}")
        raise


def modify_json_config(dataset_name, base_config, existing_workspace_dir="", config_path=CONFIG_PATH):
    """Write a fresh copy of the config with the requested dataset/workspace."""
    try:
        config = deepcopy(base_config)
        setup = config.setdefault("experiment_setup", {})
        setup["dataset_name"] = dataset_name
        setup["existing_workspace_dir"] = existing_workspace_dir or ""

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


def run_pipeline(start_from_step=None):
    """Run the pipeline command."""
    try:
        print("🚀 Starting pipeline execution...")
        command = [
            "python", "-u", "run_pipeline.py",
            "--config", str(CONFIG_PATH)
        ]
        if start_from_step:
            command.extend(["--start_from_step", str(start_from_step)])

        # 使用Popen来捕获所有输出，包括错误
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

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
    print(f"🧪 Starting batch experiments for {len(DATASET_RUNS)} dataset(s)")
    print("=" * 60)
    sys.stdout.flush()

    # Ensure we're in the correct directory
    os.chdir(SCRIPT_DIR)

    if not DATASET_RUNS:
        print("✗ DATASET_RUNS is empty. Please configure it at the top of this file.")
        sys.exit(1)

    try:
        base_config_template = load_base_config(CONFIG_PATH)
    except Exception:
        sys.exit(1)

    successful_runs = 0
    failed_runs = 0

    total_runs = len(DATASET_RUNS)
    for i, run_spec in enumerate(DATASET_RUNS, 1):
        dataset_name = run_spec.get("dataset_name")
        existing_workspace_dir = run_spec.get("existing_workspace_dir") or ""

        if not dataset_name:
            print(f"\n❌ Skipping entry #{i}: missing dataset_name")
            failed_runs += 1
            continue

        start_from_step = 2 if existing_workspace_dir else None

        print(f"\n📊 Processing dataset {i}/{total_runs}: {dataset_name}")
        print("-" * 40)
        if existing_workspace_dir:
            print(f"🔁 Using existing workspace: {existing_workspace_dir} (start_from_step=2)")
        else:
            print("🆕 Running full pipeline (start_from_step=1)")

        # Modify the JSON config
        if not modify_json_config(dataset_name, base_config_template, existing_workspace_dir):
            print(f"❌ Failed to modify config for {dataset_name}")
            failed_runs += 1
            continue

        # Run the pipeline
        if run_pipeline(start_from_step):
            successful_runs += 1
            print(f"✅ Successfully completed experiment for {dataset_name}")
        else:
            failed_runs += 1
            print(f"❌ Failed experiment for {dataset_name}")

        print(f"📈 Progress: {i}/{total_runs} datasets processed")
        sys.stdout.flush()

    # Final summary
    print("\n" + "=" * 60)
    print("📋 FINAL SUMMARY")
    print("=" * 60)
    print(f"✅ Successful runs: {successful_runs}")
    print(f"❌ Failed runs: {failed_runs}")
    print(f"📊 Total datasets: {total_runs}")

    if failed_runs > 0:
        print(f"\n⚠️  {failed_runs} experiments failed. Check the logs above for details.")
        sys.exit(1)
    else:
        print("\n🎉 All experiments completed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()
