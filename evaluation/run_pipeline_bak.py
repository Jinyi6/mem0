# ./run_pipeline.py

import json
import os
import subprocess
import argparse
import shutil
from datetime import datetime


os.environ["LOCAL_MEM0_PATH"] = "/Users/jinyi/Documents/code/memory/mem0"
# Set the OpenAI API key
os.environ['OPENAI_API_KEY'] = "sk-vyvftxtwuiznrwrfvayhfitxgpdpsykrdnukzfdtdwtjgqvo"
os.environ["OPENAI_BASE_URL"] = "https://api.siliconflow.cn/v1"
os.environ["BASE_MODEL"] = "Qwen/Qwen3-14B"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def run_command(command):
    """Executes a command, captures its output, and raises an exception if it fails."""
    print(f"\n🚀 Executing command:\n{' '.join(command)}\n")
    try:
        # 修改这里：添加 capture_output=True 和 text=True
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
    except subprocess.CalledProcessError as e:
        # 在这里打印捕获到的标准输出和标准错误
        print("-------------------- STDOUT --------------------")
        print(e.stdout)
        print("-------------------- STDERR --------------------")
        print(e.stderr)
        print("----------------------------------------------")
        print(f"❌ Command failed with exit code {e.returncode}")
        raise e
    except FileNotFoundError:
        print(f"❌ Command not found. Make sure '{command[1]}' is in the correct path.")
        raise

def main():
    parser = argparse.ArgumentParser(description="Automated experiment pipeline for memory evaluation.")
    parser.add_argument(
        "--config",
        type=str,
        default="./config/default.json",
        help="Path to the experiment configuration JSON file."
    )
    args = parser.parse_args()

    # 1. Load Configuration
    print(f"📄 Loading configuration from: {args.config}")
    with open(args.config, 'r') as f:
        config = json.load(f)

    # 2. Create Experiment Workspace
    setup_params = config["experiment_setup"]
    exp_params = config["exp_params"]
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = (
        f"{setup_params['dataset_name']}_"
        f"top_k_{exp_params['top_k']}_"
        f"filter_{exp_params['filter_memories']}_"
        f"graph_{exp_params['is_graph']}_"
        f"{timestamp}"
    )
    
    workspace_dir = os.path.join(setup_params['base_dir'], setup_params['dataset_name'], exp_name)
    os.makedirs(workspace_dir, exist_ok=True)
    
    # Copy config for reproducibility
    shutil.copy(args.config, os.path.join(workspace_dir, "config.json"))
    
    print("="*80)
    print(f"📂 Created experiment workspace at: {workspace_dir}")
    print("="*80)

    # 3. Define all file paths within the workspace
    qdrant_path = os.path.join(workspace_dir, "qdrant_data")
    # This file name is constructed by run_experiments.py, we reconstruct it here for the next steps
    search_results_filename = (
        f"mem0_{setup_params['dataset_name']}_results_top_{exp_params['top_k']}_"
        f"filter_{exp_params['filter_memories']}_graph_{exp_params['is_graph']}.json"
    )
    search_results_path = os.path.join(workspace_dir, search_results_filename)
    eval_metrics_path = os.path.join(workspace_dir, "evaluation_metrics.json")
    final_scores_path = os.path.join(workspace_dir, "final_scores.txt")

    # --- Step 1: Add Memories ---
    print("\n" + "#"*25 + " STEP 1: ADD MEMORIES " + "#"*25)
    add_command = [
        "python", "./run_experiments.py",
        "--method", "add",
        "--dataset_name", setup_params['dataset_name'],
        "--technique_type", exp_params['technique_type'],
        "--mode", exp_params['mode'],
        "--embedder_model", exp_params['embedder_model'],
        "--qdrant_path", qdrant_path,
        "--workspace_dir", workspace_dir, 
    ]
    if exp_params.get("figure_view", False):
        add_command.append("--figure_view")
    if exp_params.get("is_graph", False):
        add_command.append("--is_graph")
        
    run_command(add_command)
    print("✅ Step 1 completed successfully.")

    # --- Step 2: Search Memories ---
    print("\n" + "#"*25 + " STEP 2: SEARCH MEMORIES " + "#"*25)
    search_command = [
        "python", "./run_experiments.py",
        "--method", "search",
        "--dataset_name", setup_params['dataset_name'],
        "--output_folder", workspace_dir,
        "--technique_type", exp_params['technique_type'],
        "--mode", exp_params['mode'],
        "--top_k", str(exp_params['top_k']),
        "--embedder_model", exp_params['embedder_model'],
        "--qdrant_path", qdrant_path,
        "--workspace_dir", workspace_dir, 
    ]
    if exp_params.get("filter_memories", False):
        search_command.append("--filter_memories")
    if exp_params.get("is_graph", False):
        search_command.append("--is_graph")

    run_command(search_command)
    print("✅ Step 2 completed successfully.")

    # --- Step 3: Evaluate Results ---
    print("\n" + "#"*25 + " STEP 3: EVALUATE RESULTS " + "#"*25)
    eval_command = [
        "python", "./evals.py",
        "--input_file", search_results_path,
        "--output_file", eval_metrics_path,
        "--max_workers", str(config['eval_params']['max_workers'])
    ]
    run_command(eval_command)
    print("✅ Step 3 completed successfully.")

    # --- Step 4: Generate Scores ---
    print("\n" + "#"*25 + " STEP 4: GENERATE SCORES " + "#"*25)
    score_command = [
        "python", "./generate_scores_args.py",
        "--input_file", eval_metrics_path,
        "--output_file", final_scores_path
    ]
    run_command(score_command)
    print("✅ Step 4 completed successfully.")
    
    print("\n" + "="*80)
    print("🎉🎉🎉 Experiment pipeline finished successfully! 🎉🎉🎉")
    print(f"📊 All results, logs, and data are saved in:\n{workspace_dir}")
    print(f"📈 Final scores can be found in:\n{final_scores_path}")
    print("="*80)

if __name__ == "__main__":
    main()

    # python run_pipeline.py --config ./config/default.json
