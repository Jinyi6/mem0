# ./run_pipeline.py

import json
import os
import subprocess
import argparse
import shutil
from datetime import datetime
import sys
import argparse
import json
from collections import defaultdict

os.environ["LOCAL_MEM0_PATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# os.environ["LOCAL_MEM0_PATH"] = "/Users/jinyi/Documents/code/memory/mem0" # 也可以定义成绝对路径

# Set the OpenAI API key
os.environ['OPENAI_API_KEY'] = "sk-vyvftxtwuiznrwrfvayhfitxgpdpsykrdnukzfdtdwtjgqvo"
os.environ["OPENAI_BASE_URL"] = "https://api.siliconflow.cn/v1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["MEM0_TELEMETRY"] = "False"

import subprocess
import sys
import os

# 在 run_pipeline.py 中
import os
import subprocess

def save_git_state(workspace_dir, reference_point="origin/main"):
    """
    Captures the git state by comparing the project's working directory against
    a fixed reference point and saves it to the workspace.

    The git commands are run from the project root, which is assumed to be
    the parent directory of this script's location.
    """
    # --- NEW: Determine the project root directory ---
    # This assumes your script is in a subdirectory of the project root, like /scripts
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        # Fallback for interactive environments where __file__ is not defined
        project_root = os.getcwd()
        print(f"⚠️  '__file__' not defined. Using current working directory as project root: {project_root}")


    log_path = os.path.join(workspace_dir, "code_state.log")
    print(f"📝 Checking git status in '{project_root}'...")
    print(f"📝 Comparing code against '{reference_point}' and saving state to: {log_path}")

    # --- NEW: Check if the target directory is a git repository ---
    if not os.path.isdir(os.path.join(project_root, '.git')):
        print(f"❌ Error: The directory '{project_root}' is not a git repository.")
        # Optionally, write this error to the log file
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"Error: The target directory '{project_root}' is not a git repository.\n")
        return

    # --- MODIFIED: The helper function now accepts a 'cwd' argument ---
    def run_git_command(command, cwd):
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, encoding='utf-8', cwd=cwd
            )
            if result.returncode != 0:
                return f"Git command failed. Stderr:\n{result.stderr.strip()}"
            return result.stdout.strip()
        except FileNotFoundError:
            return "Git command not found. Make sure git is installed."

    # --- MODIFIED: All git commands are now run with `cwd=project_root` ---
    # 1. Get the commit hash of the reference point
    base_commit_hash = run_git_command(["git", "rev-parse", reference_point], cwd=project_root)

    # 2. Get the diff of the entire working directory against the reference point
    diff_output = run_git_command(["git", "diff", reference_point], cwd=project_root)
    if not diff_output or "Git command failed" in diff_output:
        diff_output = f"No differences found against '{reference_point}'."

    # 3. Get untracked files
    untracked_files = run_git_command(["git", "ls-files", "--others", "--exclude-standard"], cwd=project_root)
    if not untracked_files or "Git command failed" in untracked_files:
        untracked_files = "No untracked files."

    # Write all information to the log file (no changes here)
    try:
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f" Git State for Experiment (Compared against: {reference_point})\n")
            f.write(f" Project Root: {project_root}\n")
            f.write("="*80 + "\n\n")

            f.write(f"# 1. Base Commit Hash of '{reference_point}'\n")
            f.write("-" * (30 + len(reference_point)) + "\n")
            f.write(f"{base_commit_hash}\n\n")

            f.write(f"# 2. All Changes (Working Directory vs '{reference_point}')\n")
            f.write("-" * (50 + len(reference_point)) + "\n")
            f.write(f"{diff_output}\n\n")

            f.write("# 3. Untracked Files\n")
            f.write("-" * 22 + "\n")
            f.write(f"{untracked_files}\n")
        
        print("✅ Git state saved successfully.")
    except IOError as e:
        print(f"❌ Failed to write git state log file: {e}")

def run_command(command):
    """
    Executes a command, prints its combined stdout and stderr in real-time,
    and raises an exception on failure. This version avoids deadlocks caused by tqdm.
    """
    print(f"\n🚀 Executing command:\n{' '.join(command)}\n")
    
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            # bufsize=1, # bufsize=1 是行缓冲，这里我们需要无缓冲，所以可以去掉
        )

        print("-------------------- Real-time Output --------------------")
        
        # 核心改动：从 readline() 改为 read(1) 逐字符读取
        while True:
            # 读取一个字符
            char = process.stdout.read(1)
            # 如果没有更多字符，且子进程已结束，则退出循环
            if not char:
                # 再次检查进程状态，确保完全退出
                if process.poll() is not None:
                    break
                else:
                    # 如果进程还在运行但没有输出，可以短暂休眠避免CPU空转
                    # import time
                    # time.sleep(0.01)
                    continue

            # 直接将字符打印到标准输出
            sys.stdout.write(char)
            # 立即刷新缓冲区，确保实时显示
            sys.stdout.flush()

        print("\n----------------------------------------------------------") # 加一个换行符，让格式更好看
        
        # 等待进程完全结束
        process.wait()

        if process.returncode != 0:
            print(f"❌ Command failed with exit code {process.returncode}")
            raise subprocess.CalledProcessError(
                returncode=process.returncode,
                cmd=command
            )

    except FileNotFoundError:
        print(f"❌ Command not found. Make sure '{command[0]}' is in your PATH or the script path is correct.")
        raise
    except subprocess.CalledProcessError as e:
        raise e
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise
# def run_command(command):
#     """Executes a command, captures its output, and raises an exception if it fails."""
#     print(f"\n🚀 Executing command:\n{' '.join(command)}\n")
#     try:
#         result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
#     except subprocess.CalledProcessError as e:
#         print("-------------------- STDOUT --------------------")
#         print(e.stdout)
#         print("-------------------- STDERR --------------------")
#         print(e.stderr)
#         print("----------------------------------------------")
#         print(f"❌ Command failed with exit code {e.returncode}")
#         raise e
#     except FileNotFoundError:
#         print(f"❌ Command not found. Make sure '{command[0]}' is in your PATH or the script exists.")
#         raise

def main():
    parser = argparse.ArgumentParser(description="Automated experiment pipeline for memory evaluation.")
    parser.add_argument(
        "--config",
        type=str,
        default="./config/default.json",
        help="Path to the experiment configuration JSON file."
    )
    # --- 新增参数 ---
    parser.add_argument(
        "--start_from_step",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="The step to start the pipeline from. "
             "1: Full pipeline. "
             "2: Start from Search (requires existing_workspace_dir in config). "
             "3: Start from Evaluate (requires existing_workspace_dir in config). "
             "4: Start from Generate Scores (requires existing_workspace_dir in config)."
    )
    args = parser.parse_args()

    # 1. Load Configuration
    print(f"📄 Loading configuration from: {args.config}")
    with open(args.config, 'r') as f:
        config = json.load(f)
    setup_params = config["experiment_setup"]
    exp_params = config["exp_params"]
    os.environ["BASE_MODEL"] = exp_params.get("base_model", "Qwen/Qwen3-14B")

    # 2. Setup Experiment Workspace based on start_from_step
    workspace_dir = ""
    if args.start_from_step == 1:
        # --- 行为和原来一致：创建新工作区 ---
        print("▶️ Starting from Step 1: A new workspace will be created.")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = (
            f"{setup_params['dataset_name']}_"
            f"top_k_{exp_params['top_k']}_"
            f"filter_{exp_params['filter_memories']}_"
            f"graph_{exp_params['is_graph']}_"
            f"{exp_params['fact_extraction_mode']}_"
            f"{exp_params['memory_decision_mode']}_"
            f"{exp_params['search_mode']}_"
            f"{timestamp}"
        )
        workspace_dir = os.path.join(setup_params['base_dir'], setup_params['dataset_name'], exp_name)
        os.makedirs(workspace_dir, exist_ok=True)
        shutil.copy(args.config, os.path.join(workspace_dir, "config.json"))
        print("="*80)
        print(f"📂 Created new experiment workspace at: {workspace_dir}")
        print("="*80)
    else:
        # --- 新行为：使用已有的工作区 ---
        print(f"▶️ Starting from Step {args.start_from_step}: Using an existing workspace.")
        workspace_dir = setup_params.get("existing_workspace_dir")
        if not workspace_dir or not os.path.isdir(workspace_dir):
            raise ValueError(
                f"❌ Error: When starting from step {args.start_from_step}, "
                f"the 'existing_workspace_dir' must be specified in '{args.config}' "
                "and it must be a valid directory."
            )
        print("="*80)
        print(f"📂 Using existing experiment workspace: {workspace_dir}")
        print("="*80)

    # check code diff
    # save_git_state(workspace_dir)

    # 3. Define all file paths within the workspace
    qdrant_path = os.path.join(workspace_dir, "qdrant_data")
    search_results_filename = (
        f"mem0_{setup_params['dataset_name']}_results_top_{exp_params['top_k']}_"
        f"filter_{exp_params['filter_memories']}_graph_{exp_params['is_graph']}.json"
    )
    search_results_path = os.path.join(workspace_dir, search_results_filename)
    eval_metrics_path = os.path.join(workspace_dir, "evaluation_metrics.json")
    final_scores_path = os.path.join(workspace_dir, "final_scores.txt")

    # --- Execute Pipeline Steps Conditionally ---
    
    if args.start_from_step <= 1 and exp_params['technique_type'] not in ["full_context", "openai"]:
        print("\n" + "#"*25 + " STEP 1: ADD MEMORIES " + "#"*25, flush=True)
        add_command = [
            "python", "-u", "./run_experiments.py",
            "--method", "add",
            "--dataset_name", setup_params['dataset_name'],
            "--technique_type", exp_params['technique_type'],
            "--mode", exp_params['mode'],
            "--embedder_model", exp_params['embedder_model'],
            "--qdrant_path", qdrant_path,
            "--workspace_dir", workspace_dir,
            "--fact_extraction_mode", exp_params.get("fact_extraction_mode", "0"),
            "--memory_decision_mode", exp_params.get("memory_decision_mode", "0"),
        ]
        if exp_params.get("figure_view", False): add_command.append("--figure_view")
        if exp_params.get("is_graph", False): add_command.append("--is_graph")
        run_command(add_command)
        print("✅ Step 1 completed successfully.", flush=True)
    else:
        print("\n⏭️ Skipping Step 1: ADD MEMORIES. (Not required for '{exp_params['technique_type']}' or start_from_step > 1).", flush=True)

    if args.start_from_step <= 2:
        print("\n" + "#"*25 + " STEP 2: SEARCH MEMORIES " + "#"*25, flush=True)
        search_command = [
            "python", "-u", "./run_experiments.py",
            "--method", "search",
            "--dataset_name", setup_params['dataset_name'],
            "--output_folder", workspace_dir,
            "--technique_type", exp_params['technique_type'],
            "--mode", exp_params['mode'],
            "--top_k", str(exp_params['top_k']),
            "--embedder_model", exp_params['embedder_model'],
            "--qdrant_path", qdrant_path,
            "--workspace_dir", workspace_dir,
            "--search_mode", exp_params.get("search_mode", "0"),
        ]
        if exp_params.get("filter_memories", False): search_command.append("--filter_memories")
        if exp_params.get("is_graph", False): search_command.append("--is_graph")
        run_command(search_command)
        print("✅ Step 2 completed successfully.", flush=True)
    else:
        print("\n⏭️ Skipping Step 2: SEARCH MEMORIES.", flush=True)

    if args.start_from_step <= 3:
        print("\n" + "#"*25 + " STEP 3: EVALUATE RESULTS " + "#"*25, flush=True)
        eval_command = [
            "python", "-u", "./evals.py",
            "--input_file", search_results_path,
            "--output_file", eval_metrics_path,
            "--max_workers", str(config['eval_params']['max_workers'])
        ]
        run_command(eval_command)
        print("✅ Step 3 completed successfully.", flush=True)
    else:
        print("\n⏭️ Skipping Step 3: EVALUATE RESULTS.", flush=True)

    # if args.start_from_step <= 4:
    #     print("\n" + "#"*25 + " STEP 4: GENERATE SCORES " + "#"*25)
    #     score_command = [
    #         "python", "-u", "./generate_scores_args.py",
    #         "--input_file", eval_metrics_path,
    #         "--output_file", final_scores_path
    #     ]
    #     run_command(score_command)
    #     print("✅ Step 4 completed successfully.")
    # else:
    #     print("\n⏭️ Skipping Step 4: GENERATE SCORES.")

    print("\n" + "="*80)
    print("🎉🎉🎉 Experiment pipeline finished successfully! 🎉🎉🎉")
    print(f"📊 All results, logs, and data are saved in:\n{workspace_dir}")
    # print(f"📈 Final scores can be found in:\n{final_scores_path}")
    print("="*80)

if __name__ == "__main__":
    main()

# nohup python run_pipeline.py --config ./config/default.json > pipeline.log 2>&1 &