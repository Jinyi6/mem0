# ./run_pipeline.py

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

os.environ["LOCAL_MEM0_PATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# os.environ["LOCAL_MEM0_PATH"] = "/Users/jinyi/Documents/code/memory/mem0" # 也可以定义成绝对路径

# Set the OpenAI API key
# os.environ['OPENAI_API_KEY'] = "sk-vyvftxtwuiznrwrfvayhfitxgpdpsykrdnukzfdtdwtjgqvo"
# os.environ["OPENAI_BASE_URL"] = "https://api.siliconflow.cn/v1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["MEM0_TELEMETRY"] = "False"


def sanitize_collection_name(name: str) -> str:
    """
    Convert an arbitrary workspace name into a Qdrant collection name that is
    stable across runs and safe for concurrent experiments.
    """
    if not name:
        return "mem0"
    sanitized = re.sub(r"[^0-9a-zA-Z_]+", "_", name).strip("_")
    if not sanitized:
        sanitized = "mem0"
    if sanitized[0].isdigit():
        sanitized = f"c_{sanitized}"
    # Qdrant allows collection names up to 255 chars, but keep ours shorter.
    return sanitized[:120]

import subprocess
import sys
import os

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
            stdin=subprocess.DEVNULL, 
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
    parser.add_argument(
        "--use_msp_runner",
        action="store_true",
        help="If set, invoke run_experiments_msp.py instead of the default runner."
    )
    args = parser.parse_args()

    # 1. Load Configuration
    print(f"📄 Loading configuration from: {args.config}")
    with open(args.config, 'r') as f:
        config = json.load(f)
    setup_params = config["experiment_setup"]
    exp_params = config["exp_params"]
    dataset_name = setup_params.get("dataset_name", "dataset")
    technique_type = exp_params.get("technique_type", "mem0")

    def safe_param_value(key, fallback="na"):
        """
        Read experiment parameter `key` and coerce to a string, falling back when missing.
        """
        value = exp_params.get(key, fallback)
        if value in (None, ""):
            value = fallback
        return str(value)

    top_k_str = safe_param_value("top_k")
    filter_memories_str = safe_param_value("filter_memories")
    is_graph_str = safe_param_value("is_graph")
    fact_extraction_mode_str = safe_param_value("fact_extraction_mode")
    memory_decision_mode_str = safe_param_value("memory_decision_mode")
    search_mode_str = safe_param_value("search_mode")
    answer_mode_str = safe_param_value("answer_mode")
    llm_params = exp_params.get("llm", {}) or {}
    search_llm_params = exp_params.get("search_llm") or {}
    answer_llm_params = exp_params.get("answer_llm") or {}
    embedder_params = exp_params.get("embedder", {})
    evaluator_params = exp_params.get("evaluator", {})

    add_llm_model = llm_params.get("model") or "Qwen/Qwen3-14B"
    add_llm_base_url = llm_params.get("base_url") or "https://api.siliconflow.cn/v1"
    add_llm_api_key = llm_params.get("api_key") or ""

    search_llm_model = search_llm_params.get("model") or add_llm_model
    search_llm_base_url = search_llm_params.get("base_url") or add_llm_base_url
    search_llm_api_key = search_llm_params.get("api_key") or add_llm_api_key

    answer_llm_model = answer_llm_params.get("model") or search_llm_model
    answer_llm_base_url = answer_llm_params.get("base_url") or search_llm_base_url
    answer_llm_api_key = answer_llm_params.get("api_key") or search_llm_api_key

    embedder_model = embedder_params.get("model") or "Pro/BAAI/bge-m3"
    embedder_base_url = embedder_params.get("base_url") or search_llm_base_url
    embedder_api_key = embedder_params.get("api_key") or search_llm_api_key
    embedder_dims = embedder_params.get("embedding_dims")

    evaluator_model = evaluator_params.get("model") or answer_llm_model
    evaluator_base_url = evaluator_params.get("base_url") or answer_llm_base_url
    evaluator_api_key = evaluator_params.get("api_key") or answer_llm_api_key

    def set_llm_env(model_value, base_url_value, api_key_value):
        if model_value:
            os.environ["BASE_MODEL"] = model_value
        if base_url_value:
            os.environ["OPENAI_BASE_URL"] = base_url_value
        if api_key_value:
            os.environ["OPENAI_API_KEY"] = api_key_value

    set_llm_env(add_llm_model, add_llm_base_url, add_llm_api_key)
    if evaluator_model:
        os.environ["EVALUATOR_MODEL"] = evaluator_model

    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    llm_model_tag = sanitize_collection_name(add_llm_model)
    planned_exp_name = (
        f"{dataset_name}_"
        f"model_{llm_model_tag}_"
        f"top_k_{top_k_str}_"
        f"filter_{filter_memories_str}_"
        f"graph_{is_graph_str}_"
        f"{fact_extraction_mode_str}_"
        f"{memory_decision_mode_str}_"
        f"{search_mode_str}_"
        f"{answer_mode_str}_"
        f"{run_timestamp}"
    )
    proposed_workspace_dir = os.path.join(
        setup_params['base_dir'], dataset_name, planned_exp_name
    )

    existing_workspace_dir = setup_params.get("existing_workspace_dir")
    workspace_dir = ""
    state_workspace_dir = ""
    if args.start_from_step == 1:
        print("▶️ Starting from Step 1: A new workspace will be created.")
        workspace_dir = proposed_workspace_dir
        state_workspace_dir = workspace_dir
        os.makedirs(workspace_dir, exist_ok=True)
        print("="*80)
        print(f"📂 Created new experiment workspace at: {workspace_dir}")
        print("="*80)
    elif args.start_from_step == 2:
        print("▶️ Starting from Step 2: Preparing a new workspace for this run.")
        if not existing_workspace_dir or not os.path.isdir(existing_workspace_dir):
            raise ValueError(
                f"❌ Error: When starting from step {args.start_from_step}, "
                f"the 'existing_workspace_dir' must be specified in '{args.config}' "
                "and it must be a valid directory."
            )
        workspace_dir = proposed_workspace_dir
        state_workspace_dir = existing_workspace_dir
        os.makedirs(workspace_dir, exist_ok=True)
        with open(os.path.join(workspace_dir, "state_source.txt"), "w", encoding="utf-8") as marker:
            marker.write(existing_workspace_dir)
        print("="*80)
        print(f"📂 Created new output workspace at: {workspace_dir}")
        print(f"♻️  Reusing memories from: {existing_workspace_dir}")
        print("="*80)
    else:
        print(f"▶️ Starting from Step {args.start_from_step}: Using an existing workspace.")
        workspace_dir = existing_workspace_dir
        state_workspace_dir = existing_workspace_dir
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
    qdrant_path = os.path.join(state_workspace_dir, "qdrant_data")
    os.makedirs(qdrant_path, exist_ok=True)
    collection_name = sanitize_collection_name(Path(state_workspace_dir).name)
    mem0_state_dir = os.path.join(state_workspace_dir, ".mem0_state")
    os.makedirs(mem0_state_dir, exist_ok=True)
    os.environ["MEM0_DIR"] = mem0_state_dir
    msp_runner = args.use_msp_runner
    runner_script = "./run_experiments_msp.py" if msp_runner else "./run_experiments.py"

    if technique_type == "full_context":
        search_results_raw_filename = f"full_context_{dataset_name}_results.json"
        search_results_filename = f"full_context_{dataset_name}_results_{run_timestamp}.json"
    else:
        if msp_runner:
            search_results_raw_filename = (
                f"msp_{dataset_name}_results_top_{top_k_str}_mode_{search_mode_str}.json"
            )
            search_results_filename = (
                f"msp_{dataset_name}_results_top_{top_k_str}_mode_{search_mode_str}_"
                f"{run_timestamp}_{fact_extraction_mode_str}_{memory_decision_mode_str}_{answer_mode_str}.json"
            )
        else:
            search_results_raw_filename = (
                f"mem0_{dataset_name}_results_top_{top_k_str}_"
                f"filter_{filter_memories_str}_graph_{is_graph_str}.json"
            )
            search_results_filename = (
                f"mem0_{dataset_name}_results_top_{top_k_str}_"
                f"filter_{filter_memories_str}_graph_{is_graph_str}_"
                f"{run_timestamp}_{fact_extraction_mode_str}_{memory_decision_mode_str}_{search_mode_str}_{answer_mode_str}.json"
            )
    shutil.copy(args.config, os.path.join(workspace_dir, f"config_{run_timestamp}.json"))
    search_results_raw_path = os.path.join(workspace_dir, search_results_raw_filename)
    search_results_path = os.path.join(workspace_dir, search_results_filename)
    eval_metrics_path = os.path.join(workspace_dir, f"evaluation_metrics_{run_timestamp}.json")
    final_scores_path = os.path.join(workspace_dir, "final_scores.txt")

    # --- Execute Pipeline Steps Conditionally ---

    def append_arg(command_list, flag, value):
        if value not in (None, "", False):
            command_list.extend([flag, value])
    
    if args.start_from_step <= 1 and technique_type not in ["full_context", "openai"]:
        print("\n" + "#"*25 + " STEP 1: ADD MEMORIES " + "#"*25, flush=True)
        add_max_workers = (
            config.get("add_params", {}).get("max_workers")
            or exp_params.get("max_workers", 4)
        )
        add_batch_size = (
            config.get("add_params", {}).get("batch_size")
            or exp_params.get("batch_size")
        )
        add_mode_value = (
            config.get("add_params", {}).get("add_mode")
            or exp_params.get("add_mode")
        )
        add_command = [
            "python", "-u", runner_script,
            "--method", "add",
            "--dataset_name", dataset_name,
            "--technique_type", technique_type,
            "--mode", exp_params.get("mode", "no_client_async"),
            "--embedder_model", embedder_model,
            "--qdrant_path", qdrant_path,
            "--workspace_dir", workspace_dir,
            "--fact_extraction_mode", exp_params.get("fact_extraction_mode", "0"),
            "--memory_decision_mode", exp_params.get("memory_decision_mode", "0"),
            "--max_workers", str(add_max_workers),
            "--collection_name", collection_name,
        ]
        if add_batch_size is not None:
            append_arg(add_command, "--batch_size", str(add_batch_size))
        if add_mode_value is not None:
            append_arg(add_command, "--add_mode", str(add_mode_value))
        append_arg(add_command, "--llm_model", add_llm_model)
        append_arg(add_command, "--llm_base_url", add_llm_base_url)
        append_arg(add_command, "--llm_api_key", add_llm_api_key)
        append_arg(add_command, "--embedder_base_url", embedder_base_url)
        append_arg(add_command, "--embedder_api_key", embedder_api_key)
        if embedder_dims is not None:
            append_arg(add_command, "--embedder_dims", str(embedder_dims))
        if exp_params.get("figure_view", False): add_command.append("--figure_view")
        if exp_params.get("is_graph", False): add_command.append("--is_graph")
        run_command(add_command)
        print("✅ Step 1 completed successfully.", flush=True)
    else:
        print(f"\n⏭️ Skipping Step 1: ADD MEMORIES. (Not required for '{technique_type}' or start_from_step > 1).", flush=True)

    if args.start_from_step <= 2:
        print("\n" + "#"*25 + " STEP 2: SEARCH MEMORIES " + "#"*25, flush=True)
        set_llm_env(search_llm_model, search_llm_base_url, search_llm_api_key)
        search_max_workers = (
            config.get("search_params", {}).get("max_workers")
            or exp_params.get("max_workers", 6)
        )
        search_command = [
            "python", "-u", runner_script,
            "--method", "search",
            "--dataset_name", dataset_name,
            "--output_folder", workspace_dir,
            "--technique_type", technique_type,
            "--mode", exp_params.get("mode", "no_client_async"),
            "--top_k", str(exp_params.get("top_k", 0)),
            "--embedder_model", embedder_model,
            "--qdrant_path", qdrant_path,
            "--workspace_dir", workspace_dir,
            "--search_mode", exp_params.get("search_mode", "0"),
            "--answer_mode", exp_params.get("answer_mode", "0"),
            "--max_workers", str(search_max_workers),
            "--collection_name", collection_name,
        ]
        append_arg(search_command, "--llm_model", search_llm_model)
        append_arg(search_command, "--llm_base_url", search_llm_base_url)
        append_arg(search_command, "--llm_api_key", search_llm_api_key)
        append_arg(search_command, "--search_llm_model", search_llm_model)
        append_arg(search_command, "--search_llm_base_url", search_llm_base_url)
        append_arg(search_command, "--search_llm_api_key", search_llm_api_key)
        append_arg(search_command, "--answer_llm_model", answer_llm_model)
        append_arg(search_command, "--answer_llm_base_url", answer_llm_base_url)
        append_arg(search_command, "--answer_llm_api_key", answer_llm_api_key)
        append_arg(search_command, "--embedder_base_url", embedder_base_url)
        append_arg(search_command, "--embedder_api_key", embedder_api_key)
        if embedder_dims is not None:
            append_arg(search_command, "--embedder_dims", str(embedder_dims))
        if exp_params.get("filter_memories", False): search_command.append("--filter_memories")
        if exp_params.get("is_graph", False): search_command.append("--is_graph")
        run_command(search_command)
        # rename the output file to include timestamp when needed
        if os.path.exists(search_results_raw_path):
            if search_results_raw_path != search_results_path:
                os.rename(search_results_raw_path, search_results_path)
                print(f"Renamed search results file to include timestamp:\n{search_results_path}")
            else:
                print(f"Search results file already stored with target name:\n{search_results_path}")
        elif os.path.exists(search_results_path):
            print(f"Search results file located at:\n{search_results_path}")
        else:
            print(f"⚠️ Expected search results file not found:\n  {search_results_raw_path}")
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
        append_arg(eval_command, "--evaluator_model", evaluator_model)
        append_arg(eval_command, "--evaluator_base_url", evaluator_base_url)
        append_arg(eval_command, "--evaluator_api_key", evaluator_api_key)
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
