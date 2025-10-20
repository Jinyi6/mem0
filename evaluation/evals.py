import argparse
import concurrent.futures
import json
from collections import defaultdict
import os
from tqdm import tqdm
import sys
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

# --- 环境设置 (保持不变) ---
os.environ["LOCAL_MEM0_PATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.environ['OPENAI_API_KEY'] = "sk-vyvftxtwuiznrwrfvayhfitxgpdpsykrdnukzfdtdwtjgqvo"
os.environ["OPENAI_BASE_URL"] = "https://api.siliconflow.cn/v1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["MEM0_TELEMETRY"] = "False"

from metrics.llm_judge import evaluate_llm_judge
from utils.result_merger import merge_memory_and_scores, save_json_file
# from metrics.utils import calculate_metrics

# --- 需求 3: 将原始 for 循环中的逻辑提取为独立的函数 ---
def process_single_item(item):
    """
    处理单个数据项的函数，用于提交给线程池。
    """
    gt_answer = str(item["answer"])
    pred_answer = str(item["response"])
    if "Final Answer:" in pred_answer:
        pred_answer = pred_answer.split("Final Answer:", 1)[1].strip()
    category = str(item["category"])
    question = str(item["question"])

    # 跳过指定类别
    if category == "5":
        return None  # 返回 None 以便主循环可以跳过它

    # metrics = calculate_metrics(pred_answer, gt_answer)
    llm_score = evaluate_llm_judge(question, gt_answer, pred_answer)

    # 返回处理结果字典
    return {
        "question": question,
        "answer": gt_answer,
        "response": pred_answer,
        "category": category,
        # "f1_score": metrics["f1"],
        "llm_score": llm_score,
        # "failure_mode": x x,
        # "reasons_summary": x x,
    }

def main():
    print("Starting evaluation...")
    parser = argparse.ArgumentParser(description="Evaluate RAG results")
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to the input JSON file with model responses."
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to save the evaluation results JSON file."
    )
    parser.add_argument("--max_workers", type=int, default=5, help="Maximum number of worker threads")
    parser.add_argument(
        "--combined_output_file",
        type=str,
        default=None,
        help="Optional path for the merged (memories + scores) JSON output. Defaults to <output_file>_combined.json",
    )

    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        data = json.load(f)

    print(f"文件读取和解析完成！共加载了 {len(data)} 组数据。")
    
    results = defaultdict(list)

    total_keys = len(data)
    with tqdm(total=total_keys, desc="Processing Keys") as key_pbar:
        for key, items_to_process in data.items():
            processed_items = []

            with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = [executor.submit(process_single_item, item) for item in items_to_process]

                progress_desc = f"Processing items in '{key}'"
                inner_total = len(futures)
                disable_inner = inner_total <= 1
                with tqdm(total=inner_total, desc=progress_desc, leave=False, disable=disable_inner) as inner_pbar:
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            result = future.result()
                            if result is not None:
                                processed_items.append(result)
                        except Exception as e:
                            print(f"处理 '{key}' 中的一个项目时发生错误: {e}")
                        finally:
                            if not disable_inner:
                                inner_pbar.update(1)

            results[key] = processed_items
            key_pbar.update(1)

    # 将结果保存到JSON文件
    results_dict = {str(k): v for k, v in results.items()}
    with open(args.output_file, "w") as f:
        json.dump(results_dict, f, indent=4)

    # --- 生成合并后的完整结果文件 ---
    combined_path = (
        args.combined_output_file
        if args.combined_output_file
        else f"{os.path.splitext(args.output_file)[0]}_combined.json"
    )
    print("评估完成，正在合并评估结果...")
    combined_results = merge_memory_and_scores(data, results_dict)
    save_json_file(combined_path, combined_results)

    print(f"\n所有处理完成！结果已保存到 {args.output_file}")
    print(f"合并后的完整结果文件已保存到 {combined_path}")


if __name__ == "__main__":
    main()
