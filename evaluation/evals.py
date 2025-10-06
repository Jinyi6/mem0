import argparse
import concurrent.futures
import json
from collections import defaultdict
import os
from tqdm import tqdm

# --- 环境设置 (保持不变) ---
os.environ["LOCAL_MEM0_PATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['OPENAI_API_KEY'] = "sk-vyvftxtwuiznrwrfvayhfitxgpdpsykrdnukzfdtdwtjgqvo"
os.environ["OPENAI_BASE_URL"] = "https://api.siliconflow.cn/v1"
os.environ["BASE_MODEL"] = "Qwen/Qwen3-14B"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from metrics.llm_judge import evaluate_llm_judge
# from metrics.utils import calculate_metrics

# --- 需求 3: 将原始 for 循环中的逻辑提取为独立的函数 ---
def process_single_item(item):
    """
    处理单个数据项的函数，用于提交给线程池。
    """
    gt_answer = str(item["answer"])
    pred_answer = str(item["response"])
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
    }

def main():
    print("Starting evaluation...")
    parser = argparse.ArgumentParser(description="Evaluate RAG results")
    parser.add_argument(
        "--input_file", type=str, default="/Users/jinyi/Documents/code/memory/mem0/exp_data/locomo/exp_data_0-10_2000/mem0_results_top_30_filter_False_graph_False.json", help="Path to the input dataset file"
    )
    parser.add_argument(
        "--output_file", type=str, default="/Users/jinyi/Documents/code/memory/mem0/exp_data/locomo/exp_data_0-10_2000/all_result.json", help="Path to save the evaluation results"
    )
    parser.add_argument("--max_workers", type=int, default=5, help="Maximum number of worker threads")

    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        data = json.load(f)

    print(f"文件读取和解析完成！共加载了 {len(data)} 组数据。")
    
    results = defaultdict(list)

    # --- 需求 1: 外层循环串行，并添加进度条 ---
    # tqdm 的外层描述设为 "Processing Keys"
    for key, items_to_process in tqdm(data.items(), desc="Processing Keys"):
        
        processed_items = []
        # --- 需求 2: 在处理某一组数据时，并行处理其内部所有 data_item ---
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # 提交所有任务
            futures = [executor.submit(process_single_item, item) for item in items_to_process]
            
            # 使用 tqdm 显示内部处理进度
            # leave=False 表示这个内层进度条在完成后会消失，避免刷屏
            progress_desc = f"Processing items in '{key}'"
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=progress_desc, leave=False):
                try:
                    result = future.result()
                    # 只有当任务成功返回有效结果时才添加
                    if result is not None:
                        processed_items.append(result)
                except Exception as e:
                    print(f"处理 '{key}' 中的一个项目时发生错误: {e}")

        results[key] = processed_items

    # 将结果保存到JSON文件
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n所有处理完成！结果已保存到 {args.output_file}")


if __name__ == "__main__":
    main()