import argparse
import concurrent.futures
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
from tqdm import tqdm
import sys
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

# --- 环境设置 (保持不变) ---
os.environ["LOCAL_MEM0_PATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# os.environ['OPENAI_API_KEY'] = "sk-vyvftxtwuiznrwrfvayhfitxgpdpsykrdnukzfdtdwtjgqvo"
# os.environ["OPENAI_BASE_URL"] = "https://api.siliconflow.cn/v1"
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["MEM0_TELEMETRY"] = "False"

from metrics.llm_judge_v5 import evaluate_llm_judge, configure_evaluator
from utils.result_merger import merge_memory_and_scores, save_json_file
from metrics.utils import calculate_bleu_scores, calculate_metrics

DATASET_FIXED_PATH = Path(__file__).resolve().parent / "dataset" / "locomo10_fixed.json"


def _coerce_numeric_key(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _iter_dataset_questions(dataset: Sequence[Dict[str, object]]) -> Iterable[Dict[str, object]]:
    for item in dataset or []:
        qa_section = item.get("qa", [])
        if isinstance(qa_section, dict):
            ordered_keys = sorted(qa_section.keys(), key=_coerce_numeric_key)
            entries = [qa_section[key] for key in ordered_keys]
        elif isinstance(qa_section, list):
            entries = qa_section
        else:
            entries = []
        for qa_item in entries or []:
            if isinstance(qa_item, dict):
                yield qa_item


def load_answer_lookup(dataset_path: Path) -> Dict[str, List[str]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Fixed dataset file not found: {dataset_path}")
    with dataset_path.open("r") as f:
        dataset = json.load(f)
    lookup: Dict[str, List[str]] = {}
    for qa_item in _iter_dataset_questions(dataset):
        question = qa_item.get("question")
        if not question:
            continue
        answer_fixed = qa_item.get("answer_fixed")
        if answer_fixed is None:
            continue
        if isinstance(answer_fixed, list):
            candidates = [str(candidate) for candidate in answer_fixed if candidate not in (None, "")]
        else:
            candidates = [str(answer_fixed)]
        if candidates:
            lookup[question] = candidates
    return lookup


def _normalize_answer_candidates(raw_candidates, fallback) -> List[str]:
    candidates: List[str] = []
    if isinstance(raw_candidates, list):
        candidates.extend(str(candidate) for candidate in raw_candidates if candidate not in (None, ""))
    elif raw_candidates not in (None, ""):
        candidates.append(str(raw_candidates))

    if not candidates and fallback not in (None, ""):
        candidates.append(str(fallback))

    if not candidates:
        candidates.append("")
    return candidates


def _evaluate_candidates(question: str, prediction: str, candidates: Sequence[str]) -> tuple[str, float, float, float]:
    best_answer = ""
    best_llm_score = float("-inf")
    max_f1 = 0.0
    max_bleu1 = 0.0

    total = len(candidates)
    for idx, candidate in enumerate(candidates, start=1):
        candidate_str = str(candidate)
        try:
            metrics = calculate_metrics(prediction, candidate_str)
            bleu_scores = calculate_bleu_scores(prediction, candidate_str)
            llm_score = evaluate_llm_judge(question, candidate_str, prediction)
        except Exception as exc:
            print(
                f"[Eval Error] Candidate evaluation failed for question '{question[:80]}...' "
                f"with candidate '{candidate_str[:80]}': {exc}"
            , flush=True)
            continue

        max_f1 = max(max_f1, metrics.get("f1", 0.0))
        max_bleu1 = max(max_bleu1, bleu_scores.get("bleu1", 0.0))

        if llm_score > best_llm_score:
            best_llm_score = llm_score
            best_answer = candidate_str

    if best_llm_score == float("-inf"):
        best_llm_score = 0.0
    return best_answer, max_f1, max_bleu1, best_llm_score


# --- 需求 3: 将原始 for 循环中的逻辑提取为独立的函数 ---
def process_single_item(item):
    """
    处理单个数据项的函数，用于提交给线程池。
    """
    answer_candidates = _normalize_answer_candidates(item.get("answer_fixed"), item.get("answer"))
    pred_answer = str(item["response"])
    if "Final Answer:" in pred_answer:
        pred_answer = pred_answer.split("Final Answer:", 1)[1].strip()
    category = str(item["category"])
    question = str(item["question"])

    # 跳过指定类别
    if category == "5":
        return None  # 返回 None 以便主循环可以跳过它

    if all(candidate == "" for candidate in answer_candidates):
        print(f"⚠️ Question '{question[:80]}...' has empty gold answers. Proceeding with empty string fallback.", flush=True)

    try:
        best_answer, max_f1, max_bleu1, best_llm_score = _evaluate_candidates(
            question, pred_answer, answer_candidates
        )
    except Exception as exc:
        print(
            f"❌ Failed to evaluate question '{question[:80]}...' due to: {exc}. "
            f"Candidates: {answer_candidates}"
        , flush=True)
        raise

    # 返回处理结果字典
    return {
        "question": question,
        "answer": best_answer,
        "response": str(item["response"]),
        "category": category,
        "f1_score": max_f1,
        "bleu_score": max_bleu1,
        "llm_score": best_llm_score,
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
    parser.add_argument("--evaluator_model", type=str, default=None, help="Model name for evaluator LLM")
    parser.add_argument("--evaluator_base_url", type=str, default=None, help="Base URL for evaluator LLM API")
    parser.add_argument("--evaluator_api_key", type=str, default=None, help="API key for evaluator LLM provider")

    args = parser.parse_args()

    configure_evaluator(
        model=args.evaluator_model,
        base_url=args.evaluator_base_url,
        api_key=args.evaluator_api_key,
    )

    with open(args.input_file, "r") as f:
        data = json.load(f)

    needs_answer_fixed = False
    for items in data.values():
        for entry in items:
            if not isinstance(entry, dict):
                continue
            if "answer_fixed" not in entry or entry["answer_fixed"] in (None, [], ""):
                needs_answer_fixed = True
                break
        if needs_answer_fixed:
            break

    if needs_answer_fixed:
        print("🔄 Detected legacy evaluation file without 'answer_fixed'. Enriching from fixed dataset...")
        answer_lookup = load_answer_lookup(DATASET_FIXED_PATH)
        missing_questions = set()
        for items in data.values():
            for entry in items:
                if not isinstance(entry, dict):
                    continue
                if "answer_fixed" in entry and entry["answer_fixed"]:
                    continue
                question = entry.get("question")
                candidates = answer_lookup.get(question, [])
                if candidates:
                    entry["answer_fixed"] = candidates.copy()
                else:
                    missing_questions.add(question)
        with open(args.input_file, "w") as f:
            json.dump(data, f, indent=4)
        if missing_questions:
            print(f"⚠️  Warning: {len(missing_questions)} questions not found in fixed dataset; falling back to original answers for those entries.")
            if len(missing_questions) <= 5:
                for q in missing_questions:
                    print(f"   - {q}")
        print("✅ Evaluation file updated with 'answer_fixed'. Proceeding with metrics computation.")

    print(f"文件读取和解析完成！共加载了 {len(data)} 组数据。")
    
    results = defaultdict(list)

    total_keys = len(data)
    with tqdm(total=total_keys, desc="Processing Keys") as key_pbar:
        for key, items_to_process in data.items():
            processed_items = []

            try:
                inner_total = len(items_to_process)
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                    futures = []
                    for item_idx, item in enumerate(items_to_process, start=1):
                        future = executor.submit(process_single_item, item)
                        futures.append(future)
                        # No verbose progress unless debugging is needed

                    progress_desc = f"Processing items in '{key}'"
                    disable_inner = inner_total <= 1
                    with tqdm(total=inner_total, desc=progress_desc, leave=False, disable=disable_inner) as inner_pbar:
                        for future in concurrent.futures.as_completed(futures):
                            try:
                                result = future.result()
                                if result is not None:
                                    processed_items.append(result)
                            except KeyboardInterrupt:
                                print("\n⚠️ Evaluation interrupted by user keyboard input. Cancelling pending tasks...", flush=True)
                                executor.shutdown(wait=False, cancel_futures=True)
                                raise
                            except Exception as e:
                                print(f"处理 '{key}' 中的一个项目时发生错误: {e}", flush=True)
                            finally:
                                if not disable_inner:
                                    inner_pbar.update(1)
            except KeyboardInterrupt:
                print("⛔ KeyboardInterrupt received. Stopping evaluation loop gracefully.", flush=True)
                raise

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
