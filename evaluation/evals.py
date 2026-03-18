import argparse
import concurrent.futures
import json
import os
import re
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
from tqdm import tqdm
import sys
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

# --- 环境设置 (保持不变) ---
os.environ["LOCAL_MEM0_PATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# os.environ['OPENAI_API_KEY'] = "sk-vyvftxtwuiznrwrfvayhfitxgpdpsykrdnukzfdtdwtjgqvo"
# os.environ["OPENAI_BASE_URL"] = "https://api.siliconflow.cn/v1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["MEM0_TELEMETRY"] = "False"

from utils.result_merger import merge_memory_and_scores, save_json_file

DATASET_FIXED_PATH = Path(__file__).resolve().parent / "dataset" / "locomo10_fixed.json"


@contextmanager
def _thread_pool(max_workers: int):
    """
    ThreadPoolExecutor wrapper that cancels pending futures during shutdown to avoid thread leakage.
    """
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="eval-worker")
    try:
        yield executor
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


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


def _evaluate_candidates(
    question: str,
    prediction: str,
    candidates: Sequence[str],
    enabled_metrics: Set[str],
    metric_helpers: Dict[str, object],
) -> Tuple[str, Dict[str, float]]:
    """Compute metrics for a question/candidate set and select the best answer."""
    
    f1_func = metric_helpers.get("f1")
    bleu_func = metric_helpers.get("bleu")
    llm_submit = metric_helpers.get("llm_submit")

    if candidates is None or candidates == "" or candidates == [] or all(candidate == "" for candidate in candidates):
        metrics_summary: Dict[str, float] = {}
        if "llm" in enabled_metrics:
            metrics_summary["llm_score"] = 0.0
        if "f1" in enabled_metrics:
            metrics_summary["f1_score"] = 0.0
        if "bleu" in enabled_metrics:
            metrics_summary["bleu_score"] = 0.0

        return "skip", metrics_summary

    best_answer = ""
    selection_metric = "llm" if "llm" in enabled_metrics else (
        "f1" if "f1" in enabled_metrics else ("bleu" if "bleu" in enabled_metrics else None)
    )
    best_selection_value: float = float("-inf") if selection_metric == "llm" else -1.0

    max_f1 = 0.0
    max_bleu1 = 0.0
    best_llm_score = float("-inf")

    candidate_infos: List[Dict[str, object]] = []
    llm_future_map: Dict[concurrent.futures.Future, Tuple[str, Dict[str, float]]] = {}

    for candidate in candidates:
        candidate_str = str(candidate)
        candidate_metrics: Dict[str, float] = {}

        try:
            if "f1" in enabled_metrics and f1_func:
                metric_values = f1_func(prediction, candidate_str)
                f1_score = float(metric_values.get("f1", 0.0) or 0.0)
                candidate_metrics["f1"] = f1_score
                max_f1 = max(max_f1, f1_score)

            if "bleu" in enabled_metrics and bleu_func:
                bleu_values = bleu_func(prediction, candidate_str)
                bleu_score = float(bleu_values.get("bleu1", 0.0) or 0.0)
                candidate_metrics["bleu"] = bleu_score
                max_bleu1 = max(max_bleu1, bleu_score)

            if "llm" in enabled_metrics:
                if not llm_submit:
                    raise RuntimeError("llm metric enabled but LLM judge is unavailable (import failed).")
                future = llm_submit(question, candidate_str, prediction)
                llm_future_map[future] = (candidate_str, candidate_metrics)
        except Exception as exc:
            print(
                f"[Eval Error] Candidate evaluation failed for question '{question[:80]}...' "
                f"with candidate '{candidate_str[:80]}': {exc}",
                flush=True,
            )
            continue

        candidate_infos.append({"candidate": candidate_str, "metrics": candidate_metrics})

    if "llm" in enabled_metrics and llm_future_map:
        for future in concurrent.futures.as_completed(llm_future_map.keys()):
            candidate_str, candidate_metrics = llm_future_map[future]
            try:
                llm_score = float(future.result())
            except Exception as exc:
                print(
                    f"[Eval Error] LLM judge failed for question '{question[:80]}...' "
                    f"with candidate '{candidate_str[:80]}': {exc}",
                    flush=True,
                )
                llm_score = 0.0
            candidate_metrics["llm"] = llm_score
            best_llm_score = max(best_llm_score, llm_score)

    if candidate_infos:
        for info in candidate_infos:
            candidate_value = None
            metrics = info["metrics"]
            if selection_metric is None:
                if best_answer == "":
                    best_answer = info["candidate"]
                continue

            candidate_value = metrics.get(selection_metric)
            if candidate_value is None:
                candidate_value = float("-inf") if selection_metric == "llm" else 0.0

            if best_answer == "" or candidate_value > best_selection_value:
                best_answer = info["candidate"]
                best_selection_value = candidate_value

    metrics_summary: Dict[str, float] = {}
    if "llm" in enabled_metrics:
        metrics_summary["llm_score"] = best_llm_score if best_llm_score != float("-inf") else 0.0
    if "f1" in enabled_metrics:
        metrics_summary["f1_score"] = max_f1
    if "bleu" in enabled_metrics:
        metrics_summary["bleu_score"] = max_bleu1

    if best_answer == "":
        best_answer = str(candidates[0]) if candidates else ""

    return best_answer, metrics_summary


# --- 需求 3 新增: MCQ 解析与评估 ---

def _parse_mcq_pred_answers(text: str) -> Tuple[Set[str], bool]:
    """
    从文本中提取 MCQ 选项（支持含有空格及选项内容的形式）：
    - (A) / (b) / ( A ) / [ b ]
    - ( A. He becomes more skeptical. ) / [ B: Some text ]

    返回: (options_set, malformed)
    - options_set: 提取到的选项字母集合（统一为大写），未提取到则为空集合
    - malformed: 是否检测到不允许的紧邻形式，如 (A)(B) 或 [ A ][ B ]（无分隔符，直接判错）
    """
    text = str(text).strip()

    # 额外支持：如果模型输出“有且只有字母”，且长度 < 6，则视为选项输出。
    if re.fullmatch(r"[A-Fa-f]{1,5}", text):
        return {ch.upper() for ch in text}, False

    # 不允许紧邻形式：支持判定内部带有空格的情况，如 "( A )( B )" 或 "[A][ B ]"（无分隔符）
    if re.search(r"\(\s*[A-Fa-f]\s*\)\(\s*[A-Fa-f]\s*\)", text) or \
       re.search(r"\[\s*[A-Fa-f]\s*\]\[\s*[A-Fa-f]\s*\]", text):
        return set(), True

    options: Set[str] = set()

    token_re = re.compile(r"\([^)]*\)|\[[^\]]*\]")
    for match in token_re.finditer(text):
        token = match.group(0)
        inner = token[1:-1].strip()
        if not inner:
            continue

        single_letter_match = re.fullmatch(r"([A-Fa-f])", inner)
        if single_letter_match:
            options.add(single_letter_match.group(1).upper())
            continue

        # 允许带 label 的完整选项文本，但要求 label 后有明确分隔符，
        # 避免把 "(B-side version)" / "[C-suite leader]" 之类普通短语误判为选项。
        labeled_text_match = re.match(r"^([A-Fa-f])\s*[.:]\s*.+$", inner)
        if labeled_text_match:
            options.add(labeled_text_match.group(1).upper())
            continue

        # 对很短的括号内容做一个保守兜底：若内容只含字母/空格且总字母数 <= 5，
        # 且首字母是 A-F，则按首字母解析。
        letters_only = re.sub(r"[^A-Za-z]", "", inner)
        if (
            letters_only
            and len(letters_only) <= 5
            and inner[0].upper() in {"A", "B", "C", "D", "E", "F"}
            and re.fullmatch(r"[A-Za-z ]+", inner)
        ):
            options.add(inner[0].upper())

    return options, False

def _parse_mcq_gt_answers(text: str, *, allow_prefixed_option_text: bool = False) -> Set[str]:
    """
    解析 Ground Truth 的 MCQ 答案。

    兼容多类输入：
    - 裸字母: "A" / "c"
    - 字母+文本: "D.He was..." 或 "D. He was..." (仅在 allow_prefixed_option_text=True 时启用)
    - 括号字母: "(A)" / "[c]"（会复用 pred 的解析规则，但忽略 malformed 标记）
    """
    raw = str(text).strip()
    if not raw:
        return set()

    if allow_prefixed_option_text:
        text_match = re.match(r"^([A-Fa-f])\.", raw)
        if text_match:
            return {text_match.group(1).upper()}

    # 纯字母（单选或多选：逗号/空格分隔），且不得包含其他字符
    if re.fullmatch(r"[A-Fa-f](?:[,\s]+[A-Fa-f])*", raw):
        return {token.upper() for token in re.split(r"[,\s]+", raw) if token}

    extracted, _malformed = _parse_mcq_pred_answers(raw)
    return extracted

def print_metrics_summary(results_dict: Dict[str, List[Dict]], output_path: str):
    """
    打印按 Category 分组的指标统计摘要（仿照 generate_scores.py），保存到文件。
    """
    try:
        with open(output_path, "w") as f:
            f.write("\n" + "="*40 + "\n")
            f.write("       METRICS SUMMARY PER CATEGORY\n")
            f.write("="*40 + "\n")
            
            # 展平所有结果
            all_items = []
            for key, items in results_dict.items():
                all_items.extend(items)
                
            if not all_items:
                f.write("No results to summarize.\n")
                return

            # 按 category 分组
            grouped = defaultdict(list)
            for item in all_items:
                cat = _coerce_numeric_key(item.get("category", "Unknown"))
                grouped[cat].append(item)
                
            # 定义我们要统计的指标
            metrics_of_interest = ["llm_score", "f1_score", "bleu_score", "mcq_score"]
            
            # 表头
            headers = ["Category", "Count"] + [m.replace("_score", "").upper() for m in metrics_of_interest]
            header_str = f"{headers[0]:<10} | {headers[1]:<6} | " + " | ".join(f"{h:<8}" for h in headers[2:])
            f.write(header_str + "\n")
            f.write("-" * 80 + "\n")
            
            # 计算并打印
            # 排序：尝试按数字排序
            try:
                sorted_cats = sorted(grouped.keys(), key=lambda x: float(x))
            except:
                sorted_cats = sorted(grouped.keys(), key=str)
                
            overall_sums = defaultdict(float)
            overall_counts = defaultdict(int)
            
            for cat in sorted_cats:
                items = grouped[cat]
                count = len(items)
                row_str = f"{str(cat):<10} | {count:<6}"
                
                for metric in metrics_of_interest:
                    # 提取该指标的所有非 None 值
                    vals = [item.get(metric) for item in items if item.get(metric) is not None]
                    if vals:
                        mean_val = sum(vals) / len(vals)
                        row_str += f" | {mean_val:>8.4f}"
                        
                        overall_sums[metric] += sum(vals)
                        overall_counts[metric] += len(vals)
                    else:
                        row_str += f" | {'-':>8}"
                f.write(row_str + "\n")
                
            f.write("-" * 80 + "\n")
            
            # 打印 Overall
            overall_str = f"{'ALL':<10} | {len(all_items):<6}"
            for metric in metrics_of_interest:
                total_val = overall_sums[metric]
                total_cnt = overall_counts[metric]
                if total_cnt > 0:
                    avg = total_val / total_cnt
                    overall_str += f" | {avg:>8.4f}"
                else:
                    overall_str += f" | {'-':>8}"
            f.write(overall_str + "\n")
            f.write("="*40 + "\n")
        
        print(f"统计摘要已保存到 {output_path}")

    except Exception as e:
        print(f"⚠️ 无法写入统计摘要到文件: {e}")

def process_single_item(
    item: Dict[str, object],
    enabled_metrics: Set[str],
    metric_helpers: Dict[str, object],
):
    """
    处理单个数据项的函数，用于提交给线程池。
    """
    answer_candidates = _normalize_answer_candidates(item.get("answer_fixed"), item.get("answer"))
    pred_answer = str(item["response"])
    if "Final Answer:" in pred_answer:
        pred_answer = pred_answer.split("Final Answer:", 1)[1].strip()
    if "</think>" in pred_answer:
        pred_answer = pred_answer.split("</think>", 1)[1].strip()

    category = str(item["category"])
    question = str(item["question"])

    metrics_summary = {}
    best_answer = ""

    # --- MCQ 评估路径（只要启用了 mcq 就计算 mcq_score） ---
    if "mcq" in enabled_metrics:
        pred_options, pred_malformed = _parse_mcq_pred_answers(pred_answer)
        is_correct = False
        matched_candidate = ""
        if not pred_malformed:
            for cand in answer_candidates:
                gt_options = _parse_mcq_gt_answers(str(cand), allow_prefixed_option_text=True)
                if not gt_options:
                    continue
                if pred_options == gt_options:
                    is_correct = True
                    matched_candidate = str(cand)
                    break
        metrics_summary["mcq_score"] = 1.0 if is_correct else 0.0
        if matched_candidate:
            best_answer = matched_candidate

        # 若只启用 mcq，则不走其他指标（保持与用户预期一致）
        if enabled_metrics == {"mcq"} and not best_answer:
            best_answer = str(answer_candidates[0]) if answer_candidates else ""

    # --- 原有评估路径 (LLM, F1, BLEU) ---
    if any(m in enabled_metrics for m in ["llm", "f1", "bleu"]):
        try:
            selected_answer, metrics_res = _evaluate_candidates(
                question,
                pred_answer,
                answer_candidates,
                enabled_metrics,
                metric_helpers,
            )
            metrics_summary.update(metrics_res)
            if not best_answer:
                best_answer = selected_answer
        except Exception as exc:
            print(
                f"❌ Failed to evaluate question '{question[:80]}...' due to: {exc}. "
                f"Candidates: {answer_candidates}"
            , flush=True)
            raise

    if not best_answer:
        best_answer = str(answer_candidates[0]) if answer_candidates else ""

    # 返回处理结果字典
    result = {
        "question": question,
        "answer": best_answer,
        "response": str(item["response"]),
        "category": category,
    }
    for metric_name, value in metrics_summary.items():
        result[metric_name] = value

    return result

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
        "--metrics",
        type=str,
        nargs="+",
        default=["llm"],
        help="Metrics to compute (choices: llm, f1, bleu, mcq, all). Default uses llm only.",
    )
    parser.add_argument(
        "--combined_output_file",
        type=str,
        default=None,
        help="Optional path for the merged (memories + scores) JSON output. Defaults to <output_file>_combined.json",
    )
    parser.add_argument("--evaluator_model", type=str, default=None, help="Model name for evaluator LLM")
    parser.add_argument("--evaluator_base_url", type=str, default=None, help="Base URL for evaluator LLM API")
    parser.add_argument("--evaluator_api_key", type=str, default=None, help="API key for evaluator LLM provider")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Optional dataset name. When it contains 'locomo', enables legacy answer_fixed backfill.",
    )

    args = parser.parse_args()

    metric_options = {metric.lower() for metric in args.metrics}
    available_metrics = {"llm", "f1", "bleu", "mcq"}
    if "all" in metric_options:
        metric_options.update(available_metrics)
        metric_options.discard("all")

    unknown_metrics = metric_options - available_metrics
    if unknown_metrics:
        parser.error(
            "Unknown metric(s) requested: " + ", ".join(sorted(unknown_metrics))
        )

    if not metric_options:
        metric_options = {"llm"}

    enabled_metrics: Set[str] = set(metric_options)
    metric_helpers: Dict[str, object] = {}

    if "f1" in enabled_metrics:
        from metrics.utils import calculate_metrics as _calculate_metrics

        metric_helpers["f1"] = _calculate_metrics
    if "bleu" in enabled_metrics:
        from metrics.utils import calculate_bleu_scores as _calculate_bleu_scores

        metric_helpers["bleu"] = _calculate_bleu_scores

    if "llm" in enabled_metrics:
        try:
            from metrics.llm_judge_v5 import configure_evaluator, submit_llm_judge
        except Exception as exc:
            raise RuntimeError(
                "Failed to import LLM judge dependencies (required when using --metrics llm). "
                "Either install missing dependencies or drop 'llm' from --metrics."
            ) from exc

        configure_evaluator(
            model=args.evaluator_model,
            base_url=args.evaluator_base_url,
            api_key=args.evaluator_api_key,
        )
        metric_helpers["llm_submit"] = submit_llm_judge
    else:
        print("Skipping LLM evaluator configuration (llm metric disabled).", flush=True)

    enabled_metrics_frozen = frozenset(enabled_metrics)

    print(
        "Active metrics: " + ", ".join(sorted(enabled_metrics_frozen)) or "none",
        flush=True,
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

    dataset_name_lower = (args.dataset_name or "").lower()
    is_locomo_dataset = "locomo" in dataset_name_lower

    if needs_answer_fixed and is_locomo_dataset:
        print("🔄 Detected legacy locomo evaluation file without 'answer_fixed'. Enriching from fixed dataset...")
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
                with _thread_pool(args.max_workers) as executor:
                    futures = []
                    for item_idx, item in enumerate(items_to_process, start=1):
                        future = executor.submit(
                            process_single_item,
                            item,
                            enabled_metrics_frozen,
                            metric_helpers,
                        )
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

    # 打印统计摘要
    # 打印统计摘要
    log_path = os.path.join(os.path.dirname(combined_path), "final_result.log")
    print_metrics_summary(results, log_path)

    print(f"\n所有处理完成！结果已保存到 {args.output_file}")
    print(f"合并后的完整结果文件已保存到 {combined_path}")


if __name__ == "__main__":
    main()
