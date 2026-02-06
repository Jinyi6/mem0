"""Evaluation Script - Rubric-based LLM Judge

Use an LLM as the judge to grade model outputs with binary scores (0/1) based on rubrics.

Input File:
    - JSONL: each line contains:
      {"idx": 0, "messages": [...], "model_output": "...", "ref_answer": "...", "rubrics": [...]}
    - JSON (dict or list): typical evaluation output from the pipeline

Output File:
    - JSONL for JSONL input
    - JSON for JSON input (mirrors evals.py style)
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple

from openai import OpenAI
from tqdm import tqdm

from utils.result_merger import merge_memory_and_scores, save_json_file


def get_timestamp():
    """Get current timestamp string."""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def log(message):
    """Print log message with timestamp."""
    print(f"[{get_timestamp()}] {message}")


def load_jsonl(file_path):
    """Load JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_input_file(file_path: str) -> Tuple[Any, str]:
    """
    Load input file that can be JSON or JSONL.

    Returns:
        data: parsed data
        fmt: "json" or "jsonl"
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data, "json"
    except json.JSONDecodeError:
        data = load_jsonl(file_path)
        return data, "jsonl"


def _normalize_model_output(text: str) -> str:
    if not text:
        return ""
    cleaned = text
    if "Final Answer:" in cleaned:
        cleaned = cleaned.split("Final Answer:", 1)[1].strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()
    return cleaned.strip()


def _normalize_rubrics(rubrics: Any) -> List[str]:
    if rubrics is None:
        return []
    if isinstance(rubrics, list):
        return rubrics
    if isinstance(rubrics, tuple):
        return list(rubrics)
    if isinstance(rubrics, dict):
        return [rubrics]
    text = str(rubrics).strip()
    return [text] if text else []


def append_jsonl(item, file_path):
    """Append a single record to JSONL file."""
    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_rubrics_text(rubrics):
    """Build rubrics checklist from rubrics list."""
    if not rubrics:
        return "No specific rubrics provided."
    
    lines = []
    for i, rubric in enumerate(rubrics, 1):
        if isinstance(rubric, dict):
            criteria = rubric.get("rubric_criteria", "").strip()
        else:
            criteria = str(rubric).strip()
        if criteria:
            lines.append(f"{i}. {criteria}")
    
    return "\n".join(lines) if lines else "No specific rubrics provided."


def _extract_json_block(text: str) -> str | None:
    if not text:
        return None
    # Fast path: looks like JSON
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    # Find a JSON object by bracket matching.
    in_string = False
    escape = False
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    return text[start : i + 1].strip()
    return None


def _coerce_status_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        text = value.strip()
        # Try to parse as JSON list
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
        # Fallback: split by commas
        parts = [p.strip() for p in text.split(",") if p.strip()]
        return parts
    return []


def _coerce_score(payload: dict) -> int:
    for key in ("Overall Score", "overall_score", "overall score", "score", "Score"):
        if key in payload:
            raw = payload.get(key)
            try:
                score = int(raw)
            except (TypeError, ValueError):
                score = 0
            return 1 if score == 1 else 0
    return 0


def _compute_rubric_ratio(status_list) -> float:
    if not status_list:
        return 0.0
    total = len(status_list)
    yes = 0
    for item in status_list:
        if isinstance(item, str):
            val = item.strip().lower()
            if val in {"yes", "y", "true", "1"}:
                yes += 1
        elif isinstance(item, (int, float, bool)):
            if bool(item):
                yes += 1
    return yes / total if total > 0 else 0.0


def call_judge_api(client, model, rubrics_text, model_output, max_retries=3, retry_delay=3):
    """
    Call judge model API for grading (only handles API call, returns raw text).
    
    Args:
        client: OpenAI client instance
        model: Judge model name
        rubrics_text: Formatted rubrics text
        model_output: Model's response to be graded
        max_retries: Maximum number of retries for API call
        retry_delay: Delay between retries (seconds)
    
    Returns:
        result_text: Raw response text from API, or None if failed
    """
    grading_prompt = (
        "Starting now, you are a rigorous instruction-following grading teacher. Your task is to accurately grade and score student answers based on the 【Rubrics】.\n\n"
        "Grading Criteria\n"
        "This is a strict, all-or-nothing grading system. The final score is binary.\n"
        "To receive a score of 1, the student's answer must perfectly satisfy every single requirement listed in the 【Rubrics】.\n"
        "If even one requirement is not fully met, the final score will be 0.\n"
        "Grading Process\n"
        "Please strictly follow the steps below for analysis—no steps may be skipped:\n"
        "Step 1: Analyze the Standard Answer\n"
        "List all explicit requirements in the 【Rubrics】 item by item (including format, content, quantity, order, etc.).\n"
        "Identify implicit requirements in the 【Rubrics】 (e.g., language style, logical structure).\n"
        "Define specific evaluation criteria for each requirement (e.g., \"must include X,\" \"must not exceed Y\").\n"
        "Step 2: Check Each Requirement Against the Student's Answer\n"
        "For every requirement in the 【Rubrics】, verify one by one whether the student's answer fully satisfies it.\n"
        "Step 3: Self-Reflection\n"
        "Before giving the final score, you must conduct the following checks:\n"
        "  Completeness Check: Whether all requirements in the standard answer have been reviewed with no omissions.\n"
        "  Strictness Check: Whether the evaluation strictly adheres to the \"fully satisfied\" standard without relaxing requirements due to subjective judgment.\n"
        "  Consistency Check: Whether the grading rationale aligns logically with the final score.\n"
        "  Objectivity Check: Whether judgments are based on objective facts rather than subjective speculation.\n"
        "Output Format Requirements\n"
        "【Grading Rationale】: xxx\n"
        "【List of Requirement Satisfaction Status】: [x₁, x₂, …, xᵢ, …, xₙ] (where n is the total number of requirements in the 【Rubrics】, and xᵢ indicates whether the student's answer meets the i-th requirement, with values \"yes\"/\"no\")\n"
        "【Overall Score】: x points (x is an integer, either 0 or 1.)\n\n"
        "Content to Be Graded\n"
        f"【Rubrics】:\n{rubrics_text}\n"
        f"【Student Response】:\n{model_output}\n"
        "\nPlease strictly output ONLY the following JSON format (do not output any other content):\n"
        "{\n"
        '  "Grading Rationale": "Your detailed grading rationale",\n'
        '  "List of Requirement Satisfaction Status": ["yes", "no", ...],\n'
        '  "Overall Score": 0 or 1\n'
        "}\n"
    )
    
    messages = [{"role": "user", "content": grading_prompt}]
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            result_text = response.choices[0].message.content.strip()
            
            # Remove code block wrapper if present
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.startswith("```"):
                result_text = result_text[3:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
            result_text = result_text.strip()
            
            return result_text
                
        except Exception as e:
            error_msg = str(e)
            if attempt < max_retries - 1:
                log(f"   ⚠️ API call failed (attempt {attempt + 1}/{max_retries}): {error_msg[:100]}")
                time.sleep(retry_delay)
            else:
                log(f"   ❌ API call failed after {max_retries} attempts: {error_msg[:100]}")
                return None
    
    return None


def process_single_item(args):
    """Process a single item for grading."""
    item, client, judge_model, max_retries, idx_fallback, output_style = args
    idx = item.get("idx", idx_fallback)

    model_output = item.get("response", item.get("model_output", ""))
    model_output = _normalize_model_output(str(model_output))
    rubrics = item.get("rubrics", None)
    if rubrics is None:
        rubrics = item.get("answer", item.get("ref_answer", []))
    rubrics = _normalize_rubrics(rubrics)
    
    # Skip if no model output
    if not model_output or not model_output.strip():
        base = {
            "grading_rationale": "No model output (counted as score 0)",
            "requirement_status": [],
            "score": 0,
            "requirement_ratio": 0.0,
        }
        if output_style == "jsonl":
            result = {**item, **base}
        else:
            result = {
                "question": item.get("question", ""),
                "answer": item.get("answer", item.get("ref_answer", "")),
                "response": item.get("response", item.get("model_output", "")),
                "category": item.get("category", ""),
                "rubric_score": 0,
                "rubric_rationale": base["grading_rationale"],
                "rubric_requirement_status": base["requirement_status"],
                "rubric_requirement_ratio": base["requirement_ratio"],
            }
        return idx, result, None  # None is no error
    
    # Build rubrics text
    rubrics_text = build_rubrics_text(rubrics)
    
    # JSON parsing retry logic (re-call API if JSON parsing fails)
    for parse_attempt in range(max_retries):
        # Call judge API
        grading_result = call_judge_api(
            client, judge_model, rubrics_text, model_output, max_retries
        )
        
        if not grading_result:
            log(f"   ❌ [idx={idx}] API call failed (attempt {parse_attempt + 1}/{max_retries})")
            if parse_attempt < max_retries - 1:
                log(f"      Waiting 2s before retry...")
                time.sleep(2)
                continue
            else:
                # All retries failed
                result = {
                    **item,
                    "grading_rationale": "API call failed (counted as score 0)",
                    "requirement_status": [],
                    "score": 0
                }
                return idx, result, "API call failed" # error
        
        # Try to parse JSON
        try:
            try:
                result_json = json.loads(grading_result)
            except json.JSONDecodeError:
                extracted = _extract_json_block(grading_result)
                if not extracted:
                    raise
                result_json = json.loads(extracted)
            
            # Validate required field
            if not isinstance(result_json, dict):
                raise ValueError("Result is not a JSON object")

            # Parse success
            score = _coerce_score(result_json)
            rationale = result_json.get("Grading Rationale", result_json.get("grading_rationale", ""))
            status_list = _coerce_status_list(
                result_json.get(
                    "List of Requirement Satisfaction Status",
                    result_json.get("requirement_status", []),
                )
            )
            ratio = _compute_rubric_ratio(status_list)
            if output_style == "jsonl":
                result = {
                    **item,
                    "grading_rationale": rationale,
                    "requirement_status": status_list,
                    "score": score,
                    "requirement_ratio": ratio,
                }
            else:
                result = {
                    "question": item.get("question", ""),
                    "answer": item.get("answer", item.get("ref_answer", "")),
                    "response": item.get("response", item.get("model_output", "")),
                    "category": item.get("category", ""),
                    "rubric_score": score,
                    "rubric_rationale": rationale,
                    "rubric_requirement_status": status_list,
                    "rubric_requirement_ratio": ratio,
                }
            return idx, result, None # None is no error
            
        except (json.JSONDecodeError, ValueError) as e:
            log(f"   ⚠️ [idx={idx}] JSON parse failed (attempt {parse_attempt + 1}/{max_retries}): {e}")
            log(f"      Raw response: {grading_result[:200]}...")
            
            if parse_attempt < max_retries - 1:
                log(f"      Waiting 2s before re-grading...")
                time.sleep(2)
            else:
                log(f"   ❌ [idx={idx}] JSON parse failed after {max_retries} attempts")
                base = {
                    "grading_rationale": f"JSON parse failed ({max_retries} attempts): {grading_result[:500]}",
                    "requirement_status": [],
                    "score": 0,
                    "requirement_ratio": 0.0,
                }
                if output_style == "jsonl":
                    result = {**item, **base}
                else:
                    result = {
                        "question": item.get("question", ""),
                        "answer": item.get("answer", item.get("ref_answer", "")),
                        "response": item.get("response", item.get("model_output", "")),
                        "category": item.get("category", ""),
                        "rubric_score": 0,
                        "rubric_rationale": base["grading_rationale"],
                        "rubric_requirement_status": base["requirement_status"],
                        "rubric_requirement_ratio": base["requirement_ratio"],
                    }
                return idx, result, f"JSON parse failed: {e}" # error
    
    # Should not reach here
    base = {
        "grading_rationale": "Unknown error (counted as score 0)",
        "requirement_status": [],
        "score": 0,
        "requirement_ratio": 0.0,
    }
    if output_style == "jsonl":
        result = {**item, **base}
    else:
        result = {
            "question": item.get("question", ""),
            "answer": item.get("answer", item.get("ref_answer", "")),
            "response": item.get("response", item.get("model_output", "")),
            "category": item.get("category", ""),
            "rubric_score": 0,
            "rubric_rationale": base["grading_rationale"],
            "rubric_requirement_status": base["requirement_status"],
            "rubric_requirement_ratio": base["requirement_ratio"],
        }
    return idx, result, "Unknown error"


def main():
    parser = argparse.ArgumentParser(description="Evaluation Script - Rubric-based LLM Judge")
    parser.add_argument("--input", "--input_file", dest="input", type=str, required=True, help="Input file path")
    parser.add_argument("--output", "--output_file", dest="output", type=str, default=None, help="Output file path")
    parser.add_argument("--judge-model", "--evaluator_model", dest="judge_model", type=str, default=None, help="Judge model name")
    parser.add_argument("--base-url", "--evaluator_base_url", dest="base_url", type=str, default=None, help="API Base URL (optional)")
    parser.add_argument("--api-key", "--evaluator_api_key", dest="api_key", type=str, default=None, help="API Key (optional)")
    parser.add_argument("--workers", "--max_workers", dest="workers", type=int, default=1, help="Number of concurrent workers")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries per item")
    parser.add_argument(
        "--combined_output_file",
        type=str,
        default=None,
        help="Optional path for the merged (memories + scores) JSON output. Defaults to <output>_combined.json",
    )
    parser.add_argument("--dataset_name", type=str, default=None, help="Optional dataset name (unused)")
    args = parser.parse_args()
    
    # Set output path
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"outputs/{base_name}_graded.jsonl"
    
    log("=" * 60)
    log("🎯 Evaluation Task")
    log("=" * 60)
    log(f"📥 Input file: {args.input}")
    log(f"📤 Output file: {args.output}")
    judge_model = args.judge_model or os.getenv("EVALUATOR_MODEL") or "gpt-5.1"
    log(f"🤖 Judge model: {judge_model}")
    log(f"⚡ Workers: {args.workers}")
    log("=" * 60)
    
    # Initialize OpenAI client
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        log("❌ Error: Please set OPENAI_API_KEY or use --api-key argument")
        return
    
    client_kwargs = {"api_key": api_key}
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL")
    if base_url:
        client_kwargs["base_url"] = base_url
        log(f"🔗 Using custom API: {base_url}")
    
    client = OpenAI(**client_kwargs)
    
    # Load data
    log("📖 Loading data...")
    data, input_format = load_input_file(args.input)
    output_style = "jsonl" if input_format == "jsonl" else "json"

    if input_format == "jsonl":
        log(f"   Total {len(data)} samples")

        # Check completed samples (resume from checkpoint)
        completed_indices = set()
        if os.path.exists(args.output):
            existing_data = load_jsonl(args.output)
            completed_indices = {item.get("idx") for item in existing_data if item.get("idx") is not None}
            log(f"📌 Found {len(completed_indices)} completed, resuming remaining")

        # Filter pending tasks
        pending_items = [item for item in data if item.get("idx") not in completed_indices]

        if not pending_items:
            log("✅ All samples already evaluated")
            calculate_statistics(args.output, output_style)
            return

        log(f"🚀 Starting evaluation ({len(pending_items)} pending)...")

        # Prepare tasks
        tasks = [
            (item, client, judge_model, args.max_retries, idx, output_style)
            for idx, item in enumerate(pending_items)
        ]

        # Statistics
        success_count = 0
        fail_count = 0

        if args.workers == 1:
            for task in tqdm(tasks, desc="Evaluating"):
                _, result, error = process_single_item(task)
                if error:
                    fail_count += 1
                else:
                    append_jsonl(result, args.output)
                    success_count += 1
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(process_single_item, task): task[0].get("idx") for task in tasks}
                with tqdm(total=len(tasks), desc="Evaluating") as pbar:
                    for future in as_completed(futures):
                        try:
                            _, result, error = future.result()
                            if error:
                                fail_count += 1
                            else:
                                append_jsonl(result, args.output)
                                success_count += 1
                        except Exception as e:
                            log(f"   ❌ Exception: {str(e)}")
                            fail_count += 1
                        pbar.update(1)

        log("=" * 60)
        log("✅ Evaluation completed!")
        log(f"   Success: {success_count}")
        log(f"   Failed: {fail_count}")
        log(f"   Output: {args.output}")
        calculate_statistics(args.output, output_style)
        return

    # JSON input (dict or list)
    if isinstance(data, dict):
        items_by_key = data
        memory_results = data
    elif isinstance(data, list):
        items_by_key = {"0": data}
        memory_results = {"0": data}
    else:
        log("❌ Unsupported JSON structure. Expected dict or list.")
        return

    total_items = sum(len(v) for v in items_by_key.values() if isinstance(v, list))
    log(f"   Total {total_items} samples")
    log(f"🚀 Starting evaluation ({total_items} total)...")

    results: Dict[str, List[Dict[str, Any]]] = {}
    success_count = 0
    fail_count = 0

    for key, items in items_by_key.items():
        if not isinstance(items, list):
            results[key] = []
            continue

        tasks = [
            (item, client, judge_model, args.max_retries, idx, output_style)
            for idx, item in enumerate(items)
        ]
        processed_items: List[Tuple[int, Dict[str, Any], str]] = []

        if args.workers == 1:
            for task in tqdm(tasks, desc=f"Evaluating {key}", leave=False):
                processed_items.append(process_single_item(task))
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(process_single_item, task) for task in tasks]
                with tqdm(total=len(tasks), desc=f"Evaluating {key}", leave=False) as pbar:
                    for future in as_completed(futures):
                        processed_items.append(future.result())
                        pbar.update(1)

        ordered: List[Dict[str, Any]] = [None] * len(items)
        for item_idx, result, error in processed_items:
            if error:
                fail_count += 1
            else:
                success_count += 1
            if 0 <= item_idx < len(ordered):
                ordered[item_idx] = result
        results[key] = [item for item in ordered if item is not None]

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    combined_path = (
        args.combined_output_file
        if args.combined_output_file
        else f"{os.path.splitext(args.output)[0]}_combined.json"
    )
    try:
        combined_results = merge_memory_and_scores(memory_results, results)
        save_json_file(combined_path, combined_results)
        log(f"🧩 Combined output saved to: {combined_path}")
    except Exception as exc:
        log(f"⚠️ Combined output failed: {exc}")

    log("=" * 60)
    log("✅ Evaluation completed!")
    log(f"   Success: {success_count}")
    log(f"   Failed: {fail_count}")
    log(f"   Output: {args.output}")
    calculate_statistics(args.output, output_style)


def calculate_statistics(output_path, output_style):
    """Calculate and display final statistics."""
    if not os.path.exists(output_path):
        return

    if output_style == "jsonl":
        data = load_jsonl(output_path)
        items = data
    else:
        with open(output_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            items = []
            for value in payload.values():
                if isinstance(value, list):
                    items.extend(value)
        elif isinstance(payload, list):
            items = payload
        else:
            items = []

    total = len(items)
    score_0 = sum(1 for item in items if item.get("rubric_score", item.get("score")) == 0)
    score_1 = sum(1 for item in items if item.get("rubric_score", item.get("score")) == 1)
    
    log("\n📊 Final Statistics:")
    log(f"   Total samples: {total}")
    log(f"   Score 0: {score_0}")
    log(f"   Score 1: {score_1}")
    
    if total > 0:
        solving_rate = score_1 / total
        log(f"\n📈 Solving Rate: {solving_rate:.4f} ({score_1}/{total})")
    
    log("=" * 60)


if __name__ == "__main__":
    main()
