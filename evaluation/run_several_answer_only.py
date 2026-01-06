"""
Re-run the answer-only stage for a list of finished experiments and evaluate the outputs.

The script reads each experiment result JSON, replays every stored `answer_prompt`
with the configured answer LLM (in parallel), writes a new result file, and then
invokes `evals.py` to score the refreshed answers.
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from openai import OpenAI
from tqdm import tqdm

# ---------------------------------------------------------------------------
# LLM + experiment configurations (edit these to match your environment)
# ---------------------------------------------------------------------------

ANSWER_LLM_CONFIG = {
    "model": "Qwen/Qwen3-14B",
    "base_url": "https://api.siliconflow.cn/v1",
    "api_key": "sk-kuyhjxvgozardrjxhcmahdydxyyassjgyyxuzviivnpnrrul",
}

EVALUATOR_LLM_CONFIG = {
    "model": "Qwen/Qwen3-14B",
    "base_url": "https://api.siliconflow.cn/v1",
    "api_key": "sk-kuyhjxvgozardrjxhcmahdydxyyassjgyyxuzviivnpnrrul",
}

# Provide the list of experiment result files (relative to this script or absolute paths).
EXPERIMENT_RESULT_FILES = [
    "longmemeval_d0_model_Qwen_Qwen3_14B_top_k_30_filter_False_graph_False_0_0_0_0_20251031_015142/mem0_longmemeval_d0_results_top_30_filter_False_graph_False_20251031_015142_0_0_0_0.json"
]

DEFAULT_ANSWER_MAX_WORKERS = 8
DEFAULT_EVAL_MAX_WORKERS = 8
MAX_ANSWER_RETRIES = 8

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_SCRIPT_PATH = Path(__file__).resolve().with_name("evals.py")

os.environ.setdefault("LOCAL_MEM0_PATH", str(PROJECT_ROOT))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("MEM0_TELEMETRY", "False")


def _sanitize_model_tag(model_name: str) -> str:
    clean = re.sub(r"[^0-9a-zA-Z_]+", "_", model_name or "answer").strip("_")
    return clean or "answer"


def _extract_time_tag(path: Path) -> str:
    """
    Try to reuse the timestamp segment embedded in the original filename.
    Fallback to the entire stem when no timestamp is found.
    """
    stem = path.stem
    match = re.search(r"(\d{8}_\d{6}.*)$", stem)
    if match:
        return match.group(1)
    return stem


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path


def _build_openai_client(config: Dict[str, str]) -> OpenAI:
    kwargs = {}
    base_url = (config.get("base_url") or "").strip()
    api_key = (config.get("api_key") or "").strip()
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key
    return OpenAI(**kwargs)


def _call_answer_llm(
    client: OpenAI,
    model_name: str,
    answer_prompt: str,
    question_preview: str,
) -> Tuple[str, float]:
    """Send a single prompt to the answer LLM with retry/backoff."""
    attempt = 0
    total_sleep = 0.0
    last_error = None
    while attempt < MAX_ANSWER_RETRIES:
        attempt += 1
        start = time.time()
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": answer_prompt}],
                temperature=0.0,
            )
            duration = time.time() - start
            content = (response.choices[0].message.content or "").strip()
            if not content:
                raise ValueError("LLM returned empty content.")
            return content, duration
        except Exception as exc:  # noqa: BLE001 - want the exact error message
            last_error = exc
            if attempt >= MAX_ANSWER_RETRIES:
                break
            error_str = str(exc).lower()
            if any(keyword in error_str for keyword in ("limit", "overloaded", "token")):
                sleep_s = random.uniform(6, 12) + attempt * 4
            else:
                sleep_s = min(10.0, 1.5 * attempt) + random.uniform(0.5, 1.5)
            total_sleep += sleep_s
            print(
                f"[WARN] LLM call failed for '{question_preview}' (attempt {attempt}/{MAX_ANSWER_RETRIES}). "
                f"Retrying in {sleep_s:.1f}s... Error: {exc}",
                flush=True,
            )
            time.sleep(sleep_s)
    raise RuntimeError(f"Answer LLM failed after {MAX_ANSWER_RETRIES} attempts: {last_error}")


def _iter_questions(payload: Dict[str, List[Dict]]) -> Iterable[Tuple[str, int, Dict]]:
    for conv_key, entries in payload.items():
        if not isinstance(entries, list):
            continue
        for idx, entry in enumerate(entries):
            if isinstance(entry, dict):
                yield str(conv_key), idx, entry


def _run_evaluation(
    answer_file: Path,
    model_tag: str,
    time_tag: str,
    evaluator_config: Dict[str, str],
    max_workers: int,
) -> Path:
    metrics_path = answer_file.with_name(f"evaluation_metrics_1126_new106.json")
    cmd = [
        sys.executable,
        str(EVAL_SCRIPT_PATH),
        "--input_file",
        str(answer_file),
        "--output_file",
        str(metrics_path),
        "--max_workers",
        str(max_workers),
    ]

    evaluator_model = evaluator_config.get("model")
    evaluator_base_url = evaluator_config.get("base_url")
    evaluator_api_key = evaluator_config.get("api_key")

    if evaluator_model:
        cmd.extend(["--evaluator_model", evaluator_model])
    if evaluator_base_url:
        cmd.extend(["--evaluator_base_url", evaluator_base_url])
    if evaluator_api_key:
        cmd.extend(["--evaluator_api_key", evaluator_api_key])

    print(f"🧪 Running evaluation:\n  {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=EVAL_SCRIPT_PATH.parent)
    return metrics_path


def _answer_experiment(
    input_path: Path,
    answer_client: OpenAI,
    model_name: str,
    max_workers: int,
) -> Tuple[Path, Dict[str, int], str]:
    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    OLD_PREFIX = """You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.\n\n    # CONTEXT:\n    You have access to memories from two speakers in a conversation. These memories contain \n    timestamped information that may be relevant to answering the question.\n\n    # INSTRUCTIONS:\n    1. Carefully analyze all provided memories from both speakers\n    2. Pay special attention to the timestamps to determine the answer\n    3. If the question asks about a specific event or fact, look for direct evidence in the memories\n    4. If the memories contain contradictory information, prioritize the most recent memory\n    5. If there is a question about time references (like \"last year\", \"two months ago\", etc.), \n       calculate the actual date based on the memory timestamp. For example, if a memory from \n       4 May 2022 mentions \"went to India last year,\" then the trip occurred in 2021.\n    6. Always convert relative time references to specific dates, months, or years. For example, \n       convert \"last year\" to \"2022\" or \"two months ago\" to \"March 2023\" based on the memory \n       timestamp. Ignore the reference while answering the question.\n    7. Focus only on the content of the memories from both speakers. Do not confuse character \n       names mentioned in memories with the actual users who created those memories.\n    8. The answer should be less than 5-6 words.\n    9. When you can\u2019t find evidence, the question may be an inference question. In that case, use common sense and reasoning to answer\u2014don\u2019t refuse.\n\n    # APPROACH (Think step by step):\n    1. First, examine all memories that contain information related to the question\n    2. Examine the timestamps and content of these memories carefully\n    3. Look for explicit mentions of dates, times, locations, or events that answer the question\n    4. If the answer requires calculation (e.g., converting relative time references), show your work\n    5. Formulate a precise, concise answer based solely on the evidence in the memories\n    6. Double-check that your answer directly addresses the question asked\n    7. Ensure your final answer is specific and avoids vague time references\n\n   
    
    """
    NEW_PREFIX = """
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.\n\n    # CONTEXT:\n    You have access to memories from two speakers in a conversation. These memories contain \n    timestamped information that may be relevant to answering the question.\n\n    # INSTRUCTIONS:\n    1. Carefully analyze all provided memories from both speakers\n    2. Pay special attention to the timestamps to determine the answer\n    3. If the question asks about a specific event or fact, look for direct evidence in the memories\n    4. If the memories contain contradictory information, prioritize the most recent memory\n    5. If there is a question about time references (like \"last year\", \"two months ago\", etc.), \n       calculate the actual date based on the memory timestamp. For example, if a memory from \n       4 May 2022 mentions \"went to India last year,\" then the trip occurred in 2021.\n    6. Always convert relative time references to specific dates, months, or years. For example, \n       convert \"last year\" to \"2022\" or \"two months ago\" to \"March 2023\" based on the memory \n       timestamp. Ignore the reference while answering the question.\n    7. Focus only on the content of the memories from both speakers. Do not confuse character \n       names mentioned in memories with the actual users who created those memories.\n    8. The answer should be less than 5-6 words.\n\n    # APPROACH (Think step by step):\n    1. First, examine all memories that contain information related to the question\n    2. Examine the timestamps and content of these memories carefully\n    3. Look for explicit mentions of dates, times, locations, or events that answer the question\n    4. If the answer requires calculation (e.g., converting relative time references), show your work\n    5. Formulate a precise, concise answer based solely on the evidence in the memories\n    6. Always center your judgment on the keywords in the question\n  7. Double-check that your answer directly addresses the question asked，if anything is missing, keep adding until the list is complete\n    8. Ensure your final answer is specific and avoids vague time references\n\n  

    """

    tasks = []
    total_entries = 0
    for conv_key, idx, entry in _iter_questions(payload):
        total_entries += 1
        prompt = entry.get("answer_prompt")
        prompt = NEW_PREFIX + prompt[len(OLD_PREFIX):]
        # print(f"📊📊📊📊📊📊📊📊📊📊📊{prompt}")
        if not prompt:
            continue
        question = (entry.get("question") or "").strip().replace("\n", " ")
        if len(question) > 80:
            question = question[:77] + "..."
        tasks.append((conv_key, idx, prompt, question))

    if not tasks:
        raise ValueError(f"No answer prompts found inside {input_path}")

    model_tag = _sanitize_model_tag(model_name)
    answer_output = input_path.with_name(f"{input_path.stem}_{model_tag}_answer{input_path.suffix}")
    time_tag = _extract_time_tag(input_path)

    stats = {"completed": 0, "failed": 0, "skipped": total_entries - len(tasks)}

    print(f"📄 Loaded {len(payload)} conversations. Re-answering {len(tasks)} questions with {model_name}...")

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="answer-only") as executor:
        future_map = {
            executor.submit(_call_answer_llm, answer_client, model_name, prompt, question): (conv_key, idx, question)
            for conv_key, idx, prompt, question in tasks
        }

        for future in tqdm(as_completed(future_map), total=len(future_map), desc="Answering questions"):
            conv_key, idx, question_preview = future_map[future]
            entry = payload[conv_key][idx]
            try:
                response_text, latency = future.result()
                entry["response"] = response_text
                entry["response_time"] = latency
                entry["answer_model"] = model_name
                stats["completed"] += 1
            except Exception as exc:
                entry["response"] = f"[ERROR] answer-only run failed: {exc}"
                entry["response_time"] = 0.0
                entry["answer_model"] = model_name
                entry["answer_error"] = str(exc)
                stats["failed"] += 1
                print(f"[ERROR] Failed to answer '{question_preview}': {exc}", flush=True)

    with answer_output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4, ensure_ascii=False)

    print(f"✅ Answer-only file written to: {answer_output}")
    return answer_output, stats, time_tag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-run answers for existing experiments and evaluate them.")
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="Optional override for the experiment result files. Defaults to EXPERIMENT_RESULT_FILES.",
    )
    parser.add_argument("--answer_max_workers", type=int, default=DEFAULT_ANSWER_MAX_WORKERS)
    parser.add_argument("--eval_max_workers", type=int, default=DEFAULT_EVAL_MAX_WORKERS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_files = args.files or EXPERIMENT_RESULT_FILES
    if not candidate_files:
        print("No experiment files specified. Add paths to EXPERIMENT_RESULT_FILES or pass --files.", file=sys.stderr)
        sys.exit(1)

    answer_model = ANSWER_LLM_CONFIG.get("model") or os.getenv("BASE_MODEL") or "unknown_answer_model"
    answer_client = _build_openai_client(ANSWER_LLM_CONFIG)

    for raw_path in candidate_files:
        input_path = _resolve_path(raw_path)
        if not input_path.exists():
            print(f"[WARN] Skipping missing experiment file: {input_path}")
            continue

        print("\n" + "=" * 80)
        print(f"🚀 Processing experiment: {input_path}")
        print("=" * 80)

        try:
            answer_file, stats, time_tag = _answer_experiment(
                input_path=input_path,
                answer_client=answer_client,
                model_name=answer_model,
                max_workers=args.answer_max_workers,
            )
            print(f"Answer summary: {stats}")
        except Exception as exc:
            print(f"[FATAL] Failed to regenerate answers for {input_path}: {exc}")
            continue

        try:
            metrics_path = _run_evaluation(
                answer_file=answer_file,
                model_tag=_sanitize_model_tag(answer_model),
                time_tag=time_tag,
                evaluator_config=EVALUATOR_LLM_CONFIG,
                max_workers=args.eval_max_workers,
            )
            print(f"📊 Evaluation metrics saved to: {metrics_path}")
        except subprocess.CalledProcessError as exc:
            print(f"[FATAL] Evaluation failed for {answer_file}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
