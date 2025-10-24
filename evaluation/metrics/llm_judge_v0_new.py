# -*- coding: utf-8 -*-
import argparse
import asyncio
import json
from collections import defaultdict
from contextvars import ContextVar
import os
import sys
import threading
from typing import Optional

import numpy as np
from openai import AsyncOpenAI

# --- 本地 mem0 路径注入 ---
LOCAL_MEM0_PATH = os.getenv("LOCAL_MEM0_PATH")
if not LOCAL_MEM0_PATH:
    raise ValueError("环境变量 LOCAL_MEM0_PATH 未设置，请在 .env 文件中配置。")
if not os.path.exists(LOCAL_MEM0_PATH):
    raise ImportError(f"指定的本地 mem0 路径不存在: {LOCAL_MEM0_PATH}")
if LOCAL_MEM0_PATH not in sys.path:
    sys.path.insert(0, LOCAL_MEM0_PATH)

print("="*80)
print(f"✅ 成功将本地 mem0 库路径添加到环境中: {LOCAL_MEM0_PATH}")
print("="*80)

from mem0.memory.utils import extract_json

# ---- Evaluator 可配置参数（沿用“文件版”风格）----
FALLBACK_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", "Qwen/Qwen3-14B")
DEFAULT_EVALUATOR_BASE_URL = os.getenv("OPENAI_BASE_URL") or FALLBACK_BASE_URL
DEFAULT_EVALUATOR_API_KEY = os.getenv("OPENAI_API_KEY")

_evaluator_model = DEFAULT_EVALUATOR_MODEL
_evaluator_base_url = DEFAULT_EVALUATOR_BASE_URL
_evaluator_api_key = DEFAULT_EVALUATOR_API_KEY
_client: Optional[AsyncOpenAI] = None
_client_lock = threading.Lock()

_event_loop: Optional[asyncio.AbstractEventLoop] = None
_event_loop_thread: Optional[threading.Thread] = None
_event_loop_lock = threading.Lock()
_event_loop_ready = threading.Event()

def _build_client() -> AsyncOpenAI:
    client_kwargs = {}
    if _evaluator_base_url:
        client_kwargs["base_url"] = _evaluator_base_url
    if _evaluator_api_key:
        client_kwargs["api_key"] = _evaluator_api_key
    return AsyncOpenAI(**client_kwargs)


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = _build_client()
    return _client


def _ensure_event_loop() -> asyncio.AbstractEventLoop:
    """Ensure a background event loop is running for async evaluator calls."""
    global _event_loop, _event_loop_thread

    with _event_loop_lock:
        if _event_loop and not _event_loop.is_closed() and _event_loop.is_running():
            return _event_loop

        loop = asyncio.new_event_loop()
        _event_loop_ready.clear()

        def _run_loop() -> None:
            asyncio.set_event_loop(loop)
            _event_loop_ready.set()
            loop.run_forever()

        thread = threading.Thread(target=_run_loop, name="llm-judge-event-loop", daemon=True)
        thread.start()

        _event_loop = loop
        _event_loop_thread = thread

    _event_loop_ready.wait()
    return _event_loop

def configure_evaluator(model=None, base_url=None, api_key=None):
    """
    运行期更新评测端参数。传 None 表示保持现值。
    """
    global _client, _evaluator_model, _evaluator_base_url, _evaluator_api_key
    if model:
        _evaluator_model = model
    if base_url is not None:
        _evaluator_base_url = base_url or None
    if api_key is not None:
        _evaluator_api_key = api_key or None
    _client = None  # 触发下次调用时重建

# ---- 采用 v5 的严格判定规则（ACCURACY_PROMPT）----
ACCURACY_PROMPT = """Your task is to label an answer as ’CORRECT’ or ’WRONG’ given:
(1) a question,
(2) a gold (ground truth) answer,
(3) a generated answer.

Core principle — Inclusion + Non-contradiction
- Be GENEROUS: if the generated answer clearly includes the gold’s key content (or a clear paraphrase of the same content) and does not contradict it, mark CORRECT — even if extra details are added.
- Mark WRONG only when the generated answer does not include the gold’s content, changes it, or contradicts it.

TIME (strict granularity; relative form equivalence; no calendar math)
- Granularity must match exactly: HOUR↔HOUR, DAY↔DAY, MONTH↔MONTH, YEAR↔YEAR.
  Do not answer a gold at a different time unit — even if the numeric value overlaps. Do not answer a month-level gold with a specific day, nor a year with a specific month/day/hour, etc.
  (e.g., gold = "July 26, 2019" [DAY]; generated = "2019-07-26 08:09:17" [includes Second] → WRONG)
- Do NOT convert relative ↔ absolute. If the gold uses a relative time expression, the generated answer must also use a relative form (or a clear paraphrase of that same form), not a computed date/range.
- Treat harmless modifiers in relative forms (e.g., “the/last/previous/just prior”) as equivalent when both the anchor date and the time unit are the same.

- Lists of DISTINCT facts:
- If the gold answer lists multiple distinct facts (joined by "and", commas, or slashes), the generated answer must cover **all** of them.
- Extra non-contradictory items **generally count as WRONG**.
    - Example: gold = A, B, C ; gen = A, B, C → CORRECT
    - Example: gold = A, B, C ; gen = A, B, C, D → WRONG
- Exception: If a gold element is elaborated or split into finer details in the generated answer (e.g., C → C, C′), it is still considered CORRECT.

Preference/Benefit Questions (e.g., "what X likes/values most")
- If gold lists multiple reasons/aspects, the generated answer only needs to include **any one** of them without contradiction to be CORRECT.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. 
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label":

```json
{{
    "label": "CORRECT" or "WRONG"
}}
"""
# ---- 错误重试：tenacity 装饰 ----
from tenacity import AsyncRetrying, RetryCallState, stop_after_attempt
from tenacity import wait_fixed, wait_exponential

_retry_details: ContextVar[Optional[str]] = ContextVar("_retry_details", default=None)


def _log_retry(retry_state: RetryCallState):
    exception = retry_state.outcome.exception() if retry_state.outcome else None
    wait_action = getattr(retry_state.next_action, "sleep", None)
    max_attempts = getattr(getattr(retry_state.retry_object, "stop", None), "max_attempt_number", None)
    attempt = retry_state.attempt_number

    message = f"⏳ Evaluator retry attempt {attempt}"
    if max_attempts:
        message += f"/{max_attempts}"
    if exception:
        message += f" due to: {exception}"
    if wait_action is not None:
        message += f"; waiting {wait_action:.2f}s before next attempt"
    context = _retry_details.get()
    if context:
        message += f" | Context: {context}"
    print(message, flush=True)

def _custom_wait(retry_state: RetryCallState):
    """
    碰到配额/速率（tpm/limit）错误 → 指数退避（40~80 秒区间，逐步增大）；
    其他错误 → 固定 1 秒。
    """
    exception = retry_state.outcome.exception() if retry_state.outcome else None
    if exception and any(x in str(exception).lower() for x in ("tpm", "limit")):
        # 指数退避，限制在 40~80s 区间活动
        # tenacity 的 wait_exponential 返回 seconds；我们再 clip 一下到 [40, 80]
        base = wait_exponential(multiplier=1, min=40, max=80)
        return base(retry_state)
    # 其他错误：1s 固定等待
    return wait_fixed(1)(retry_state)

async def _evaluate_single_gold_async(question: str, gold_answer_str: str, generated_answer: str) -> int:
    """Evaluate a single gold answer asynchronously with retries."""
    context_token = _retry_details.set(
        f"question='{question[:60]}', gold='{gold_answer_str[:40]}', generated='{generated_answer[:40]}'"
    )
    try:
        async for attempt in AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(10),
            wait=_custom_wait,
            before_sleep=_log_retry,
        ):
            with attempt:
                response = await get_client().chat.completions.create(
                    model=_evaluator_model or "Qwen/Qwen3-14B",
                    messages=[
                        {
                            "role": "user",
                            "content": ACCURACY_PROMPT.format(
                                question=question,
                                gold_answer=gold_answer_str,
                                generated_answer=generated_answer,
                            ),
                        }
                    ],
                    temperature=0.0,
                )
                content = response.choices[0].message.content
                label = json.loads(extract_json(content))["label"]
                return 1 if label == "CORRECT" else 0
    finally:
        _retry_details.reset(context_token)


async def evaluate_llm_judge_async(question, gold_answer, generated_answer):
    """Async helper supporting both single and multi-gold evaluation."""
    try:
        if isinstance(gold_answer, (list, tuple)):
            tasks = [
                _evaluate_single_gold_async(question, str(gold), generated_answer)
                for gold in gold_answer
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            best = 0
            for outcome in results:
                if isinstance(outcome, Exception):
                    print(
                        f"⚠️ 单 gold 评测异常（将由外层继续处理或进入下一个 gold）：{outcome}",
                        flush=True,
                    )
                    continue
                value = int(outcome)
                if value == 1:
                    return 1
                best = max(best, value)
            return best

        return await _evaluate_single_gold_async(question, str(gold_answer), generated_answer)
    except Exception as exc:
        print(f"❌ 评测失败（超过最大重试或不可恢复错误）：{exc}", flush=True)
        return 0


def submit_llm_judge(question, gold_answer, generated_answer):
    """Schedule an async LLM judge evaluation and return a concurrent.futures.Future."""
    loop = _ensure_event_loop()
    try:
        return asyncio.run_coroutine_threadsafe(
            evaluate_llm_judge_async(question, gold_answer, generated_answer),
            loop,
        )
    except RuntimeError:
        # Loop might not be running yet; try to restart it once.
        loop = _ensure_event_loop()
        return asyncio.run_coroutine_threadsafe(
            evaluate_llm_judge_async(question, gold_answer, generated_answer),
            loop,
        )


def evaluate_llm_judge(question, gold_answer, generated_answer):
    """Synchronous wrapper for async evaluation (keeps legacy API)."""
    future = submit_llm_judge(question, gold_answer, generated_answer)
    return future.result()

def main():
    """Evaluate RAG results using LLM judge."""
    parser = argparse.ArgumentParser(description="Evaluate RAG results using LLM judge")
    parser.add_argument(
        "--input_file",
        type=str,
        default="results/default_run_v4_k30_new_graph.json",
        help="Path to the input dataset file",
    )
    args = parser.parse_args()

    dataset_path = args.input_file
    output_path = f"results/llm_judge_{os.path.basename(dataset_path)}"

    with open(dataset_path, "r") as f:
        data = json.load(f)

    LLM_JUDGE = defaultdict(list)
    RESULTS = defaultdict(list)

    index = 0
    for k, v in data.items():
        for x in v:
            question = x["question"]
            gold_answer = x["answer"]           # 允许为 str 或 list[str]
            generated_answer = x["response"]
            category = x["category"]

            # Skip category 5
            if int(category) == 5:
                continue

            # Evaluate
            label = evaluate_llm_judge(question, gold_answer, generated_answer)
            LLM_JUDGE[category].append(label)

            # Store
            RESULTS[index].append(
                {
                    "question": question,
                    "gt_answer": gold_answer,
                    "response": generated_answer,
                    "category": category,
                    "llm_label": label,
                }
            )

            # Save intermediate
            with open(output_path, "w") as f:
                json.dump(RESULTS, f, indent=4, ensure_ascii=False)

            # Live accuracy by category
            print("All categories accuracy:")
            for cat, results in LLM_JUDGE.items():
                if results:
                    print(f"  Category {cat}: {np.mean(results):.4f} ({sum(results)}/{len(results)})")
            print("------------------------------------------")
        index += 1

    # Final save
    with open(output_path, "w") as f:
        json.dump(RESULTS, f, indent=4, ensure_ascii=False)

    # Summary
    print("PATH: ", dataset_path)
    print("------------------------------------------")
    for k, v in LLM_JUDGE.items():
        print(k, np.mean(v))

if __name__ == "__main__":
    main()
