import argparse
import json
from collections import defaultdict

import numpy as np
from openai import OpenAI, AsyncClient
import os


LOCAL_MEM0_PATH = os.getenv("LOCAL_MEM0_PATH")
if not LOCAL_MEM0_PATH:
    raise ValueError("环境变量 LOCAL_MEM0_PATH 未设置，请在 .env 文件中配置。")

if not os.path.exists(LOCAL_MEM0_PATH):
    raise ImportError(f"指定的本地 mem0 路径不存在: {LOCAL_MEM0_PATH}")

import sys
if LOCAL_MEM0_PATH not in sys.path:
    sys.path.insert(0, LOCAL_MEM0_PATH)

print("="*80)
print(f"✅ 成功将本地 mem0 库路径添加到环境中: {LOCAL_MEM0_PATH}")
print("="*80)

from mem0.memory.utils import extract_json

client = AsyncClient(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))

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

import time
import random
from typing import Callable, Awaitable, TypeVar, ParamSpec
from tenacity import retry, stop_after_attempt, wait_fixed, wait_exponential, retry_if_exception_type, RetryCallState
from functools import partial, wraps
import asyncio
import threading


P = ParamSpec('P')
K = TypeVar("K")
T = TypeVar("T")
U = TypeVar("U")


def make_sync(async_func: Callable[P, Awaitable[T]]) -> Callable[P, T]:
    @wraps(async_func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(async_func(*args, **kwargs))
        result = None
        exception = None
        def runner():
            nonlocal result, exception
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(async_func(*args, **kwargs))
            except Exception as e:
                exception = e
            finally:
                loop.close()
        thread = threading.Thread(target=runner)
        thread.start()
        thread.join()
        if exception:
            raise exception
        return result

    return wrapper


def custom_wait(retry_state: RetryCallState):
    exception = retry_state.outcome.exception() if retry_state.outcome else None
    retries = retry_state.attempt_number
    max_retries = retry_state.retry_object.stop.max_attempt_number
    if exception and any(x in str(exception).lower() for x in ("tpm", "limit")):
        # 指数退避：1, 2, 4, 8... capped at 80
        wait_strategy = wait_exponential(multiplier=1, min=40, max=80)
        sleep_duration = wait_strategy(retry_state)
        print(
            f"   -> Rate limit error detected. "
            f"Waiting for {sleep_duration:.1f}s... (Attempt {retries}/{max_retries})"
        )
        return sleep_duration
    else:
        # 固定等待 1 秒
        wait_strategy = wait_fixed(1)
        sleep_duration = wait_strategy(retry_state)
        print(
            f"   -> Other error detected. "
            f"Waiting for {sleep_duration:.1f}s... (Attempt {retries}/{max_retries})"
        )
        return sleep_duration


@retry(
    reraise=True,
    stop=stop_after_attempt(10),
    wait=custom_wait,
)
async def _aevaluate_llm_judge_core(question, gold_answer: str, generated_answer):
    response = await client.chat.completions.create(
        model="Qwen/Qwen3-14B",
        messages=[
            {
                "role": "user",
                "content": ACCURACY_PROMPT.format(
                    question=question, gold_answer=gold_answer, generated_answer=generated_answer
                ),
            }
        ],
        temperature=0.0,
    )
    label = json.loads(extract_json(response.choices[0].message.content))["label"]
    return 1 if label == "CORRECT" else 0


async def _aevaluate_llm_judge(question, gold_answer: str, generated_answer):
    try:
        return await _aevaluate_llm_judge_core(question, gold_answer, generated_answer)
    except Exception as e:
        print(f"LLM Judge evaluation failed after retries: {e}")
        return 0  # 默认返回 WRONG


async def aevaluate_llm_judge(question, gold_answer: list[str], generated_answer):
    tasks = [
        _aevaluate_llm_judge(question, gold, generated_answer) for gold in gold_answer
    ]
    results = await asyncio.gather(*tasks)
    return max(results)

def evaluate_llm_judge(question, gold_answer: list[str], generated_answer):
    sync_func = make_sync(aevaluate_llm_judge)
    return sync_func(question, gold_answer, generated_answer)

def main():
    """Main function to evaluate RAG results using LLM judge."""
    parser = argparse.ArgumentParser(description="Evaluate RAG results using LLM judge")
    parser.add_argument(
        "--input_file",
        type=str,
        default="results/default_run_v4_k30_new_graph.json",
        help="Path to the input dataset file",
    )

    args = parser.parse_args()

    dataset_path = args.input_file
    output_path = f"results/llm_judge_{dataset_path.split('/')[-1]}"

    with open(dataset_path, "r") as f:
        data = json.load(f)

    LLM_JUDGE = defaultdict(list)
    RESULTS = defaultdict(list)

    index = 0
    for k, v in data.items():
        for x in v:
            question = x["question"]
            gold_answer = x["answer"]
            generated_answer = x["response"]
            category = x["category"]

            # Skip category 5
            if int(category) == 5:
                continue

            # Evaluate the answer
            label = evaluate_llm_judge(question, gold_answer, generated_answer)
            LLM_JUDGE[category].append(label)

            # Store the results
            RESULTS[index].append(
                {
                    "question": question,
                    "gt_answer": gold_answer,
                    "response": generated_answer,
                    "category": category,
                    "llm_label": label,
                }
            )

            # Save intermediate results
            with open(output_path, "w") as f:
                json.dump(RESULTS, f, indent=4)

            # Print current accuracy for all categories
            print("All categories accuracy:")
            for cat, results in LLM_JUDGE.items():
                if results:  # Only print if there are results for this category
                    print(f"  Category {cat}: {np.mean(results):.4f} ({sum(results)}/{len(results)})")
            print("------------------------------------------")
        index += 1

    # Save final results
    with open(output_path, "w") as f:
        json.dump(RESULTS, f, indent=4)

    # Print final summary
    print("PATH: ", dataset_path)
    print("------------------------------------------")
    for k, v in LLM_JUDGE.items():
        print(k, np.mean(v))


if __name__ == "__main__":
    main()
