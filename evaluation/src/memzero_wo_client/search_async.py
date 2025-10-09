import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import json
import logging
import uuid

from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI
from prompts import ANSWER_PROMPT, ANSWER_PROMPT_GRAPH
from tqdm import tqdm
import random 
from mem0 import Memory

load_dotenv()

# Set the OpenAI API key

model_name = os.getenv("BASE_MODEL", "Qwen/Qwen3-14B")
os.environ["MODEL"] = model_name

class MemorySearch:
    def __init__(self, output_path="results.json", top_k=10, filter_memories=False, is_graph=False, logger=None, qdrant_path=None):
        config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": model_name,
                    "openai_base_url": os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1"),
                    "temperature": 0.1,
                    "max_tokens": 2000,
                }
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "BAAI/bge-m3",
                    "openai_base_url": os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1"),
                }
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "path": qdrant_path,
                    "on_disk": True,
                    "embedding_model_dims": 1024
                }
            },
            "version": "v1.1",
        }
        self.top_k = top_k
        self.openai_client = OpenAI()
        self.results = defaultdict(list)
        self.output_path = output_path
        self.logger = logger if logger else logging.getLogger(__name__)
        self.filter_memories = filter_memories
        self.is_graph = is_graph
        self.lock = None
        # Create the memory object first
        self.memory = Memory.from_config(config)
        # Then, set the logger attribute on the created instance
        self.memory.logger = self.logger        # self.lock = threading.Lock()

        if self.is_graph:
            self.ANSWER_PROMPT = ANSWER_PROMPT_GRAPH
        else:
            self.ANSWER_PROMPT = ANSWER_PROMPT

    def search_memory(self, user_id, query, max_retries=5, pbar=None):
        start_time = time.time()
        retries = 0
        while retries < max_retries:
            try:
                memories = self.memory.search(
                    query,
                    user_id=user_id,
                    limit=self.top_k,
                )
                break
            except Exception as e:
                self.logger.warning(f"Retrying search for user {user_id}...{retries+1}/{max_retries}\tError: {str(e)}")
                retries += 1
                if retries >= max_retries:
                    raise e
                time.sleep(random.randint(1, 3))

        end_time = time.time()

        semantic_memories = [
            {
                "memory": memory["memory"],
                "timestamp": memory["metadata"].get("timestamp") if memory.get("metadata") else None,
                "score": round(memory["score"], 2),
            }
            for memory in memories["results"]
        ]
        graph_memories = None

        return semantic_memories, graph_memories, end_time - start_time
    
    def _log_llm_call(self, request_id, attempt, max_retries, prompt_components, full_prompt, response_content, status):
        """
        一个专门的方法来格式化和记录LLM的完整交互。
        """
        log_message = f"""
========================= LLM Call Start =========================
Request ID: {request_id}
Attempt: {attempt}/{max_retries}
Status: {status}
--------------------------- INPUT ----------------------------
[Question]: {prompt_components['question']}

[Speaker 1 ({prompt_components['speaker_1_user_id']}) Memories]:
{json.dumps(prompt_components['speaker_1_memories'], indent=2)}

[Speaker 2 ({prompt_components['speaker_2_user_id']}) Memories]:
{json.dumps(prompt_components['speaker_2_memories'], indent=2)}

[Full Prompt to LLM]:
{full_prompt}
--------------------------- OUTPUT ---------------------------
{response_content if response_content else 'N/A'}
========================== LLM Call End ==========================
"""
        if "Success" in status:
            self.logger.info(log_message)
        else:
            self.logger.error(log_message)


    def answer_question(self, speaker_1_user_id, speaker_2_user_id, question, answer, category, pbar=None, max_retries=51):
        # TODO @gangyi 方案3 and 5，config中的参数控制
        speaker_1_memories, speaker_1_graph_memories, speaker_1_memory_time = self.search_memory(
            speaker_1_user_id, question, pbar=pbar
        )
        speaker_2_memories, speaker_2_graph_memories, speaker_2_memory_time = self.search_memory(
            speaker_2_user_id, question, pbar=pbar
        )

        search_1_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_1_memories]
        search_2_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_2_memories]

        prompt_components = {
            "speaker_1_user_id": speaker_1_user_id.split("_")[0],
            "speaker_2_user_id": speaker_2_user_id.split("_")[0],
            "question": question,
            "speaker_1_memories": search_1_memory,
            "speaker_2_memories": search_2_memory,
            "speaker_1_graph_memories": speaker_1_graph_memories,
            "speaker_2_graph_memories": speaker_2_graph_memories,
        }
        
        template = Template(self.ANSWER_PROMPT)
        answer_prompt = template.render(
            speaker_1_user_id=prompt_components["speaker_1_user_id"],
            speaker_2_user_id=prompt_components["speaker_2_user_id"],
            question=prompt_components["question"],
            speaker_1_memories=json.dumps(prompt_components["speaker_1_memories"], indent=4),
            speaker_2_memories=json.dumps(prompt_components["speaker_2_memories"], indent=4),
            speaker_1_graph_memories=json.dumps(prompt_components["speaker_1_graph_memories"], indent=4),
            speaker_2_graph_memories=json.dumps(prompt_components["speaker_2_graph_memories"], indent=4),
        )
        response_content = None
        request_id = f"answer-q-{uuid.uuid4()}"
        # 细化重试逻辑
        llm_error_retries = 0
        other_error_retries = 0
        MAX_OTHER_ERROR_RETRIES = 6
        while True:
            try:
                t1 = time.time()
                response = self.openai_client.chat.completions.create(
                    model=os.getenv("MODEL"), messages=[{"role": "system", "content": answer_prompt}], temperature=0.0
                )
                t2 = time.time()
                response_time = t2 - t1
                response_content = response.choices[0].message.content
                self._log_llm_call(request_id, llm_error_retries+other_error_retries, max_retries, prompt_components, answer_prompt, response_content, "Success")
                break 
            except Exception as e:
                error_str = str(e).lower()
                if "rate limit" in error_str or "limit" in error_str or "overloaded" in error_str or "token" in error_str:
                    # 识别为LLM相关的限流错误
                    llm_error_retries += 1
                    other_error_retries = 0 # 重置其他错误计数
                    sleep_duration = random.uniform(2, 20) + 5 * llm_error_retries
                    error_message = f"LLM Rate Limit related Error. Retrying in {sleep_duration:.2f}s... Error: {e}"
                    self._log_llm_call(request_id, llm_error_retries, max_retries, prompt_components, answer_prompt, error_message, "Failed Attempt (Rate Limit)")
                    time.sleep(sleep_duration)
                else:
                    # 识别为其他错误
                    other_error_retries += 1
                    if other_error_retries >= MAX_OTHER_ERROR_RETRIES:
                        self.logger.error(f"Request ID [{request_id}] - Exceeded max retries ({MAX_OTHER_ERROR_RETRIES}) for non-rate-limit errors. Failing permanently. Error: {e}")
                        response_content = "Error: Default response due to unrecoverable error." # 设置默认值
                        break # 达到最大次数，跳出循环
                    
                    error_message = f"An unexpected error occurred. Retrying immediately (attempt {other_error_retries}/{MAX_OTHER_ERROR_RETRIES})... Error: {e}"
                    self._log_llm_call(request_id, other_error_retries, max_retries, prompt_components, answer_prompt, error_message, "Failed Attempt (Other Error)")
                    self.logger.warning(error_message)

                # error_message = f"LLM API call failed. Error: {e}"
                # self._log_llm_call(request_id, attempt, max_retries, prompt_components, answer_prompt, error_message, f"Failed Attempt")
                
                # if attempt < max_retries:
                #     time.sleep(random.randint(5, 15)+5*attempt)
                # else:
                #     self.logger.error(f"Request ID [{request_id}] - LLM call failed permanently after {max_retries} attempts.")
                    
        return (    
            response_content,
            speaker_1_memories,
            speaker_2_memories,
            speaker_1_memory_time,
            speaker_2_memory_time,
            speaker_1_graph_memories,
            speaker_2_graph_memories,
            response_time,
            answer_prompt,
        )

    def process_question(self, val, speaker_a_user_id, speaker_b_user_id, idx, pbar=None, lock=None):
        question = val.get("question", "")
        answer = val.get("answer", "")
        category = val.get("category", -1)
        evidence = val.get("evidence", [])
        adversarial_answer = val.get("adversarial_answer", "")

        (
            response,
            speaker_1_memories,
            speaker_2_memories,
            speaker_1_memory_time,
            speaker_2_memory_time,
            speaker_1_graph_memories,
            speaker_2_graph_memories,
            response_time,
            answer_prompt,
        ) = self.answer_question(speaker_a_user_id, speaker_b_user_id, question, answer, category, pbar)

        result = {
            "question": question,
            "answer": answer,
            "category": category,
            "evidence": evidence,
            "response": response,
            "adversarial_answer": adversarial_answer,
            "speaker_1_memories": speaker_1_memories,
            "speaker_2_memories": speaker_2_memories,
            "num_speaker_1_memories": len(speaker_1_memories),
            "num_speaker_2_memories": len(speaker_2_memories),
            "speaker_1_memory_time": speaker_1_memory_time,
            "speaker_2_memory_time": speaker_2_memory_time,
            "speaker_1_graph_memories": speaker_1_graph_memories,
            "speaker_2_graph_memories": speaker_2_graph_memories,
            "response_time": response_time,
            "answer_prompt": answer_prompt
        }


        if lock:
            with lock:
                self.results[idx].append(result)
                with open(self.output_path, "w") as f:
                    json.dump(self.results, f, indent=4)
        else: 
            self.results[idx].append(result)
            with open(self.output_path, "w") as f:
                json.dump(self.results, f, indent=4)
        
        if pbar:
            pbar.update(1)

        return result

    def process_data_file(self, file_path, max_workers=5):
        with open(file_path, "r") as f:
            data = json.load(f)

        total_questions = sum(len(item.get("qa", [])) for item in data)
        if total_questions == 0:
            print("No questions found to process.")
            return
            
        print(f"--- 预计总共需要处理 {total_questions} 个问题 ---")

        successful_count = 0
        failed_count = 0
        
        with tqdm(total=total_questions, desc="💡Total Questions Progress") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for idx, item in enumerate(data):
                    qa = item.get("qa", [])
                    conversation = item["conversation"]
                    speaker_a = conversation["speaker_a"]
                    speaker_b = conversation["speaker_b"]
                    speaker_a_user_id = f"{speaker_a}_{idx}"
                    speaker_b_user_id = f"{speaker_b}_{idx}"

                    for question_item in qa:
                        future = executor.submit(
                            self.process_question, 
                            question_item, 
                            speaker_a_user_id, 
                            speaker_b_user_id,
                            idx, 
                            pbar, 
                            self.lock
                        )
                        futures[future] = f"Conv {idx} - Q: {question_item.get('question', '')[:30]}..."

                for future in as_completed(futures):
                    task_id = futures[future]
                    try:
                        future.result() 
                        successful_count += 1
                    except Exception as e:
                        failed_count += 1
                        pbar.write(f"\n--- ❌ Error processing task '{task_id}': {e} ---\n")

        print(f"\n✅ All questions processed. Success: {successful_count}, Failed: {failed_count}")