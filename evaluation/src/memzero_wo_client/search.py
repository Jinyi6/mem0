import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI
from prompts import ANSWER_PROMPT, ANSWER_PROMPT_GRAPH
from tqdm import tqdm
import random 
from mem0 import Memory
import uuid
from src.utils import normalize_dataset_records
load_dotenv()

model_name = os.getenv("BASE_MODEL", "Qwen/Qwen3-14B")
os.environ["MODEL"] = model_name

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
            "path": "./qdrant_data/tmp/qdrant_data_locomo1_6",
            "on_disk": True,
            "embedding_model_dims":1024
        }
    },
    "version": "v1.1",
}


class MemorySearch:
    def __init__(self, output_path="results.json", top_k=10, filter_memories=False, is_graph=False, enable_memory_summary=True):
        self._io_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="mem0-search-io")
        self.memory = Memory.from_config(config, shared_executor=self._io_executor)
        self.top_k = top_k
        self.openai_client = OpenAI()
        self.results = defaultdict(list)
        self.output_path = output_path
        self.filter_memories = filter_memories
        self.is_graph = is_graph
        self.enable_memory_summary = enable_memory_summary
        # self.enable_memory_summary = True

        if self.is_graph:
            self.ANSWER_PROMPT = ANSWER_PROMPT_GRAPH
        else:
            self.ANSWER_PROMPT = ANSWER_PROMPT

    def search_memory(self, user_id, query, max_retries=11):
        request_id = f"search-mem-{uuid.uuid4()}"
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
                print(f"Request ID [{request_id}] - Retrying search for user {user_id}...{retries+1}/{max_retries}\tError: {str(e)}")
                retries += 1
                if retries >= max_retries:
                    raise e
                time.sleep(int(random.uniform(20, 40) + 15 * retries))  # Wait before retrying

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

    def summarize_memories(self, memories, speaker_id, question, max_retries=3):
        """
        对检索到的记忆列表进行摘要总结
        
        Args:
            memories: 记忆列表，每个元素包含 memory, timestamp, score
            speaker_id: 说话者ID
            question: 当前问题
            max_retries: 最大重试次数
            
        Returns:
            summary: 摘要文本，如果失败则返回 None
        """
        if not memories or not self.enable_memory_summary:
            return None
        
        # 构建记忆文本
        memory_texts = []
        for idx, mem in enumerate(memories, 1):
            timestamp_str = f"[{mem['timestamp']}]" if mem.get('timestamp') else "[无时间戳]"
            memory_texts.append(f"{idx}. {timestamp_str} {mem['memory']}")
        
        memories_text = "\n".join(memory_texts)

        system_prompt = (
            "You are a memory summarizer. Your task is to create a concise summary of the provided memories "
            "that are relevant to answering a specific question. The summary should:\n"
            "1. Consolidate related information from multiple memories\n"
            "2. Preserve important details like names, dates, locations, and specific facts\n"
            "3. Remove redundancy while maintaining completeness\n"
            "4. Focus on information that directly relates to the question\n"
            "5. Keep the summary concise (3-5 sentences or bullet points)\n"
            "Output plain text only, no JSON format."
        )
        
        user_prompt = (
            f"Question: {question}\n\n" 
            f"Memories for {speaker_id}:\n{memories_text}\n\n"
            f"Please provide a concise summary of these memories relevant to the question."
        )
        
        retries = 0
        while retries < max_retries:
            try:
                response = self.openai_client.chat.completions.create(
                    model=os.getenv("MODEL"),
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=500
                )
                summary = response.choices[0].message.content.strip()
                return summary if summary else None
            except Exception as e:
                retries += 1
                if retries >= max_retries:
                    print(f"Failed to summarize memories for {speaker_id} after {max_retries} retries: {e}")
                    return None
                time.sleep(1)
        
        return None

    def answer_question(self, speaker_1_user_id, speaker_2_user_id, question, answer, category):
        speaker_1_memories, speaker_1_graph_memories, speaker_1_memory_time = self.search_memory(
            speaker_1_user_id, question
        )
        speaker_2_memories, speaker_2_graph_memories, speaker_2_memory_time = self.search_memory(
            speaker_2_user_id, question
        )

        # 对记忆进行摘要总结（如果启用）
        speaker_1_summary = None
        speaker_2_summary = None
        if self.enable_memory_summary:
            speaker_1_summary = self.summarize_memories(
                speaker_1_memories, speaker_1_user_id.split("_")[0], question
            )
            speaker_2_summary = self.summarize_memories(
                speaker_2_memories, speaker_2_user_id.split("_")[0], question
            )

        # 始终使用原始记忆列表
        search_1_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_1_memories]
        search_2_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_2_memories]

        template = Template(self.ANSWER_PROMPT)
        answer_prompt = template.render(
            speaker_1_user_id=speaker_1_user_id.split("_")[0],
            speaker_2_user_id=speaker_2_user_id.split("_")[0],
            speaker_1_memories=json.dumps(search_1_memory, indent=4),
            speaker_2_memories=json.dumps(search_2_memory, indent=4),
            speaker_1_graph_memories=json.dumps(speaker_1_graph_memories, indent=4),
            speaker_2_graph_memories=json.dumps(speaker_2_graph_memories, indent=4),
            question=question,
        )
        
        # 将摘要添加到prompt中（与原始记忆一起使用）
        if self.enable_memory_summary and speaker_1_summary and speaker_2_summary:
            summary_section = f"""

# Memory Summaries (for reference):
## Summary for {speaker_1_user_id.split('_')[0]}:
{speaker_1_summary}

## Summary for {speaker_2_user_id.split('_')[0]}:
{speaker_2_summary}

---
Note: The summaries above provide a condensed overview of the memories. Please refer to the detailed memories above for complete information.
"""
            answer_prompt = answer_prompt + summary_section
        
        t1 = time.time()
        response = self.openai_client.chat.completions.create(
            model=os.getenv("MODEL"), messages=[{"role": "system", "content": answer_prompt}], temperature=0.0
        )
        t2 = time.time()
        response_time = t2 - t1
        return (
            response.choices[0].message.content,
            speaker_1_memories,
            speaker_2_memories,
            speaker_1_memory_time,
            speaker_2_memory_time,
            speaker_1_graph_memories,
            speaker_2_graph_memories,
            response_time,
            speaker_1_summary,
            speaker_2_summary,
        )

    def close(self):
        if getattr(self, "_io_executor", None):
            self._io_executor.shutdown(wait=True, cancel_futures=True)
            self._io_executor = None
            if hasattr(self.memory, "set_shared_executor"):
                self.memory.set_shared_executor(None)

    def process_question(self, val, speaker_a_user_id, speaker_b_user_id):
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
            speaker_1_summary,
            speaker_2_summary,
        ) = self.answer_question(speaker_a_user_id, speaker_b_user_id, question, answer, category)

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
        }
        
        # 添加记忆摘要（如果启用）
        if self.enable_memory_summary:
            result["speaker_1_memory_summary"] = speaker_1_summary
            result["speaker_2_memory_summary"] = speaker_2_summary

        # Save results after each question is processed
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

        return result

    def process_data_file(self, file_path):
        with open(file_path, "r") as f:
            raw_data = json.load(f)
        data = normalize_dataset_records(raw_data)

        for idx, item in tqdm(enumerate(data), total=len(data), desc="Processing conversations"):
            qa = item.get("qa", [])
            conversation = item["conversation"]
            speaker_a = conversation["speaker_a"]
            speaker_b = conversation["speaker_b"]

            speaker_a_user_id = f"{speaker_a}_{idx}"
            speaker_b_user_id = f"{speaker_b}_{idx}"

            for question_item in tqdm(
                qa, total=len(qa), desc=f"Processing questions for conversation {idx}", leave=False
            ):
                result = self.process_question(question_item, speaker_a_user_id, speaker_b_user_id)
                self.results[idx].append(result)

                # Save results after each question is processed
                with open(self.output_path, "w") as f:
                    json.dump(self.results, f, indent=4)

        # Final save at the end
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

    def process_questions_parallel(self, qa_list, speaker_a_user_id, speaker_b_user_id, max_workers=1):
        def process_single_question(val):
            result = self.process_question(val, speaker_a_user_id, speaker_b_user_id)
            # Save results after each question is processed
            with open(self.output_path, "w") as f:
                json.dump(self.results, f, indent=4)
            return result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                tqdm(executor.map(process_single_question, qa_list), total=len(qa_list), desc="Answering Questions")
            )

        # Final save at the end
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

        return results
