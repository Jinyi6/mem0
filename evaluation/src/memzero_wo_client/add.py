import json
import logging
import os
import random
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import traceback

from dotenv import load_dotenv
from tqdm import tqdm
from src.utils import normalize_dataset_records
load_dotenv()  # Load environment variables from .env file
LOCAL_MEM0_PATH = os.getenv("LOCAL_MEM0_PATH")
if not LOCAL_MEM0_PATH:
    raise ValueError("环境变量 LOCAL_MEM0_PATH 未设置，请在 .env 文件中配置。")
# LOCAL_MEM0_PATH = "/Users/jinyi/Documents/code/memory/mem0" # ATTENTION: Change this to your local mem0 path

if not os.path.exists(LOCAL_MEM0_PATH):
    raise ImportError(f"指定的本地 mem0 路径不存在: {LOCAL_MEM0_PATH}")

import sys
if LOCAL_MEM0_PATH not in sys.path:
    sys.path.insert(0, LOCAL_MEM0_PATH)

print("="*80)
print(f"✅ 成功将本地 mem0 库路径添加到环境中: {LOCAL_MEM0_PATH}")
print("="*80)
from mem0 import Memory
load_dotenv()

model_name = os.getenv("BASE_MODEL", "Qwen/Qwen3-14B")
DEFAULT_EMBEDDER_MODEL = "Pro/BAAI/bge-m3"
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"



# Update custom instructions
custom_instructions = """
Generate personal memories that follow these guidelines:

1. Each memory should be self-contained with complete context, including:
   - The person's name, do not use "user" while creating memories
   - Personal details (career aspirations, hobbies, life circumstances)
   - Emotional states and reactions
   - Ongoing journeys or future plans
   - Specific dates when events occurred

2. Include meaningful personal narratives focusing on:
   - Identity and self-acceptance journeys
   - Family planning and parenting
   - Creative outlets and hobbies
   - Mental health and self-care activities
   - Career aspirations and education goals
   - Important life events and milestones

3. Make each memory rich with specific details rather than general statements
   - Include timeframes (exact dates when possible)
   - Name specific activities (e.g., "charity race for mental health" rather than just "exercise")
   - Include emotional context and personal growth elements

4. Extract memories only from user messages, not incorporating assistant responses

5. Format each memory as a paragraph with a clear narrative structure that captures the person's experience, challenges, and aspirations
"""



class MemoryADD:
    def __init__(self, data_path=None, batch_size=6, is_graph=False, logger=None, **kwargs):
        llm_config = kwargs.get("llm_config") or {}
        embedder_config = kwargs.get("embedder_config") or {}

        llm_model = llm_config.get("model") or model_name
        llm_base_url = llm_config.get("base_url") or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL
        llm_api_key = llm_config.get("api_key") or os.getenv("OPENAI_API_KEY")

        legacy_embedder_model = kwargs.get("embedder_model")
        legacy_embedder_dims = kwargs.get("embedding_dims")
        embedder_model = embedder_config.get("model") or legacy_embedder_model or DEFAULT_EMBEDDER_MODEL
        embedder_base_url = embedder_config.get("base_url") or llm_base_url
        embedder_api_key = embedder_config.get("api_key") or llm_api_key
        embedder_dims = embedder_config.get("embedding_dims")
        if embedder_dims is None:
            embedder_dims = legacy_embedder_dims
        if embedder_dims is not None:
            try:
                embedder_dims = int(embedder_dims)
            except (TypeError, ValueError):
                embedder_dims = None

        vector_store_dims = embedder_dims if embedder_dims is not None else 1024

        config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": llm_model,
                    "openai_base_url": llm_base_url,
                    "temperature": 0.1,
                    "max_tokens": 2000,
                    # "prompts": {
                    #     "memory_creation": custom_instructions
                    # }
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": embedder_model,
                    "openai_base_url": embedder_base_url,
                    "embedding_dims": embedder_dims if embedder_dims is not None else 1536,
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "path": kwargs.get("qdrant_path", "./qdrant_data/tmp"),
                    "on_disk": True,
                    "embedding_model_dims": vector_store_dims,
                },
            },
            "version": "v1.1",
        }
        if llm_api_key:
            config["llm"]["config"]["api_key"] = llm_api_key
        if embedder_api_key:
            config["embedder"]["config"]["api_key"] = embedder_api_key

        self.logger = logger if logger else logging.getLogger(__name__)
        self.batch_size = batch_size
        self.data_path = data_path
        self.data = None
        self.is_graph = is_graph
        self.figure_view = kwargs.get("figure_view", False)
        self.fact_extraction_mode = int(kwargs.get("fact_extraction_mode", "0"))
        self.memory_decision_mode = int(kwargs.get("memory_decision_mode", "0"))
        self.llm_model = llm_model

        qdrant_path = config["vector_store"]["config"]["path"]
        os.makedirs(qdrant_path, exist_ok=True)
        self.collection_name = kwargs.get("collection_name") or self._derive_collection_name(qdrant_path)
        config["vector_store"]["config"]["collection_name"] = self.collection_name
        self.logger.info(
            f"Using Qdrant path '{qdrant_path}' with collection '{self.collection_name}' for memory writes."
        )

        if self.fact_extraction_mode == 0:
            from mem0.configs.prompts import FACT_RETRIEVAL_PROMPT

            config["custom_fact_extraction_prompt"] = FACT_RETRIEVAL_PROMPT
        elif self.fact_extraction_mode == 1:
            from mem0.configs.prompts import FACT_RETRIEVAL_PROMPT_1

            config["custom_fact_extraction_prompt"] = FACT_RETRIEVAL_PROMPT_1
        elif self.fact_extraction_mode == 2:
            from mem0.configs.prompts import FACT_RETRIEVAL_PROMPT_2

            config["custom_fact_extraction_prompt"] = FACT_RETRIEVAL_PROMPT_2
        elif self.fact_extraction_mode == 3:
            from mem0.configs.prompts import FACT_RETRIEVAL_PROMPT_3

            config["custom_fact_extraction_prompt"] = FACT_RETRIEVAL_PROMPT_3

        if self.memory_decision_mode == 0:
            from mem0.configs.prompts import DEFAULT_UPDATE_MEMORY_PROMPT

            config["custom_memory_decision_prompt"] = DEFAULT_UPDATE_MEMORY_PROMPT
        elif self.memory_decision_mode == 1:
            from mem0.configs.prompts import UPDATE_MEMORY_PROMPT_1

            config["custom_memory_decision_prompt"] = UPDATE_MEMORY_PROMPT_1
        # please modify the prompt in mem0/configs/prompts.py if you want to change the memory decision prompt

        self._max_parallelism_cap = max(1, min(os.cpu_count() or 4, 6))
        self._pbar_lock = threading.Lock()
        self._memory_lock = threading.Lock()
        self._memory_semaphore = threading.Semaphore(self._max_parallelism_cap)

        if data_path:
            self.load_data()

        self.memory = Memory.from_config(config)
        # Then, set the logger attribute on the created instance
        self.memory.logger = self.logger
        self._ensure_collection_exists()

    def load_data(self):
        with open(self.data_path, "r") as f:
            raw_data = json.load(f)
        self.data = normalize_dataset_records(raw_data)
        return self.data

    @staticmethod
    def _derive_collection_name(qdrant_path: str) -> str:
        if not qdrant_path:
            return "mem0"
        workspace_name = Path(qdrant_path).resolve().parent.name
        candidate = re.sub(r"[^0-9a-zA-Z_]+", "_", workspace_name).strip("_")
        if not candidate:
            candidate = "mem0"
        if candidate[0].isdigit():
            candidate = f"c_{candidate}"
        return candidate[:120]

    def _ensure_collection_exists(self):
        vector_store = getattr(self.memory, "vector_store", None)
        if not vector_store:
            return
        try:
            vector_store.col_info()
        except Exception as exc:
            self.logger.warning(
                "Qdrant collection '%s' unavailable. Attempting to (re)create it. Reason: %s",
                self.collection_name,
                exc,
            )
            try:
                vector_store.create_col(
                    vector_store.embedding_model_dims,
                    getattr(vector_store, "on_disk", True),
                )
            except Exception as create_exc:
                raise RuntimeError(
                    f"Failed to initialize Qdrant collection '{self.collection_name}'."
                ) from create_exc

    def _resolve_max_workers(self, requested: int) -> int:
        if requested is None or requested <= 0:
            self.logger.warning("Received invalid max_workers=%s. Falling back to 1.", requested)
            return 1
        resolved = min(requested, self._max_parallelism_cap)
        if resolved != requested:
            self.logger.info(
                "Capping max_workers from %s to %s to respect parallelism limits and avoid rate limits.",
                requested,
                resolved,
            )
        return resolved

    def add_memory(self, user_id, message, metadata, retries=2):
        request_id = f"add-mem-{uuid.uuid4()}"
        max_attempts = retries + 1

        for attempt in range(1, max_attempts + 1):
            try:
                with self._memory_semaphore:
                    self.memory.add(
                        message,
                        user_id=user_id,
                        metadata=metadata,
                        fact_extraction_mode=self.fact_extraction_mode,
                        memory_decision_mode=self.memory_decision_mode,
                    )
                return
            except Exception as exc:
                if attempt < max_attempts:
                    sleep_time = random.uniform(1, 3)
                    self.logger.warning(
                        "Request ID [%s] - Memory add failed (attempt %s/%s). Retrying in %.2fs. Error: %s",
                        request_id,
                        attempt,
                        max_attempts,
                        sleep_time,
                        exc,
                    )
                    time.sleep(sleep_time)
                    continue

                self.logger.error(
                    "Request ID [%s] - Failed to add memory after %s attempts. Error: %s",
                    request_id,
                    max_attempts,
                    exc,
                )
                raise

    def add_memories_for_speaker(self, speaker, messages, timestamp, message_pbar=None):
        for i in range(0, len(messages), self.batch_size):
            batch_messages = messages[i : i + self.batch_size]
            self.add_memory(speaker, batch_messages, metadata={"timestamp": timestamp})
            if message_pbar:
                with self._pbar_lock:
                    message_pbar.update(self.batch_size/2)

    def process_conversation(self, item, idx, session_pbar=None, message_pbar=None):

        max_retries = 2  # 定义最大重试次数 (总共尝试 1 + 2 = 3 次)

        for attempt in range(max_retries + 1):
            try:
                conversation = item["conversation"]
                speaker_a = conversation["speaker_a"]
                speaker_b = conversation["speaker_b"]

                speaker_a_user_id = f"{speaker_a}_{idx}"
                speaker_b_user_id = f"{speaker_b}_{idx}"

                # delete all memories for the two users (使用锁确保线程安全)
                with self._memory_lock:
                    self.memory.delete_all(user_id=speaker_a_user_id)
                    self.memory.delete_all(user_id=speaker_b_user_id)

                session_keys = [
                    key
                    for key in conversation.keys()
                    if key.startswith("session_") and not key.endswith("_date_time")
                ]
                total_session_count = len(session_keys)
                total_dialogue_count = sum(
                    len(conversation.get(key, []))
                    if isinstance(conversation.get(key, []), list)
                    else 0
                    for key in session_keys
                )
                sessions_processed = 0
                dialogues_processed = 0

                for key in session_keys:
                    date_time_key = key + "_date_time"
                    timestamp = conversation[date_time_key]
                    chats = conversation[key]

                    messages = []
                    messages_reverse = []
                    for chat in chats:
                        context = chat['text']
                        if self.figure_view:
                            if "img_url" in chat and "blip_caption" in chat:
                                context += f" [Image: {chat['img_url']}] with caption: {chat['blip_caption']}"
                        if chat["speaker"] == speaker_a:
                            messages.append({"role": "user", "content": f"{speaker_a}: {context}"})
                            messages_reverse.append({"role": "assistant", "content": f"{speaker_a}: {context}"})
                        elif chat["speaker"] == speaker_b:
                            messages.append({"role": "assistant", "content": f"{speaker_b}: {context}"})
                            messages_reverse.append({"role": "user", "content": f"{speaker_b}: {context}"})
                        else:
                            raise ValueError(f"Unknown speaker: {chat['speaker']}")

                    self.add_memories_for_speaker(speaker_a_user_id, messages, timestamp, message_pbar)
                    self.add_memories_for_speaker(speaker_b_user_id, messages_reverse, timestamp, message_pbar)

                    sessions_processed += 1
                    dialogues_processed += len(chats)
                    if session_pbar:
                        with self._pbar_lock:
                            session_pbar.update(1)
                    

                self.logger.info(f"Conversation {idx} processed successfully on attempt {attempt + 1}.")
                return  # 成功后直接退出函数，不再重试

            except Exception as e:
                # --- 如果 try 块中任何地方发生错误，都会进入这里 ---
                error_details = traceback.format_exc()
                self.logger.warning(f"An error occurred on attempt {attempt + 1}/{max_retries + 1} for conversation {idx}. Error: {e}")
                self.logger.warning(f"Full error traceback: {error_details}")
                
                if attempt < max_retries:
                    # 如果还未达到最大重试次数，则等待一小段时间后重试
                    self.logger.info(f"Retrying conversation {idx}...")
                    time.sleep(random.uniform(1, 3))  # 随机等待1-3秒，避免同时重试
                else:
                    # 如果已经达到最大重试次数，记录严重错误并放弃
                    self.logger.error(f"Failed to process conversation {idx} after {max_retries + 1} attempts.")
                    self.logger.error(f"Final error details: {error_details}")
                    if session_pbar:
                        remaining_sessions = total_session_count - sessions_processed
                        if remaining_sessions > 0:
                            with self._pbar_lock:
                                session_pbar.update(remaining_sessions)
                    if message_pbar:
                        remaining_dialogues = total_dialogue_count - dialogues_processed
                        if remaining_dialogues > 0:
                            with self._pbar_lock:
                                message_pbar.update(remaining_dialogues)
                    # 这里可以选择是抛出异常让整个程序停止，还是仅仅跳过这个item
                    # raise e # 如果希望程序停止，取消这一行的注释
                    return # 如果希望仅仅跳过这个失败的item，保留这行

                
    # def process_all_conversations(self, max_workers=4):
    #     if not self.data:
    #         raise ValueError("No data loaded. Please set data_path and call load_data() first.")
    #     with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #         futures = [executor.submit(self.process_conversation, item, idx) for idx, item in enumerate(self.data)]

    #         for future in futures:
    #             future.result()

    def process_all_conversations(self, max_workers=10):
        if not self.data:
            raise ValueError("No data loaded. Please set data_path and call load_data() first.")

        total_conversations = len(self.data)
        total_sessions = 0
        total_dialogues = 0
        for item in self.data:
            conversation = item.get("conversation", {})
            session_keys = [
                key for key in conversation.keys() if key.startswith("session_") and not key.endswith("_date_time")
            ]
            total_sessions += len(session_keys)
            for key in session_keys:
                chats = conversation.get(key, [])
                if isinstance(chats, list):
                    total_dialogues += len(chats)

        resolved_workers = self._resolve_max_workers(max_workers)
        print(f"--- 总共需要处理 {total_conversations} 组对话 ---")
        if total_sessions:
            print(f"--- 覆盖 {total_sessions} 个会话，{total_dialogues} 条对话 ---")
        if resolved_workers != max_workers:
            print(f"⚙️ 调整 max_workers: 从 {max_workers} -> {resolved_workers}")
        else:
            print(f"⚙️ 使用 max_workers = {resolved_workers}")

        successful_count = 0
        failed_count = 0

        conversation_pbar = tqdm(total=total_conversations, desc="📦 Conversation Groups")
        session_pbar = (
            tqdm(total=total_sessions, desc="🧵 Session Progress", position=1, leave=False) if total_sessions else None
        )
        message_pbar = (
            tqdm(total=total_dialogues, desc="🗣️ Dialogue Turns", position=2, leave=False)
            if total_dialogues
            else None
        )

        try:
            with ThreadPoolExecutor(max_workers=resolved_workers) as executor:
                futures = {
                    executor.submit(self.process_conversation, item, idx, session_pbar, message_pbar): f"Conversation {idx}"
                    for idx, item in enumerate(self.data)
                }

                for future in as_completed(futures):
                    conversation_id = futures[future]
                    try:
                        future.result()
                        successful_count += 1
                    except Exception as e:
                        error_details = traceback.format_exc()
                        failed_count += 1
                        conversation_pbar.write(
                            f"\n---[Error {failed_count}] ❌ Error processing {conversation_id}: {str(e)} ---\n"
                        )
                        conversation_pbar.write(f"{error_details}\n")
                    finally:
                        conversation_pbar.update(1)
        finally:
            conversation_pbar.close()
            if session_pbar:
                session_pbar.close()
            if message_pbar:
                message_pbar.close()

        print(f"\n✅ All conversations processed. Success: {successful_count}, Failed: {failed_count}")
