import logging
import os
import random
import re
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
import traceback

from dotenv import load_dotenv
from tqdm import tqdm
from src.utils import compute_dataset_stats, stream_normalized_dataset
load_dotenv()  # Load environment variables from .env file
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
from mem0 import Memory

model_name = os.getenv("BASE_MODEL", "Qwen/Qwen3-14B")
DEFAULT_EMBEDDER_MODEL = "Pro/BAAI/bge-m3"
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"

class _StreamingDataset:
    """
    Lightweight iterable wrapper that streams normalized dataset records on demand.
    """

    def __init__(self, owner: "MemoryADD"):
        self._owner = owner

    def __iter__(self):
        if not self._owner.data_path:
            return iter(())
        return stream_normalized_dataset(self._owner.data_path)

    def __len__(self):
        stats = self._owner._dataset_stats or self._owner.load_data()
        return stats.get("total_conversations", 0)


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
                    "max_tokens": 8000,
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
                    "path": kwargs.get("qdrant_path", "./qdrant_data/mem0_global_msp"), # Default to global msp
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
        self._dataset_stats = None
        self.data = _StreamingDataset(self) if data_path else None
        self.is_graph = is_graph
        self.figure_view = kwargs.get("figure_view", False)
        self.fact_extraction_mode = self._normalize_mode(kwargs.get("fact_extraction_mode", "0"))
        self.memory_decision_mode = self._normalize_mode(kwargs.get("memory_decision_mode", "0"))
        self.add_mode = self._normalize_mode(kwargs.get("add_mode", "0"))
        self.llm_model = llm_model

        qdrant_path = config["vector_store"]["config"]["path"]
        os.makedirs(qdrant_path, exist_ok=True)
        self.collection_name = kwargs.get("collection_name") or self._derive_collection_name(qdrant_path)
        config["vector_store"]["config"]["collection_name"] = self.collection_name
        self.logger.info(
            f"Using Qdrant path '{qdrant_path}' with collection '{self.collection_name}' for memory writes."
        )

        # Prompt selection logic including MSP versions
        if self.fact_extraction_mode == "0":
            from mem0.configs.prompts import FACT_RETRIEVAL_PROMPT
            config["custom_fact_extraction_prompt"] = FACT_RETRIEVAL_PROMPT
        elif self.fact_extraction_mode == "15":
            from mem0.configs.prompts import FACT_RETRIEVAL_PROMPT_15_MSP
            config["custom_fact_extraction_prompt"] = FACT_RETRIEVAL_PROMPT_15_MSP
        # Add other modes as needed...

        if self.memory_decision_mode == "0":
            from mem0.configs.prompts import DEFAULT_UPDATE_MEMORY_PROMPT
            config["custom_update_memory_prompt"] = DEFAULT_UPDATE_MEMORY_PROMPT

        elif self.memory_decision_mode == "14.7":
             from mem0.configs.prompts import UPDATE_MEMORY_PROMPT_147_MSP
             config["custom_update_memory_prompt"] = UPDATE_MEMORY_PROMPT_147_MSP
        elif self.memory_decision_mode == "14.9":
             from mem0.configs.prompts import UPDATE_MEMORY_PROMPT_149_MSP
             config["custom_update_memory_prompt"] = UPDATE_MEMORY_PROMPT_149_MSP

        self._max_parallelism_cap = max(1, min(os.cpu_count() or 4, 12))
        self._pbar_lock = threading.Lock()
        self._memory_lock = threading.Lock()
        self._memory_semaphore = threading.Semaphore(self._max_parallelism_cap)
        self._io_executor = ThreadPoolExecutor(
            max_workers=max(2, min(self._max_parallelism_cap, 4)),
            thread_name_prefix="mem0-io",
        )

        if data_path:
            self.load_data()

        self.memory = Memory.from_config(
            config,
            logger=self.logger,
            shared_executor=self._io_executor,
        )
        self.memory.logger = self.logger
        self._ensure_collection_exists()

    def close(self):
        if getattr(self, "_io_executor", None):
            self._io_executor.shutdown(wait=True, cancel_futures=True)
            self._io_executor = None
            if hasattr(self.memory, "set_shared_executor"):
                self.memory.set_shared_executor(None)

    @staticmethod
    def _normalize_mode(value):
        normalized = "0" if value is None else str(value).strip()
        if not normalized:
            normalized = "0"
        return normalized

    def _build_timestamp_metadata(self, timestamp_value):
        if timestamp_value in (None, ""):
            return {}
        return {"timestamp": str(timestamp_value)}

    def load_data(self):
        if not self.data_path:
            raise ValueError("No data path configured for MemoryADD.")
        self._dataset_stats = compute_dataset_stats(self.data_path)
        self.data = _StreamingDataset(self)
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
                        # Pass modes if supported by add, otherwise they are in config
                    )
                return
            except Exception as exc:
                if attempt < max_attempts:
                    sleep_time = random.uniform(1, 3)
                    self.logger.warning(
                        "Request ID [%s] - Memory add failed (attempt %s/%s). Retrying in %.2fs. Error: %s",
                        request_id, attempt, max_attempts, sleep_time, exc,
                    )
                    time.sleep(sleep_time)
                    continue
                self.logger.error("Request ID [%s] - Failed to add memory. Error: %s", request_id, exc)
                raise

    def add_memories_for_global_observer(
        self,
        user_id,
        messages,
        timestamp,
        message_pbar=None,
    ):
        """
        Add messages to the global observer memory in batches (message-count based),
        matching the non-MSP ingestion semantics.

        When add_mode == "1", batches overlap by 1 message.
        """
        overlap_mode = self.add_mode == "1"
        for i in range(0, len(messages), self.batch_size):
            start_index = i
            if overlap_mode and i > 0:
                start_index = max(0, i - 1)
            end_index = min(len(messages), i + self.batch_size)
            batch_messages = messages[start_index:end_index]
            metadata = self._build_timestamp_metadata(timestamp)
            self.add_memory(user_id, batch_messages, metadata=metadata or None)
            if message_pbar:
                increment = len(batch_messages)
                if overlap_mode and i > 0:
                    increment = max(0, increment - 1)
                with self._pbar_lock:
                    message_pbar.update(increment)

    def process_conversation(self, item, idx, session_pbar=None, message_pbar=None):
        max_retries = 2
        for attempt in range(max_retries + 1):
            total_session_count = 0
            total_dialogue_count = 0
            sessions_processed = 0
            try:
                conversation = item.get("conversation") or {}
                
                # Global Observer ID for this conversation group (Static as per user request)
                global_user_id = "global_observer"

                session_keys = [
                    key for key in conversation.keys()
                    if key.startswith("session_") and not key.endswith("_date_time")
                ]
                total_session_count = len(session_keys)
                
                for key in session_keys:
                    chats = conversation.get(key, [])
                    if isinstance(chats, list):
                        total_dialogue_count += len(chats)

                for key in session_keys:
                    date_time_key = key + "_date_time"
                    timestamp = conversation.get(date_time_key)
                    chats = conversation.get(key, [])
                    if not isinstance(chats, list):
                        continue

                    sessions_processed += 1
                    if session_pbar:
                        with self._pbar_lock:
                            session_pbar.update(1)

                    messages = []
                    for chat in chats:
                        context = chat["text"]
                        speaker = chat.get("speaker", "Unknown")
                        if self.figure_view:
                            if "img_url" in chat and "blip_caption" in chat:
                                context += f" [Image: {chat['img_url']}] with caption: {chat['blip_caption']}"
                        messages.append({"role": "user", "content": f"{speaker}: {context}"})

                    self.add_memories_for_global_observer(
                        global_user_id,
                        messages,
                        timestamp,
                        message_pbar,
                    )

                self.logger.info(f"Conversation {idx} processed successfully.")
                return

            except Exception as e:
                self.logger.warning(f"Error on attempt {attempt + 1} for conversation {idx}: {e}")
                if attempt < max_retries:
                    time.sleep(random.uniform(1, 3))
                else:
                    self.logger.error(f"Failed to process conversation {idx}.")
                    return

    def process_all_conversations(self, max_workers=10):
        if not self.data_path:
            raise ValueError("No data path configured.")
        
        if not self._dataset_stats:
            self.load_data()
        
        # ... (Similar tqdm logic as original) ...
        # Simplified for brevity in this step, verifying essential logic
        
        stats = self._dataset_stats or {}
        total_conversations = stats.get("total_conversations", 0)
        
        # Reusing the loop structure from template
        successful_count = 0
        failed_count = 0
        
        resolved_workers = self._resolve_max_workers(max_workers)
        
        # ... Pbars ...
        conversation_pbar = tqdm(total=total_conversations, desc="📦 Conversation Groups")
        session_pbar = tqdm(total=stats.get("total_sessions", 0), desc="🧵 Session Progress", position=1, leave=False)
        message_pbar = tqdm(total=stats.get("total_dialogues", 0), desc="🗣️ Dialogue Turns", position=2, leave=False)

        futures = {}
        # ... logic ...
        
        try:
            with ThreadPoolExecutor(max_workers=resolved_workers, thread_name_prefix="mem-add") as executor:
                for idx, item in enumerate(stream_normalized_dataset(self.data_path)):
                    future = executor.submit(self.process_conversation, item, idx, session_pbar, message_pbar)
                    futures[future] = f"Conversation {idx}"
                    # ... draining logic ...
                    if len(futures) >= resolved_workers * 2:
                         done, _ = wait(tuple(futures.keys()), return_when=FIRST_COMPLETED)
                         for f in done:
                             futures.pop(f)
                             conversation_pbar.update(1)

                while futures:
                    done, _ = wait(tuple(futures.keys()))
                    for f in done:
                        futures.pop(f)
                        conversation_pbar.update(1)
        finally:
            conversation_pbar.close()
            session_pbar.close()
            message_pbar.close()
