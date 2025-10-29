import json
import logging
import os
import random
import re
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
import math
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI
from prompts import ANSWER_PROMPT, ANSWER_PROMPT_GRAPH
from tqdm import tqdm
from mem0 import Memory
from src.utils import compute_dataset_stats, stream_normalized_dataset

load_dotenv()

# Set the OpenAI API key

DEFAULT_LLM_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen3-14B")
DEFAULT_EMBEDDER_MODEL = "Pro/BAAI/bge-m3"
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
os.environ["MODEL"] = DEFAULT_LLM_MODEL


# 新增：用于 reranking 和关键词提取
try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    print("⚠️ sentence-transformers not installed. Reranking will be disabled.")

try:
    from keybert import KeyBERT
    KEYBERT_AVAILABLE = True
except ImportError:
    KEYBERT_AVAILABLE = False
    print("⚠️ keybert not installed. Will use fallback keyword extraction.")

# 全局 KeyBERT 实例（单例模式，避免重复加载）
_KEYBERT_MODEL = None
_KEYBERT_LOCK = threading.Lock()

def get_keybert_model():
    """获取全局 KeyBERT 单例，线程安全"""
    global _KEYBERT_MODEL
    if _KEYBERT_MODEL is None:
        with _KEYBERT_LOCK:
            if _KEYBERT_MODEL is None:  # Double-check locking
                if KEYBERT_AVAILABLE:
                    print("📥 首次加载 KeyBERT 模型（只需一次）...")
                    try:
                        # 🔥 配置 HuggingFace Hub 使用镜像站
                        import huggingface_hub
                        # 设置镜像端点
                        huggingface_hub.constants.HUGGINGFACE_CO_URL_TEMPLATE = "https://hf-mirror.com/{repo_id}/resolve/{revision}/{filename}"
                        huggingface_hub.constants.HUGGINGFACE_CO_URL_HOME = "https://hf-mirror.com"
                        
                        # 使用默认模型（会使用代理下载或从缓存加载）
                        _KEYBERT_MODEL = KeyBERT()
                        print("✅ KeyBERT 模型加载成功！")
                            
                    except Exception as e:
                        print(f"❌ KeyBERT 加载失败: {e}")
                        print(f"   将使用 fallback 关键词提取")
                        _KEYBERT_MODEL = False  # 标记为失败，避免重复尝试
    return _KEYBERT_MODEL if _KEYBERT_MODEL is not False else None


# 全局 Reranker 实例（单例模式，避免重复加载和 meta tensor 错误）
_RERANKER_MODEL = None
_RERANKER_LOCK = threading.Lock()

def get_reranker_model(model_name="cross-encoder/ms-marco-MiniLM-L-12-v2"):
    """获取全局 Reranker 单例，线程安全，解决 meta tensor 和 502 错误"""
    global _RERANKER_MODEL
    if _RERANKER_MODEL is None:
        with _RERANKER_LOCK:
            if _RERANKER_MODEL is None:
                if RERANKER_AVAILABLE:
                    print(f"📥 首次加载 Reranker 模型: {model_name}（只需一次）...")
                    try:
                        import os
                        # 设置镜像站环境变量，避免 502 错误
                        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
                        
                        # 使用 device_map="cpu" 避免 meta tensor 问题
                        _RERANKER_MODEL = CrossEncoder(
                            model_name, 
                            max_length=512,
                            device='cpu'  # 先加载到 CPU，避免 meta tensor 错误
                        )
                        print("✅ Reranker 模型加载成功！")
                    except Exception as e:
                        print(f"❌ Reranker 加载失败: {e}")
                        print("   将使用基于分数的排序作为 fallback")
                        _RERANKER_MODEL = False
    return _RERANKER_MODEL if _RERANKER_MODEL is not False else None


class IncrementalResultsWriter:
    """
    Append-only writer that buffers conversation results and materializes the
    final JSON payload once, avoiding repeated full rewrites under concurrency.
    """

    def __init__(self, output_path: str, flush_every: int = 8):
        self._output_path = Path(output_path)
        self._temp_path = self._output_path.with_name(self._output_path.name + ".partial")
        self._flush_every = max(1, flush_every)
        self._buffer = []
        self._lock = threading.Lock()

        parent = self._output_path.parent
        if parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
        temp_parent = self._temp_path.parent
        if temp_parent and not temp_parent.exists():
            temp_parent.mkdir(parents=True, exist_ok=True)
        if self._temp_path.exists():
            self._temp_path.unlink()

    def append(self, conversation_idx: int, records):
        payload = {"idx": conversation_idx, "results": records}
        with self._lock:
            self._buffer.append(payload)
            if len(self._buffer) >= self._flush_every:
                self._flush_locked()

    def _flush_locked(self):
        if not self._buffer:
            return
        with self._temp_path.open("a", encoding="utf-8") as handle:
            for item in self._buffer:
                json.dump(item, handle, ensure_ascii=False)
                handle.write("\n")
        self._buffer.clear()

    def flush(self):
        with self._lock:
            self._flush_locked()

    def finalize(self):
        self.flush()
        aggregated = {}
        if self._temp_path.exists():
            with self._temp_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    key = str(payload.get("idx"))
                    aggregated.setdefault(key, []).extend(payload.get("results", []))

        tmp_path = self._output_path.with_name(self._output_path.name + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(aggregated, handle, indent=4, ensure_ascii=False)
        tmp_path.replace(self._output_path)

        if self._temp_path.exists():
            self._temp_path.unlink()


class MemorySearch:
    """
    记忆搜索类，用于处理基于对话的问答任务
    
    支持多种搜索方法：
    - 方法3: 问题分解 + 多查询搜索 + Reranking
    - 方法5: 关键词提取 + PRF扩展 + Reranking + MMR多样化
    """
    
    def __init__(
        self,
        output_path="results.json",
        top_k=10,
        filter_memories=False,
        is_graph=False,
        logger=None,
        qdrant_path=None,
        search_method="5",
        answer_mode="0",
        collection_name=None,
        llm_config=None,
        embedder_config=None,
        search_llm_config=None,
        answer_llm_config=None,
    ):
        llm_config = llm_config or {}
        search_llm_config = search_llm_config or llm_config
        answer_llm_config = answer_llm_config or search_llm_config
        embedder_config = embedder_config or {}
        env_base_model = os.getenv("BASE_MODEL", DEFAULT_LLM_MODEL)
        env_base_url = os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL
        env_api_key = os.getenv("OPENAI_API_KEY")

        self.llm_model = search_llm_config.get("model") or env_base_model
        self.search_llm_model = self.llm_model
        llm_base_url = search_llm_config.get("base_url") or env_base_url
        llm_api_key = search_llm_config.get("api_key") or env_api_key

        self.answer_llm_model = answer_llm_config.get("model") or self.llm_model
        answer_llm_base_url = answer_llm_config.get("base_url") or llm_base_url
        answer_llm_api_key = answer_llm_config.get("api_key") or llm_api_key

        embedder_model = embedder_config.get("model") or DEFAULT_EMBEDDER_MODEL
        embedder_base_url = embedder_config.get("base_url") or llm_base_url
        embedder_api_key = embedder_config.get("api_key") or llm_api_key
        embedder_dims = embedder_config.get("embedding_dims")
        if embedder_dims is not None:
            try:
                embedder_dims = int(embedder_dims)
            except (TypeError, ValueError):
                embedder_dims = None

        vector_store_dims = embedder_dims if embedder_dims is not None else 1024

        qdrant_path = qdrant_path or "./qdrant_data/tmp"
        os.makedirs(qdrant_path, exist_ok=True)
        self.logger = logger if logger else logging.getLogger(__name__)
        self.collection_name = collection_name or self._derive_collection_name(qdrant_path)
        os.environ["MODEL"] = self.llm_model
        self.embedder_model = embedder_model
        search_client_kwargs = {}
        if llm_base_url:
            search_client_kwargs["base_url"] = llm_base_url
        if llm_api_key:
            search_client_kwargs["api_key"] = llm_api_key
        answer_client_kwargs = {}
        if answer_llm_base_url:
            answer_client_kwargs["base_url"] = answer_llm_base_url
        if answer_llm_api_key:
            answer_client_kwargs["api_key"] = answer_llm_api_key
        config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": self.llm_model,
                    "openai_base_url": llm_base_url,
                    "temperature": 0.1,
                    "max_tokens": 2000,
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
                    "path": qdrant_path,
                    "on_disk": True,
                    "embedding_model_dims": vector_store_dims,
                    "collection_name": self.collection_name,
                },
            },
            "version": "v1.1",
        }
        if llm_api_key:
            config["llm"]["config"]["api_key"] = llm_api_key
        if embedder_api_key:
            config["embedder"]["config"]["api_key"] = embedder_api_key
        self.top_k = top_k
        self.search_client = OpenAI(**search_client_kwargs)
        if (
            answer_llm_base_url == llm_base_url
            and answer_llm_api_key == llm_api_key
        ):
            self.answer_client = self.search_client
        else:
            self.answer_client = OpenAI(**answer_client_kwargs)
        self.output_path = output_path
        self.filter_memories = filter_memories
        self.is_graph = is_graph
        self.search_method = self._normalize_mode(search_method)
        answer_mode = self._normalize_mode(answer_mode)
        self.answer_mode = answer_mode
        self.qdrant_path = qdrant_path
        self._max_parallelism_cap = max(1, min(os.cpu_count() or 4, 12))
        self.logger.info(
            "Using Qdrant path '%s' with collection '%s' for search reads.",
            self.qdrant_path,
            self.collection_name,
        )

        self._io_executor = ThreadPoolExecutor(
            max_workers=max(2, min(self._max_parallelism_cap, 4)),
            thread_name_prefix="mem0-search-io",
        )

        self.memory = Memory.from_config(
            config,
            logger=self.logger,
            shared_executor=self._io_executor,
        )
        self.memory.logger = self.logger
        self._ensure_collection_exists()

        self._results_state_lock = threading.Lock()
        self._results_buffer = {}
        self._expected_results_per_conversation = []
        self._results_writer = IncrementalResultsWriter(output_path)
        self._dataset_stats = None

        self._keybert_checked = False
        self._keybert_model = None
        self._reranker_checked = False
        self._reranker_model_instance = None
        self._rerank_executor = None
        self._rerank_executor_lock = threading.Lock()

        if self.is_graph:
            self.ANSWER_PROMPT = ANSWER_PROMPT_GRAPH
        else:
            if answer_mode == "0":
                self.ANSWER_PROMPT = ANSWER_PROMPT
            elif answer_mode == "1":
                from prompts import ANSWER_PROMPT_1

                self.ANSWER_PROMPT = ANSWER_PROMPT_1
            elif answer_mode == "2":
                from prompts import ANSWER_PROMPT_2

                self.ANSWER_PROMPT = ANSWER_PROMPT_2
            elif answer_mode == "3":
                from prompts import ANSWER_PROMPT_3

                self.ANSWER_PROMPT = ANSWER_PROMPT_3
            elif answer_mode == "4":
                from prompts import ANSWER_PROMPT_4

                self.ANSWER_PROMPT = ANSWER_PROMPT_4
            elif answer_mode == "5":
                from prompts import ANSWER_PROMPT_5
                self.ANSWER_PROMPT = ANSWER_PROMPT_5
            elif answer_mode == "6":
                from prompts import ANSWER_PROMPT_6

                self.ANSWER_PROMPT = ANSWER_PROMPT_6
            elif answer_mode == "7":
                from prompts import ANSWER_PROMPT_7

                self.ANSWER_PROMPT = ANSWER_PROMPT_7
            elif answer_mode == "10":
                from prompts import ANSWER_PROMPT_10

                self.ANSWER_PROMPT = ANSWER_PROMPT_10
            else:
                self.logger.warning("Unknown answer_mode '%s'. Falling back to default prompt.", answer_mode)
                self.ANSWER_PROMPT = ANSWER_PROMPT

        self.speaker_1_full_memories = []
        self.speaker_2_full_memories = []
            
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
                "Qdrant collection '%s' not found for search. Attempting to recreate it. Reason: %s",
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
                    f"Failed to ensure Qdrant collection '{self.collection_name}' exists for search."
                ) from create_exc

    @staticmethod
    def _normalize_mode(value):
        """
        Normalize mode identifiers to non-empty strings.
        """
        normalized = "0" if value is None else str(value).strip()
        if not normalized:
            normalized = "0"
        return normalized

    @staticmethod
    @contextmanager
    def _thread_pool(max_workers: int, prefix: str):
        """
        ThreadPoolExecutor wrapper that guarantees shutdown with cancelled futures on errors.
        """
        executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=prefix)
        try:
            yield executor
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    def _ensure_keybert_model(self):
        if not self._keybert_checked:
            self._keybert_model = get_keybert_model()
            self._keybert_checked = True
        return self._keybert_model

    def _ensure_reranker_model(self):
        if not self._reranker_checked:
            self._reranker_model_instance = get_reranker_model()
            self._reranker_checked = True
        return self._reranker_model_instance

    def _get_rerank_executor(self):
        if self._rerank_executor is None:
            with self._rerank_executor_lock:
                if self._rerank_executor is None:
                    self._rerank_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="mem-search-rerank")
        return self._rerank_executor

    @staticmethod
    def _format_memory_line(item):
        """
        Render memory entries with a consistent timestamp display.
        """
        display_ts = item.get("timestamp_display") or item.get("timestamp") or ""
        memory_text = item.get("memory", "")
        return f"{display_ts}: {memory_text}"

    def _record_result(self, conversation_idx: int, result):
        """
        Buffer per-conversation results and flush them in batches to the writer
        once the expected number of QA pairs have completed.
        """
        payload = None
        with self._results_state_lock:
            bucket = self._results_buffer.setdefault(conversation_idx, [])
            bucket.append(result)
            expected = 0
            if (
                self._expected_results_per_conversation
                and conversation_idx < len(self._expected_results_per_conversation)
            ):
                expected = self._expected_results_per_conversation[conversation_idx]
            if expected and len(bucket) >= expected:
                payload = (conversation_idx, list(bucket))
                self._results_buffer.pop(conversation_idx, None)
        if payload:
            idx, records = payload
            self._results_writer.append(idx, records)

    def _resolve_max_workers(self, requested: int) -> int:
        if requested is None or requested <= 0:
            self.logger.warning("Received invalid max_workers=%s. Falling back to 1.", requested)
            return 1
        resolved = min(requested, self._max_parallelism_cap)
        if resolved != requested:
            self.logger.info(
                "Capping search max_workers from %s to %s to stay within safe parallelism limits.",
                requested,
                resolved,
            )
        return resolved


    def search_memory(self, user_id, query, max_retries=5, pbar=None, limit=None):
        """
        搜索指定用户的记忆
        
        Args:
            user_id: 用户ID
            query: 搜索查询
            max_retries: 最大重试次数
            pbar: 进度条对象
            
        Returns:
            tuple: (semantic_memories, graph_memories, search_time)
        """
        if limit is None:
            limit = self.top_k
        start_time = time.perf_counter()
        sleep_penalty = 0.0
        attempts = 0
        retries = 0
        retry_sleeps = []
        while retries < max_retries:
            attempts += 1
            try:
                memory_payload = self.memory.search(
                    query,
                    user_id=user_id,
                    limit=limit,
                )
                break
            except Exception as e:
                retries += 1
                error_message = str(e)
                error_lower = error_message.lower()
                if "collection" in error_lower and "not found" in error_lower:
                    self.logger.warning(
                        "Search collection missing for user %s (attempt %s/%s). Recreating and retrying. Error: %s",
                        user_id,
                        retries,
                        max_retries,
                        error_message,
                    )
                    self._ensure_collection_exists()
                    continue

                backoff = min(8.0, 0.75 * (2 ** (retries - 1))) + random.uniform(0.1, 0.6)
                retry_sleeps.append(backoff)
                self.logger.warning(
                    "Retrying search for user %s...%s/%s | backoff=%.2fs | error_type=%s | error=%s\n%s",
                    user_id,
                    retries,
                    max_retries,
                    backoff,
                    type(e).__name__,
                    error_message,
                    traceback.format_exc(),
                )
                if retries >= max_retries:
                    raise
                sleep_penalty += backoff
                time.sleep(backoff)  # 减少重试延迟

        end_time = time.perf_counter()
        search_duration = max(0.0, end_time - start_time - sleep_penalty)

        if "memory_payload" not in locals():
            memory_payload = {"results": []}

        raw_memories = []
        search_metrics = {}
        graph_payload = None
        if isinstance(memory_payload, dict):
            raw_memories = memory_payload.get("results") or []
            search_metrics = memory_payload.get("metrics") or {}
            graph_payload = memory_payload.get("relations")
        else:
            raw_memories = memory_payload or []

        postprocess_start = time.perf_counter()
        semantic_memories = []
        for memory in raw_memories:
            metadata = memory.get("metadata") or {}
            timestamp_value = metadata.get("timestamp")
            if timestamp_value is None:
                timestamp_value = memory.get("created_at")
            timestamp_epoch = metadata.get("timestamp_epoch")

            semantic_memories.append(
                {
                    "memory": memory["memory"],
                    "timestamp": timestamp_value,
                    "timestamp_display": timestamp_value,
                    "timestamp_epoch": timestamp_epoch,
                    "score": round(memory["score"], 2),
                }
            )
        postprocess_end = time.perf_counter()
        postprocess_time = postprocess_end - postprocess_start

        embedding_time = float(search_metrics.get("embedding_sec") or 0.0)
        vector_time = float(search_metrics.get("vector_query_sec") or 0.0) + float(
            search_metrics.get("vector_postprocess_sec") or 0.0
        )
        graph_time = float(search_metrics.get("graph_query_sec") or 0.0)
        accounted_time = embedding_time + vector_time + graph_time + postprocess_time
        overhead_time = max(0.0, search_duration - accounted_time)

        cache_hit = search_metrics.get("embedding_cache_hit")
        cache_note = f", cache_hit={cache_hit}" if cache_hit is not None else ""
        retry_note = f", retry_sleeps={['%.2f' % s for s in retry_sleeps]}" if retry_sleeps else ""

        load_note = ""
        load_tuple = None
        if hasattr(os, "getloadavg"):
            try:
                load_tuple = os.getloadavg()
                load_note = f", load_avg=({load_tuple[0]:.1f},{load_tuple[1]:.1f},{load_tuple[2]:.1f})"
            except (OSError, ValueError):
                load_tuple = None

        message = (
            f"Search success for user {user_id} in {search_duration:.2f}s "
            f"(attempts={attempts}, retry_sleep={sleep_penalty:.2f}s{cache_note}{retry_note}{load_note}) | "
            f"breakdown: embedding={embedding_time:.2f}s, vector={vector_time:.2f}s, "
            f"graph={graph_time:.2f}s, postprocess={postprocess_time:.2f}s, other={overhead_time:.2f}s"
        )
        self.logger.info(message)

        if load_tuple and load_tuple[0] >= 200:
            self.logger.warning(
                "High system load detected during search for user %s (1m=%.1f, 5m=%.1f, 15m=%.1f). "
                "Embedding concurrency capped at %s to mitigate pressure.",
                user_id,
                load_tuple[0],
                load_tuple[1],
                load_tuple[2],
                getattr(self.memory, "_max_inflight_embeddings", "n/a"),
            )

        graph_memories = graph_payload

        return semantic_memories, graph_memories, search_duration
    
    def _log_llm_call(
        self,
        request_id,
        attempt,
        max_retries,
        prompt_components,
        full_prompt,
        response_content,
        status,
    ):
        """
        精简记录LLM调用信息，避免将完整提示和记忆写入日志导致的巨量输出。
        """
        question_preview = (prompt_components.get("question") or "").strip().replace("\n", " ")
        if len(question_preview) > 120:
            question_preview = question_preview[:117] + "..."

        speaker_1_memories = prompt_components.get("speaker_1_memories") or []
        speaker_2_memories = prompt_components.get("speaker_2_memories") or []
        mem1_count = len(speaker_1_memories)
        mem2_count = len(speaker_2_memories)

        response_preview = (response_content or "").strip().replace("\n", " ")
        if len(response_preview) > 160:
            response_preview = response_preview[:157] + "..."

        prompt_chars = len(full_prompt or "")

        log_message = (
            "LLM Call [%s] attempt %d/%d status=%s | question='%s' | mem1=%d mem2=%d | prompt_chars=%d | response='%s'"
        )
        if "Success" in status:
            self.logger.info(
                log_message,
                request_id,
                attempt,
                max_retries,
                status,
                question_preview,
                mem1_count,
                mem2_count,
                prompt_chars,
                response_preview or "N/A",
            )
        else:
            self.logger.error(
                log_message,
                request_id,
                attempt,
                max_retries,
                status,
                question_preview,
                mem1_count,
                mem2_count,
                prompt_chars,
                response_preview or "N/A",
            )

    def safe_chat(self, model=None, messages=None, temperature=0.0, sleep_time=20):
        """
        安全的LLM调用，自动处理速率限制
        
        Args:
            model: 模型名称
            messages: 消息列表
            temperature: 温度参数
            sleep_time: 触发速率限制时的等待时间（秒）
            
        Returns:
            LLM响应对象
        """
        if messages is None:
            raise ValueError("messages 必须提供。")
        model_name = model or self.llm_model or os.getenv("MODEL", "Qwen/Qwen3-14B")
        while True:
            try:
                return self.search_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                )
            except Exception as e:
                s = str(e)
                if ("429" in s) or ("TPM" in s) or ("rate limit" in s.lower()):
                    print(f"⚠️ 触发速率限制，等待 {sleep_time} 秒后重试...")
                    time.sleep(sleep_time)
                    continue
                raise

    def __load_full_memories(self, speaker_1_user_id, speaker_2_user_id):
        t1 = time.time()
        speaker1full = self.search_memory(user_id=speaker_1_user_id, query="get all", limit=500)
        t2 = time.time()

        print(f'Load full memories in speaker 1 cost {t2-t1} seconds')

        t1 = time.time()
        speaker2full = self.search_memory(user_id=speaker_2_user_id, query="get all", limit=500)
        t2 = time.time()

        print(f'Load full memories in speaker 1 cost {t2-t1} seconds')

        # for item in speaker1full:
        #     record = str(item['timestamp'] + ' : ' + item['memory'])
        #     self.speaker_1_full_memories.append(record)
        #
        # for item in speaker2full:
        #     record = str(item['timestamp'] + ' : ' + item['memory'])
        #     self.speaker_2_full_memories.append(record)

    def vibe_rerank(self, question, documents , top_k=30) -> list:
        docs_with_idx = "\n".join([f"[{i}] {d}" for i, d in enumerate(documents)])
        prompt = f"""You are an intelligent reranker.
                    Question: {question}
                    Documents:
                    {docs_with_idx}
                    Please identify the most relevant documents. 
                    If there are no such texts just return "Not found"
                    Return only the indices in parentheses, e.g. (2,5,1)
                    Top indices:"""
        raw=""
        try:
            resp = self.safe_chat(
                model=self.llm_model,
                messages=[{"role": "system", "content": prompt}],
                temperature=0.8,
            )
            if resp.choices[0].message.content is not None:
                raw = resp.choices[0].message.content.strip()

            nums = re.findall(r'\d+', raw)
            indices = []
            seen = set()
            for n in nums:
                idx = int(n)
                if idx not in seen and 0 <= idx < len(documents):
                    indices.append(idx)
                    seen.add(idx)
            if not indices:
                raise ValueError("No valid indices parsed")
            return indices[:top_k]

        except Exception as e:
            print(f"[vibe_rerank] {e} | raw LLM output: {raw[:80]}...")
            return []


    def Search(self, speaker_1_user_id, speaker_2_user_id, question, search_method, top_k_rerank=15, pbar=None):
        """
        执行记忆搜索，支持多种搜索策略
        
        Args:
            speaker_1_user_id: 说话者1的用户ID
            speaker_2_user_id: 说话者2的用户ID
            question: 问题文本
            search_method: 搜索方法编号 (3=问题分解, 5=PRF+MMR, 其他=默认搜索)
            top_k_rerank: 重排序后返回的top-k记忆数量
            pbar: 进度条对象
            
        Returns:
            tuple: (search_1_memory, search_2_memory) - 两个说话者的记忆列表
        """
        top_k_rerank = self.top_k
        search_method = self._normalize_mode(search_method)

        if search_method == "3":
            # ========== 方法3: 问题分解 + 多查询搜索 + Reranking ==========
            NUM_SUB_QUESTIONS = 5  # 子问题数量
            MAX_WORKERS = min(3, self._max_parallelism_cap)  # 并发工作线程数
            
            question_list = []
            question_list.append(question)
            q_prompt = f"""
            You are a precise decomposition model. Split the following user question into exactly five (5) sub-questions that together cover all information in the original.

            Hard constraints (follow ALL):
            1) Fixed count: produce exactly five sub-questions.
            2) No omissions: preserve every fact, constraint, number, date, name, location, condition, and qualifier.
            3) Preserve wording cues: keep key terms/keywords from the original; do not paraphrase them away.
            4) No invention: do not add assumptions or external knowledge.
            5) Non-overlap: minimize duplication between sub-questions; each should target a distinct aspect.
            6) Standalone: each sub-question must be a complete, answerable question.
            7) Language: keep the same language as the input question.
            8) Entities/numerals: retain original names, quantities, units, and times.
            9) Temporal/logical constraints: keep if/when/before/after/except/unless conditions intact.

            QUESTION:
            {question}

            OUTPUT FORMAT (return ONLY these five lines, nothing else):
            1. [SUB-QUESTION 1]
            2. [SUB-QUESTION 2]
            3. [SUB-QUESTION 3]
            4. [SUB-QUESTION 4]
            5. [SUB-QUESTION 5]
            """

            q_response = self.safe_chat(
                    model=self.llm_model,
                    messages=[{"role": "system", "content": q_prompt}],
                    temperature=0.8,
            )
            if q_response.choices[0].message.content is not None:
                raw_text = q_response.choices[0].message.content.strip()
            else:
                raw_text = ""
            question_list = re.findall(r'^\s*\d+\.\s*(.+)', raw_text, flags=re.M)
            
            # 收集所有子问题的搜索结果（去重）
            a_mem_map = {}  # speaker_1 的记忆映射
            b_mem_map = {}  # speaker_2 的记忆映射
            
            search_time_by_user = {speaker_1_user_id: 0.0, speaker_2_user_id: 0.0}

            def search_(uid, question):
                """辅助函数：搜索单个用户的记忆"""
                memories, graph_memories, duration = self.search_memory(uid, question)
                return uid, memories, graph_memories, duration
            
            # 并发搜索所有子问题
            with self._thread_pool(MAX_WORKERS, "mem-search-subq") as executor:
                futures = []
                for q in question_list:
                    futures.append(executor.submit(search_, speaker_1_user_id, q))
                    futures.append(executor.submit(search_, speaker_2_user_id, q))
                
                # 合并结果，保留每条记忆的最高分数
                for f in as_completed(futures):
                    uid, mems, graph_mems, duration = f.result()
                    search_time_by_user[uid] = search_time_by_user.get(uid, 0.0) + float(duration or 0.0)
                    for m in mems:
                        key = m["memory"]
                        if uid == speaker_1_user_id:
                            if key not in a_mem_map or m["score"] > a_mem_map[key]["score"]:
                                a_mem_map[key] = m
                        else:
                            if key not in b_mem_map or m["score"] > b_mem_map[key]["score"]:
                                b_mem_map[key] = m
            
            # 使用 Reranker 重新排序
            reranker = self._ensure_reranker_model()
            
            if reranker is not None:
                try:
                    # Rerank for speaker A
                    if a_mem_map:
                        a_candidates = list(a_mem_map.values())
                        a_pairs = [[question, m["memory"]] for m in a_candidates]
                        a_scores = reranker.predict(a_pairs, show_progress_bar=False)
                        
                        # Update scores with reranker scores
                        for i, mem in enumerate(a_candidates):
                            mem["rerank_score"] = float(a_scores[i])
                        
                        # Sort by rerank score
                        a_top = sorted(a_candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k_rerank]
                    else:
                        a_top = []
                    
                    # Rerank for speaker B
                    if b_mem_map:
                        b_candidates = list(b_mem_map.values())
                        b_pairs = [[question, m["memory"]] for m in b_candidates]
                        b_scores = reranker.predict(b_pairs, show_progress_bar=False)
                        
                        # Update scores with reranker scores
                        for i, mem in enumerate(b_candidates):
                            mem["rerank_score"] = float(b_scores[i])
                        
                        # Sort by rerank score
                        b_top = sorted(b_candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k_rerank]
                    else:
                        b_top = []
                        
                except Exception as e:
                    print(f"⚠️ Reranking failed: {e}. Falling back to score-based ranking.")
                    # Fallback to original scoring
                    a_top = sorted(a_mem_map.values(), key=lambda x: x["score"], reverse=True)[:top_k_rerank]
                    b_top = sorted(b_mem_map.values(), key=lambda x: x["score"], reverse=True)[:top_k_rerank]
            else:
                # Fallback when reranker is not available
                a_top = sorted(a_mem_map.values(), key=lambda x: x["score"], reverse=True)[:top_k_rerank]
                b_top = sorted(b_mem_map.values(), key=lambda x: x["score"], reverse=True)[:top_k_rerank]
            
            # 格式化输出
            search_1_memory = [self._format_memory_line(m) for m in a_top]
            search_2_memory = [self._format_memory_line(m) for m in b_top]
            
            return (
                search_1_memory,
                search_2_memory,
                None,
                None,
                search_time_by_user.get(speaker_1_user_id, 0.0),
                search_time_by_user.get(speaker_2_user_id, 0.0),
            )
           
        elif search_method == "5":
            # ========== 方法5: PRF + 关键词提取 + Reranking + MMR多样化 ==========
            # 配置参数
            PRF_K = 20  # PRF使用的文档数
            MAX_KEYWORDS = 5  # 基础关键词数量
            PRF_KEYWORDS = 6  # PRF关键词数量
            MAX_WORKERS = min(3, self._max_parallelism_cap)  # 并发线程数
            LAMBDA_DIV = 0.6  # MMR多样性参数
            RERANK_WEIGHT = 0.8  # 重排序分数权重
            RECENCY_WEIGHT = 0.2  # 时间衰减权重
            HALF_LIFE_DAYS = 7.0  # 时间衰减半衰期（天）

            # ---------- 工具函数（仅在此函数作用域内） ----------
            def _safe_lower_tokens(text):
                """安全地提取文本的单词token（小写）"""
                try:
                    return re.findall(r"\b\w+\b", (text or "").lower())
                except Exception:
                    return []

            def _jaccard_sim(a_text, b_text):
                """计算两个文本的Jaccard相似度"""
                a = set(_safe_lower_tokens(a_text))
                b = set(_safe_lower_tokens(b_text))
                if not a or not b:
                    return 0.0
                return len(a & b) / float(len(a | b))

            def _mmr_select(candidates, relevance_scores, k, lambda_div=0.6):
                """
                使用MMR算法选择多样化的候选项
                
                Args:
                    candidates: 候选记忆列表，每项必须包含'memory'字段
                    relevance_scores: 相关性分数字典
                    k: 选择的数量
                    lambda_div: 多样性参数 (0-1)，越大越注重相关性
                """
                selected = []
                selected_texts = []
                remaining = list(candidates)
                while remaining and len(selected) < k:
                    best_item = None
                    best_val = -1e9
                    for item in remaining:
                        mem_text = item.get("memory", "")
                        rel = relevance_scores.get(mem_text, 0.0)
                        if selected_texts:
                            max_sim = max(_jaccard_sim(mem_text, t) for t in selected_texts)
                        else:
                            max_sim = 0.0
                        val = lambda_div * rel - (1.0 - lambda_div) * max_sim
                        if val > best_val:
                            best_val = val
                            best_item = item
                    if best_item is None:
                        break
                    selected.append(best_item)
                    selected_texts.append(best_item.get("memory", ""))
                    remaining.remove(best_item)
                return selected

            def _extract_timestamp_seconds(item):
                """
                Retrieve a timestamp in epoch seconds from a memory item.
                """
                if not item:
                    return 0.0
                ts_epoch = item.get("timestamp_epoch")
                if ts_epoch is not None:
                    try:
                        return float(ts_epoch)
                    except Exception:
                        pass
                ts_value = item.get("timestamp")
                if ts_value is None:
                    return 0.0
                try:
                    return float(ts_value)
                except Exception:
                    pass
                try:
                    parsed = datetime.fromisoformat(str(ts_value).replace("Z", "+00:00"))
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=timezone.utc)
                    return parsed.timestamp()
                except Exception:
                    return 0.0

            def _recency_score(ts, now_ts, half_life_days=7.0):
                """
                计算时间新近度分数（指数衰减）
                
                Args:
                    ts: 记忆的时间戳
                    now_ts: 当前时间戳
                    half_life_days: 半衰期天数
                    
                Returns:
                    float: 0-1之间的新近度分数
                """
                if ts <= 0 or now_ts <= 0:
                    return 0.0
                delta_days = max(0.0, (now_ts - ts) / 86400.0)
                return math.exp(-math.log(2.0) * (delta_days / max(1e-6, half_life_days)))

            def _add_to_map(dst_map, item):
                """将记忆项添加到映射中，保留最高分数"""
                key = item.get("memory")
                if not key:
                    return
                if key not in dst_map or float(item.get("score", 0.0)) > float(dst_map[key].get("score", 0.0)):
                    dst_map[key] = item

            def _search(uid, q):
                """辅助函数：搜索单个用户的记忆并返回耗时"""
                memories, graph_memories, duration = self.search_memory(uid, q)
                return uid, memories, graph_memories, duration

            # ---------- 第一阶段：基础多查询召回 ----------
            # 使用 KeyBERT 提取关键词
            keywords = []
            kw_model = self._ensure_keybert_model()
            if kw_model is not None:
                try:
                    extracted = kw_model.extract_keywords(
                        question,
                        keyphrase_ngram_range=(1, 3),
                        stop_words='english',
                        top_n=MAX_KEYWORDS,
                        use_mmr=True,
                        diversity=0.7
                    )
                    keywords = [kw[0] for kw in extracted]
                except Exception as e:
                    print(f"⚠️ KeyBERT extraction failed: {e}. Using fallback.")
            
            # 降级方案：简单的停用词过滤
            if not keywords:
                stop_words = {'what', 'when', 'where', 'who', 'why', 'how', 'is', 'are', 'was', 'were',
                            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
                words = re.findall(r'\b\w+\b', question.lower())
                keywords = [w for w in words if w not in stop_words and len(w) > 2][:MAX_KEYWORDS]

            # 构建查询列表：原问题 + 关键词
            base_queries = [question]
            for kw in keywords:
                base_queries.append(kw)

            # 初始化记忆映射
            a_map = {}  # speaker_1 的记忆
            b_map = {}  # speaker_2 的记忆
            search_time_by_user = {speaker_1_user_id: 0.0, speaker_2_user_id: 0.0}

            # 并发执行所有基础查询
            with self._thread_pool(MAX_WORKERS, "mem-search-base") as executor:
                futures = []
                for q in base_queries:
                    futures.append(executor.submit(_search, speaker_1_user_id, q))
                    futures.append(executor.submit(_search, speaker_2_user_id, q))
                for f in as_completed(futures):
                    uid, mems, graph_mems, duration = f.result()
                    search_time_by_user[uid] = search_time_by_user.get(uid, 0.0) + float(duration or 0.0)
                    for m in mems:
                        if uid == speaker_1_user_id:
                            _add_to_map(a_map, m)
                        else:
                            _add_to_map(b_map, m)

            # ---------- 第二阶段：PRF (伪相关反馈) 生成扩展查询 ----------
            # 从初始检索结果中选取top-k文档用于查询扩展
            global_pool = list(a_map.values()) + list(b_map.values())
            global_pool_sorted = sorted(global_pool, key=lambda x: float(x.get("score", 0.0)), reverse=True)
            prf_docs = global_pool_sorted[:min(PRF_K, len(global_pool_sorted))]

            # 从PRF文档中提取关键词
            prf_text = " \n".join(m.get("memory", "") for m in prf_docs)
            prf_terms = []
            if kw_model is not None and prf_text:
                try:
                    extracted = kw_model.extract_keywords(
                        prf_text,
                        keyphrase_ngram_range=(1, 3),
                        stop_words='english',
                        top_n=PRF_KEYWORDS,
                        use_mmr=True,
                        diversity=0.7
                    )
                    prf_terms = [kw[0] for kw in extracted]
                except Exception as e:
                    print(f"⚠️ PRF KeyBERT failed: {e}. Using fallback.")
            
            # 降级方案：简单的token提取
            if not prf_terms:
                prf_terms = [t for t in _safe_lower_tokens(prf_text) if len(t) > 2][:PRF_KEYWORDS]

            # 构建PRF扩展查询
            prf_queries = []
            if prf_terms:
                prf_queries.append(" ".join([str(term) for term in prf_terms[:3]]))
            if len(prf_terms) >= 4:
                prf_queries.append(" ".join([str(term) for term in prf_terms[2:6]]))

            # 执行PRF扩展查询
            if prf_queries:
                with self._thread_pool(MAX_WORKERS, "mem-search-prf") as executor:
                    futures = []
                    for q in prf_queries:
                        futures.append(executor.submit(_search, speaker_1_user_id, q))
                        futures.append(executor.submit(_search, speaker_2_user_id, q))
                    for f in as_completed(futures):
                        uid, mems, graph_mems, duration = f.result()
                        search_time_by_user[uid] = search_time_by_user.get(uid, 0.0) + float(duration or 0.0)
                        for m in mems:
                            if uid == speaker_1_user_id:
                                _add_to_map(a_map, m)
                            else:
                                _add_to_map(b_map, m)

            # ---------- 第三阶段：Reranking + 时间加权 ----------
            reranker = self._ensure_reranker_model()  # 使用全局单例
            now_ts = datetime.now(tz=timezone.utc).timestamp()

            def _score_after_rerank(cands):
                """
                对候选记忆进行重排序和时间加权
                
                Args:
                    cands: 候选记忆列表
                    
                Returns:
                    tuple: (排序后的候选列表, 最终分数字典)
                """
                if not cands:
                    return [], {}
                base_scores = [float(m.get("score", 0.0)) for m in cands]
                rerank_scores = None
                
                # 使用 Cross-Encoder 模型重排序
                if reranker is not None:
                    try:
                        pairs = [[question, m.get("memory", "")] for m in cands]
                        rerank_scores = reranker.predict(pairs, show_progress_bar=False)
                    except Exception as e:
                        print(f"⚠️ Reranking failed: {e}. Fallback to base score.")
                
                # 计算最终分数：重排序分数 + 时间衰减分数
                final_scores = {}
                for i, m in enumerate(cands):
                    mem_text = m.get("memory", "")
                    rr = float(rerank_scores[i]) if rerank_scores is not None else base_scores[i]
                    ts = _extract_timestamp_seconds(m)
                    rec = _recency_score(ts, now_ts, half_life_days=HALF_LIFE_DAYS)
                    final_scores[mem_text] = RERANK_WEIGHT * rr + RECENCY_WEIGHT * rec
                    m["rerank_score"] = rr
                    m["final_score"] = final_scores[mem_text]
                return sorted(cands, key=lambda x: x.get("final_score", 0.0), reverse=True), final_scores
                

            # 并发重排序两个说话人的记忆
            a_candidates, b_candidates = list(a_map.values()), list(b_map.values())
            rerank_executor = self._get_rerank_executor()
            future_a = rerank_executor.submit(_score_after_rerank, a_candidates)
            future_b = rerank_executor.submit(_score_after_rerank, b_candidates)
            a_sorted, a_scores = future_a.result()
            b_sorted, b_scores = future_b.result()

            # ---------- 第四阶段：MMR 多样化选择 ----------
            a_top = _mmr_select(a_sorted, a_scores, k=top_k_rerank, lambda_div=LAMBDA_DIV) if a_sorted else []
            b_top = _mmr_select(b_sorted, b_scores, k=top_k_rerank, lambda_div=LAMBDA_DIV) if b_sorted else []

            # ---------- 格式化输出 ----------
            search_1_memory = [self._format_memory_line(m) for m in a_top]
            search_2_memory = [self._format_memory_line(m) for m in b_top]
            speaker_1_time = search_time_by_user.get(speaker_1_user_id, 0.0)
            speaker_2_time = search_time_by_user.get(speaker_2_user_id, 0.0)
            return (
                search_1_memory,
                search_2_memory,
                None,
                None,
                speaker_1_time,
                speaker_2_time,
            )
        elif search_method == "6":
            if not len(self.speaker_1_full_memories) or not len(self.speaker_2_full_memories):
                self.__load_full_memories(speaker_1_user_id, speaker_2_user_id)

            speaker_1_indexs = self.vibe_rerank(question=question, documents=self.speaker_1_full_memories, top_k=top_k_rerank)
            speaker_2_indexs = self.vibe_rerank(question=question, documents=self.speaker_2_full_memories, top_k=top_k_rerank)

            speaker_1_memory = []
            speaker_2_memory = []

            if not speaker_1_indexs:
                speaker_1_memories, speaker_1_graph_memories, speaker_1_memory_time = self.search_memory(
                     speaker_1_user_id, question, pbar=pbar
                 )
                search_1_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_1_memories]
            else:
                search_1_memory = [self.speaker_1_full_memories[i] for i in speaker_1_indexs]

            if not speaker_2_indexs:
                speaker_2_memories, speaker_2_graph_memories, speaker_2_memory_time = self.search_memory(
                     speaker_2_user_id, question, pbar=pbar
                 )
                search_2_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_2_memories]
            else:
                search_2_memory = [self.speaker_2_full_memories[i] for i in speaker_2_indexs]

            return (
                search_1_memory,
                search_2_memory,
                None,
                None,
                0.0,
                0.0,
            )

        elif search_method == "10":
            # ========== 方法7: 问题分解 + 关键词 + 多视角搜索融合 ==========
            MAX_WORKERS = min(4, self._max_parallelism_cap)
            SUBQ_COUNT = 4
            MAX_KEYWORDS = 5
            LAMBDA_DIV = 0.65
            RERANK_WEIGHT = 0.75
            RECENCY_WEIGHT = 0.25
            HALF_LIFE_DAYS = 10.0

            def _safe_lower_tokens(text):
                try:
                    return re.findall(r"\b\w+\b", (text or "").lower())
                except Exception:
                    return []

            def _mmr_select(candidates, relevance_scores, k, lambda_div=0.65):
                selected = []
                selected_texts = []
                remaining = list(candidates)
                while remaining and len(selected) < k:
                    best_item = None
                    best_val = -1e9
                    for item in remaining:
                        mem_text = item.get("memory", "")
                        rel = relevance_scores.get(mem_text, 0.0)
                        if selected_texts:
                            max_sim = max(_jaccard_sim(mem_text, t) for t in selected_texts)
                        else:
                            max_sim = 0.0
                        val = lambda_div * rel - (1.0 - lambda_div) * max_sim
                        if val > best_val:
                            best_val = val
                            best_item = item
                    if best_item is None:
                        break
                    selected.append(best_item)
                    selected_texts.append(best_item.get("memory", ""))
                    remaining.remove(best_item)
                return selected

            def _jaccard_sim(a_text, b_text):
                a = set(_safe_lower_tokens(a_text))
                b = set(_safe_lower_tokens(b_text))
                if not a or not b:
                    return 0.0
                return len(a & b) / float(len(a | b))

            def _extract_timestamp_seconds(item):
                if not item:
                    return 0.0
                ts_epoch = item.get("timestamp_epoch")
                if ts_epoch is not None:
                    try:
                        return float(ts_epoch)
                    except Exception:
                        pass
                ts_value = item.get("timestamp")
                if ts_value is None:
                    return 0.0
                try:
                    return float(ts_value)
                except Exception:
                    pass
                try:
                    parsed = datetime.fromisoformat(str(ts_value).replace("Z", "+00:00"))
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=timezone.utc)
                    return parsed.timestamp()
                except Exception:
                    return 0.0

            def _recency_score(ts, now_ts, half_life_days=10.0):
                if ts <= 0 or now_ts <= 0:
                    return 0.0
                delta_days = max(0.0, (now_ts - ts) / 86400.0)
                return math.exp(-math.log(2.0) * (delta_days / max(1e-6, half_life_days)))

            def _add_to_map(dst_map, item):
                key = item.get("memory")
                if not key:
                    return
                if key not in dst_map or float(item.get("score", 0.0)) > float(dst_map[key].get("score", 0.0)):
                    dst_map[key] = item

            def _search(uid, q):
                memories, graph_memories, duration = self.search_memory(uid, q)
                return uid, memories, graph_memories, duration

            # ---------- 构建查询集合 ----------
            candidate_queries = [question]

            subq_prompt = f"""
            You are a query planner. Rewrite the question below into {SUBQ_COUNT} complementary sub-questions that together cover every constraint and detail. Keep the same language as the question. Each sub-question must be standalone, specific, and avoid duplication.

            QUESTION:
            {question}

            Return exactly {SUBQ_COUNT} lines using the format:
            1. <sub-question>
            2. <sub-question>
            ...
            """

            try:
                subq_response = self.safe_chat(
                    model=self.llm_model,
                    messages=[{"role": "system", "content": subq_prompt}],
                    temperature=0.2,
                )
                subq_text = subq_response.choices[0].message.content.strip() if subq_response.choices else ""
            except Exception as exc:
                self.logger.warning("Sub-question generation failed: %s", exc)
                subq_text = ""

            if subq_text:
                extracted = re.findall(r"^\s*\d+\.\s*(.+)", subq_text, flags=re.M)
                for sq in extracted:
                    sq_clean = sq.strip()
                    if sq_clean:
                        candidate_queries.append(sq_clean)

            # 追加关键词查询
            keywords = []
            kw_model = self._ensure_keybert_model()
            if kw_model is not None:
                try:
                    pairs = kw_model.extract_keywords(
                        question,
                        keyphrase_ngram_range=(1, 3),
                        stop_words='english',
                        top_n=MAX_KEYWORDS,
                        use_mmr=True,
                        diversity=0.65,
                    )
                    keywords = [kw[0] for kw in pairs]
                except Exception as exc:
                    self.logger.warning("KeyBERT keyword extraction failed: %s", exc)

            if not keywords:
                stop_words = {
                    'what', 'when', 'where', 'who', 'why', 'how', 'is', 'are', 'was', 'were',
                    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'about'
                }
                tokens = [w for w in re.findall(r"\b\w+\b", question.lower()) if len(w) > 2 and w not in stop_words]
                keywords = tokens[:MAX_KEYWORDS]

            for kw in keywords:
                candidate_queries.append(kw)

            if len(keywords) >= 3:
                candidate_queries.append(" ".join(keywords[:3]))

            # 增强：移除疑问词的精简查询
            stripped = re.sub(r"\b(what|when|where|who|why|how)\b", "", question, flags=re.I)
            stripped = re.sub(r"\s+", " ", stripped).strip()
            if stripped:
                candidate_queries.append(stripped)

            # 去重并保持顺序
            seen_queries = set()
            deduped_queries = []
            for q in candidate_queries:
                q_norm = re.sub(r"\s+", " ", (q or "").strip())
                if not q_norm:
                    continue
                key = q_norm.lower()
                if key in seen_queries:
                    continue
                seen_queries.add(key)
                deduped_queries.append(q_norm)

            # ---------- 执行多视角查询 ----------
            a_map, b_map = {}, {}
            search_time_by_user = {speaker_1_user_id: 0.0, speaker_2_user_id: 0.0}

            with self._thread_pool(MAX_WORKERS, "mem-search-multi") as executor:
                futures = []
                for q in deduped_queries:
                    futures.append(executor.submit(_search, speaker_1_user_id, q))
                    futures.append(executor.submit(_search, speaker_2_user_id, q))
                for f in as_completed(futures):
                    uid, mems, graph_mems, duration = f.result()
                    search_time_by_user[uid] = search_time_by_user.get(uid, 0.0) + float(duration or 0.0)
                    for m in mems:
                        if uid == speaker_1_user_id:
                            _add_to_map(a_map, m)
                        else:
                            _add_to_map(b_map, m)

            # ---------- Rerank + Recency 加权 ----------
            reranker = self._ensure_reranker_model()
            now_ts = datetime.now(tz=timezone.utc).timestamp()

            def _score_candidates(cands):
                if not cands:
                    return [], {}
                base_scores = [float(m.get("score", 0.0)) for m in cands]
                rerank_scores = None
                if reranker is not None:
                    try:
                        pairs = [[question, m.get("memory", "")] for m in cands]
                        rerank_scores = reranker.predict(pairs, show_progress_bar=False)
                    except Exception as exc:
                        self.logger.warning("Reranker failed, fallback to base scores: %s", exc)
                        rerank_scores = None

                final_scores = {}
                for idx, item in enumerate(cands):
                    mem_text = item.get("memory", "")
                    rr_score = float(rerank_scores[idx]) if rerank_scores is not None else base_scores[idx]
                    ts = _extract_timestamp_seconds(item)
                    rec = _recency_score(ts, now_ts, half_life_days=HALF_LIFE_DAYS)
                    final_scores[mem_text] = RERANK_WEIGHT * rr_score + RECENCY_WEIGHT * rec
                    item["rerank_score"] = rr_score
                    item["final_score"] = final_scores[mem_text]
                sorted_items = sorted(cands, key=lambda x: x.get("final_score", 0.0), reverse=True)
                return sorted_items, final_scores

            a_candidates, b_candidates = list(a_map.values()), list(b_map.values())
            rerank_executor = self._get_rerank_executor()
            future_a = rerank_executor.submit(_score_candidates, a_candidates)
            future_b = rerank_executor.submit(_score_candidates, b_candidates)
            a_sorted, a_scores = future_a.result()
            b_sorted, b_scores = future_b.result()

            a_top = _mmr_select(a_sorted, a_scores, k=self.top_k, lambda_div=LAMBDA_DIV) if a_sorted else []
            b_top = _mmr_select(b_sorted, b_scores, k=self.top_k, lambda_div=LAMBDA_DIV) if b_sorted else []

            search_1_memory = [self._format_memory_line(m) for m in a_top]
            search_2_memory = [self._format_memory_line(m) for m in b_top]
            speaker_1_time = search_time_by_user.get(speaker_1_user_id, 0.0)
            speaker_2_time = search_time_by_user.get(speaker_2_user_id, 0.0)

            return (
                search_1_memory,
                search_2_memory,
                None,
                None,
                speaker_1_time,
                speaker_2_time,
            )

        else:
            # ========== 默认搜索方法 ==========
            speaker_1_memories, speaker_1_graph_memories, speaker_1_memory_time = self.search_memory(
                speaker_1_user_id, question, pbar=pbar
            )
            speaker_2_memories, speaker_2_graph_memories, speaker_2_memory_time = self.search_memory(
                speaker_2_user_id, question, pbar=pbar
            )
            search_1_memory = [self._format_memory_line(item) for item in speaker_1_memories]
            search_2_memory = [self._format_memory_line(item) for item in speaker_2_memories]
            return (
                search_1_memory,
                search_2_memory,
                speaker_1_graph_memories,
                speaker_2_graph_memories,
                speaker_1_memory_time,
                speaker_2_memory_time,
            )



    def answer_question(self, speaker_1_user_id, speaker_2_user_id, question, answer, category, pbar=None, max_retries=51):
        """
        处理单个问题，搜索记忆并生成答案
        
        Args:
            speaker_1_user_id: 说话者1的用户ID
            speaker_2_user_id: 说话者2的用户ID
            question: 问题文本
            answer: 参考答案
            category: 问题类别
            pbar: 进度条对象
            max_retries: 最大重试次数
        """
        # 执行搜索并获取耗时
        (
            search_1_memory,
            search_2_memory,
            speaker_1_graph_memories,
            speaker_2_graph_memories,
            speaker_1_memory_time,
            speaker_2_memory_time,
        ) = self.Search(
            speaker_1_user_id, speaker_2_user_id, question, self.search_method, pbar=pbar
        )

        speaker_1_graph_memories = speaker_1_graph_memories or []
        speaker_2_graph_memories = speaker_2_graph_memories or []
        speaker_1_memory_time = float(speaker_1_memory_time or 0.0)
        speaker_2_memory_time = float(speaker_2_memory_time or 0.0)
        response_time = 0.0
        total_search_time = speaker_1_memory_time + speaker_2_memory_time

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
        answer_start = time.time()
        answer_sleep_penalty = 0.0
        answer_attempts = 0
        while True:
            try:
                answer_attempts += 1
                response = self.answer_client.chat.completions.create(
                    model=self.answer_llm_model, 
                    messages=[{"role": "system", "content": answer_prompt}], 
                    temperature=0.0
                )
                response_content = response.choices[0].message.content
                self._log_llm_call(
                    request_id,
                    llm_error_retries + other_error_retries,
                    max_retries,
                    prompt_components,
                    answer_prompt,
                    response_content,
                    "Success",
                )
                break 
            except Exception as e:
                error_str = str(e).lower()
                if "rate limit" in error_str or "limit" in error_str or "overloaded" in error_str or "token" in error_str:
                    # 识别为LLM相关的限流错误
                    llm_error_retries += 1
                    other_error_retries = 0 # 重置其他错误计数
                    sleep_duration = random.uniform(2, 20) + 5 * llm_error_retries
                    answer_sleep_penalty += sleep_duration
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
                    
        response_time = max(0.0, time.time() - answer_start - answer_sleep_penalty)
        self.logger.info(
            "Answer generated for request %s in %.2fs (attempts=%d, retry_sleep_skipped=%.2fs)",
            request_id,
            response_time,
            answer_attempts,
            answer_sleep_penalty,
        )

        return (    
            response_content,
            search_1_memory,  
            search_2_memory,  
            speaker_1_memory_time,
            speaker_2_memory_time,
            speaker_1_graph_memories,
            speaker_2_graph_memories,
            response_time,
            answer_prompt,
            total_search_time,
        )

    def process_question(self, val, speaker_a_user_id, speaker_b_user_id, idx, pbar=None):
        """
        处理单个问答对，包括搜索记忆和生成答案
        
        Args:
            val: 问答对数据字典
            speaker_a_user_id: 说话者A的用户ID
            speaker_b_user_id: 说话者B的用户ID
            idx: 对话索引
            pbar: 进度条对象
            
        Returns:
            dict: 包含问题、答案、记忆等信息的结果字典
        """
        question = val.get("question", "")
        answer = val.get("answer", "")
        answer_fixed = val.get("answer_fixed", "")
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
            search_time_total,
        ) = self.answer_question(speaker_a_user_id, speaker_b_user_id, question, answer, category, pbar)

        result = {
            "question": question,
            "answer": answer,
            "answer_fixed": answer_fixed,
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
            "answer_prompt": answer_prompt,
            "search_time": search_time_total,
        }


        self._record_result(idx, result)

        if pbar:
            pbar.update(1)

        return result

    def process_data_file(self, file_path, max_workers=5):
        """
        处理整个数据文件，并发处理所有问答对

        Args:
            file_path: 数据文件路径
            max_workers: 最大并发工作线程数
        """
        dataset_path = Path(file_path)
        stats = compute_dataset_stats(dataset_path)
        self._dataset_stats = stats

        total_questions = stats.get("total_questions", 0)
        if total_questions == 0:
            print("No questions found to process.")
            self._results_writer = IncrementalResultsWriter(self.output_path)
            self._results_writer.finalize()
            return

        print(f"--- 预计总共需要处理 {total_questions} 个问题 ---")

        resolved_workers = self._resolve_max_workers(max_workers)
        if resolved_workers != max_workers:
            print(f"⚙️ 调整 max_workers: 从 {max_workers} -> {resolved_workers}")
        else:
            print(f"⚙️ 使用 max_workers = {resolved_workers}")

        self._expected_results_per_conversation = stats.get("qa_per_conversation", [])
        self._results_buffer = {}
        self._results_writer = IncrementalResultsWriter(self.output_path)

        successful_count = 0
        failed_count = 0
        futures = {}
        drain_threshold = max(resolved_workers, 1) * 4

        def consume_one():
            nonlocal successful_count, failed_count
            if not futures:
                return
            done, _ = wait(tuple(futures.keys()), return_when=FIRST_COMPLETED)
            for finished in done:
                conv_idx, task_label = futures.pop(finished)
                try:
                    finished.result()
                    successful_count += 1
                except Exception as exc:
                    failed_count += 1
                    error_details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                    pbar.write(f"\n--- ❌ Error processing task '{task_label}' ---")
                    pbar.write(f"{exc}\n")
                    pbar.write(f"{error_details}\n")
                    pbar.update(1)

        with tqdm(total=total_questions, desc="💡Total Questions Progress") as pbar:
            try:
                with ThreadPoolExecutor(max_workers=resolved_workers, thread_name_prefix="mem-search-main") as executor:
                    for conv_idx, item in enumerate(stream_normalized_dataset(dataset_path)):
                        conversation = item.get("conversation") or {}
                        speaker_a = conversation.get("speaker_a")
                        speaker_b = conversation.get("speaker_b")
                        if not speaker_a or not speaker_b:
                            self.logger.warning("Conversation %s 缺少说话者信息，已跳过。", conv_idx)
                            continue

                        speaker_a_user_id = f"{speaker_a}_{conv_idx}"
                        speaker_b_user_id = f"{speaker_b}_{conv_idx}"
                        qa_list = item.get("qa", [])

                        for question_item in qa_list:
                            future = executor.submit(
                                self.process_question,
                                question_item,
                                speaker_a_user_id,
                                speaker_b_user_id,
                                conv_idx,
                                pbar,
                            )
                            question_preview = (question_item.get("question") or "").strip().replace("\n", " ")
                            if len(question_preview) > 40:
                                question_preview = question_preview[:37] + "..."
                            futures[future] = (conv_idx, f"Conv {conv_idx} - {question_preview}")

                            if len(futures) >= drain_threshold:
                                consume_one()

                    while futures:
                        consume_one()
            except Exception as exc:
                raise RuntimeError("Failed during threaded question processing.") from exc
            finally:
                pending_flush = []
                with self._results_state_lock:
                    for conv_idx, bucket in self._results_buffer.items():
                        if bucket:
                            pending_flush.append((conv_idx, list(bucket)))
                    self._results_buffer.clear()
                for conv_idx, bucket in pending_flush:
                    self._results_writer.append(conv_idx, bucket)
                self._results_writer.finalize()

        print(f"\n✅ All questions processed. Success: {successful_count}, Failed: {failed_count}")

    def close(self):
        if self._rerank_executor:
            self._rerank_executor.shutdown(wait=True, cancel_futures=True)
            self._rerank_executor = None
        if getattr(self, "_io_executor", None):
            self._io_executor.shutdown(wait=True, cancel_futures=True)
            self._io_executor = None
            if hasattr(self.memory, "set_shared_executor"):
                self.memory.set_shared_executor(None)
