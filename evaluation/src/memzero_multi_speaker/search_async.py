import json
import logging
import math
import os
import random
import re
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from jinja2 import Template
from mem0 import Memory
from mem0.utils.factory import LlmFactory
from prompts import ANSWER_PROMPT_0_MSP, ANSWER_PROMPT_15_MSP, ANSWER_PROMPT_GRAPH
from tqdm import tqdm
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

_NOISE_PATTERNS = [
    r"\bcongrats\b", r"\bcongratulations\b", r"\bgood luck\b", r"\bgreat job\b",
    r"\bnice!\b", r"\bawesome\b", r"\bamazing\b", r"\bthat'?s (great|awesome)\b",
    r"\bsorry\b", r"\bapolog(y|ize|ies)\b", r"\bthanks\b", r"\bthank you\b",
    r"\bglad to\b", r"\bcheer(s|ing)?\b"
]
_NOISE_QUESTION_PAT = r"\?\s*$"

_ACTION_CANON = {
    # 轻量同义簇（可持续补充）
    "job_loss": [r"\blost (his|her|their)? job\b", r"\bfired\b", r"\blaid off\b"],
    "business_open": [r"\b(open|start)(ed)? (an? )?(online )?(store|business|studio)\b"],
    "studio_open": [r"\b(open|launch|set\s*up|establish|start|kick\s*off)(ed)? (an? )?(dance )?(studio|workshop)\b", r"\b(opening|launching) (an? )?(dance )?studio\b"],
    "gym_start": [r"\bstart(ed)? (to )?go to the gym\b", r"\b(started|began) going to the gym\b", r"\bgo(es|ing)? to the gym\b"],
    "color_pref": [r"\bfavorite color\b", r"\bfavourite colour\b"],
}

_MONTHS = {
    "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
    "july":7,"august":8,"september":9,"october":10,"november":11,"december":12
}

_TIME_KEYWORDS_CN = [
    "什么时候", "几月", "几年", "哪年", "哪月", "哪天", "哪一天", "哪一周", "日期", "时间", "几点", "哪刻", "何时", "何年", "何月", "何日",
    "星期几", "周几", "近日", "近期", "最近", "刚刚", "刚才", "方才", "不久前", "快要", "即将", "马上", "立刻",
    "下周", "上周", "本周", "这周", "之前", "之后"
]

_TIME_KEYWORDS_EN = [
    "when", "what date", "which year", "which month", "which day", "what day", "which week", "what time", "date", "time", "day",
    "recent", "recently", "lately", "just", "soon", "upcoming", "shortly", "next week", "last week", "earlier", "later", "before", "after",
    "since", "during", "until", "throughout", "this week", "this month", "this year", "past few", "over the past", "over the last",
    "in the last", "these days", "afterwards", "previously", "once", "twice", "every day", "each week"
]

_NUMERIC_KEYWORDS_CN = [
    "多少", "几", "第几", "多长时间", "几岁", "几次", "多少次", "几天", "多少天", "几周", "多少周", "几个月", "多少个月", "几人", "多少人", "几位",
    "几个人", "人数", "几种", "几类", "几号", "几块钱", "多少金额", "多大", "多高", "多重"
]

_NUMERIC_KEYWORDS_EN = [
    "how many", "how much", "how long", "how old", "how often", "how far", "how many people", "how many times", "how many days",
    "how many years", "how many months", "how many hours", "how much time", "how much money", "how big", "how tall", "how heavy", "how far away",
    "at least", "at most", "no more than", "no less than", "less than", "greater than", "more than", "fewer than", "under", "over",
    "minimum", "maximum", "budget", "price", "cost", "limit"
]

class MemorySearch:
    """
    记忆搜索类，用于处理基于对话的问答任务
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
        config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": self.llm_model,
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
        self._llm_provider = config["llm"]["provider"]
        self._llm_config = deepcopy(config["llm"]["config"])
        self.top_k = top_k
        self.output_path = output_path
        self.filter_memories = filter_memories
        self.is_graph = is_graph
        self.search_method = self._normalize_mode(search_method)
        answer_mode = self._normalize_mode(answer_mode)
        self.answer_mode = answer_mode
        self.qdrant_path = qdrant_path
        self._max_parallelism_cap = max(1, min(os.cpu_count() * 2 or 8, 18))
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
        self.llm = self.memory.llm
        self.answer_llm = self.llm
        if (
            self.answer_llm_model != self.llm_model
            or answer_llm_base_url != llm_base_url
            or answer_llm_api_key != llm_api_key
        ):
            answer_llm_config = deepcopy(self._llm_config)
            answer_llm_config.update(
                {
                    "model": self.answer_llm_model,
                    "openai_base_url": answer_llm_base_url,
                }
            )
            if answer_llm_api_key:
                answer_llm_config["api_key"] = answer_llm_api_key
            self.answer_llm = LlmFactory.create(self._llm_provider, answer_llm_config)
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
            if answer_mode == "15":
                self.ANSWER_PROMPT = ANSWER_PROMPT_15_MSP
            elif answer_mode == "0":
                self.ANSWER_PROMPT = ANSWER_PROMPT_0_MSP
            else:
                self.logger.warning(
                    "Unknown answer_mode '%s'. Falling back to MSP prompt 0.", answer_mode
                )
                self.ANSWER_PROMPT = ANSWER_PROMPT_0_MSP

        # CrossEncoder score cache与方案148配置
        self._ce_cache = {}
        self._ce_cache_order = []
        self._ce_cache_max = 4096
        self._cfg_148 = {
            "per_query_limit": 48,
            "max_base_queries": 8,
            "max_prf_queries": 2,
            "mmr_lambda_time": 0.75,
            "mmr_lambda_default": 0.60,
            "noise_penalty": 0.35,
            "rrf_c_small": 10,
            "rrf_c_large": 40,
            "small_pool_thresh": 30,
            "timeline_subject_cap": 12,
        }
            
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

    def _record_result(self, conversation_idx: int, result):
        """
        Buffer per-conversation results and flush once all QA pairs finish.
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

    @staticmethod
    def _format_memory_line(item):
        """
        Render memory entries with a consistent timestamp display.
        """
        display_ts = item.get("timestamp_display") or item.get("timestamp") or ""
        memory_text = item.get("memory", "")
        return f"{display_ts}: {memory_text}"

    def _format_exception_details(self, exc, body_preview_chars=2000):
        """Format detailed information about HTTP/API exceptions for logging."""
        try:
            preview_limit = max(0, int(body_preview_chars))
        except (TypeError, ValueError):
            preview_limit = 2000

        def _truncate(text: str) -> str:
            if not text:
                return ""
            if preview_limit and len(text) > preview_limit:
                return f"{text[:preview_limit]}... [truncated {len(text) - preview_limit} chars]"
            return text

        details = [f"{type(exc).__name__}: {exc}"]

        for attr in ("code", "error_code", "status_code", "http_status"):
            value = getattr(exc, attr, None)
            if value:
                details.append(f"{attr}={value}")

        response = getattr(exc, "response", None) or getattr(exc, "http_response", None)
        if response is not None:
            status = getattr(response, "status_code", None)
            reason = getattr(response, "reason_phrase", None) or getattr(response, "reason", None)
            request_obj = getattr(response, "request", None)
            method = getattr(request_obj, "method", None)
            url = getattr(request_obj, "url", None)
            response_line_parts = []
            if status is not None:
                response_line_parts.append(f"HTTP {status}")
            if reason:
                response_line_parts.append(str(reason))
            if method:
                response_line_parts.append(method)
            if url:
                response_line_parts.append(str(url))
            if response_line_parts:
                details.append(" ".join(response_line_parts))

            body_text = ""
            if hasattr(response, "text"):
                try:
                    body_text = response.text or ""
                except Exception:
                    body_text = ""
            if not body_text and hasattr(response, "content"):
                content = getattr(response, "content")
                if isinstance(content, (bytes, bytearray)):
                    body_text = content.decode("utf-8", errors="replace")
                elif content is not None:
                    body_text = str(content)
            if not body_text and hasattr(response, "json"):
                try:
                    json_payload = response.json()
                    body_text = json.dumps(json_payload, ensure_ascii=False)
                except Exception:
                    body_text = ""

            if body_text:
                try:
                    parsed = json.loads(body_text)
                    body_text = json.dumps(parsed, ensure_ascii=False, indent=2)
                except Exception:
                    pass
                details.append(f"response_body={_truncate(body_text)}")
        elif hasattr(exc, "body"):
            body_payload = getattr(exc, "body")
            if body_payload:
                if isinstance(body_payload, (dict, list)):
                    body_content = json.dumps(body_payload, ensure_ascii=False, indent=2)
                else:
                    body_content = str(body_payload)
                details.append(f"response_body={_truncate(body_content)}")

        extra = getattr(exc, "error", None) or getattr(exc, "details", None)
        if extra:
            if isinstance(extra, (dict, list)):
                extra = json.dumps(extra, ensure_ascii=False, indent=2)
            details.append(f"extra={_truncate(str(extra))}")

        return "\n".join(details)

    def _normalize_chat_messages(self, messages, fallback_instruction=None):
        """Ensure chat payload contains at least one user/assistant message."""
        if not messages:
            raise ValueError("messages 必须提供。")

        has_non_system = any((msg or {}).get("role") != "system" for msg in messages)
        if has_non_system:
            return messages

        fallback_instruction = fallback_instruction or "You are a helpful assistant."
        normalized = []
        converted = False
        for msg in messages:
            text = msg.get("content", "") if isinstance(msg, dict) else ""
            if text:
                normalized.append({"role": "user", "content": text})
                converted = True
        if not converted:
            normalized.append({"role": "user", "content": fallback_instruction})
        return normalized

    def _extract_llm_text(self, response) -> str:
        """Normalize LLM responses (str/dict) into a plain string."""
        if response is None:
            return ""
        if isinstance(response, str):
            return response
        if isinstance(response, dict):
            content = response.get("content")
            if isinstance(content, str):
                return content
            return ""
        return str(response)

    def safe_chat(self, model=None, messages=None, temperature=0.0, sleep_time=20, llm=None, response_format=None):
        """
        安全的LLM调用，自动处理速率限制并复用 mem0 提供的 LLM 封装。
        """
        if messages is None:
            raise ValueError("messages 必须提供。")
        llm_client = llm
        if llm_client is None:
            if model and model == getattr(self, "answer_llm_model", None) and getattr(self, "answer_llm", None):
                llm_client = self.answer_llm
            else:
                llm_client = getattr(self, "llm", None)
        if llm_client is None:
            raise RuntimeError("LLM client is not initialized.")
        normalized_messages = self._normalize_chat_messages(messages)
        while True:
            try:
                return llm_client.generate_response(
                    messages=normalized_messages,
                    temperature=temperature,
                    response_format=response_format,
                )
            except Exception as e:
                s = str(e)
                if ("429" in s) or ("TPM" in s) or ("rate limit" in s.lower()):
                    self.logger.warning("Rate limit hit, sleeping %.1fs before retry...", sleep_time)
                    time.sleep(sleep_time)
                    continue
                raise

    def search_memory(self, user_id, query, max_retries=5, limit=None):
        """
        搜索指定用户的记忆，返回 (memories, graph_memories, search_time)。
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
                error_lower = str(e).lower()
                if "collection" in error_lower and "not found" in error_lower:
                    self.logger.warning(
                        "Search collection missing for user %s (attempt %s/%s). Recreating and retrying.",
                        user_id,
                        retries,
                        max_retries,
                    )
                    self._ensure_collection_exists()
                    continue
                backoff = min(8.0, 0.75 * (2 ** (retries - 1))) + random.uniform(0.1, 0.6)
                retry_sleeps.append(backoff)
                self.logger.warning(
                    "Retrying search for user %s...%s/%s | backoff=%.2fs\n%s",
                    user_id,
                    retries,
                    max_retries,
                    backoff,
                    self._format_exception_details(e),
                )
                if retries >= max_retries:
                    raise
                sleep_penalty += backoff
                time.sleep(backoff)

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
            timestamp_value = metadata.get("timestamp") or memory.get("created_at")
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
        postprocess_time = time.perf_counter() - postprocess_start

        embedding_time = float(search_metrics.get("embedding_sec") or 0.0)
        vector_time = float(search_metrics.get("vector_query_sec") or 0.0) + float(
            search_metrics.get("vector_postprocess_sec") or 0.0
        )
        graph_time = float(search_metrics.get("graph_query_sec") or 0.0)
        accounted_time = embedding_time + vector_time + graph_time + postprocess_time
        overhead_time = max(0.0, search_duration - accounted_time)

        cache_hit = search_metrics.get("embedding_cache_hit")
        retry_note = f", retry_sleeps={['%.2f' % s for s in retry_sleeps]}" if retry_sleeps else ""
        cache_note = f", cache_hit={cache_hit}" if cache_hit is not None else ""
        self.logger.info(
            "Search success for user %s in %.2fs (attempts=%s, retry_sleep=%.2fs%s%s) | breakdown: "
            "embedding=%.2fs, vector=%.2fs, graph=%.2fs, postprocess=%.2fs, other=%.2fs",
            user_id,
            search_duration,
            attempts,
            sleep_penalty,
            cache_note,
            retry_note,
            embedding_time,
            vector_time,
            graph_time,
            postprocess_time,
            overhead_time,
        )
        return semantic_memories, graph_payload, search_duration

    def _is_noise_text(self, text: str) -> bool:
        t = (text or "").lower().strip()
        if not t:
            return True
        import re
        for pat in _NOISE_PATTERNS:
            if re.search(pat, t):
                return True
        if re.search(_NOISE_QUESTION_PAT, t):  # 纯问句
            return True
        if re.search(r"!{2,}\s*$", t):  # 强情绪感叹句
            return True
        # 轻量规则：表达同情/询问
        if "expressed sympathy" in t or t.startswith(("did ", "do ", "what ", "why ", "how ", "when ")):
            return True
        return False

    def _extract_candidate_names(self, text: str):
        import re

        if not text:
            return []

        names = set()
        try:
            lower_text = text.lower()

            stop_words = {
                "I", "The", "A", "An", "And", "But", "For", "From", "This", "That", "Those", "These",
                "He", "She", "They", "We", "You", "It", "His", "Her", "Their", "Our", "Its",
                "When", "What", "Where", "Why", "How", "If", "In", "On", "At", "By", "With",
                "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
                "January", "February", "March", "April", "May", "June", "July", "August", "September",
                "October", "November", "December", "Earlier", "Later"
            }
            stop_words_lower = {w.lower() for w in stop_words}

            if re.search(r"\buser\b", lower_text):
                names.add("user")

            colon_pattern = re.compile(r"^\s*([A-Za-z0-9_\-\u4e00-\u9fff]{2,})\s*:", re.MULTILINE)
            for raw in colon_pattern.findall(text):
                token_lower = raw.strip().lower()
                if token_lower and token_lower not in stop_words_lower:
                    names.add(token_lower)

            courtesy_pattern = re.compile(r"\b(Mr|Mrs|Ms|Miss|Dr|Prof)\.?\s+([A-Za-z][A-Za-z\-]+)\b", re.IGNORECASE)
            for _, raw in courtesy_pattern.findall(text):
                token_lower = raw.lower()
                if token_lower not in stop_words_lower:
                    names.add(token_lower)

            apostrophe_pattern = re.compile(r"\b([A-Za-z\u4e00-\u9fff][A-Za-z\u4e00-\u9fff]+)'s\b")
            for raw in apostrophe_pattern.findall(text):
                token_lower = raw.lower()
                if token_lower not in stop_words_lower:
                    names.add(token_lower)

            capital_tokens = re.findall(r"\b[A-Z][a-z]+(?:-[A-Z][a-z]+)?\b", text)
            for token in capital_tokens:
                token_lower = token.lower()
                if token_lower not in stop_words_lower:
                    names.add(token_lower)

            chinese_label_pattern = re.compile(r"([\u4e00-\u9fff]{2,4})(?:：|:)")
            for token in chinese_label_pattern.findall(text):
                names.add(token)

            return sorted(names)
        except Exception:
            return sorted(names)

    def _extract_subjects_from_question(self, q: str):
        return self._extract_candidate_names(q)

    def _subject_bonus(self, text: str, subjects: list) -> float:
        if not subjects:
            return 0.0
        t = (text or "").lower()
        unique_subjects = {s for s in subjects if s}
        hits = sum(1 for s in unique_subjects if s in t)
        return min(0.6, 0.3 * hits)

    def _lexical_score(self, q: str, text: str) -> float:
        import re, math
        def toks(s):
            return re.findall(r"\b\w+\b", (s or "").lower())
        q_tokens = toks(q)
        t_tokens = toks(text)
        if not q_tokens or not t_tokens:
            return 0.0
        q_set, t_set = set(q_tokens), set(t_tokens)
        overlap = len(q_set & t_set)
        # bigram overlap（弱 BM25）
        def bigrams(xs): return set(zip(xs, xs[1:])) if len(xs) > 1 else set()
        bo = len(bigrams(q_tokens) & bigrams(t_tokens))
        return overlap * 0.6 + bo * 0.9

    def _extract_negative_terms(self, text: str):
        """
        提取“排除/不包含”语义中的关键词、原始短语以及去除后的问题文本。
        返回 (neg_terms, sanitized_text, neg_metadata)。
        """
        import re

        raw = text or ""
        neg_terms = set()
        spans = []
        neg_infos = []
        relation_terms = {
            "girlfriend",
            "boyfriend",
            "partner",
            "wife",
            "husband",
            "friend",
            "friends",
            "family",
            "parents",
            "kids",
            "children",
            "son",
            "daughter",
            "brother",
            "sister",
            "coworker",
            "coworkers",
            "colleague",
            "colleagues",
            "roommate",
            "roommates",
            "fiance",
            "fiancee",
            "spouse",
        }
        stop_tokens = {
            "and",
            "or",
            "but",
            "the",
            "a",
            "an",
            "other",
            "than",
            "besides",
            "except",
            "rather",
            "instead",
            "without",
            "with",
            "of",
            "any",
            "my",
            "your",
            "his",
            "her",
            "our",
            "their",
            "me",
            "you",
            "him",
            "them",
            "ours",
            "yours",
            "hers",
            "mine",
            "ourselves",
            "yourselves",
            "myself",
            "yourself",
            "herself",
            "himself",
            "themselves",
        }

        def _collect(segment: str):
            tokens = re.findall(r"\b[\w']+\b", (segment or "").lower())
            for tok in tokens:
                tok = tok.strip("'\"")
                if not tok or tok in stop_tokens or tok in relation_terms:
                    continue
                if tok.isdigit():
                    continue
                neg_terms.add(tok)

        def _record_span(start: int, end: int, phrase_text: str, target_text: str):
            spans.append((start, end))
            prefix = raw[:start]
            suffix = raw[end:]
            neg_infos.append(
                {
                    "phrase": (phrase_text or "").strip(),
                    "target": (target_text or "").strip(),
                    "prefix": prefix.strip(),
                    "suffix": suffix.strip(),
                }
            )

        english_patterns = [
            r"\bother than\s+([^\.;,!?]+)",
            r"\bexcept(?: for)?\s+([^\.;,!?]+)",
            r"\b(?:anything|things|activities)\s+besides\s+([^\.;,!?]+)",
            r"\bapart from\s+([^\.;,!?]+)",
            r"\bwithout\s+([^\.;,!?]+)",
            r"\brather than\s+([^\.;,!?]+)",
            r"\bexcluding\s+([^\.;,!?]+)",
        ]
        for pat in english_patterns:
            for match in re.finditer(pat, raw, flags=re.IGNORECASE):
                group_text = match.group(1)
                if group_text:
                    _collect(group_text)
                _record_span(match.start(), match.end(), match.group(0), group_text)

        chinese_patterns = [
            r"除了(.+?)以外",
            r"不包括(.+?)(?:，|,|。|？|\?|$)",
            r"不含(.+?)(?:，|,|。|？|\?|$)",
            r"别([^，。,？?]+)不要",
        ]
        for pat in chinese_patterns:
            for match in re.finditer(pat, raw):
                group_text = match.group(1)
                if group_text:
                    _collect(group_text)
                _record_span(match.start(), match.end(), match.group(0), group_text)

        sanitized = raw
        if spans:
            chars = list(raw)
            for start, end in sorted(spans, key=lambda x: x[0], reverse=True):
                for idx in range(start, min(end, len(chars))):
                    chars[idx] = ""
            sanitized = "".join(chars)
        sanitized = re.sub(r"\s+", " ", sanitized).strip()
        if not sanitized:
            sanitized = raw.strip()

        return neg_terms, sanitized, neg_infos

    def _question_is_time_intent(self, q: str) -> bool:
        q = q or ""
        ql = q.lower()
        return any(kw in q for kw in _TIME_KEYWORDS_CN) or any(kw in ql for kw in _TIME_KEYWORDS_EN)

    def _parse_month_in_question(self, q: str):
        # 返回 {year: int|None, month: int|None}
        import re
        ql = (q or "").lower()
        # 年
        year = None
        m = re.search(r"\b(20\d{2})\b", ql)
        if m:
            year = int(m.group(1))
        # 月
        month = None
        for name, mi in _MONTHS.items():
            if name in ql:
                month = mi
                break
        return {"year": year, "month": month}

    def _extract_norm_date_from_memory(self, text: str):
        """解析记忆中的时间片段：返回 (year, month, day)，若不存在则为 None"""
        import re

        t = text or ""

        # ISO / 数字型日期：2023-03-05、2023/3/5、2023.03、03/2023
        m = re.search(r"\b(20\d{2})[-/.](\d{1,2})(?:[-/.](\d{1,2}))?\b", t)
        if m:
            year = int(m.group(1))
            month = int(m.group(2))
            day = int(m.group(3)) if m.group(3) else None
            return year, month, day

        m = re.search(r"\b(\d{1,2})[/-](20\d{2})\b", t)
        if m:
            month = int(m.group(1))
            year = int(m.group(2))
            return year, month, None

        # 英文日期：March 5, 2023 / 5 March 2023 / March 2023
        mon = r"(january|february|march|april|may|june|july|august|september|october|november|december)"
        m = re.search(fr"\b{mon}\s+(\d{{1,2}})(?:st|nd|rd|th)?\,?\s+(20\d{{2}})\b", t, flags=re.I)
        if m:
            year = int(m.group(3))
            month = _MONTHS.get(m.group(1).lower())
            day = int(m.group(2))
            return year, month, day
        m = re.search(fr"\b(\d{{1,2}})\s+{mon}\s+(20\d{{2}})\b", t, flags=re.I)
        if m:
            day = int(m.group(1))
            month = _MONTHS.get(m.group(2).lower())
            year = int(m.group(3))
            return year, month, day
        m = re.search(fr"\b{mon}\s+(20\d{{2}})\b", t, flags=re.I)
        if m:
            year = int(m.group(2))
            month = _MONTHS.get(m.group(1).lower())
            return year, month, None

        # 中文日期：2023年3月5日 / 2023年3月 / 3月5日2023年
        m = re.search(r"(20\d{2})年(\d{1,2})月(\d{1,2})?日?", t)
        if m:
            year = int(m.group(1))
            month = int(m.group(2))
            day = int(m.group(3)) if m.group(3) else None
            return year, month, day
        m = re.search(r"(\d{1,2})月(\d{1,2})?日?,?\s*(20\d{2})年?", t)
        if m:
            month = int(m.group(1))
            day = int(m.group(2)) if m.group(2) else None
            year = int(m.group(3))
            return year, month, day
        m = re.search(r"(20\d{2})年", t)
        if m:
            return int(m.group(1)), None, None

        return None, None, None

    def _time_bonus(self, q: str, mem_text: str) -> float:
        if not self._question_is_time_intent(q):
            return 0.0
        wanted = self._parse_month_in_question(q)  # 可能只有 year/month 之一
        y, m, d = self._extract_norm_date_from_memory(mem_text)
        score = 0.0
        if wanted["year"] and y == wanted["year"]:
            score += 0.6
        if wanted["month"] and m == wanted["month"]:
            score += 0.8
        return score

    def _canonical_action(self, text: str):
        import re
        t = (text or "").lower()
        for canon, pats in _ACTION_CANON.items():
            for pat in pats:
                if re.search(pat, t):
                    return canon
        return None

    def _action_bonus(self, q: str, mem_text: str) -> float:
        q_act = self._canonical_action(q)
        if not q_act:
            return 0.0
        m_act = self._canonical_action(mem_text)
        return 0.7 if m_act and (m_act == q_act) else 0.0

    def _rrf(self, ranks: list, c: int = 60) -> float:
        """ranks: [rank_from_embed (1-based or None), rank_from_lex (…), rank_from_xenc (…)]"""
        s = 0.0
        for r in ranks:
            if r is not None and r > 0:
                s += 1.0 / (c + r)
        return s

    def _final_score_145(self, q, mem, rank_embed, rank_lex, rank_xenc):
        base = self._rrf([rank_embed, rank_lex, rank_xenc], c=60)
        bonus = self._subject_bonus(mem["memory"], self._extract_subjects_from_question(q))
        bonus += self._action_bonus(q, mem["memory"])
        bonus += 0.15 * float(mem.get("score", 0.0))  # 轻度保留原 embedding 分
        if self._is_noise_text(mem["memory"]):
            base -= 0.35  # 噪声轻扣
        return base + bonus

    def _final_score_146(self, q, mem, rank_embed, rank_lex, rank_xenc):
        base = self._rrf([rank_embed, rank_lex, rank_xenc], c=40)
        bonus = self._time_bonus(q, mem["memory"])
        bonus += self._action_bonus(q, mem["memory"])
        bonus += self._subject_bonus(mem["memory"], self._extract_subjects_from_question(q))
        if self._is_noise_text(mem["memory"]):
            base -= 0.35
        return base + bonus

    def vibe_rerank(self, question, documents, top_k=30) -> list:
        """
        调用 LLM 对候选记忆进行轻量 re-rank，返回索引列表。
        """
        docs_with_idx = "\n".join([f"[{i}] {d}" for i, d in enumerate(documents)])
        prompt = f"""You are an intelligent reranker.
Question: {question}
Documents:
{docs_with_idx}
Please identify the most relevant documents.
If there are no such texts just return "Not found"
Return only the indices in parentheses, e.g. (2,5,1)
Top indices:"""
        raw = ""
        try:
            resp = self.safe_chat(
                model=self.llm_model,
                messages=[{"role": "system", "content": prompt}],
                temperature=0.8,
            )
            raw = self._extract_llm_text(resp).strip()
            nums = re.findall(r"\d+", raw)
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
            self.logger.warning("[vibe_rerank] %s | raw output: %s", e, raw[:80])
            return []

    # ==== 方案 148：意图驱动混合检索工具 ====
    def _intent_148(self, q: str):
        import re

        q = q or ""
        ql = q.lower()
        is_time = any(kw in q for kw in _TIME_KEYWORDS_CN) or any(kw in ql for kw in _TIME_KEYWORDS_EN)
        time_regexes = [
            r"\b(before|after|during|since|until|earlier|later)\b",
            r"\b(in|during|over)\s+the\s+(last|past)\s+\d+\s+(days|weeks|months|years)\b",
            r"\b(from|since)\s+(last|this)\s+(year|month|week)\b",
            r"\bover the (past|last)\s+few\b",
        ]
        if not is_time:
            is_time = any(re.search(pat, ql) for pat in time_regexes)
        has_year = bool(re.search(r"\b(20\d{2})\b", ql))
        has_month = any(m in ql for m in _MONTHS.keys()) or bool(re.search(r"\b(0?[1-9]|1[0-2])\b", ql))
        recent_cn = ["近期", "最近", "刚刚", "刚才", "不久前"]
        recent_en = ["recent", "recently", "lately", "just", "in the last", "over the past", "past few", "these days"]
        want_recent = any(kw in q for kw in recent_cn) or any(kw in ql for kw in recent_en)
        if not want_recent:
            want_recent = any(re.search(pat, ql) for pat in time_regexes[1:3])
        numeric_regexes = [
            r"\b(at least|at most|no more than|no less than|less than|fewer than|more than|greater than|over|under)\s+\d+",
            r"\b\d+\s+(times|people|persons|years|months|weeks|days|hours)\b",
            r"\$\s*\d+",
        ]
        is_numeric = any(kw in q for kw in _NUMERIC_KEYWORDS_CN) or any(kw in ql for kw in _NUMERIC_KEYWORDS_EN)
        if not is_numeric:
            is_numeric = any(re.search(pat, ql) for pat in numeric_regexes)
        subjects = self._extract_subjects_from_question(q)
        action = self._canonical_action(q)
        return {
            "is_time": is_time,
            "has_year": has_year,
            "has_month": has_month,
            "want_recent": want_recent,
            "is_numeric": is_numeric,
            "subjects": subjects,
            "action": action,
        }

    def _gen_queries_148(self, q: str, intent: dict):
        import re

        queries = []
        seen = set()

        def add(candidate: str):
            if not candidate:
                return
            norm = re.sub(r"\s+", " ", candidate.strip())
            if not norm:
                return
            key = norm.lower()
            if key in seen:
                return
            seen.add(key)
            queries.append(norm)

        add(q)

        stripped = re.sub(r"\b(what|when|where|who|why|how)\b", "", q, flags=re.I)
        stripped = re.sub(r"\s+", " ", stripped).strip()
        if stripped and stripped.lower() != q.lower():
            add(stripped)

        ym = re.findall(r"(20\d{2})|(\d{1,2}\s*月)", q)
        if ym:
            add(" ".join([x for pair in ym for x in pair if x]))

        action = intent.get("action")
        if action in {"business_open", "studio_open"}:
            add("open start launch set up establish studio business store dance studio")
        elif action == "gym_start":
            add("start going to the gym go to gym started gym")

        keywords = []
        kw_model = self._ensure_keybert_model()
        if kw_model:
            try:
                pairs = kw_model.extract_keywords(
                    q,
                    keyphrase_ngram_range=(1, 3),
                    stop_words="english",
                    top_n=3,
                    use_mmr=True,
                    diversity=0.7,
                )
                keywords = [item[0] for item in pairs][:3]
            except Exception:
                keywords = []
        if not keywords:
            stop_words = {
                "what",
                "when",
                "where",
                "who",
                "why",
                "how",
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "with",
                "about",
                "is",
                "are",
                "was",
                "were",
            }
            tokens = [w for w in re.findall(r"\b\w+\b", q.lower()) if len(w) > 2 and w not in stop_words]
            keywords = tokens[:3]
        for kw in keywords:
            add(kw)

        return queries[: self._cfg_148["max_base_queries"]]

    def _ce_predict_cached(self, q: str, cands: list):
        """
        获取 CrossEncoder 分数并复用类内缓存，失败时返回 None。
        """
        xenc = self._ensure_reranker_model()
        if not xenc or not cands:
            return None

        import hashlib

        def _hash(text: str) -> str:
            return hashlib.md5((text or "").encode("utf-8")).hexdigest()

        q_hash = _hash(q)
        scores = [None] * len(cands)
        missing = []
        for idx, mem in enumerate(cands):
            key = (q_hash, _hash(mem.get("memory", "")))
            cached = self._ce_cache.get(key)
            if cached is not None:
                scores[idx] = cached
            else:
                missing.append((idx, key))

        if missing:
            pairs = [[q, cands[idx]["memory"]] for idx, _ in missing]
            try:
                preds = xenc.predict(pairs, show_progress_bar=False)
            except Exception:
                return None
            for (idx, key), val in zip(missing, preds):
                score = float(val)
                scores[idx] = score
                self._ce_cache[key] = score
                self._ce_cache_order.append(key)
                if len(self._ce_cache_order) > self._ce_cache_max:
                    old_key = self._ce_cache_order.pop(0)
                    self._ce_cache.pop(old_key, None)

        for idx, score in enumerate(scores):
            if score is None:
                scores[idx] = float(cands[idx].get("score", 0.0))
        return scores

    def _ce_rank_map(self, q: str, cands: list):
        xenc = self._ensure_reranker_model()
        if not xenc or not cands:
            return {}

        import hashlib

        def _hash(text: str) -> str:
            return hashlib.md5((text or "").encode("utf-8")).hexdigest()

        q_hash = _hash(q)
        scores = [None] * len(cands)
        missing = []
        for idx, mem in enumerate(cands):
            key = (q_hash, _hash(mem.get("memory", "")))
            cached = self._ce_cache.get(key)
            if cached is not None:
                scores[idx] = cached
            else:
                missing.append((idx, key))

        if missing:
            pairs = [[q, cands[idx]["memory"]] for idx, _ in missing]
            try:
                preds = xenc.predict(pairs, show_progress_bar=False)
                for (idx, key), val in zip(missing, preds):
                    score = float(val)
                    scores[idx] = score
                    self._ce_cache[key] = score
                    self._ce_cache_order.append(key)
                    if len(self._ce_cache_order) > self._ce_cache_max:
                        old_key = self._ce_cache_order.pop(0)
                        self._ce_cache.pop(old_key, None)
            except Exception:
                return {}

        ordering = sorted(range(len(cands)), key=lambda j: float(scores[j]), reverse=True)
        return {id(cands[j]): rank + 1 for rank, j in enumerate(ordering)}

    def _search_5(self, user_id, question, top_k):
        # ========== Method 5: Simplified Hybrid Search (Adapted from 14.5/14.9) ==========
        safe_enhanced = True
        PRF_K = 20
        MAX_KEYWORDS = 5
        PRF_KEYWORDS = 6
        MAX_WORKERS = min(3, self._max_parallelism_cap)
        LAMBDA_DIV = 0.8
        RERANK_WEIGHT = 0.8
        RECENCY_WEIGHT = 0.2
        HALF_LIFE_DAYS = 7.0
        BOOST_ALPHA = 0.3
        NOISE_PENALTY = -0.2
        STRUCTURED_MARKER_PREFIXES = ("NUM:", "YEAR:", "MONTH:", "YM:")

        def _safe_lower_tokens(text):
            try:
                return re.findall(r"\b\w+\b", (text or "").lower())
            except Exception:
                return []

        def _jaccard_sim(a_text, b_text):
            a = set(_safe_lower_tokens(a_text))
            b = set(_safe_lower_tokens(b_text))
            if not a or not b:
                return 0.0
            return len(a & b) / float(len(a | b))

        def _mmr_select(candidates, relevance_scores, k, lambda_div=0.6):
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
            if not item: return 0.0
            ts_epoch = item.get("timestamp_epoch")
            if ts_epoch is not None:
                try: return float(ts_epoch)
                except: pass
            ts_value = item.get("timestamp")
            if ts_value is None: return 0.0
            try:
                parsed = datetime.fromisoformat(str(ts_value).replace("Z", "+00:00"))
                if parsed.tzinfo is None: parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed.timestamp()
            except: return 0.0

        def _recency_score(ts, now_ts, half_life_days=7.0):
            if ts <= 0 or now_ts <= 0: return 0.0
            delta_days = max(0.0, (now_ts - ts) / 86400.0)
            return math.exp(-math.log(2.0) * (delta_days / max(1e-6, half_life_days)))

        def _add_to_map(dst_map, item):
            key = item.get("memory")
            if not key: return
            if key not in dst_map or float(item.get("score", 0.0)) > float(dst_map[key].get("score", 0.0)):
                dst_map[key] = item

        # Build Queries
        candidate_queries = [question]
        kw_model = self._ensure_keybert_model()
        keywords = []
        if kw_model:
            try:
                pairs = kw_model.extract_keywords(question, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=MAX_KEYWORDS, use_mmr=True, diversity=0.65)
                keywords = [kw[0] for kw in pairs]
            except Exception as e:
                self.logger.warning(f"KeyBERT failed: {e}")
        
        if not keywords:
             stop_words = {'what', 'when', 'where', 'who', 'why', 'how', 'is', 'are', 'was', 'were', 'the', 'a', 'an'}
             keywords = [w for w in re.findall(r"\b\w+\b", question.lower()) if len(w) > 2 and w not in stop_words][:MAX_KEYWORDS]
        
        candidate_queries.extend(keywords)
        stripped = re.sub(r"\b(what|when|where|who|why|how)\b", "", question, flags=re.I).strip()
        if stripped: candidate_queries.append(stripped)

        final_queries = list(dict.fromkeys(candidate_queries))[:10]

        # Execute Search
        mem_map = {}
        search_time = 0.0
        
        def _search_op(q):
            mems, _, dur = self.search_memory(user_id, q, limit=top_k * 2)
            return mems, dur

        with self._thread_pool(MAX_WORKERS, "mem-search-5") as executor:
            futures = [executor.submit(_search_op, q) for q in final_queries]
            for f in as_completed(futures):
                mems, dur = f.result()
                search_time += float(dur or 0.0)
                for m in mems:
                    _add_to_map(mem_map, m)

        # Rerank
        reranker = self._ensure_reranker_model()
        now_ts = datetime.now(tz=timezone.utc).timestamp()
        
        candidates = list(mem_map.values())
        if not candidates:
            return [], [], search_time

        rerank_scores = []
        if reranker:
            try:
                pairs = [[question, m.get("memory", "")] for m in candidates]
                rerank_scores = reranker.predict(pairs, show_progress_bar=False)
            except Exception:
                 rerank_scores = [0.0] * len(candidates)
        
        processed_candidates = []
        final_scores_map = {}
        
        for idx, item in enumerate(candidates):
            rr = float(rerank_scores[idx]) if reranker else float(item.get("score", 0.0))
            ts = _extract_timestamp_seconds(item)
            rec = _recency_score(ts, now_ts, half_life_days)
            final_val = RERANK_WEIGHT * rr + RECENCY_WEIGHT * rec
            item["final_score"] = final_val
            final_scores_map[item.get("memory", "")] = final_val
            processed_candidates.append(item)
            
        processed_candidates.sort(key=lambda x: x["final_score"], reverse=True)
        
        selected = _mmr_select(processed_candidates, final_scores_map, k=top_k, lambda_div=LAMBDA_DIV)
        formatted = [self._format_memory_line(m) for m in selected]
        return formatted, [], search_time

    def _search_6(self, user_id, question, top_k):
        # ========== Method 6: Vibe Rerank on Full Memories ==========
        mems, _, time_val = self.search_memory(user_id, "get all", limit=1000)
        if not mems:
             return [], [], time_val
        
        doc_texts = [m.get("memory", "") for m in mems]
        idxs = self.vibe_rerank(question=question, documents=doc_texts, top_k=top_k)
        selected = [mems[i] for i in idxs]
        formatted = [self._format_memory_line(m) for m in selected]
        return formatted, [], time_val

    def _search_1427(self, user_id, question, top_k):
        # ========== Method 14.27: Reliable Hybrid (Single User) ==========
        BASE_LIMIT = max(top_k * 2, 40)
        PROTECTED_COUNT = min(5, max(2, top_k // 2))
        kw_model = self._ensure_keybert_model()
        reranker = self._ensure_reranker_model()
        intent = self._intent_148(question)

        mem_map = {}
        base_mems, _, time_val = self.search_memory(user_id, question, limit=BASE_LIMIT)
        protected_keys = {m.get("memory") for m in base_mems[:PROTECTED_COUNT] if m.get("memory")}
        for m in base_mems:
            mem_map[m.get("memory")] = m

        extra_queries = []
        if kw_model:
            try:
                pairs = kw_model.extract_keywords(question, keyphrase_ngram_range=(1,3), stop_words='english', top_n=3, use_mmr=True)
                extra_queries = [p[0] for p in pairs]
            except: pass
        if not extra_queries:
             tokens = re.findall(r"\b\w+\b", question.lower())
             if len(tokens) > 3: extra_queries.append(" ".join(tokens[:3]))
        stripped = re.sub(r"\b(what|when|where|who|why|how)\b", "", question, flags=re.I).strip()
        if stripped: extra_queries.append(stripped)

        with self._thread_pool(4, "mem-search-1427") as executor:
            futures = [executor.submit(self.search_memory, user_id, q, limit=BASE_LIMIT) for q in extra_queries[:4]]
            for f in as_completed(futures):
                mems, _, dur = f.result()
                time_val += dur
                for m in mems:
                    k = m.get("memory")
                    if k and (k not in mem_map or m["score"] > mem_map[k]["score"]):
                         mem_map[k] = m
                         
        cands = list(mem_map.values())
        if not cands: return [], [], time_val
        
        rerank_scores = None
        if reranker:
            try:
                pairs = [[question, m.get("memory", "")] for m in cands]
                rerank_scores = reranker.predict(pairs, show_progress_bar=False)
            except: pass
            
        now_ts = datetime.now(tz=timezone.utc).timestamp()
        recency_weight = 0.25 if intent.get("want_recent") else (0.18 if intent.get("is_time") else 0.1)
        
        scored = []
        for idx, mem in enumerate(cands):
            mem_text = mem.get("memory", "")
            base = float(rerank_scores[idx]) if rerank_scores is not None else float(mem.get("score", 0.0))
            lex = self._lexical_score(question, mem_text)
            lex_norm = math.log1p(max(0.0, lex))
            
            ts_val = mem.get("timestamp_epoch")
            try: ts = float(ts_val) if ts_val else 0.0
            except: ts = 0.0
            rec = math.exp(-math.log(2.0) * max(0.0, (now_ts - ts) / 86400.0 / 10.0)) if ts > 0 else 0.0
            
            final_val = 0.72 * base + 0.18 * lex_norm + recency_weight * rec
            if intent.get("is_time"): final_val += 0.08 if self._search_1427_time_hint(mem_text) else -0.04
            if mem.get("memory") in protected_keys: final_val += 0.2
            
            mem["_final_score"] = final_val
            scored.append(mem)
            
        scored.sort(key=lambda x: x["_final_score"], reverse=True)
        selected = scored[:top_k]
        formatted = [self._format_memory_line(m) for m in selected]
        return formatted, [], time_val

    def _search_1427_time_hint(self, text):
        return bool(re.search(r"\b20\d{2}\b", text or ""))

    def search(self, user_id, question, search_method, top_k_rerank=15, pbar=None):
        search_method = str(search_method)
        if search_method == "5" or search_method == "14.5":
            return self._search_5(user_id, question, top_k_rerank)
        elif search_method == "6":
            return self._search_6(user_id, question, top_k_rerank)
        elif search_method == "1427" or search_method == "14.27":
            return self._search_1427(user_id, question, top_k_rerank)
        else:
            mems, graphs, dur = self.search_memory(user_id, question, limit=top_k_rerank)
            formatted = [self._format_memory_line(m) for m in mems]
            return formatted, graphs, dur

    def answer_question(self, global_user_id, question, answer, category, pbar=None, max_retries=5):
        memories, graph_memories, search_time = self.search(
            global_user_id, question, self.search_method, pbar=pbar
        )
        
        if self.answer_mode == "15":
            prompt_template = ANSWER_PROMPT_15_MSP
        else:
            prompt_template = ANSWER_PROMPT_0_MSP
            
        template = Template(prompt_template)
        
        rendered_prompt = template.render(
            user_id=global_user_id,
            question=question,
            memories=json.dumps(memories, indent=4),
            graph_memories=json.dumps(graph_memories, indent=4)
        )
        
        base_instruction = "You are an intelligent memory assistant."
        combined_prompt = f"{base_instruction}\n\n{rendered_prompt}"

        answer_start = time.time()
        raw_response = self.safe_chat(
            model=self.answer_llm_model,
            messages=[{"role": "user", "content": combined_prompt}],
            temperature=0.0,
        )
        response_time = max(0.0, time.time() - answer_start)
        response_content = self._extract_llm_text(raw_response).strip()
        return response_content, memories, graph_memories, search_time, rendered_prompt, response_time

    def process_question(self, val, global_user_id, idx, pbar=None):
        question = val.get("question", "")
        answer = val.get("answer", "")
        category = val.get("category", -1)
        evidence = val.get("evidence", [])
        adversarial_answer = val.get("adversarial_answer", "")
        
        (
            response,
            memories,
            graph_memories,
            search_time,
            answer_prompt,
            response_time,
        ) = self.answer_question(global_user_id, question, answer, category, pbar)
        
        result = {
            "question": question,
            "answer": answer,
            "category": category,
            "evidence": evidence,
            "adversarial_answer": adversarial_answer,
            "response": response,
            "memories": memories,
            "num_memories": len(memories),
            "graph_memories": graph_memories,
            "search_time": search_time,
            "response_time": response_time,
            "answer_prompt": answer_prompt,
        }
        
        self._record_result(idx, result)
        if pbar: pbar.update(1)
        return result

    def process_data_file(self, file_path, max_workers=5):
        dataset_path = Path(file_path)
        stats = compute_dataset_stats(dataset_path)
        self._dataset_stats = stats
        total_questions = stats.get("total_questions", 0)
        
        if total_questions == 0:
            print("No questions found.")
            self._results_writer = IncrementalResultsWriter(self.output_path)
            self._results_writer.finalize()
            return
            
        print(f"Processing {total_questions} questions.")
        resolved_workers = self._resolve_max_workers(max_workers)
        self._expected_results_per_conversation = stats.get("qa_per_conversation", [])
        self._results_buffer = {}
        self._results_writer = IncrementalResultsWriter(self.output_path)
        
        successful_count = 0
        failed_count = 0
        futures = {}
        drain_threshold = max(resolved_workers, 1) * 4
        
        def consume_one():
            nonlocal successful_count, failed_count
            if not futures: return
            done, _ = wait(tuple(futures.keys()), return_when=FIRST_COMPLETED)
            for finished in done:
                futures.pop(finished)
                try:
                    finished.result()
                    successful_count += 1
                except Exception as exc:
                    failed_count += 1
                    print(f"Error: {exc}")

        with tqdm(total=total_questions, desc="Progress") as pbar:
            try:
                with ThreadPoolExecutor(max_workers=resolved_workers, thread_name_prefix="mem-search-main") as executor:
                    for conv_idx, item in enumerate(stream_normalized_dataset(dataset_path)):
                        # Logic for global observer ID (Static)
                        global_user_id = "global_observer"
                        qa_list = item.get("qa", [])
                        
                        for question_item in qa_list:
                            future = executor.submit(
                                self.process_question,
                                question_item,
                                global_user_id,
                                conv_idx,
                                pbar,
                            )
                            futures[future] = conv_idx
                            if len(futures) >= drain_threshold:
                                consume_one()
                                
                    while futures:
                        consume_one()
                        
            except Exception as exc:
                print(f"Runtime error: {exc}")
            finally:
                self._results_writer.flush() # Simplify flush
                self._results_writer.finalize()
        
        print(f"Processed: Success={successful_count}, Failed={failed_count}")
