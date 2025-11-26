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
            elif answer_mode == "12":
                from prompts import ANSWER_PROMPT_12
                self.ANSWER_PROMPT = ANSWER_PROMPT_12
            elif answer_mode == "13":
                from prompts import ANSWER_PROMPT_13
                self.ANSWER_PROMPT = ANSWER_PROMPT_13
            elif answer_mode == "14":
                from prompts import ANSWER_PROMPT_14
                self.ANSWER_PROMPT = ANSWER_PROMPT_14
            elif answer_mode == "14.6":
                from prompts import ANSWER_PROMPT_14_6
                self.ANSWER_PROMPT = ANSWER_PROMPT_14_6
            elif answer_mode == "14.7":
                from prompts import ANSWER_PROMPT_14_7
                self.ANSWER_PROMPT = ANSWER_PROMPT_14_7
            elif answer_mode == "14.8":
                from prompts import ANSWER_PROMPT_14_8
                self.ANSWER_PROMPT = ANSWER_PROMPT_14_8
            elif answer_mode == "14.9":
                from prompts import ANSWER_PROMPT_14_9
                self.ANSWER_PROMPT = ANSWER_PROMPT_14_9
            else:
                self.logger.warning("Unknown answer_mode '%s'. Falling back to default prompt.", answer_mode)
                self.ANSWER_PROMPT = ANSWER_PROMPT

        self.speaker_1_full_memories = []
        self.speaker_2_full_memories = []

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

    @staticmethod
    def _format_memory_line(item):
        """
        Render memory entries with a consistent timestamp display.
        """
        display_ts = item.get("timestamp_display") or item.get("timestamp") or ""
        memory_text = item.get("memory", "")
        return f"{display_ts}: {memory_text}"

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

    def _numeric_bonus_148(self, intent: dict, text: str) -> float:
        if not intent.get("is_numeric"):
            return 0.0
        if not text:
            return 0.0
        import re

        has_num = bool(re.search(r"\d", text))
        units_pattern = r"(次|天|周|月|年|小时|分钟|岁|人|people|times|days|weeks|months|years|hours|minutes)"
        has_unit = bool(re.search(units_pattern, text.lower()))
        if has_num and has_unit:
            return 0.5
        if has_num:
            return 0.25
        return 0.0

    def _recency_weight_148(self, intent: dict) -> float:
        if intent.get("want_recent") and not (intent.get("has_year") or intent.get("has_month")):
            return 0.25
        return 0.0

    def _rrf_sum(self, ranks: list, c: int) -> float:
        total = 0.0
        for rank in ranks:
            if rank and rank > 0:
                total += 1.0 / (c + rank)
        return total

    def _final_score_148(self, q: str, mem: dict, ranks: dict, intent: dict, pool_size: int) -> float:
        c_value = self._cfg_148["rrf_c_small"] if pool_size < self._cfg_148["small_pool_thresh"] else self._cfg_148["rrf_c_large"]
        base = self._rrf_sum([ranks.get("embed"), ranks.get("lex"), ranks.get("xenc")], c=c_value)
        base += 0.12 * float(mem.get("score", 0.0))

        if self._is_noise_text(mem.get("memory", "")):
            base -= self._cfg_148["noise_penalty"]

        base += self._subject_bonus(mem.get("memory", ""), intent.get("subjects") or [])
        base += self._action_bonus(q, mem.get("memory", ""))

        time_bonus = self._time_bonus(q, mem.get("memory", ""))
        if intent.get("is_time"):
            if intent.get("has_month") or intent.get("has_year"):
                base += 1.1 * time_bonus
            else:
                base += 0.7 * time_bonus

        if intent.get("is_time"):
            y, m, d = self._extract_norm_date_from_memory(mem.get("memory", ""))
            if not (y or m or d):
                base -= 0.3

        base += self._numeric_bonus_148(intent, mem.get("memory", ""))

        rec_weight = self._recency_weight_148(intent)
        if rec_weight > 0.0:
            now_ts = datetime.now(tz=timezone.utc).timestamp()
            ts = mem.get("timestamp_epoch") or 0.0
            if ts > 0.0:
                decay = math.exp(-math.log(2.0) * max(0.0, (now_ts - ts) / 86400.0 / 7.0))
                base += rec_weight * decay

        return base

    def _mmr_select_148(self, candidates: list, k: int, lambda_div: float) -> list:
        import re

        def tokens(text: str):
            return set(re.findall(r"\b\w+\b", (text or "").lower()))

        def jaccard(a: str, b: str) -> float:
            ta, tb = tokens(a), tokens(b)
            if not ta or not tb:
                return 0.0
            return len(ta & tb) / float(len(ta | tb))

        remaining = list(candidates)
        selected = []
        chosen_texts = []
        while remaining and len(selected) < k:
            best_item = None
            best_val = -1e9
            for item in remaining:
                relevance = float(item.get("__final__", 0.0))
                diversity = max((jaccard(item.get("memory", ""), txt) for txt in chosen_texts), default=0.0)
                value = lambda_div * relevance - (1.0 - lambda_div) * diversity
                if value > best_val:
                    best_val = value
                    best_item = item
            selected.append(best_item)
            chosen_texts.append(best_item.get("memory", ""))
            remaining.remove(best_item)
        return selected

    def _search_148(self, speaker_1_user_id, speaker_2_user_id, question, top_k):
        cfg = self._cfg_148
        intent = self._intent_148(question)
        base_queries = self._gen_queries_148(question, intent)

        time_map = {speaker_1_user_id: 0.0, speaker_2_user_id: 0.0}

        def recall(uid):
            collected = {}
            spent = 0.0
            for q in base_queries:
                mems, _, elapsed = self.search_memory(uid, q, limit=cfg["per_query_limit"])
                spent += float(elapsed or 0.0)
                for mem in mems:
                    key = mem["memory"]
                    if key not in collected or float(mem["score"]) > float(collected[key]["score"]):
                        collected[key] = mem
            return collected, spent

        a_map, time_map[speaker_1_user_id] = recall(speaker_1_user_id)
        b_map, time_map[speaker_2_user_id] = recall(speaker_2_user_id)

        def prf_expand(collected, uid, spent):
            if len(collected) >= max(3 * top_k, 60):
                return collected, spent
            top_pool = sorted(collected.values(), key=lambda x: float(x.get("score", 0.0)), reverse=True)[:20]
            prf_text = " \n".join(mem["memory"] for mem in top_pool)
            terms = []
            kw_model = self._ensure_keybert_model()
            if kw_model and prf_text:
                try:
                    pairs = kw_model.extract_keywords(
                        prf_text,
                        keyphrase_ngram_range=(1, 3),
                        stop_words="english",
                        top_n=4,
                        use_mmr=True,
                        diversity=0.7,
                    )
                    terms = [p[0] for p in pairs][:4]
                except Exception:
                    terms = []
            if not terms:
                import re

                tokens = [w for w in re.findall(r"\b\w+\b", prf_text.lower()) if len(w) > 2]
                terms = tokens[:4]
            prf_queries = [" ".join(terms[:2]), " ".join(terms[2:4])]
            for q in prf_queries[: cfg["max_prf_queries"]]:
                if not q:
                    continue
                mems, _, elapsed = self.search_memory(uid, q, limit=cfg["per_query_limit"])
                spent += float(elapsed or 0.0)
                for mem in mems:
                    key = mem["memory"]
                    if key not in collected or float(mem["score"]) > float(collected[key]["score"]):
                        collected[key] = mem
            return collected, spent

        a_map, time_map[speaker_1_user_id] = prf_expand(a_map, speaker_1_user_id, time_map[speaker_1_user_id])
        b_map, time_map[speaker_2_user_id] = prf_expand(b_map, speaker_2_user_id, time_map[speaker_2_user_id])

        def rank(uid_map):
            candidates = list(uid_map.values())
            if not candidates:
                return []

            embed_rank = {
                id(mem): idx + 1
                for idx, mem in enumerate(sorted(candidates, key=lambda x: float(x.get("score", 0.0)), reverse=True))
            }
            lex_rank = {
                id(mem): idx + 1
                for idx, mem in enumerate(sorted(candidates, key=lambda x: self._lexical_score(question, x["memory"]), reverse=True))
            }
            x_rank = self._ce_rank_map(question, candidates)

            scored = []
            for mem in candidates:
                ranks = {
                    "embed": embed_rank.get(id(mem)),
                    "lex": lex_rank.get(id(mem)),
                    "xenc": x_rank.get(id(mem)),
                }
                score = self._final_score_148(question, mem, ranks, intent, pool_size=len(candidates))
                mem["__final__"] = score
                scored.append(mem)

            scored.sort(key=lambda m: m["__final__"], reverse=True)
            lam = cfg["mmr_lambda_time"] if intent.get("is_time") else cfg["mmr_lambda_default"]
            picked = self._mmr_select_148(scored, k=top_k, lambda_div=lam)
            return [self._format_memory_line(mem) for mem in picked]

        return rank(a_map), rank(b_map), time_map.get(speaker_1_user_id, 0.0), time_map.get(speaker_2_user_id, 0.0)

    def _search_1410(self, speaker_1_user_id, speaker_2_user_id, question, top_k):
        PRF_K = 20
        MAX_KEYWORDS = 5
        PRF_KEYWORDS = 6
        MAX_WORKERS = min(3, self._max_parallelism_cap)
        RERANK_WEIGHT = 0.8
        BOOST_ALPHA = 0.3
        NOISE_PENALTY = -0.2
        STRUCTURED_MARKER_PREFIXES = ("NUM:", "YEAR:", "MONTH:", "YM:")

        intent = self._intent_148(question)
        if intent.get("want_recent"):
            RECENCY_WEIGHT = 0.35
            HALF_LIFE_DAYS = 3.0
        elif intent.get("is_time"):
            RECENCY_WEIGHT = 0.25
            HALF_LIFE_DAYS = 5.0
        else:
            RECENCY_WEIGHT = 0.15
            HALF_LIFE_DAYS = 14.0
        LAMBDA_DIV = 0.85 if intent.get("is_time") else 0.75

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

        _MONTH_ALIAS = {
            "jan": 1,
            "january": 1,
            "feb": 2,
            "february": 2,
            "mar": 3,
            "march": 3,
            "apr": 4,
            "april": 4,
            "may": 5,
            "jun": 6,
            "june": 6,
            "jul": 7,
            "july": 7,
            "aug": 8,
            "august": 8,
            "sep": 9,
            "sept": 9,
            "september": 9,
            "oct": 10,
            "october": 10,
            "nov": 11,
            "november": 11,
            "dec": 12,
            "december": 12,
        }
        _CN_MONTH_WORDS = {
            "一月": 1,
            "二月": 2,
            "三月": 3,
            "四月": 4,
            "五月": 5,
            "六月": 6,
            "七月": 7,
            "八月": 8,
            "九月": 9,
            "十月": 10,
            "十一月": 11,
            "十二月": 12,
        }

        def _extract_time_number_markers(text):
            markers = set()
            if not text:
                return markers
            lower_text = text.lower()

            def _add_num(value):
                try:
                    num = int(value)
                except (TypeError, ValueError):
                    return
                markers.add(f"NUM:{num}")

            def _add_year(value):
                try:
                    year = int(value)
                except (TypeError, ValueError):
                    return
                markers.add(f"YEAR:{year}")
                _add_num(year)

            def _add_month(value):
                try:
                    month = int(value)
                except (TypeError, ValueError):
                    return
                if 1 <= month <= 12:
                    markers.add(f"MONTH:{month}")

            def _add_year_month(year, month):
                try:
                    y = int(year)
                    m = int(month)
                except (TypeError, ValueError):
                    return
                if 1 <= m <= 12:
                    markers.add(f"YM:{y}-{m:02d}")
                    _add_year(y)
                    _add_month(m)

            for raw in re.findall(r"\b\d{1,4}\b", text):
                _add_num(raw)
                if len(raw) == 4 and raw.startswith("20"):
                    _add_year(raw)

            for match in re.finditer(r"\b(20\d{2})[-/.](\d{1,2})(?:[-/.]\d{1,2})?\b", text):
                _add_year_month(match.group(1), match.group(2))
            for match in re.finditer(r"\b(\d{1,2})[/-](20\d{2})\b", text):
                _add_year_month(match.group(2), match.group(1))
            for match in re.finditer(r"(20\d{2})年(\d{1,2})月", text):
                _add_year_month(match.group(1), match.group(2))
            for match in re.finditer(r"(\d{1,2})月(20\d{2})年?", text):
                _add_year_month(match.group(2), match.group(1))
            for match in re.finditer(r"(20\d{2})年", text):
                _add_year(match.group(1))
            for match in re.finditer(r"(\d{1,2})月", text):
                _add_month(match.group(1))

            for alias, month_idx in _MONTH_ALIAS.items():
                pattern = r"\b" + re.escape(alias) + r"\b"
                if re.search(pattern, lower_text):
                    _add_month(month_idx)
                combo_pattern = pattern + r"\s+(20\d{2})\b"
                for year in re.findall(combo_pattern, lower_text):
                    _add_year_month(year, month_idx)
            for word, month_idx in _CN_MONTH_WORDS.items():
                if word in text:
                    _add_month(month_idx)

            for match in re.finditer(
                r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
                lower_text,
            ):
                token = match.group(1)
                base = token[:3]
                if base == "sep":
                    month_idx = 9
                else:
                    month_idx = _MONTH_ALIAS.get(base, _MONTHS.get(token, None))
                if month_idx:
                    _add_month(month_idx)
            for match in re.finditer(
                r"\b(20\d{2})\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
                lower_text,
            ):
                year = match.group(1)
                token = match.group(2)
                base = token[:3]
                if base == "sep":
                    month_idx = 9
                else:
                    month_idx = _MONTH_ALIAS.get(base, _MONTHS.get(token, None))
                if month_idx:
                    _add_year_month(year, month_idx)

            return markers

        def _has_structured_tokens(markers):
            return any(token.startswith(STRUCTURED_MARKER_PREFIXES) for token in markers)

        question_markers = _extract_time_number_markers(question)

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

        def _recency_score(ts, now_ts, half_life_days=7.0):
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

        keywords = []
        kw_model = self._ensure_keybert_model()
        if kw_model is not None:
            try:
                extracted = kw_model.extract_keywords(
                    question,
                    keyphrase_ngram_range=(1, 3),
                    stop_words="english",
                    top_n=MAX_KEYWORDS,
                    use_mmr=True,
                    diversity=0.7,
                )
                keywords = [kw[0] for kw in extracted]
            except Exception as e:
                print(f"⚠️ KeyBERT extraction failed: {e}. Using fallback.")

        if not keywords:
            stop_words = {
                "what",
                "when",
                "where",
                "who",
                "why",
                "how",
                "is",
                "are",
                "was",
                "were",
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
            }
            words = re.findall(r"\b\w+\b", question.lower())
            keywords = [w for w in words if w not in stop_words and len(w) > 2][:MAX_KEYWORDS]

        base_queries = [question]
        base_queries.extend(keywords)

        a_map = {}
        b_map = {}
        search_time_by_user = {speaker_1_user_id: 0.0, speaker_2_user_id: 0.0}

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

        global_pool = list(a_map.values()) + list(b_map.values())
        global_pool_sorted = sorted(global_pool, key=lambda x: float(x.get("score", 0.0)), reverse=True)
        prf_docs = global_pool_sorted[: min(PRF_K, len(global_pool_sorted))]

        prf_text = " \n".join(m.get("memory", "") for m in prf_docs)
        prf_terms = []
        if kw_model is not None and prf_text:
            try:
                extracted = kw_model.extract_keywords(
                    prf_text,
                    keyphrase_ngram_range=(1, 3),
                    stop_words="english",
                    top_n=PRF_KEYWORDS,
                    use_mmr=True,
                    diversity=0.7,
                )
                prf_terms = [kw[0] for kw in extracted]
            except Exception as e:
                print(f"⚠️ PRF KeyBERT failed: {e}. Using fallback.")

        if not prf_terms:
            prf_terms = [t for t in _safe_lower_tokens(prf_text) if len(t) > 2][:PRF_KEYWORDS]

        prf_queries = []
        if prf_terms:
            prf_queries.append(" ".join([str(term) for term in prf_terms[:3]]))
        if len(prf_terms) >= 4:
            prf_queries.append(" ".join([str(term) for term in prf_terms[2:6]]))

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

        reranker = self._ensure_reranker_model()
        now_ts = datetime.now(tz=timezone.utc).timestamp()
        subjects = intent.get("subjects") or []

        def _score_after_rerank(cands):
            if not cands:
                return [], {}
            base_scores = [float(m.get("score", 0.0)) for m in cands]
            rerank_scores = self._ce_predict_cached(question, cands)
            if rerank_scores is None and reranker is not None:
                try:
                    pairs = [[question, m.get("memory", "")] for m in cands]
                    rerank_scores = reranker.predict(pairs, show_progress_bar=False)
                except Exception as e:
                    print(f"⚠️ Reranking failed: {e}. Fallback to base score.")
                    rerank_scores = None

            final_scores = {}
            for i, m in enumerate(cands):
                mem_text = m.get("memory", "")
                rr = float(rerank_scores[i]) if rerank_scores is not None else base_scores[i]
                ts = _extract_timestamp_seconds(m)
                rec = _recency_score(ts, now_ts, half_life_days=HALF_LIFE_DAYS)
                final_val = RERANK_WEIGHT * rr + RECENCY_WEIGHT * rec

                mem_markers = _extract_time_number_markers(mem_text)
                overlap = question_markers & mem_markers
                if question_markers and overlap:
                    frac = len(overlap) / max(1, len(question_markers))
                    boost = BOOST_ALPHA * min(1.0, 0.5 + frac)
                    final_val += boost
                    m["marker_boost"] = boost

                if NOISE_PENALTY < 0.0 and self._is_noise_text(mem_text):
                    penalty = NOISE_PENALTY
                    has_structured = _has_structured_tokens(mem_markers)
                    if not has_structured:
                        has_structured = bool(self._extract_candidate_names(mem_text))
                    if has_structured:
                        penalty = max(penalty, 0.0)
                    final_val += penalty
                    m["noise_penalty"] = penalty

                sb = self._subject_bonus(mem_text, subjects)
                ab = self._action_bonus(question, mem_text)
                final_val += 0.12 * sb + 0.15 * ab

                time_hit = self._time_bonus(question, mem_text)
                final_val += 0.30 * time_hit
                if intent.get("is_time"):
                    y, mn, dd = self._extract_norm_date_from_memory(mem_text)
                    if not (y or mn or dd):
                        final_val -= 0.15

                lx = self._lexical_score(question, mem_text)
                final_val += 0.05 * math.log1p(max(0.0, lx))

                final_scores[mem_text] = final_val
                m["rerank_score"] = rr
                m["final_score"] = final_val

            sorted_items = sorted(cands, key=lambda x: x.get("final_score", 0.0), reverse=True)
            return sorted_items, final_scores

        a_candidates, b_candidates = list(a_map.values()), list(b_map.values())
        rerank_executor = self._get_rerank_executor()
        future_a = rerank_executor.submit(_score_after_rerank, a_candidates)
        future_b = rerank_executor.submit(_score_after_rerank, b_candidates)
        a_sorted, a_scores = future_a.result()
        b_sorted, b_scores = future_b.result()

        a_top = _mmr_select(a_sorted, a_scores, k=top_k, lambda_div=LAMBDA_DIV) if a_sorted else []
        b_top = _mmr_select(b_sorted, b_scores, k=top_k, lambda_div=LAMBDA_DIV) if b_sorted else []

        search_1_memory = [self._format_memory_line(m) for m in a_top]
        search_2_memory = [self._format_memory_line(m) for m in b_top]
        speaker_1_time = search_time_by_user.get(speaker_1_user_id, 0.0)
        speaker_2_time = search_time_by_user.get(speaker_2_user_id, 0.0)
        return search_1_memory, search_2_memory, speaker_1_time, speaker_2_time

    def _search_1421(self, speaker_1_user_id, speaker_2_user_id, question, top_k):
        """
        14.20 增强版：在原 recall/PRF/多视角基础上增加模式识别 + 置信度 gating 的软融合，
        旨在避免“跷跷板”并提升通用性。
        """
        MAX_KEYWORDS = 8
        MAX_FALLBACK_KEYWORDS = 8
        MAX_KEYWORD_TOTAL = 10
        PRF_K = 25
        PRF_KEYWORDS = 8
        MAX_MULTIHOP_QUERIES = 3
        NEG_PENALTY = 0.45
        STRUCTURED_MARKER_PREFIXES = ("NUM:", "YEAR:", "MONTH:", "YM:")
        MAX_WORKERS = min(4, self._max_parallelism_cap)

        # Gating + 融合控制
        CONF_HIGH = 0.64
        CONF_LOW = 0.32
        DELTA_CLIP = 0.9
        LAMBDA_BASE = 0.18
        PROTECT_TOP = 1  # 高置信度下保护前 N

        intent = self._intent_148(question)
        if intent.get("is_time") and (intent.get("has_year") or intent.get("has_month")):
            return self._search_147(speaker_1_user_id, speaker_2_user_id, question, top_k)
        if intent.get("is_time") and not intent.get("want_recent"):
            return self._search_146(speaker_1_user_id, speaker_2_user_id, question, top_k)

        neg_terms, sanitized_question, neg_infos = self._extract_negative_terms(question)
        neg_terms = {t for t in neg_terms if t}
        neg_infos = neg_infos or []
        question_for_keywords = sanitized_question or question

        if intent.get("want_recent"):
            recency_weight = 0.32
            half_life_days = 3.0
        elif intent.get("is_time"):
            recency_weight = 0.24
            half_life_days = 5.0
        else:
            recency_weight = 0.16
            half_life_days = 10.0
        rerank_weight = 0.66
        lambda_div = 0.8 if intent.get("is_time") else 0.74

        base_limit = max(self.top_k, 40)
        keyword_limit = min(80, max(self.top_k * 2, base_limit))
        phrase_limit = min(100, max(self.top_k * 3, keyword_limit))
        followup_limit = min(120, max(self.top_k * 4, phrase_limit))

        def _safe_lower_tokens(text):
            try:
                return re.findall(r"\b\w+\b", (text or "").lower())
            except Exception:
                return []

        def _filtered_tokens(text):
            toks = _safe_lower_tokens(text)
            if not neg_terms:
                return toks
            return [t for t in toks if t not in neg_terms]

        def _normalize_for_lex(text):
            filtered = _filtered_tokens(text)
            if filtered:
                return " ".join(filtered)
            return (text or "").strip()

        def _extract_time_number_markers(text):
            markers = set()
            if not text:
                return markers
            lower_text = text.lower()

            def _add_num(value):
                try:
                    num = int(value)
                except (TypeError, ValueError):
                    return
                markers.add(f"NUM:{num}")

            def _add_year(value):
                try:
                    year = int(value)
                except (TypeError, ValueError):
                    return
                markers.add(f"YEAR:{year}")
                _add_num(year)

            def _add_month(value):
                try:
                    month = int(value)
                except (TypeError, ValueError):
                    return
                if 1 <= month <= 12:
                    markers.add(f"MONTH:{month:02d}")
                    _add_num(month)

            def _add_year_month(year, month):
                try:
                    y = int(year)
                    m = int(month)
                except (TypeError, ValueError):
                    return
                if 1 <= m <= 12:
                    markers.add(f"YM:{y}-{m:02d}")
                    _add_year(y)
                    _add_month(m)

            for raw in re.findall(r"\b\d{1,4}\b", text):
                _add_num(raw)
                if len(raw) == 4 and raw.startswith("20"):
                    _add_year(raw)
            for match in re.finditer(r"\b(20\d{2})[-/.](\d{1,2})(?:[-/.]\d{1,2})?\b", text):
                _add_year_month(match.group(1), match.group(2))
            for match in re.finditer(r"\b(\d{1,2})[/-](20\d{2})\b", text):
                _add_year_month(match.group(2), match.group(1))
            for match in re.finditer(r"(20\d{2})年(\d{1,2})月", text):
                _add_year_month(match.group(1), match.group(2))
            for match in re.finditer(r"(\d{1,2})月(20\d{2})年?", text):
                _add_year_month(match.group(2), match.group(1))
            for match in re.finditer(r"(20\d{2})年", text):
                _add_year(match.group(1))
            for match in re.finditer(r"(\d{1,2})月", text):
                _add_month(match.group(1))
            for alias, month_idx in {
                "jan": 1,
                "january": 1,
                "feb": 2,
                "february": 2,
                "mar": 3,
                "march": 3,
                "apr": 4,
                "april": 4,
                "may": 5,
                "jun": 6,
                "june": 6,
                "jul": 7,
                "july": 7,
                "aug": 8,
                "august": 8,
                "sep": 9,
                "sept": 9,
                "september": 9,
                "oct": 10,
                "october": 10,
                "nov": 11,
                "november": 11,
                "dec": 12,
                "december": 12,
            }.items():
                pattern = r"\b" + re.escape(alias) + r"\b"
                if re.search(pattern, lower_text):
                    _add_month(month_idx)
                combo_pattern = pattern + r"\s+(20\d{2})\b"
                for year in re.findall(combo_pattern, lower_text):
                    _add_year_month(year, month_idx)
            for match in re.finditer(
                r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
                lower_text,
            ):
                token = match.group(1)
                base = token[:3]
                month_idx = 9 if base == "sep" else {
                    "jan": 1,
                    "feb": 2,
                    "mar": 3,
                    "apr": 4,
                    "may": 5,
                    "jun": 6,
                    "jul": 7,
                    "aug": 8,
                    "oct": 10,
                    "nov": 11,
                    "dec": 12,
                }.get(base, _MONTHS.get(token, None))
                if month_idx:
                    _add_month(month_idx)
            for match in re.finditer(
                r"\b(20\d{2})\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
                lower_text,
            ):
                year = match.group(1)
                token = match.group(2)
                base = token[:3]
                month_idx = 9 if base == "sep" else {
                    "jan": 1,
                    "feb": 2,
                    "mar": 3,
                    "apr": 4,
                    "may": 5,
                    "jun": 6,
                    "jul": 7,
                    "aug": 8,
                    "oct": 10,
                    "nov": 11,
                    "dec": 12,
                }.get(base, _MONTHS.get(token, None))
                if month_idx:
                    _add_year_month(year, month_idx)
            return markers

        def _has_structured_tokens(markers):
            return any(token.startswith(STRUCTURED_MARKER_PREFIXES) for token in markers)

        question_markers = _extract_time_number_markers(question)

        def _add_to_map(dst_map, item):
            key = item.get("memory")
            if not key:
                return
            existing = dst_map.get(key)
            if existing is None or float(item.get("score", 0.0)) > float(existing.get("score", 0.0)):
                dst_map[key] = item

        def _search(uid, q, limit):
            memories, _, duration = self.search_memory(uid, q, limit=limit)
            return uid, memories, duration

        kw_model = self._ensure_keybert_model()
        keybert_terms = []
        if kw_model is not None:
            try:
                extracted = kw_model.extract_keywords(
                    question_for_keywords,
                    keyphrase_ngram_range=(1, 3),
                    stop_words="english",
                    top_n=MAX_KEYWORDS,
                    use_mmr=True,
                    diversity=0.7,
                )
                keybert_terms = [kw[0] for kw in extracted]
            except Exception as exc:
                print(f"⚠️ KeyBERT extraction failed: {exc}.")

        stop_words = {
            "what",
            "when",
            "where",
            "who",
            "why",
            "how",
            "is",
            "are",
            "was",
            "were",
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
        }
        tokens = [w for w in re.findall(r"\b\w+\b", question_for_keywords.lower()) if len(w) > 2 and w not in stop_words]
        fallback_terms = tokens[:MAX_FALLBACK_KEYWORDS]

        merged_keywords = []
        seen_kw = set()
        for term in fallback_terms + keybert_terms:
            normalized = (term or "").strip()
            if not normalized:
                continue
            low = normalized.lower()
            if low in seen_kw or low in neg_terms:
                continue
            seen_kw.add(low)
            merged_keywords.append(normalized)
            if len(merged_keywords) >= MAX_KEYWORD_TOTAL:
                break

        def _generate_phrase_queries():
            phrases = []
            base_text = question_for_keywords or ""
            lowered_question = base_text.lower()
            filtered_tokens = _filtered_tokens(base_text)
            nouns_of_interest = {
                "activities",
                "activity",
                "things",
                "ideas",
                "plans",
                "places",
                "options",
                "spots",
                "games",
                "events",
                "restaurants",
                "meals",
                "hobbies",
                "recommendations",
                "destinations",
            }
            qualifier_terms = []
            for idx in range(len(filtered_tokens) - 1):
                nxt = filtered_tokens[idx + 1]
                if nxt in nouns_of_interest:
                    token = filtered_tokens[idx]
                    if token not in neg_terms and len(token) > 2:
                        qualifier_terms.append(token)
            qualifier_terms = list(dict.fromkeys(qualifier_terms))

            relation_vocab = [
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
                "coworker",
                "coworkers",
                "colleague",
                "colleagues",
                "roommate",
                "roommates",
            ]
            found_relations = [rel for rel in relation_vocab if rel in lowered_question]

            for n in (2, 3):
                for i in range(len(filtered_tokens) - n + 1):
                    span = filtered_tokens[i : i + n]
                    phrase = " ".join(span)
                    if phrase:
                        phrases.append(phrase)

            for qual in qualifier_terms[:6]:
                for noun in ("activities", "things", "ideas"):
                    phrases.append(f"{qual} {noun}")

            for rel in found_relations[:5]:
                phrases.append(f"with my {rel}")
                phrases.append(f"{rel} activities")
                for qual in qualifier_terms[:4] or ["fun", "best"]:
                    phrases.append(f"{qual} things to do with my {rel}")
                    phrases.append(f"{qual} activities with {rel}")

            prep_patterns = [
                r"\bwith\s+(?:my|your|his|her|our|their|the)?\s*[A-Za-z][A-Za-z\s]{1,40}",
                r"\bfor\s+(?:my|your|his|her|our|their|the)?\s*[A-Za-z][A-Za-z\s]{1,40}",
            ]
            for pat in prep_patterns:
                for match in re.finditer(pat, base_text, flags=re.IGNORECASE):
                    phrase = re.sub(r"\s+", " ", match.group(0).strip())
                    if phrase:
                        phrases.append(phrase)

            for qual in qualifier_terms[:4]:
                for kw in merged_keywords[:3]:
                    if qual and kw and qual not in kw.lower():
                        phrases.append(f"{qual} {kw}")

            return phrases

        queries = []
        seen_queries = set()

        def _add_query(text, limit):
            norm = re.sub(r"\s+", " ", (text or "").strip())
            if not norm:
                return
            key = norm.lower()
            if key in seen_queries:
                return
            seen_queries.add(key)
            queries.append({"text": norm, "limit": limit})

        _add_query(question, base_limit)
        if sanitized_question and sanitized_question.lower() != question.lower():
            _add_query(sanitized_question, base_limit)

        for info in neg_infos:
            phrase = info.get("phrase")
            target = info.get("target")
            prefix = info.get("prefix") or ""
            suffix = info.get("suffix") or ""
            positive_focus = prefix.strip() or sanitized_question
            if prefix:
                _add_query(prefix, base_limit)
            if suffix:
                _add_query(suffix, base_limit)
            if phrase:
                _add_query(phrase, keyword_limit)
                if positive_focus:
                    _add_query(f"{positive_focus} {phrase}", phrase_limit)
            if target:
                cleaned_target = re.sub(r"\s+", " ", target).strip()
                if cleaned_target and positive_focus:
                    _add_query(f"{positive_focus} {cleaned_target}", phrase_limit)
            if sanitized_question and phrase:
                _add_query(f"{sanitized_question} {phrase}", phrase_limit)
            if sanitized_question and target:
                cleaned_target = re.sub(r"\s+", " ", target or "").strip()
                if cleaned_target:
                    _add_query(f"{sanitized_question} {cleaned_target}", phrase_limit)

        for kw in merged_keywords:
            _add_query(kw, keyword_limit)
        for phrase in _generate_phrase_queries():
            _add_query(phrase, phrase_limit)

        a_map = {}
        b_map = {}
        search_time_by_user = {speaker_1_user_id: 0.0, speaker_2_user_id: 0.0}

        with self._thread_pool(MAX_WORKERS, "mem-search-1421-base") as executor:
            futures = []
            for payload in queries:
                q_text = payload["text"]
                limit = payload["limit"]
                futures.append(executor.submit(_search, speaker_1_user_id, q_text, limit))
                futures.append(executor.submit(_search, speaker_2_user_id, q_text, limit))
            for future in as_completed(futures):
                uid, mems, duration = future.result()
                search_time_by_user[uid] = search_time_by_user.get(uid, 0.0) + float(duration or 0.0)
                for mem in mems:
                    if uid == speaker_1_user_id:
                        _add_to_map(a_map, mem)
                    else:
                        _add_to_map(b_map, mem)

        global_pool = list(a_map.values()) + list(b_map.values())
        global_pool_sorted = sorted(global_pool, key=lambda x: float(x.get("score", 0.0)), reverse=True)
        prf_docs = global_pool_sorted[: min(PRF_K, len(global_pool_sorted))]
        prf_text = " \n".join(m.get("memory", "") for m in prf_docs)

        prf_terms = []
        if kw_model is not None and prf_text:
            try:
                extracted = kw_model.extract_keywords(
                    prf_text,
                    keyphrase_ngram_range=(1, 3),
                    stop_words="english",
                    top_n=PRF_KEYWORDS,
                    use_mmr=True,
                    diversity=0.7,
                )
                prf_terms = [kw[0] for kw in extracted]
            except Exception as exc:
                print(f"⚠️ PRF KeyBERT failed: {exc}.")
        if not prf_terms:
            prf_terms = [t for t in _filtered_tokens(prf_text) if len(t) > 2][:PRF_KEYWORDS]
        prf_terms = [term for term in prf_terms if term.lower() not in neg_terms]

        prf_queries = []
        if prf_terms:
            prf_queries.append(" ".join(prf_terms[:3]))
        if len(prf_terms) >= 4:
            prf_queries.append(" ".join(prf_terms[2:6]))

        if prf_queries:
            with self._thread_pool(MAX_WORKERS, "mem-search-1421-prf") as executor:
                futures = []
                for q_text in prf_queries:
                    futures.append(executor.submit(_search, speaker_1_user_id, q_text, phrase_limit))
                    futures.append(executor.submit(_search, speaker_2_user_id, q_text, phrase_limit))
                for future in as_completed(futures):
                    uid, mems, duration = future.result()
                    search_time_by_user[uid] = search_time_by_user.get(uid, 0.0) + float(duration or 0.0)
                    for mem in mems:
                        if uid == speaker_1_user_id:
                            _add_to_map(a_map, mem)
                        else:
                            _add_to_map(b_map, mem)

        global_pool = list(a_map.values()) + list(b_map.values())
        global_pool_sorted = sorted(global_pool, key=lambda x: float(x.get("score", 0.0)), reverse=True)

        def _multi_hop_queries(base_question, docs):
            if not docs or not getattr(self, "search_client", None):
                return []
            doc_lines = "\n".join(f"- {m.get('memory', '')}" for m in docs[:10])
            constraint_line = ""
            if neg_terms:
                constraint_line = f"\nAvoid asking about: {', '.join(sorted(neg_terms))}."
            prompt = f"""You are a retrieval planner that proposes follow-up search queries.
User question:
{base_question}

Currently retrieved memories:
{doc_lines}

Suggest up to {MAX_MULTIHOP_QUERIES} additional short search queries that could retrieve missing complementary facts.{constraint_line}
Format:
Q1: ...
Q2: ...
Q3: ..."""
            try:
                resp = self.safe_chat(
                    model=self.llm_model,
                    messages=[{"role": "system", "content": prompt}],
                    temperature=0.2,
                )
            except Exception as exc:
                print(f"⚠️ Multi-hop planning failed: {exc}")
                return []
            text = resp.choices[0].message.content if resp.choices and resp.choices[0].message else ""
            if not text:
                return []
            queries = []
            for line in text.splitlines():
                match = re.search(r"Q\d+:\s*(.+)", line.strip())
                if not match:
                    continue
                candidate = match.group(1).strip()
                if not candidate:
                    continue
                low = candidate.lower()
                if neg_terms and any(term in low for term in neg_terms):
                    continue
                if candidate in queries:
                    continue
                queries.append(candidate)
                if len(queries) >= MAX_MULTIHOP_QUERIES:
                    break
            return queries

        coarse_docs = global_pool_sorted[:20]
        hop_queries = _multi_hop_queries(question_for_keywords, coarse_docs)
        if hop_queries:
            with self._thread_pool(MAX_WORKERS, "mem-search-1421-hop") as executor:
                futures = []
                for q_text in hop_queries:
                    futures.append(executor.submit(_search, speaker_1_user_id, q_text, followup_limit))
                    futures.append(executor.submit(_search, speaker_2_user_id, q_text, followup_limit))
                for future in as_completed(futures):
                    uid, mems, duration = future.result()
                    search_time_by_user[uid] = search_time_by_user.get(uid, 0.0) + float(duration or 0.0)
                    for mem in mems:
                        if uid == speaker_1_user_id:
                            _add_to_map(a_map, mem)
                        else:
                            _add_to_map(b_map, mem)

        global_pool = list(a_map.values()) + list(b_map.values())
        global_pool_sorted = sorted(global_pool, key=lambda x: float(x.get("score", 0.0)), reverse=True)

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

        def _recency_score(ts, now_ts_value):
            if ts <= 0 or now_ts_value <= 0:
                return 0.0
            delta_days = max(0.0, (now_ts_value - ts) / 86400.0)
            return math.exp(-math.log(2.0) * (delta_days / max(1e-6, half_life_days)))

        reranker = self._ensure_reranker_model()
        now_ts = datetime.now(tz=timezone.utc).timestamp()
        subjects = intent.get("subjects") or []
        lex_question = _normalize_for_lex(question_for_keywords)

        def _classify_query(q):
            toks = _safe_lower_tokens(q)
            has_id_like = any(re.search(r"[A-Za-z]*\d", t) for t in toks)
            nav_like = len(toks) <= 4 and (has_id_like or any(len(t) > 5 for t in toks))
            structured = any(t in {"v1", "v2", "online", "offline"} for t in toks) or intent.get("has_year") or intent.get("has_month")
            domain_terms = {t for t in toks if (len(t) > 3 and (t[0].isupper() or any(ch.isdigit() for ch in t) or "_" in t))}
            return {
                "nav": nav_like,
                "structured": structured,
                "domain_terms": domain_terms,
                "id_tokens": {t for t in toks if any(ch.isdigit() for ch in t)},
            }

        q_profile = _classify_query(question)

        def _estimate_confidence(sorted_items):
            if not sorted_items:
                return 0.0
            top = float(sorted_items[0].get("final_score", 0.0))
            second = float(sorted_items[1].get("final_score", top - 0.01)) if len(sorted_items) > 1 else (top - 0.01)
            margin = max(0.0, top - second)
            denom = max(abs(top), abs(second), 1.0)
            margin_conf = min(1.0, margin / denom)
            hit_bonus = 0.05 if question_markers else 0.0
            return max(0.0, min(1.0, margin_conf + hit_bonus))

        def _pattern_delta(mem_text, base_rr, base_sem, base_lex):
            delta = 0.0
            toks = set(_safe_lower_tokens(mem_text))
            # Field imbalance: 强词匹配但语义弱 → 轻扣；语义强且词少 → 轻抬
            if base_lex > 2.5 and base_rr < 0.15:
                delta -= 0.22
            elif base_rr > 0.6 and base_lex < 0.6:
                delta += 0.18
            # Domain term coverage
            domain_terms = q_profile.get("domain_terms") or set()
            if domain_terms:
                overlap = len(domain_terms & toks) / float(len(domain_terms))
                if overlap > 0:
                    delta += 0.25 * overlap
                else:
                    delta -= 0.12
            # Entity/id consistency
            id_tokens = q_profile.get("id_tokens") or set()
            if id_tokens and not (id_tokens & toks):
                delta -= 0.18
            # Structured/time alignment
            if q_profile.get("structured") or intent.get("is_time"):
                delta += 0.1 * self._time_bonus(question, mem_text)
            # Subject/action soft alignment
            delta += 0.08 * self._subject_bonus(mem_text, subjects)
            delta += 0.08 * self._action_bonus(question, mem_text)
            return delta

        def _score_after_rerank(cands):
            if not cands:
                return [], {}
            base_scores = [float(m.get("score", 0.0)) for m in cands]
            rerank_scores = self._ce_predict_cached(question, cands)
            normalized = None
            if rerank_scores is not None:
                normalized = _normalize_scores([float(x) for x in rerank_scores])
            if normalized is None:
                normalized = _normalize_scores(base_scores)
            if normalized is None:
                normalized = base_scores

            base_final_scores = {}
            for idx, mem in enumerate(cands):
                mem_text = mem.get("memory", "")
                rr = normalized[idx] if idx < len(normalized) else 0.0
                ts = _extract_timestamp_seconds(mem)
                rec = _recency_score(ts, now_ts)

                mem_markers = _extract_time_number_markers(mem_text)
                overlap = question_markers & mem_markers
                marker_boost = 0.24 * min(1.0, len(overlap) / max(1, len(question_markers))) if question_markers and overlap else 0.0
                noise_penalty = -0.18 if self._is_noise_text(mem_text) and not _has_structured_tokens(mem_markers) else 0.0

                base_val = rerank_weight * rr + recency_weight * rec + marker_boost + noise_penalty
                base_val += 0.11 * self._subject_bonus(mem_text, subjects)
                base_val += 0.13 * self._action_bonus(question, mem_text)
                base_val += 0.18 * self._time_bonus(question, mem_text)
                base_val += 0.07 * math.log1p(max(0.0, self._lexical_score(lex_question, _normalize_for_lex(mem_text))))

                base_final_scores[id(mem)] = base_val
                mem["rerank_score"] = rr
                mem["final_score"] = base_val

            base_sorted = sorted(cands, key=lambda x: x.get("final_score", 0.0), reverse=True)
            base_conf = _estimate_confidence(base_sorted)

            # 高置信度完全信任基础排序
            if base_conf >= CONF_HIGH:
                return base_sorted, {id(m): m.get("final_score", 0.0) for m in base_sorted}

            # 低置信度时的软调权
            lambda_eff = LAMBDA_BASE * max(0.0, 1.0 - base_conf)
            adjusted_scores = {}
            for mem in base_sorted:
                mem_text = mem.get("memory", "")
                rr = mem.get("rerank_score", 0.0)
                sem = mem.get("score", 0.0)
                lex = self._lexical_score(lex_question, _normalize_for_lex(mem_text))
                delta = _pattern_delta(mem_text, rr, sem, lex)
                delta = max(-DELTA_CLIP, min(DELTA_CLIP, delta))
                adjusted = mem.get("final_score", 0.0) + lambda_eff * delta
                mem["final_score_1421"] = adjusted
                adjusted_scores[id(mem)] = adjusted

            sorted_adjusted = sorted(base_sorted, key=lambda x: x.get("final_score_1421", x.get("final_score", 0.0)), reverse=True)

            # 避免跷跷板：保护前 PROTECT_TOP 来自基础排序
            if PROTECT_TOP > 0 and base_sorted:
                locked = base_sorted[:PROTECT_TOP]
                locked_ids = {id(m) for m in locked}
                remainder = [m for m in sorted_adjusted if id(m) not in locked_ids]
                sorted_adjusted = locked + remainder
                for m in locked:
                    adjusted_scores[id(m)] = m.get("final_score", adjusted_scores.get(id(m), 0.0))

            return sorted_adjusted, adjusted_scores

        def _normalize_scores(values):
            if not values:
                return None
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std = math.sqrt(variance)
            if std < 1e-6:
                return [0.0 for _ in values]
            return [(v - mean) / std for v in values]

        def _jaccard_filtered(a_text, b_text):
            a = set(_filtered_tokens(a_text))
            b = set(_filtered_tokens(b_text))
            if not a or not b:
                return 0.0
            return len(a & b) / float(len(a | b))

        def _mmr_select(candidates, score_map, k, lambda_value):
            selected = []
            selected_texts = []
            remaining = list(candidates)
            while remaining and len(selected) < k:
                best = None
                best_val = -1e9
                for item in remaining:
                    mem_text = item.get("memory", "")
                    rel = score_map.get(id(item), 0.0)
                    if not selected_texts:
                        div_penalty = 0.0
                    else:
                        div_penalty = max(_jaccard_filtered(mem_text, prev) for prev in selected_texts)
                    # 基于置信度动态调节 lambda：低置信度 → 更多相关性
                    eff_lambda = lambda_value
                    val = eff_lambda * rel - (1.0 - eff_lambda) * div_penalty
                    if val > best_val:
                        best_val = val
                        best = item
                if best is None:
                    break
                selected.append(best)
                selected_texts.append(best.get("memory", ""))
                remaining.remove(best)
            return selected

        a_candidates, b_candidates = list(a_map.values()), list(b_map.values())
        rerank_executor = self._get_rerank_executor()
        future_a = rerank_executor.submit(_score_after_rerank, a_candidates)
        future_b = rerank_executor.submit(_score_after_rerank, b_candidates)
        a_sorted, a_scores = future_a.result()
        b_sorted, b_scores = future_b.result()

        a_conf = _estimate_confidence(a_sorted)
        b_conf = _estimate_confidence(b_sorted)
        lambda_a = lambda_div + 0.05 * (CONF_LOW - a_conf) if a_conf < CONF_LOW else lambda_div
        lambda_b = lambda_div + 0.05 * (CONF_LOW - b_conf) if b_conf < CONF_LOW else lambda_div
        lambda_a = max(0.6, min(0.9, lambda_a))
        lambda_b = max(0.6, min(0.9, lambda_b))

        a_top = _mmr_select(a_sorted, a_scores, k=top_k, lambda_value=lambda_a) if a_sorted else []
        b_top = _mmr_select(b_sorted, b_scores, k=top_k, lambda_value=lambda_b) if b_sorted else []

        search_1_memory = [self._format_memory_line(m) for m in a_top]
        search_2_memory = [self._format_memory_line(m) for m in b_top]
        speaker_1_time = search_time_by_user.get(speaker_1_user_id, 0.0)
        speaker_2_time = search_time_by_user.get(speaker_2_user_id, 0.0)
        return search_1_memory, search_2_memory, speaker_1_time, speaker_2_time

    def _search_1420(self, speaker_1_user_id, speaker_2_user_id, question, top_k):
        MAX_KEYWORDS = 8
        MAX_FALLBACK_KEYWORDS = 8
        MAX_KEYWORD_TOTAL = 10
        PRF_K = 25
        PRF_KEYWORDS = 8
        MAX_MULTIHOP_QUERIES = 3
        NEG_PENALTY = 0.55
        STRUCTURED_MARKER_PREFIXES = ("NUM:", "YEAR:", "MONTH:", "YM:")
        MAX_WORKERS = min(4, self._max_parallelism_cap)

        intent = self._intent_148(question)
        if intent.get("is_time") and (intent.get("has_year") or intent.get("has_month")):
            return self._search_147(speaker_1_user_id, speaker_2_user_id, question, top_k)
        if intent.get("is_time") and not intent.get("want_recent"):
            return self._search_146(speaker_1_user_id, speaker_2_user_id, question, top_k)

        neg_terms, sanitized_question, neg_infos = self._extract_negative_terms(question)
        neg_terms = {t for t in neg_terms if t}
        neg_infos = neg_infos or []
        question_for_keywords = sanitized_question or question

        if intent.get("want_recent"):
            recency_weight = 0.35
            half_life_days = 3.0
        elif intent.get("is_time"):
            recency_weight = 0.25
            half_life_days = 5.0
        else:
            recency_weight = 0.18
            half_life_days = 10.0
        rerank_weight = 0.65
        lambda_div = 0.82 if intent.get("is_time") else 0.72

        base_limit = max(self.top_k, 40)
        keyword_limit = min(80, max(self.top_k * 2, base_limit))
        phrase_limit = min(100, max(self.top_k * 3, keyword_limit))
        followup_limit = min(120, max(self.top_k * 4, phrase_limit))

        def _safe_lower_tokens(text):
            try:
                return re.findall(r"\b\w+\b", (text or "").lower())
            except Exception:
                return []

        def _filtered_tokens(text):
            toks = _safe_lower_tokens(text)
            if not neg_terms:
                return toks
            return [t for t in toks if t not in neg_terms]

        def _normalize_for_lex(text):
            filtered = _filtered_tokens(text)
            if filtered:
                return " ".join(filtered)
            return (text or "").strip()

        def _extract_time_number_markers(text):
            markers = set()
            if not text:
                return markers
            lower_text = text.lower()

            def _add_num(value):
                try:
                    num = int(value)
                except (TypeError, ValueError):
                    return
                markers.add(f"NUM:{num}")

            def _add_year(value):
                try:
                    year = int(value)
                except (TypeError, ValueError):
                    return
                markers.add(f"YEAR:{year}")
                _add_num(year)

            def _add_month(value):
                try:
                    month = int(value)
                except (TypeError, ValueError):
                    return
                if 1 <= month <= 12:
                    markers.add(f"MONTH:{month:02d}")
                    _add_num(month)

            def _add_year_month(year, month):
                try:
                    y = int(year)
                    m = int(month)
                except (TypeError, ValueError):
                    return
                if 1 <= m <= 12:
                    markers.add(f"YM:{y}-{m:02d}")
                    _add_year(y)
                    _add_month(m)

            for raw in re.findall(r"\b\d{1,4}\b", text):
                _add_num(raw)
                if len(raw) == 4 and raw.startswith("20"):
                    _add_year(raw)
            for match in re.finditer(r"\b(20\d{2})[-/.](\d{1,2})(?:[-/.]\d{1,2})?\b", text):
                _add_year_month(match.group(1), match.group(2))
            for match in re.finditer(r"\b(\d{1,2})[/-](20\d{2})\b", text):
                _add_year_month(match.group(2), match.group(1))
            for match in re.finditer(r"(20\d{2})年(\d{1,2})月", text):
                _add_year_month(match.group(1), match.group(2))
            for match in re.finditer(r"(\d{1,2})月(20\d{2})年?", text):
                _add_year_month(match.group(2), match.group(1))
            for match in re.finditer(r"(20\d{2})年", text):
                _add_year(match.group(1))
            for match in re.finditer(r"(\d{1,2})月", text):
                _add_month(match.group(1))
            for alias, month_idx in {
                "jan": 1,
                "january": 1,
                "feb": 2,
                "february": 2,
                "mar": 3,
                "march": 3,
                "apr": 4,
                "april": 4,
                "may": 5,
                "jun": 6,
                "june": 6,
                "jul": 7,
                "july": 7,
                "aug": 8,
                "august": 8,
                "sep": 9,
                "sept": 9,
                "september": 9,
                "oct": 10,
                "october": 10,
                "nov": 11,
                "november": 11,
                "dec": 12,
                "december": 12,
            }.items():
                pattern = r"\b" + re.escape(alias) + r"\b"
                if re.search(pattern, lower_text):
                    _add_month(month_idx)
                combo_pattern = pattern + r"\s+(20\d{2})\b"
                for year in re.findall(combo_pattern, lower_text):
                    _add_year_month(year, month_idx)
            for match in re.finditer(
                r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
                lower_text,
            ):
                token = match.group(1)
                base = token[:3]
                month_idx = 9 if base == "sep" else {
                    "jan": 1,
                    "feb": 2,
                    "mar": 3,
                    "apr": 4,
                    "may": 5,
                    "jun": 6,
                    "jul": 7,
                    "aug": 8,
                    "oct": 10,
                    "nov": 11,
                    "dec": 12,
                }.get(base, _MONTHS.get(token, None))
                if month_idx:
                    _add_month(month_idx)
            for match in re.finditer(
                r"\b(20\d{2})\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
                lower_text,
            ):
                year = match.group(1)
                token = match.group(2)
                base = token[:3]
                month_idx = 9 if base == "sep" else {
                    "jan": 1,
                    "feb": 2,
                    "mar": 3,
                    "apr": 4,
                    "may": 5,
                    "jun": 6,
                    "jul": 7,
                    "aug": 8,
                    "oct": 10,
                    "nov": 11,
                    "dec": 12,
                }.get(base, _MONTHS.get(token, None))
                if month_idx:
                    _add_year_month(year, month_idx)
            return markers

        def _has_structured_tokens(markers):
            return any(token.startswith(STRUCTURED_MARKER_PREFIXES) for token in markers)

        question_markers = _extract_time_number_markers(question)

        def _add_to_map(dst_map, item):
            key = item.get("memory")
            if not key:
                return
            existing = dst_map.get(key)
            if existing is None or float(item.get("score", 0.0)) > float(existing.get("score", 0.0)):
                dst_map[key] = item

        def _search(uid, q, limit):
            memories, _, duration = self.search_memory(uid, q, limit=limit)
            return uid, memories, duration

        kw_model = self._ensure_keybert_model()
        keybert_terms = []
        if kw_model is not None:
            try:
                extracted = kw_model.extract_keywords(
                    question_for_keywords,
                    keyphrase_ngram_range=(1, 3),
                    stop_words="english",
                    top_n=MAX_KEYWORDS,
                    use_mmr=True,
                    diversity=0.7,
                )
                keybert_terms = [kw[0] for kw in extracted]
            except Exception as exc:
                print(f"⚠️ KeyBERT extraction failed: {exc}.")

        stop_words = {
            "what",
            "when",
            "where",
            "who",
            "why",
            "how",
            "is",
            "are",
            "was",
            "were",
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
        }
        tokens = [w for w in re.findall(r"\b\w+\b", question_for_keywords.lower()) if len(w) > 2 and w not in stop_words]
        fallback_terms = tokens[:MAX_FALLBACK_KEYWORDS]

        merged_keywords = []
        seen_kw = set()
        for term in fallback_terms + keybert_terms:
            normalized = (term or "").strip()
            if not normalized:
                continue
            low = normalized.lower()
            if low in seen_kw or low in neg_terms:
                continue
            seen_kw.add(low)
            merged_keywords.append(normalized)
            if len(merged_keywords) >= MAX_KEYWORD_TOTAL:
                break

        def _generate_phrase_queries():
            phrases = []
            base_text = question_for_keywords or ""
            lowered_question = base_text.lower()
            filtered_tokens = _filtered_tokens(base_text)
            nouns_of_interest = {
                "activities",
                "activity",
                "things",
                "ideas",
                "plans",
                "places",
                "options",
                "spots",
                "games",
                "events",
                "restaurants",
                "meals",
                "hobbies",
                "recommendations",
                "destinations",
            }
            qualifier_terms = []
            for idx in range(len(filtered_tokens) - 1):
                nxt = filtered_tokens[idx + 1]
                if nxt in nouns_of_interest:
                    token = filtered_tokens[idx]
                    if token not in neg_terms and len(token) > 2:
                        qualifier_terms.append(token)
            qualifier_terms = list(dict.fromkeys(qualifier_terms))

            relation_vocab = [
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
                "coworker",
                "coworkers",
                "colleague",
                "colleagues",
                "roommate",
                "roommates",
            ]
            found_relations = [rel for rel in relation_vocab if rel in lowered_question]

            for n in (2, 3):
                for i in range(len(filtered_tokens) - n + 1):
                    span = filtered_tokens[i : i + n]
                    phrase = " ".join(span)
                    if phrase:
                        phrases.append(phrase)

            for qual in qualifier_terms[:6]:
                for noun in ("activities", "things", "ideas"):
                    phrases.append(f"{qual} {noun}")

            for rel in found_relations[:5]:
                phrases.append(f"with my {rel}")
                phrases.append(f"{rel} activities")
                for qual in qualifier_terms[:4] or ["fun", "best"]:
                    phrases.append(f"{qual} things to do with my {rel}")
                    phrases.append(f"{qual} activities with {rel}")

            prep_patterns = [
                r"\bwith\s+(?:my|your|his|her|our|their|the)?\s*[A-Za-z][A-Za-z\s]{1,40}",
                r"\bfor\s+(?:my|your|his|her|our|their|the)?\s*[A-Za-z][A-Za-z\s]{1,40}",
            ]
            for pat in prep_patterns:
                for match in re.finditer(pat, base_text, flags=re.IGNORECASE):
                    phrase = re.sub(r"\s+", " ", match.group(0).strip())
                    if phrase:
                        phrases.append(phrase)

            for qual in qualifier_terms[:4]:
                for kw in merged_keywords[:3]:
                    if qual and kw and qual not in kw.lower():
                        phrases.append(f"{qual} {kw}")

            return phrases

        queries = []
        seen_queries = set()

        def _add_query(text, limit):
            norm = re.sub(r"\s+", " ", (text or "").strip())
            if not norm:
                return
            key = norm.lower()
            if key in seen_queries:
                return
            seen_queries.add(key)
            queries.append({"text": norm, "limit": limit})

        _add_query(question, base_limit)
        if sanitized_question and sanitized_question.lower() != question.lower():
            _add_query(sanitized_question, base_limit)

        for info in neg_infos:
            phrase = info.get("phrase")
            target = info.get("target")
            prefix = info.get("prefix") or ""
            suffix = info.get("suffix") or ""
            positive_focus = prefix.strip() or sanitized_question
            if prefix:
                _add_query(prefix, base_limit)
            if suffix:
                _add_query(suffix, base_limit)
            if phrase:
                _add_query(phrase, keyword_limit)
                if positive_focus:
                    _add_query(f"{positive_focus} {phrase}", phrase_limit)
            if target:
                cleaned_target = re.sub(r"\s+", " ", target).strip()
                if cleaned_target and positive_focus:
                    _add_query(f"{positive_focus} {cleaned_target}", phrase_limit)
            if sanitized_question and phrase:
                _add_query(f"{sanitized_question} {phrase}", phrase_limit)
            if sanitized_question and target:
                cleaned_target = re.sub(r"\s+", " ", target or "").strip()
                if cleaned_target:
                    _add_query(f"{sanitized_question} {cleaned_target}", phrase_limit)

        for kw in merged_keywords:
            _add_query(kw, keyword_limit)
        for phrase in _generate_phrase_queries():
            _add_query(phrase, phrase_limit)

        a_map = {}
        b_map = {}
        search_time_by_user = {speaker_1_user_id: 0.0, speaker_2_user_id: 0.0}

        with self._thread_pool(MAX_WORKERS, "mem-search-1420-base") as executor:
            futures = []
            for payload in queries:
                q_text = payload["text"]
                limit = payload["limit"]
                futures.append(executor.submit(_search, speaker_1_user_id, q_text, limit))
                futures.append(executor.submit(_search, speaker_2_user_id, q_text, limit))
            for future in as_completed(futures):
                uid, mems, duration = future.result()
                search_time_by_user[uid] = search_time_by_user.get(uid, 0.0) + float(duration or 0.0)
                for mem in mems:
                    if uid == speaker_1_user_id:
                        _add_to_map(a_map, mem)
                    else:
                        _add_to_map(b_map, mem)

        global_pool = list(a_map.values()) + list(b_map.values())
        global_pool_sorted = sorted(global_pool, key=lambda x: float(x.get("score", 0.0)), reverse=True)
        prf_docs = global_pool_sorted[: min(PRF_K, len(global_pool_sorted))]
        prf_text = " \n".join(m.get("memory", "") for m in prf_docs)

        prf_terms = []
        if kw_model is not None and prf_text:
            try:
                extracted = kw_model.extract_keywords(
                    prf_text,
                    keyphrase_ngram_range=(1, 3),
                    stop_words="english",
                    top_n=PRF_KEYWORDS,
                    use_mmr=True,
                    diversity=0.7,
                )
                prf_terms = [kw[0] for kw in extracted]
            except Exception as exc:
                print(f"⚠️ PRF KeyBERT failed: {exc}.")
        if not prf_terms:
            prf_terms = [t for t in _filtered_tokens(prf_text) if len(t) > 2][:PRF_KEYWORDS]
        prf_terms = [term for term in prf_terms if term.lower() not in neg_terms]

        prf_queries = []
        if prf_terms:
            prf_queries.append(" ".join(prf_terms[:3]))
        if len(prf_terms) >= 4:
            prf_queries.append(" ".join(prf_terms[2:6]))

        if prf_queries:
            with self._thread_pool(MAX_WORKERS, "mem-search-1420-prf") as executor:
                futures = []
                for q_text in prf_queries:
                    futures.append(executor.submit(_search, speaker_1_user_id, q_text, phrase_limit))
                    futures.append(executor.submit(_search, speaker_2_user_id, q_text, phrase_limit))
                for future in as_completed(futures):
                    uid, mems, duration = future.result()
                    search_time_by_user[uid] = search_time_by_user.get(uid, 0.0) + float(duration or 0.0)
                    for mem in mems:
                        if uid == speaker_1_user_id:
                            _add_to_map(a_map, mem)
                        else:
                            _add_to_map(b_map, mem)

        global_pool = list(a_map.values()) + list(b_map.values())
        global_pool_sorted = sorted(global_pool, key=lambda x: float(x.get("score", 0.0)), reverse=True)

        def _multi_hop_queries(base_question, docs):
            if not docs or not getattr(self, "search_client", None):
                return []
            doc_lines = "\n".join(f"- {m.get('memory', '')}" for m in docs[:10])
            constraint_line = ""
            if neg_terms:
                constraint_line = f"\nAvoid asking about: {', '.join(sorted(neg_terms))}."
            prompt = f"""You are a retrieval planner that proposes follow-up search queries.
User question:
{base_question}

Currently retrieved memories:
{doc_lines}

Suggest up to {MAX_MULTIHOP_QUERIES} additional short search queries that could retrieve missing complementary facts.{constraint_line}
Format:
Q1: ...
Q2: ...
Q3: ..."""
            try:
                resp = self.safe_chat(
                    model=self.llm_model,
                    messages=[{"role": "system", "content": prompt}],
                    temperature=0.2,
                )
            except Exception as exc:
                print(f"⚠️ Multi-hop planning failed: {exc}")
                return []
            text = resp.choices[0].message.content if resp.choices and resp.choices[0].message else ""
            if not text:
                return []
            queries = []
            for line in text.splitlines():
                match = re.search(r"Q\d+:\s*(.+)", line.strip())
                if not match:
                    continue
                candidate = match.group(1).strip()
                if not candidate:
                    continue
                low = candidate.lower()
                if neg_terms and any(term in low for term in neg_terms):
                    continue
                if candidate in queries:
                    continue
                queries.append(candidate)
                if len(queries) >= MAX_MULTIHOP_QUERIES:
                    break
            return queries

        coarse_docs = global_pool_sorted[:20]
        hop_queries = _multi_hop_queries(question_for_keywords, coarse_docs)
        if hop_queries:
            with self._thread_pool(MAX_WORKERS, "mem-search-1420-hop") as executor:
                futures = []
                for q_text in hop_queries:
                    futures.append(executor.submit(_search, speaker_1_user_id, q_text, followup_limit))
                    futures.append(executor.submit(_search, speaker_2_user_id, q_text, followup_limit))
                for future in as_completed(futures):
                    uid, mems, duration = future.result()
                    search_time_by_user[uid] = search_time_by_user.get(uid, 0.0) + float(duration or 0.0)
                    for mem in mems:
                        if uid == speaker_1_user_id:
                            _add_to_map(a_map, mem)
                        else:
                            _add_to_map(b_map, mem)

        reranker = self._ensure_reranker_model()
        now_ts = datetime.now(tz=timezone.utc).timestamp()
        lex_question = _normalize_for_lex(question_for_keywords)
        subjects = intent.get("subjects") or []

        def _recency_score(ts, now_ts_value):
            if ts <= 0 or now_ts_value <= 0:
                return 0.0
            delta_days = max(0.0, (now_ts_value - ts) / 86400.0)
            return math.exp(-math.log(2.0) * (delta_days / max(1e-6, half_life_days)))

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

        def _lexical_without_neg(text):
            return self._lexical_score(lex_question, _normalize_for_lex(text))

        def _normalize_scores(values):
            if not values:
                return None
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std = math.sqrt(variance)
            if std < 1e-6:
                return [0.0 for _ in values]
            return [(v - mean) / std for v in values]

        def _score_after_rerank(cands):
            if not cands:
                return [], {}
            base_scores = [float(m.get("score", 0.0)) for m in cands]
            rerank_scores = self._ce_predict_cached(question, cands)
            normalized = None
            if rerank_scores is not None:
                normalized = _normalize_scores([float(x) for x in rerank_scores])
            if normalized is None:
                normalized = _normalize_scores(base_scores)
            if normalized is None:
                normalized = base_scores

            final_scores = {}
            for idx, mem in enumerate(cands):
                mem_text = mem.get("memory", "")
                rr = normalized[idx] if idx < len(normalized) else 0.0
                ts = _extract_timestamp_seconds(mem)
                rec = _recency_score(ts, now_ts)
                final_val = rerank_weight * rr + recency_weight * rec

                mem_markers = _extract_time_number_markers(mem_text)
                overlap = question_markers & mem_markers
                if question_markers and overlap:
                    frac = len(overlap) / max(1, len(question_markers))
                    final_val += 0.25 * min(1.0, frac + 0.3)
                if self._is_noise_text(mem_text):
                    penalty = -0.25
                    has_structured = _has_structured_tokens(mem_markers) or bool(self._extract_candidate_names(mem_text))
                    if has_structured:
                        penalty = penalty / 2.0
                    final_val += penalty

                if neg_terms:
                    mem_tokens = set(_safe_lower_tokens(mem_text))
                    if mem_tokens & neg_terms:
                        final_val -= NEG_PENALTY

                final_val += 0.1 * self._subject_bonus(mem_text, subjects)
                final_val += 0.12 * self._action_bonus(question, mem_text)
                time_hit = self._time_bonus(question, mem_text)
                final_val += 0.25 * time_hit
                final_val += 0.08 * math.log1p(max(0.0, _lexical_without_neg(mem_text)))

                mem_id = id(mem)
                final_scores[mem_id] = final_val
                mem["final_score"] = final_val

            sorted_items = sorted(cands, key=lambda x: x.get("final_score", 0.0), reverse=True)
            return sorted_items, final_scores

        def _jaccard_filtered(a_text, b_text):
            a = set(_filtered_tokens(a_text))
            b = set(_filtered_tokens(b_text))
            if not a or not b:
                return 0.0
            return len(a & b) / float(len(a | b))

        def _mmr_select(candidates, score_map, k, lambda_value):
            selected = []
            selected_texts = []
            remaining = list(candidates)
            while remaining and len(selected) < k:
                best = None
                best_val = -1e9
                for item in remaining:
                    mem_text = item.get("memory", "")
                    rel = score_map.get(id(item), 0.0)
                    if not selected_texts:
                        div_penalty = 0.0
                    else:
                        div_penalty = max(_jaccard_filtered(mem_text, prev) for prev in selected_texts)
                    val = lambda_value * rel - (1.0 - lambda_value) * div_penalty
                    if val > best_val:
                        best_val = val
                        best = item
                if best is None:
                    break
                selected.append(best)
                selected_texts.append(best.get("memory", ""))
                remaining.remove(best)
            return selected

        a_candidates, b_candidates = list(a_map.values()), list(b_map.values())
        rerank_executor = self._get_rerank_executor()
        future_a = rerank_executor.submit(_score_after_rerank, a_candidates)
        future_b = rerank_executor.submit(_score_after_rerank, b_candidates)
        a_sorted, a_scores = future_a.result()
        b_sorted, b_scores = future_b.result()

        a_top = _mmr_select(a_sorted, a_scores, k=top_k, lambda_value=lambda_div) if a_sorted else []
        b_top = _mmr_select(b_sorted, b_scores, k=top_k, lambda_value=lambda_div) if b_sorted else []

        search_1_memory = [self._format_memory_line(m) for m in a_top]
        search_2_memory = [self._format_memory_line(m) for m in b_top]
        speaker_1_time = search_time_by_user.get(speaker_1_user_id, 0.0)
        speaker_2_time = search_time_by_user.get(speaker_2_user_id, 0.0)
        return search_1_memory, search_2_memory, speaker_1_time, speaker_2_time

    def _search_llm0(self, speaker_1_user_id, speaker_2_user_id, question, top_k):
        """
        LLM 直排版本：直接让 LLM 在全量记忆中选择相关项。
        失败自动重试；限流时随机等待 20~40 秒，其它异常立即重试。
        """
        def _llm_choose(memories):
            if not memories:
                return []
            lines = [f"[{idx}] {self._format_memory_line(m)}" for idx, m in enumerate(memories)]
            doc_block = "\n".join(lines[:400])  # 防止提示过长
            prompt = f"""You are a precise retrieval assistant.
Question: {question}
Memories:
{doc_block}

Select up to {top_k} most relevant memory indices. Respond ONLY with indices in ascending order, separated by commas (e.g., 0,2,5)."""
            empty_hits = 0
            while True:
                try:
                    resp = self.search_client.chat.completions.create(
                        model=self.llm_model,
                        messages=[{"role": "system", "content": prompt}],
                        temperature=0.0,
                    )
                    text = (resp.choices[0].message.content or "").strip()
                    nums = re.findall(r"\d+", text)
                    seen = set()
                    idxs = []
                    for n in nums:
                        val = int(n)
                        if 0 <= val < len(memories) and val not in seen:
                            idxs.append(val)
                            seen.add(val)
                    if idxs:
                        return [memories[i] for i in idxs[:top_k]]
                    empty_hits += 1
                    if empty_hits >= 50:
                        break
                except Exception as e:
                    msg = str(e).lower()
                    if "rate limit" in msg or "limit" in msg or "overloaded" in msg or "token" in msg:
                        time.sleep(random.uniform(20, 40))
                    continue
            # fallback to lexical排序
            memories_sorted = sorted(memories, key=lambda m: self._lexical_score(question, m.get("memory", "")), reverse=True)
            return memories_sorted[:top_k]

        # 拉取全量记忆
        s1_start = time.perf_counter()
        speaker_1_memories, _, s1_time = self.search_memory(speaker_1_user_id, "get all", limit=600)
        s1_total = max(0.0, time.perf_counter() - s1_start) if s1_time is None else s1_time

        s2_start = time.perf_counter()
        speaker_2_memories, _, s2_time = self.search_memory(speaker_2_user_id, "get all", limit=600)
        s2_total = max(0.0, time.perf_counter() - s2_start) if s2_time is None else s2_time

        top1 = _llm_choose(speaker_1_memories)
        top2 = _llm_choose(speaker_2_memories)

        search_1_memory = [self._format_memory_line(m) for m in top1]
        search_2_memory = [self._format_memory_line(m) for m in top2]
        return search_1_memory, search_2_memory, s1_total, s2_total

    # ==== 方案 145：噪声感知混合检索 ====
    def _search_145(self, speaker_1_user_id, speaker_2_user_id, question, top_k):
        # 1) 先用基础检索拿候选（沿用你已有的 search_memory + 关键词/子问扩展都可）
        base_qs = []
        seen_qs = set()

        def _add_query(candidate: str):
            if not candidate:
                return
            norm = re.sub(r"\s+", " ", candidate.strip())
            if not norm:
                return
            key = norm.lower()
            if key in seen_qs:
                return
            seen_qs.add(key)
            base_qs.append(norm)

        _add_query(question)

        stripped = re.sub(r"\b(what|when|where|who|why|how)\b", "", question, flags=re.I)
        stripped = re.sub(r"\s+", " ", stripped).strip()
        _add_query(stripped)

        short_terms = []
        short_terms.extend(re.findall(r"20\d{2}", question))
        short_terms.extend(re.findall(r"\d{1,2}\s*月", question))
        short_terms.extend(re.findall(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", question, flags=re.I))
        if short_terms:
            _add_query(" ".join(short_terms[:4]))

        a_map, b_map = {}, {}
        time_map = {speaker_1_user_id: 0.0, speaker_2_user_id: 0.0}
        for q in base_qs:
            for uid in [speaker_1_user_id, speaker_2_user_id]:
                mems, _, elapsed = self.search_memory(uid, q, limit=max(60, top_k*5))
                time_map[uid] = time_map.get(uid, 0.0) + float(elapsed or 0.0)
                for m in mems:
                    # 过滤明显噪声
                    if self._is_noise_text(m["memory"]):
                        continue
                    key = m["memory"]
                    dst = a_map if uid == speaker_1_user_id else b_map
                    if key not in dst or float(m["score"]) > float(dst[key]["score"]):
                        dst[key] = m

        # 2) 为每一侧做 lexical 与可选 CrossEncoder 评分
        def score_side(q, mem_map):
            cands = list(mem_map.values())
            # lexical ranks
            lex_rank = {
                id(m): r + 1
                for r, m in enumerate(
                    sorted(cands, key=lambda x: self._lexical_score(q, x["memory"]), reverse=True)
                )
            }
            # cross-encoder (可用则启用)
            xenc = self._ensure_reranker_model()
            x_rank = {}
            if xenc is not None and cands:
                pairs = [[q, m["memory"]] for m in cands]
                try:
                    xs = xenc.predict(pairs, show_progress_bar=False)
                    order = sorted(range(len(cands)), key=lambda j: float(xs[j]), reverse=True)
                    x_rank = {id(cands[j]): idx + 1 for idx, j in enumerate(order)}
                except Exception:
                    x_rank = {}
            # embed rank 用原 score 排序
            embed_rank = {
                id(m): r + 1
                for r, m in enumerate(
                    sorted(cands, key=lambda x: float(x.get("score", 0)), reverse=True)
                )
            }
            # 组合分
            scored = []
            for m in cands:
                rid = id(m)
                r_embed = embed_rank.get(rid)
                r_lex = lex_rank.get(rid)
                fs = self._final_score_145(q, m, r_embed, r_lex, x_rank.get(rid))
                m["__final__"] = fs
                scored.append(m)
            return [self._format_memory_line(m) for m in sorted(scored, key=lambda x: x["__final__"], reverse=True)[:top_k]]

        s1 = score_side(question, a_map)
        s2 = score_side(question, b_map)
        return s1, s2, time_map.get(speaker_1_user_id, 0.0), time_map.get(speaker_2_user_id, 0.0)

    # ==== 方案 146：时间目标化检索（TIME intent 强化） ====
    def _search_146(self, speaker_1_user_id, speaker_2_user_id, question, top_k):
        # 仅时间意图题目启用强化；否则退化到 145
        if not self._question_is_time_intent(question):
            return self._search_145(speaker_1_user_id, speaker_2_user_id, question, top_k)
        # 初检
        a_map, b_map = {}, {}
        time_map = {speaker_1_user_id: 0.0, speaker_2_user_id: 0.0}
        for uid in [speaker_1_user_id, speaker_2_user_id]:
            mems, _, elapsed = self.search_memory(uid, question, limit=max(80, top_k*6))
            time_map[uid] = time_map.get(uid, 0.0) + float(elapsed or 0.0)
            for m in mems:
                # 噪声先过滤
                if self._is_noise_text(m["memory"]):
                    continue
                key = m["memory"]
                dst = a_map if uid == speaker_1_user_id else b_map
                if key not in dst or float(m["score"]) > float(dst[key]["score"]):
                    dst[key] = m

        # 时间门控：优先保留含规范化时间/月份/年份的记忆
        def time_gate(mem_map):
            cands = list(mem_map.values())
            gated = []
            for m in cands:
                y,mn,dd = self._extract_norm_date_from_memory(m["memory"])
                if y or mn or dd:
                    gated.append(m)
            return gated if gated else cands  # 没有也不空

        def score_side(q, mem_map):
            cands = time_gate(mem_map)
            # 计算各自 rank
            embed_rank = {id(m): r+1 for r,m in enumerate(sorted(cands, key=lambda x: float(x.get("score",0)), reverse=True))}
            lex_rank = {id(m): r+1 for r,m in enumerate(sorted(cands, key=lambda x: self._lexical_score(q, x["memory"]), reverse=True))}
            xenc = self._ensure_reranker_model()
            ranks_x = {}
            if xenc and cands:
                pairs = [[q, m["memory"]] for m in cands]
                try:
                    xs = xenc.predict(pairs, show_progress_bar=False)
                    order = sorted(range(len(cands)), key=lambda j: float(xs[j]), reverse=True)
                    for pos, j in enumerate(order):
                        ranks_x[id(cands[j])] = pos+1
                except Exception:
                    pass
            # 组合
            scored = []
            for m in cands:
                rid = id(m)
                fs = self._final_score_146(q, m, embed_rank.get(rid), lex_rank.get(rid), ranks_x.get(rid))
                m["__final__"] = fs
                scored.append(m)
            return [self._format_memory_line(m) for m in sorted(scored, key=lambda x: x["__final__"], reverse=True)[:top_k]]

        return (
            score_side(question, a_map),
            score_side(question, b_map),
            time_map.get(speaker_1_user_id, 0.0),
            time_map.get(speaker_2_user_id, 0.0),
        )

    # ==== 方案 147：图/时间线联合检索（轻量） ====
    def _build_timeline_index(self, all_mems):
        """
        构建: (subject, action_canonical) -> [mem_items sorted by time]
        all_mems: [{'memory': str, 'timestamp':..., 'timestamp_epoch':..., 'score':...}, ...]
        """
        from collections import defaultdict, Counter
        bucket = defaultdict(list)
        subject_counts = Counter()
        for m in all_mems:
            mem_text = m.get("memory") or ""
            names = self._extract_candidate_names(mem_text)
            if not names:
                continue
            act = self._canonical_action(mem_text)
            if not act:
                continue
            y, mn, dd = self._extract_norm_date_from_memory(mem_text)
            ts = m.get("timestamp_epoch") or 0
            for name in names:
                key_name = name.lower() if isinstance(name, str) else name
                bucket[(key_name, act)].append((ts, y, mn, dd, m))
                subject_counts[key_name] += 1
        # sort by time epoch (or keep insertion)
        for k in bucket:
            bucket[k].sort(key=lambda x: ((x[0] or 0) == 0, x[0] or 0.0))
        return bucket, subject_counts

    def _search_147(self, speaker_1_user_id, speaker_2_user_id, question, top_k):
        # 拉两边较多的记忆（一次即可，可以做缓存）
        a_all, _, a_time = self.search_memory(speaker_1_user_id, "get all", limit=500)
        b_all, _, b_time = self.search_memory(speaker_2_user_id, "get all", limit=500)
        idx_a, counts_a = self._build_timeline_index(a_all)
        idx_b, counts_b = self._build_timeline_index(b_all)
        # 解析问题主体与动作
        available_subjects = {sub for (sub, _) in idx_a.keys()} | {sub for (sub, _) in idx_b.keys()}
        subs = list(dict.fromkeys(self._extract_subjects_from_question(question)))
        combined_counts = counts_a + counts_b
        if not subs:
            cap = max(1, self._cfg_148.get("timeline_subject_cap", 12))
            subs = [name for name, _ in combined_counts.most_common(cap)]
        subs = [s for s in subs if s in available_subjects][: max(1, self._cfg_148.get("timeline_subject_cap", 12))]
        act = self._canonical_action(question)
        if not act:
            # 回退 146（若时间题）或 145
            if self._question_is_time_intent(question):
                return self._search_146(speaker_1_user_id, speaker_2_user_id, question, top_k)
            return self._search_145(speaker_1_user_id, speaker_2_user_id, question, top_k)
        if not subs:
            if self._question_is_time_intent(question):
                return self._search_146(speaker_1_user_id, speaker_2_user_id, question, top_k)
            return self._search_145(speaker_1_user_id, speaker_2_user_id, question, top_k)

        def pick_from(idx):
            cands = []
            for s in subs:
                cands.extend(idx.get((s, act), []))
            cands = [m[-1] for m in cands]  # strip meta
            if not cands:
                return []
            # 进一步用 cross-encoder 验证 + 轻量词匹配
            xenc = self._ensure_reranker_model()
            xs = None
            if xenc:
                pairs = [[question, m["memory"]] for m in cands]
                try:
                    xs = xenc.predict(pairs, show_progress_bar=False)
                except Exception:
                    xs = None
            scored = []
            for i, m in enumerate(cands):
                rr = float(xs[i]) if xs is not None else 0.0
                lx = self._lexical_score(question, m["memory"])
                tb = self._time_bonus(question, m["memory"])
                fs = 0.6*rr + 0.3*lx + 0.4*tb  # 时间题强权重
                m["__final__"] = fs
                scored.append(m)
            return [self._format_memory_line(m) for m in sorted(scored, key=lambda x: x["__final__"], reverse=True)[:top_k]]

        return pick_from(idx_a), pick_from(idx_b), float(a_time or 0.0), float(b_time or 0.0)

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

     # ==== 方案 1422: 可靠性增强版混合检索 ====
    def _search_1422(self, speaker_1_user_id, speaker_2_user_id, question, top_k):
        """
        14.22 可靠性增强版 (Reliable Hybrid Search)
        核心逻辑：
        1. 强制混合召回 (Mandatory Hybrid Recall): 即使向量检索置信度高，也强制并入 lexical search 结果。
        2. 意图感知门控 (Intent-Aware Gating): 仅对宽泛意图启用快速返回，对时间/数字/实体类问题强制进入精细重排。
        3. 硬约束校验 (Hard Constraints): 对主体错位（如 Jon vs Gina）进行强惩罚。
        """
        # 配置参数
        CONF_HIGH = 0.80  # 提高置信度阈值（原0.64），更加保守
        MANDATORY_LEXICAL_RESCUE = 5  # 强制召回的 Lexical Top-K
        INITIAL_RECALL_LIMIT = max(80, top_k * 8)  # 扩大初始召回池，为 Lexical 留出空间
        RERANK_WEIGHT = 0.65
        LEXICAL_WEIGHT = 0.15
        SUBJECT_MISMATCH_PENALTY = 0.5  # 主体不匹配时的惩罚系数
        
        intent = self._intent_148(question)
        
        # 判断是否为“硬事实”查询：时间、数字、或有明确主体的查询
        is_hard_fact_query = (
            intent.get("is_time") 
            or intent.get("is_numeric") 
            or (intent.get("subjects") and len(intent.get("subjects")) > 0)
        )
        
        neg_terms, sanitized_question, _ = self._extract_negative_terms(question)
        q_for_search = sanitized_question if sanitized_question else question

        # 提取关键词
        keywords = []
        kw_model = self._ensure_keybert_model()
        if kw_model:
            try:
                extracted = kw_model.extract_keywords(
                    q_for_search, 
                    keyphrase_ngram_range=(1, 2), 
                    stop_words='english', 
                    top_n=4
                )
                keywords = [kw[0] for kw in extracted]
            except Exception:
                pass
        
        if not keywords:
             keywords = [w for w in re.findall(r"\b\w+\b", q_for_search.lower()) if len(w) > 3][:4]

        search_queries = [question] + keywords[:2]
        
        # 辅助函数：执行搜索
        def _execute_search(uid, queries, limit):
            all_mems = {}
            dur = 0.0
            for q in queries:
                mems, _, t = self.search_memory(uid, q, limit=limit)
                dur += float(t or 0.0)
                for m in mems:
                    key = m["memory"]
                    if key not in all_mems or m["score"] > all_mems[key]["score"]:
                        all_mems[key] = m
            return list(all_mems.values()), dur

        # 辅助函数：处理单侧用户的记忆（召回 -> 混合 -> 重排）
        def _process_side(uid, queries):
            # 1. 广度召回
            candidates, duration = _execute_search(uid, queries, INITIAL_RECALL_LIMIT)
            if not candidates:
                return [], duration
            
            # 2. 强制 Lexical Rescue (从召回池中捞取字面匹配高的)
            # 注意：这里是在 candidates 内部捞，因为我们之前的 limit 很大
            # 如果 vector search 彻底漏掉了 lexical match，这里也救不回来，但 limit=80 通常够了
            lexical_scored = []
            for m in candidates:
                lx = self._lexical_score(question, m["memory"])
                # 时间问题特别加权 Lexical
                if intent.get("is_time"):
                    tb = self._time_bonus(question, m["memory"])
                    lx += tb * 1.5 
                m["lexical_score"] = lx
                lexical_scored.append(m)
            
            lexical_scored.sort(key=lambda x: x["lexical_score"], reverse=True)
            lexical_rescue_candidates = lexical_scored[:MANDATORY_LEXICAL_RESCUE]
            
            # 3. 向量 Top-K
            vector_candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_k*2]
            
            # 4. 合并池 (去重)
            merged_map = {id(m): m for m in vector_candidates}
            for m in lexical_rescue_candidates:
                merged_map[id(m)] = m
            final_pool = list(merged_map.values())
            
            # 5. 门控检查 (Gating)
            # 只有当: 置信度极高 AND 不是硬事实查询 AND 字面匹配也不差 时，才允许快速返回
            # 否则，强制进入重排序
            if vector_candidates:
                top1_score = vector_candidates[0]["score"]
                top2_score = vector_candidates[1]["score"] if len(vector_candidates) > 1 else 0.0
                margin = top1_score - top2_score
                # Lexical check: Top-1 vector result should have decent lexical overlap
                top1_lex = self._lexical_score(question, vector_candidates[0]["memory"])
                
                can_fast_return = (
                    margin > 0.15  # 向量区分度大
                    and top1_score > CONF_HIGH  # 绝对分数高
                    and not is_hard_fact_query  # 不是查时间/数字/实体
                    and top1_lex > 0.2  # 字面匹配不至于太离谱
                )
                
                if can_fast_return:
                    return [self._format_memory_line(m) for m in vector_candidates[:top_k]], duration

            # 6. 精细重排序 (Rerank)
            reranker = self._ensure_reranker_model()
            rerank_scores = [0.0] * len(final_pool)
            
            if reranker:
                try:
                    pairs = [[question, m["memory"]] for m in final_pool]
                    rerank_scores = reranker.predict(pairs, show_progress_bar=False)
                except Exception:
                    rerank_scores = [m["score"] for m in final_pool] # Fallback
                rerank_scores = [1 / (1 + math.exp(-float(s))) for s in rerank_scores]
            # 7. 综合打分 (Scoring with Hard Constraints)
            scored_results = []
            now_ts = datetime.now(tz=timezone.utc).timestamp()
            
            q_subjects = set([s.lower() for s in intent.get("subjects") or []])

            for idx, m in enumerate(final_pool):
                mem_text = m["memory"]
                rr = float(rerank_scores[idx]) if reranker else m["score"]
                lx = m.get("lexical_score", self._lexical_score(question, mem_text))
                
                # 基础分
                final_score = RERANK_WEIGHT * rr + LEXICAL_WEIGHT * lx
                
                # 时间衰减 (对于非历史查询)
                ts_epoch = m.get("timestamp_epoch")
                if ts_epoch:
                    recency = math.exp(-math.log(2.0) * max(0.0, (now_ts - float(ts_epoch)) / 86400.0 / 14.0))
                    final_score += 0.1 * recency

                # --- 硬约束惩罚/奖励 ---
                
                # A. 主体匹配 (Hard Subject Constraint)
                if q_subjects:
                    m_subjects = set([s.lower() for s in self._extract_candidate_names(mem_text)])
                    # 如果问题有主体，但记忆中完全没有重叠，大幅降权
                    if not (q_subjects & m_subjects):
                        # 但要小心: "Gina asked Jon..." (Gina is sub, Jon is obj)
                        # 如果记忆里出现了 query subject，哪怕只是作为宾语，也不惩罚
                        # 简单的 contains check
                        # hits = sum(1 for s in q_subjects if s in mem_text.lower())
                        m_tokens = set(re.findall(r"\b\w+\b", mem_text.lower()))
                        hits = sum(1 for s in q_subjects if s in m_tokens)
                        if hits == 0:
                            final_score *= SUBJECT_MISMATCH_PENALTY
                        else:
                            final_score += 0.1  # Bonus for hit
                
                # B. 时间匹配 (Hard Time Constraint)
                if intent.get("is_time"):
                    t_bonus = self._time_bonus(question, mem_text)
                    final_score += t_bonus * 0.4  # 强加权
                    
                # C. 否定词惩罚
                if neg_terms:
                    # 简单的 token 检查
                    toks = set(re.findall(r"\b\w+\b", mem_text.lower()))
                    if toks & neg_terms:
                        final_score -= 0.3
                
                m["_final_score_1422"] = final_score
                scored_results.append(m)
            
            scored_results.sort(key=lambda x: x["_final_score_1422"], reverse=True)
            
            # 8. MMR 多样性筛选 (防止重复)
            # 使用简单的 Jaccard MMR
            selected = []
            selected_indices = []
            
            for i, item in enumerate(scored_results):
                if len(selected) >= top_k:
                    break
                
                is_dup = False
                for sel_item in selected:
                    sim = self._lexical_score(item["memory"], sel_item["memory"]) # Reuse lexical score function roughly
                    if sim > 0.75: # High overlap
                        is_dup = True
                        break
                
                if not is_dup:
                    selected.append(item)
            
            return [self._format_memory_line(m) for m in selected], duration

        # 并行处理两方
        with self._thread_pool(2, "mem-search-1422") as executor:
            f1 = executor.submit(_process_side, speaker_1_user_id, search_queries)
            f2 = executor.submit(_process_side, speaker_2_user_id, search_queries)
            
            res1, t1 = f1.result()
            res2, t2 = f2.result()
            
        return res1, res2, t1, t2
    
    # ==== 方案 1423: HyDE Lite & Precision Enhanced ====
    def _search_1423(self, speaker_1_user_id, speaker_2_user_id, question, top_k):
        """
        14.23 HyDE Lite (Generative Expansion)
        核心改进:
        1. Generative Query Expansion: 对于"martial arts"泛词查询，LLM生成"Karate, Judo..."，解决向量召回缺失。
        2. Date Anchoring: 针对"May 3"等具体日期，生成显式日期查询。
        3. 继承 14.22 的 ranking 逻辑确保准确性。
        """
        
        # --- Step 1: 意图与基础特征提取 ---
        intent = self._intent_148(question)
        neg_terms, sanitized_question, _ = self._extract_negative_terms(question)
        q_for_search = sanitized_question if sanitized_question else question
        
        # --- Step 2: 生成式查询扩展 (HyDE Lite) ---
        # 仅使用极短的 prompt 快速获取扩展词
        expanded_terms = []
        # 若包含明确日期，添加显式日期查询
        if intent.get("is_time"):
             # 提取日期字符串，这里复用正则逻辑简单提取
             # e.g. "May 3", "2023"
             date_matches = re.findall(r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?", q_for_search, re.I)
             date_matches += re.findall(r"\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*", q_for_search, re.I)
             date_matches += re.findall(r"\d{4}", q_for_search)
             expanded_terms.extend(list(set(date_matches)))
        
        # 调用 LLM 进行关键词/实例扩展 (低延迟模式)
        # Prompt 旨在让 LLM 脑补具体的例子 (martial arts -> karate) 或同义词
        expansion_prompt = f"List 5 specific keywords, entities, short synonyms, or dates appearing in a likely answer to: '{q_for_search}'. Return comma-separated list only."
        
        try:
            # 使用极低的 temperature 和 max_tokens 限制延迟
            resp = self.safe_chat(
                model=self.llm_model,
                messages=[{"role": "system", "content": expansion_prompt}],
                temperature=0.1,
            )
            content = resp.choices[0].message.content.strip()
            # 简单的清洗
            llm_terms = [t.strip() for t in content.split(',') if t.strip()]
            expanded_terms.extend(llm_terms[:5]) # 限制数量
        except Exception as e:
            # 降级到 14.22 的 KeyBERT 提取
            kw_model = self._ensure_keybert_model()
            if kw_model:
                try:
                    extracted = kw_model.extract_keywords(q_for_search, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=3)
                    expanded_terms.extend([kw[0] for kw in extracted])
                except:
                    pass
        
        # --- Step 3: 组合查询并执行搜索 ---
        # 去重
        unique_queries = []
        seen = set()
        # 原问题最重要
        unique_queries.append(q_for_search)
        seen.add(q_for_search.lower())
        
        for t in expanded_terms:
            if t and t.lower() not in seen:
                unique_queries.append(t)
                seen.add(t.lower())
        
        # 限制查询数量防止爆炸
        final_queries = unique_queries[:5] 
        
        # --- Step 4: 复用 14.22 的优良排序逻辑，但使用扩展后的查询列表 ---
        
        MANDATORY_LEXICAL_RESCUE = 5
        INITIAL_RECALL_LIMIT = max(80, top_k * 6)
        RERANK_WEIGHT = 0.65
        LEXICAL_WEIGHT = 0.15
        SUBJECT_MISMATCH_PENALTY = 0.5

        def _process_side_1423(uid):
            all_mems = {}
            total_dur = 0.0
            
            # 并行执行所有扩展查询 (Broad Recall)
            with self._thread_pool(min(len(final_queries), 3), "mem-search-1423-sub") as sub_exec:
                futs = [sub_exec.submit(self.search_memory, uid, q, limit=INITIAL_RECALL_LIMIT) for q in final_queries]
                for f in as_completed(futs):
                    mems, _, t = f.result()
                    total_dur += float(t or 0.0)
                    for m in mems:
                        key = m["memory"]
                        # 简单的分数融合：如果多次召回，取最大分
                        if key not in all_mems or m["score"] > all_mems[key]["score"]:
                            all_mems[key] = m
            
            candidates = list(all_mems.values())
            if not candidates:
                return [], total_dur

            # Lexical Rescue (针对原问题)
            lexical_scored = []
            for m in candidates:
                lx = self._lexical_score(question, m["memory"]) # 用原问题算
                
                # Bonus: 如果包含 HyDE 扩展词，额外加分！(HyDE Hit Bonus)
                for term in expanded_terms:
                    if term.lower() in m["memory"].lower():
                        lx += 0.25 # 强关联奖励

                if intent.get("is_time"):
                    tb = self._time_bonus(question, m["memory"])
                    lx += tb * 2.0  # 时间题更强力的字面挽救
                m["lexical_score"] = lx
                lexical_scored.append(m)
            
            lexical_scored.sort(key=lambda x: x["lexical_score"], reverse=True)
            rescue_cands = lexical_scored[:MANDATORY_LEXICAL_RESCUE]
            
            # Vector Top-K
            # 保留更大候选集，减少过早裁剪导致的召回损失
            vector_cap = max(top_k * 4, 80)
            vector_cands = sorted(candidates, key=lambda x: x["score"], reverse=True)[:vector_cap]
            
            # Merge
            merged_map = {id(m): m for m in vector_cands}
            for m in rescue_cands:
                merged_map[id(m)] = m
            final_pool = list(merged_map.values())

            # Rerank
            reranker = self._ensure_reranker_model()
            rerank_scores = [0.0] * len(final_pool)
            if reranker:
                try:
                    pairs = [[question, m["memory"]] for m in final_pool]
                    rerank_scores = reranker.predict(pairs, show_progress_bar=False)
                except:
                    rerank_scores = [m["score"] for m in final_pool]

            # Scoring
            scored = []
            now_ts = datetime.now(tz=timezone.utc).timestamp()
            q_subjects = set([s.lower() for s in intent.get("subjects") or []])

            for idx, m in enumerate(final_pool):
                mem_text = m["memory"]
                rr = float(rerank_scores[idx]) if reranker else m["score"]
                lx = m.get("lexical_score", 0.0)
                
                final_score = RERANK_WEIGHT * rr + LEXICAL_WEIGHT * lx
                
                # 时间衰减
                ts_epoch = m.get("timestamp_epoch")
                if ts_epoch:
                    recency = math.exp(-math.log(2.0) * max(0.0, (now_ts - float(ts_epoch)) / 86400.0 / 14.0))
                    final_score += 0.1 * recency

                # 硬约束
                if q_subjects:
                    # 检查是否包含主体（大小写不敏感）
                    hits = sum(1 for s in q_subjects if s in mem_text.lower())
                    
                    # 特别针对 "Who" 问题：如果问题问 Who，且记忆里包含大写人名实体，给予奖励
                    # 这是一个启发式规则，解决 "Who gave money" (Aunt) 的问题
                    if "who" in question.lower() and not hits:
                        # 检查是否有大写单词（可能是人名）
                        caps = re.findall(r"\b[A-Z][a-z]+\b", mem_text)
                        if caps:
                             final_score += 0.1 # 潜在实体奖励

                    if hits == 0:
                         # 如果问题有明确主体，且记忆完全不包含，则惩罚
                         # 但如果 "Who" 问题，可能主体就是未知的，所以要小心
                        if "who" not in question.lower():
                             final_score *= SUBJECT_MISMATCH_PENALTY
                    else:
                        final_score += 0.15 # 命中奖励
                
                if intent.get("is_time"):
                    t_bonus = self._time_bonus(question, mem_text)
                    final_score += t_bonus * 0.5 # 更强时间奖励
                
                if neg_terms:
                    toks = set(re.findall(r"\b\w+\b", mem_text.lower()))
                    if toks & neg_terms:
                        final_score -= 0.3
                
                m["_final_score_1423"] = final_score
                scored.append(m)

            scored.sort(key=lambda x: x["_final_score_1423"], reverse=True)
            
            # MMR Selection (Stronger Diversity for "List" questions)
            # 如果问题看起来是列举型的 (What areas, What items, Who people)，增强多样性
            is_list_question = any(w in question.lower() for w in ["what items", "what areas", "what people", "list", "and"])
            mmr_thresh = 0.65 if is_list_question else 0.85  # 更宽松，避免过度去重

            selected = []
            leftovers = []
            for item in scored:
                is_dup = False
                for sel in selected:
                    if self._lexical_score(item["memory"], sel["memory"]) > mmr_thresh:
                        is_dup = True
                        break
                if not is_dup:
                    selected.append(item)
                else:
                    leftovers.append(item)
                if len(selected) >= top_k:
                    break

            # 如果因为多样性约束导致数量不足，回填剩余候选，确保尽量凑满 top_k
            if len(selected) < top_k:
                need = min(top_k, len(scored)) - len(selected)
                if need > 0:
                    selected.extend(leftovers[:need])
            
            return [self._format_memory_line(m) for m in selected], total_dur

        with self._thread_pool(2, "mem-search-1423-main") as executor:
            f1 = executor.submit(_process_side_1423, speaker_1_user_id)
            f2 = executor.submit(_process_side_1423, speaker_2_user_id)
            r1, t1 = f1.result()
            r2, t2 = f2.result()
        
        return r1, r2, t1, t2
        
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
           
        elif search_method in {"5", "14.9"}:
            # ========== 方法5: PRF + 关键词提取 + Reranking + MMR多样化 ==========
            # 配置参数
            safe_enhanced = search_method == "14.9"
            PRF_K = 20  # PRF使用的文档数
            MAX_KEYWORDS = 5  # 基础关键词数量
            PRF_KEYWORDS = 6  # PRF关键词数量
            MAX_WORKERS = min(3, self._max_parallelism_cap)  # 并发线程数
            LAMBDA_DIV = 0.8 if safe_enhanced else 0.6  # MMR多样性参数
            RERANK_WEIGHT = 0.8  # 重排序分数权重
            RECENCY_WEIGHT = 0.2  # 时间衰减权重
            HALF_LIFE_DAYS = 7.0  # 时间衰减半衰期（天）
            BOOST_ALPHA = 0.3 if safe_enhanced else 0.0  # 数字/日期匹配额外加分
            NOISE_PENALTY = -0.2 if safe_enhanced else 0.0  # 噪声轻扣分
            STRUCTURED_MARKER_PREFIXES = ("NUM:", "YEAR:", "MONTH:", "YM:")

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

            _MONTH_ALIAS = {
                "jan": 1,
                "january": 1,
                "feb": 2,
                "february": 2,
                "mar": 3,
                "march": 3,
                "apr": 4,
                "april": 4,
                "may": 5,
                "jun": 6,
                "june": 6,
                "jul": 7,
                "july": 7,
                "aug": 8,
                "august": 8,
                "sep": 9,
                "sept": 9,
                "september": 9,
                "oct": 10,
                "october": 10,
                "nov": 11,
                "november": 11,
                "dec": 12,
                "december": 12,
            }
            _CN_MONTH_WORDS = {
                "一月": 1,
                "二月": 2,
                "三月": 3,
                "四月": 4,
                "五月": 5,
                "六月": 6,
                "七月": 7,
                "八月": 8,
                "九月": 9,
                "十月": 10,
                "十一月": 11,
                "十二月": 12,
            }

            def _extract_time_number_markers(text):
                markers = set()
                if not text:
                    return markers
                lower_text = text.lower()

                def _add_num(value):
                    try:
                        num = int(value)
                    except (TypeError, ValueError):
                        return
                    markers.add(f"NUM:{num}")

                def _add_year(value):
                    try:
                        year = int(value)
                    except (TypeError, ValueError):
                        return
                    markers.add(f"YEAR:{year}")
                    _add_num(year)

                def _add_month(value):
                    try:
                        month = int(value)
                    except (TypeError, ValueError):
                        return
                    if 1 <= month <= 12:
                        markers.add(f"MONTH:{month}")

                def _add_year_month(year, month):
                    try:
                        y = int(year)
                        m = int(month)
                    except (TypeError, ValueError):
                        return
                    if 1 <= m <= 12:
                        markers.add(f"YM:{y}-{m:02d}")
                        _add_year(y)
                        _add_month(m)

                for raw in re.findall(r"\b\d{1,4}\b", text):
                    _add_num(raw)
                    if len(raw) == 4 and raw.startswith("20"):
                        _add_year(raw)

                for match in re.finditer(r"\b(20\d{2})[-/.](\d{1,2})(?:[-/.]\d{1,2})?\b", text):
                    _add_year_month(match.group(1), match.group(2))
                for match in re.finditer(r"\b(\d{1,2})[/-](20\d{2})\b", text):
                    _add_year_month(match.group(2), match.group(1))
                for match in re.finditer(r"(20\d{2})年(\d{1,2})月", text):
                    _add_year_month(match.group(1), match.group(2))
                for match in re.finditer(r"(\d{1,2})月(20\d{2})年?", text):
                    _add_year_month(match.group(2), match.group(1))
                for match in re.finditer(r"(20\d{2})年", text):
                    _add_year(match.group(1))
                for match in re.finditer(r"(\d{1,2})月", text):
                    _add_month(match.group(1))

                for alias, month_idx in _MONTH_ALIAS.items():
                    pattern = r"\b" + re.escape(alias) + r"\b"
                    if re.search(pattern, lower_text):
                        _add_month(month_idx)
                    combo_pattern = pattern + r"\s+(20\d{2})\b"
                    for year in re.findall(combo_pattern, lower_text):
                        _add_year_month(year, month_idx)
                for word, month_idx in _CN_MONTH_WORDS.items():
                    if word in text:
                        _add_month(month_idx)

                for match in re.finditer(
                    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
                    lower_text,
                ):
                    token = match.group(1)
                    base = token[:3]
                    if base == "sep":
                        month_idx = 9
                    else:
                        month_idx = _MONTH_ALIAS.get(base, _MONTHS.get(token, None))
                    if month_idx:
                        _add_month(month_idx)
                for match in re.finditer(
                    r"\b(20\d{2})\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
                    lower_text,
                ):
                    year = match.group(1)
                    token = match.group(2)
                    base = token[:3]
                    if base == "sep":
                        month_idx = 9
                    else:
                        month_idx = _MONTH_ALIAS.get(base, _MONTHS.get(token, None))
                    if month_idx:
                        _add_year_month(year, month_idx)

                return markers

            def _has_structured_tokens(markers):
                return any(token.startswith(STRUCTURED_MARKER_PREFIXES) for token in markers)

            question_markers = _extract_time_number_markers(question) if safe_enhanced else set()

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
                    final_val = RERANK_WEIGHT * rr + RECENCY_WEIGHT * rec
                    if safe_enhanced:
                        mem_markers = _extract_time_number_markers(mem_text)
                        if question_markers and (question_markers & mem_markers):
                            final_val += BOOST_ALPHA
                            m["marker_boost"] = BOOST_ALPHA
                        if NOISE_PENALTY < 0.0 and self._is_noise_text(mem_text):
                            penalty = NOISE_PENALTY
                            has_structured = _has_structured_tokens(mem_markers)
                            if not has_structured:
                                has_structured = bool(self._extract_candidate_names(mem_text))
                            if has_structured:
                                penalty = max(penalty, 0.0)
                            final_val += penalty
                            m["noise_penalty"] = penalty
                    final_scores[mem_text] = final_val
                    m["rerank_score"] = rr
                    m["final_score"] = final_val
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
        elif search_method == "llm0":
            s1, s2, t1, t2 = self._search_llm0(speaker_1_user_id, speaker_2_user_id, question, top_k_rerank)
            return (s1, s2, None, None, t1, t2)
        elif search_method == "14.21":
            s1, s2, t1, t2 = self._search_1421(speaker_1_user_id, speaker_2_user_id, question, top_k_rerank)
            return (s1, s2, None, None, t1, t2)
        elif search_method == "14.20":
            s1, s2, t1, t2 = self._search_1420(speaker_1_user_id, speaker_2_user_id, question, top_k_rerank)
            return (s1, s2, None, None, t1, t2)
        elif search_method == "14.10":
            s1, s2, t1, t2 = self._search_1410(speaker_1_user_id, speaker_2_user_id, question, top_k_rerank)
            return (s1, s2, None, None, t1, t2)
        elif search_method == "14.22":
            s1, s2, t1, t2 = self._search_1422(speaker_1_user_id, speaker_2_user_id, question, top_k_rerank)
            return (s1, s2, None, None, t1, t2)
        elif search_method == "14.24":
             s1, s2, t1, t2 = self._search_1423(speaker_1_user_id, speaker_2_user_id, question, top_k_rerank)
             return (s1, s2, None, None, t1, t2)
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
        
        elif search_method == "14.5":
            s1, s2, t1, t2 = self._search_145(speaker_1_user_id, speaker_2_user_id, question, self.top_k)
            return (s1, s2, None, None, t1, t2)

        elif search_method == "14.6":
            s1, s2, t1, t2 = self._search_146(speaker_1_user_id, speaker_2_user_id, question, self.top_k)
            return (s1, s2, None, None, t1, t2)

        elif search_method == "14.7":
            s1, s2, t1, t2 = self._search_147(speaker_1_user_id, speaker_2_user_id, question, self.top_k)
            return (s1, s2, None, None, t1, t2)

        elif search_method == "14.8":
            s1, s2, t1, t2 = self._search_148(speaker_1_user_id, speaker_2_user_id, question, self.top_k)
            return (s1, s2, None, None, t1, t2)

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
