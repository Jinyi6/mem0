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
        search_method="14.23",
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
            try:
                import prompts
                prompt_var_name = f"ANSWER_PROMPT_{answer_mode}" if answer_mode != "0" else "ANSWER_PROMPT"
                self.ANSWER_PROMPT = getattr(prompts, prompt_var_name, prompts.ANSWER_PROMPT)
            except ImportError:
                self.ANSWER_PROMPT = ANSWER_PROMPT
            except AttributeError:
                self.ANSWER_PROMPT = ANSWER_PROMPT

        self.speaker_1_full_memories = []
        self.speaker_2_full_memories = []

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
        normalized = "0" if value is None else str(value).strip()
        if not normalized:
            normalized = "0"
        return normalized

    @staticmethod
    @contextmanager
    def _thread_pool(max_workers: int, prefix: str):
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
        if re.search(_NOISE_QUESTION_PAT, t):
            return True
        if re.search(r"!{2,}\s*$", t):
            return True
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
        def bigrams(xs): return set(zip(xs, xs[1:])) if len(xs) > 1 else set()
        bo = len(bigrams(q_tokens) & bigrams(t_tokens))
        return overlap * 0.6 + bo * 0.9

    def _extract_negative_terms(self, text: str):
        import re

        raw = text or ""
        neg_terms = set()
        spans = []
        neg_infos = []
        relation_terms = {
            "girlfriend", "boyfriend", "partner", "wife", "husband", "friend", "friends", "family",
            "parents", "kids", "children", "son", "daughter", "brother", "sister", "coworker",
            "coworkers", "colleague", "colleagues", "roommate", "roommates", "fiance", "fiancee", "spouse",
        }
        stop_tokens = {
            "and", "or", "but", "the", "a", "an", "other", "than", "besides", "except", "rather",
            "instead", "without", "with", "of", "any", "my", "your", "his", "her", "our", "their",
            "me", "you", "him", "them", "ours", "yours", "hers", "mine", "ourselves", "yourselves",
            "myself", "yourself", "herself", "himself", "themselves",
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
        import re
        ql = (q or "").lower()
        year = None
        m = re.search(r"\b(20\d{2})\b", ql)
        if m:
            year = int(m.group(1))
        month = None
        for name, mi in _MONTHS.items():
            if name in ql:
                month = mi
                break
        return {"year": year, "month": month}

    def _extract_norm_date_from_memory(self, text: str):
        import re
        t = text or ""

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
        wanted = self._parse_month_in_question(q)
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
        s = 0.0
        for r in ranks:
            if r is not None and r > 0:
                s += 1.0 / (c + r)
        return s

    def _ce_predict_cached(self, q: str, cands: list):
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
    
    # ==== 方案 1422: 可靠性增强版混合检索 ====
    def _search_1422(self, speaker_1_user_id, speaker_2_user_id, question, top_k):
        # 配置参数
        CONF_HIGH = 0.80  # 提高置信度阈值（原0.64），更加保守
        MANDATORY_LEXICAL_RESCUE = 5  # 强制召回的 Lexical Top-K
        INITIAL_RECALL_LIMIT = max(80, top_k * 8)  # 扩大初始召回池，为 Lexical 留出空间
        RERANK_WEIGHT = 0.65
        LEXICAL_WEIGHT = 0.15
        SUBJECT_MISMATCH_PENALTY = 0.5  # 主体不匹配时的惩罚系数
        
        intent = self._intent_148(question)
        
        is_hard_fact_query = (
            intent.get("is_time") 
            or intent.get("is_numeric") 
            or (intent.get("subjects") and len(intent.get("subjects")) > 0)
        )
        
        neg_terms, sanitized_question, _ = self._extract_negative_terms(question)
        q_for_search = sanitized_question if sanitized_question else question

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

        def _process_side(uid, queries):
            candidates, duration = _execute_search(uid, queries, INITIAL_RECALL_LIMIT)
            if not candidates:
                return [], duration
            
            lexical_scored = []
            for m in candidates:
                lx = self._lexical_score(question, m["memory"])
                if intent.get("is_time"):
                    tb = self._time_bonus(question, m["memory"])
                    lx += tb * 1.5 
                m["lexical_score"] = lx
                lexical_scored.append(m)
            
            lexical_scored.sort(key=lambda x: x["lexical_score"], reverse=True)
            lexical_rescue_candidates = lexical_scored[:MANDATORY_LEXICAL_RESCUE]
            
            vector_candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_k*2]
            
            merged_map = {id(m): m for m in vector_candidates}
            for m in lexical_rescue_candidates:
                merged_map[id(m)] = m
            final_pool = list(merged_map.values())
            
            if vector_candidates:
                top1_score = vector_candidates[0]["score"]
                top2_score = vector_candidates[1]["score"] if len(vector_candidates) > 1 else 0.0
                margin = top1_score - top2_score
                top1_lex = self._lexical_score(question, vector_candidates[0]["memory"])
                
                can_fast_return = (
                    margin > 0.15  # 向量区分度大
                    and top1_score > CONF_HIGH  # 绝对分数高
                    and not is_hard_fact_query  # 不是查时间/数字/实体
                    and top1_lex > 0.2  # 字面匹配不至于太离谱
                )
                
                if can_fast_return:
                    return [self._format_memory_line(m) for m in vector_candidates[:top_k]], duration

            reranker = self._ensure_reranker_model()
            rerank_scores = [0.0] * len(final_pool)
            
            if reranker:
                try:
                    pairs = [[question, m["memory"]] for m in final_pool]
                    rerank_scores = reranker.predict(pairs, show_progress_bar=False)
                except Exception:
                    rerank_scores = [m["score"] for m in final_pool] # Fallback
            
            scored_results = []
            now_ts = datetime.now(tz=timezone.utc).timestamp()
            
            q_subjects = set([s.lower() for s in intent.get("subjects") or []])

            for idx, m in enumerate(final_pool):
                mem_text = m["memory"]
                rr = float(rerank_scores[idx]) if reranker else m["score"]
                lx = m.get("lexical_score", self._lexical_score(question, mem_text))
                
                final_score = RERANK_WEIGHT * rr + LEXICAL_WEIGHT * lx
                
                ts_epoch = m.get("timestamp_epoch")
                if ts_epoch:
                    recency = math.exp(-math.log(2.0) * max(0.0, (now_ts - float(ts_epoch)) / 86400.0 / 14.0))
                    final_score += 0.1 * recency

                if q_subjects:
                    m_subjects = set([s.lower() for s in self._extract_candidate_names(mem_text)])
                    if not (q_subjects & m_subjects):
                        hits = sum(1 for s in q_subjects if s in mem_text.lower())
                        if hits == 0:
                            final_score *= SUBJECT_MISMATCH_PENALTY
                        else:
                            final_score += 0.1  # Bonus for hit
                
                if intent.get("is_time"):
                    t_bonus = self._time_bonus(question, mem_text)
                    final_score += t_bonus * 0.4  # 强加权
                    
                if neg_terms:
                    toks = set(re.findall(r"\b\w+\b", mem_text.lower()))
                    if toks & neg_terms:
                        final_score -= 0.3
                
                m["_final_score_1422"] = final_score
                scored_results.append(m)
            
            scored_results.sort(key=lambda x: x["_final_score_1422"], reverse=True)
            
            selected = []
            
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
            vector_cands = sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_k*2]
            
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
            mmr_thresh = 0.65 if is_list_question else 0.75

            selected = []
            for item in scored:
                if len(selected) >= top_k: break
                is_dup = False
                for sel in selected:
                    if self._lexical_score(item["memory"], sel["memory"]) > mmr_thresh:
                        is_dup = True
                        break
                if not is_dup:
                    selected.append(item)
            
            return [self._format_memory_line(m) for m in selected], total_dur

        with self._thread_pool(2, "mem-search-1423-main") as executor:
            f1 = executor.submit(_process_side_1423, speaker_1_user_id)
            f2 = executor.submit(_process_side_1423, speaker_2_user_id)
            r1, t1 = f1.result()
            r2, t2 = f2.result()
        
        return r1, r2, t1, t2

    def Search(self, speaker_1_user_id, speaker_2_user_id, question, search_method, top_k_rerank=15, pbar=None):
        top_k_rerank = self.top_k
        search_method = self._normalize_mode(search_method)

        # 路由更新：优先 14.23
        if search_method == "14.23":
             s1, s2, t1, t2 = self._search_1423(speaker_1_user_id, speaker_2_user_id, question, top_k_rerank)
             return (s1, s2, None, None, t1, t2)

        if search_method == "14.22":
            s1, s2, t1, t2 = self._search_1422(speaker_1_user_id, speaker_2_user_id, question, top_k_rerank)
            return (s1, s2, None, None, t1, t2)
            
        # Fallback
        return self._search_1423(speaker_1_user_id, speaker_2_user_id, question, top_k_rerank)


    def answer_question(self, speaker_1_user_id, speaker_2_user_id, question, answer, category, pbar=None, max_retries=51):
        # 指向 Search 统一入口
        (
            search_1_memory,
            search_2_memory,
            speaker_1_graph_memories,
            speaker_2_graph_memories,
            speaker_1_memory_time,
            speaker_2_memory_time,
        ) = self.Search(speaker_1_user_id, speaker_2_user_id, question, self.search_method, self.top_k)

        speaker_1_graph_memories = speaker_1_graph_memories or []
        speaker_2_graph_memories = speaker_2_graph_memories or []
        speaker_1_memory_time = float(speaker_1_memory_time or 0.0)
        speaker_2_memory_time = float(speaker_2_memory_time or 0.0)
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
        
        request_id = f"answer-q-{uuid.uuid4()}"
        answer_start = time.time()
        
        try:
             response = self.answer_client.chat.completions.create(
                model=self.answer_llm_model, 
                messages=[{"role": "system", "content": answer_prompt}], 
                temperature=0.0
            )
             response_content = response.choices[0].message.content
        except Exception as e:
             self.logger.error(f"Answer generation failed: {e}")
             response_content = "Error generating response."

        response_time = max(0.0, time.time() - answer_start)

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

    def _record_result(self, conversation_idx: int, result):
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
            return 1
        return min(requested, self._max_parallelism_cap)

    def search_memory(self, user_id, query, max_retries=5, pbar=None, limit=None):
        if limit is None:
            limit = self.top_k
        # 简单封装 memory.search，省略了复杂的重试日志逻辑以保持简洁，实际使用建议保留原版 robust retry
        try:
            start = time.perf_counter()
            res = self.memory.search(query, user_id=user_id, limit=limit)
            dur = time.perf_counter() - start
            return res.get("results", []), res.get("relations", None), dur
        except Exception as e:
            print(f"Search failed for {user_id}: {e}")
            return [], None, 0.0

    def process_data_file(self, file_path, max_workers=5):
        dataset_path = Path(file_path)
        stats = compute_dataset_stats(dataset_path)
        self._dataset_stats = stats
        total_questions = stats.get("total_questions", 0)
        
        resolved_workers = self._resolve_max_workers(max_workers)
        self._expected_results_per_conversation = stats.get("qa_per_conversation", [])
        self._results_buffer = {}
        self._results_writer = IncrementalResultsWriter(self.output_path)

        with tqdm(total=total_questions, desc="Processing") as pbar:
            with ThreadPoolExecutor(max_workers=resolved_workers) as executor:
                futures = []
                for conv_idx, item in enumerate(stream_normalized_dataset(dataset_path)):
                    conversation = item.get("conversation") or {}
                    speaker_a = conversation.get("speaker_a")
                    speaker_b = conversation.get("speaker_b")
                    if not speaker_a or not speaker_b:
                        continue
                    
                    # 构造 user_id
                    s_a_id = f"{speaker_a}_{conv_idx}"
                    s_b_id = f"{speaker_b}_{conv_idx}"
                    
                    qa_list = item.get("qa", [])
                    for q_item in qa_list:
                        futures.append(executor.submit(
                            self.process_question, 
                            q_item, s_a_id, s_b_id, conv_idx, pbar
                        ))
                
                for f in as_completed(futures):
                    try:
                        f.result()
                    except Exception as e:
                        print(f"Task failed: {e}")
        
        self._results_writer.finalize()

    def close(self):
        if self._rerank_executor:
            self._rerank_executor.shutdown(wait=True, cancel_futures=True)
            self._rerank_executor = None
        if getattr(self, "_io_executor", None):
            self._io_executor.shutdown(wait=True, cancel_futures=True)
            self._io_executor = None
            if hasattr(self.memory, "set_shared_executor"):
                self.memory.set_shared_executor(None)