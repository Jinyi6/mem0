import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
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





class MemorySearch:
    """
    记忆搜索类，用于处理基于对话的问答任务
    
    支持多种搜索方法：
    - 方法3: 问题分解 + 多查询搜索 + Reranking
    - 方法5: 关键词提取 + PRF扩展 + Reranking + MMR多样化
    """
    
    def __init__(self, output_path="results.json", top_k=10, filter_memories=False, is_graph=False, logger=None, qdrant_path=None, search_method=5):
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
        self.search_method = search_method
        # Create the memory object first
        self.memory = Memory.from_config(config)
        # Then, set the logger attribute on the created instance
        self.memory.logger = self.logger        # self.lock = threading.Lock()

        self.keybert_model = get_keybert_model() 
        self.reranker_model = get_reranker_model()

        if self.is_graph:
            self.ANSWER_PROMPT = ANSWER_PROMPT_GRAPH
        else:
            self.ANSWER_PROMPT = ANSWER_PROMPT

    def search_memory(self, user_id, query, max_retries=5, pbar=None):
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

    def safe_chat(self, model, messages, temperature, sleep_time=20):
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
        while True:
            try:
                return self.openai_client.chat.completions.create(
                    model=model or os.getenv("MODEL", "Qwen/Qwen3-14B"),
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

        if search_method == 3:
            # ========== 方法3: 问题分解 + 多查询搜索 + Reranking ==========
            NUM_SUB_QUESTIONS = 5  # 子问题数量
            MAX_WORKERS = 3  # 并发工作线程数
            
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
                    model=os.getenv("MODEL", "Qwen/Qwen3-14B"),
                    messages=[{"role": "system", "content": q_prompt}],
                    temperature=0.8,
            )
            if q_response.choices[0].message.content is not None:
                raw_text = q_response.choices[0].message.content.strip()
            else:
                raw_text = ""
            import re
            question_list = re.findall(r'^\s*\d+\.\s*(.+)', raw_text, flags=re.M)
            
            # 收集所有子问题的搜索结果（去重）
            a_mem_map = {}  # speaker_1 的记忆映射
            b_mem_map = {}  # speaker_2 的记忆映射
            
            def search_(uid, question):
                """辅助函数：搜索单个用户的记忆"""
                mems = self.search_memory(uid, question)
                return uid, mems[0]
            
            # 并发搜索所有子问题
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = []
                for q in question_list:
                    futures.append(executor.submit(search_, speaker_1_user_id, q))
                    futures.append(executor.submit(search_, speaker_2_user_id, q))
                
                # 合并结果，保留每条记忆的最高分数
                for f in as_completed(futures):
                    uid, mems = f.result()
                    for m in mems:
                        key = m["memory"]
                        if uid == speaker_1_user_id:
                            if key not in a_mem_map or m["score"] > a_mem_map[key]["score"]:
                                a_mem_map[key] = m
                        else:
                            if key not in b_mem_map or m["score"] > b_mem_map[key]["score"]:
                                b_mem_map[key] = m
            
            # 使用 Reranker 重新排序
            reranker = self.reranker_model
            
            if reranker is not None:
                try:
                    # Rerank for speaker A
                    if a_mem_map:
                        a_candidates = list(a_mem_map.values())
                        a_pairs = [[question, m["memory"]] for m in a_candidates]
                        a_scores = reranker.predict(a_pairs)
                        
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
                        b_scores = reranker.predict(b_pairs)
                        
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
            search_1_memory = [f"{m['timestamp']}: {m['memory']}" for m in a_top]
            search_2_memory = [f"{m['timestamp']}: {m['memory']}" for m in b_top]
            
            return search_1_memory, search_2_memory
           
        elif search_method == 5:
            # ========== 方法5: PRF + 关键词提取 + Reranking + MMR多样化 ==========
            import re
            import math
            from datetime import datetime, timezone
            
            # 配置参数
            PRF_K = 20  # PRF使用的文档数
            MAX_KEYWORDS = 5  # 基础关键词数量
            PRF_KEYWORDS = 6  # PRF关键词数量
            MAX_WORKERS = 3  # 并发线程数
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

            def _parse_timestamp(ts):
                """
                解析时间戳为epoch秒
                
                Args:
                    ts: 时间戳（可以是数字或ISO格式字符串）
                    
                Returns:
                    float: epoch秒，解析失败返回0.0
                """
                if ts is None:
                    return 0.0
                try:
                    return float(ts)
                except Exception:
                    pass
                try:
                    return datetime.fromisoformat(str(ts)).replace(tzinfo=timezone.utc).timestamp()
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
                """辅助函数：搜索单个用户的记忆"""
                mems = self.search_memory(uid, q)
                return uid, mems[0]

            # ---------- 第一阶段：基础多查询召回 ----------
            # 使用 KeyBERT 提取关键词
            keywords = []
            kw_model = self.keybert_model
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

            # 并发执行所有基础查询
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = []
                for q in base_queries:
                    futures.append(executor.submit(_search, speaker_1_user_id, q))
                    futures.append(executor.submit(_search, speaker_2_user_id, q))
                for f in as_completed(futures):
                    uid, mems = f.result()
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
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    futures = []
                    for q in prf_queries:
                        futures.append(executor.submit(_search, speaker_1_user_id, q))
                        futures.append(executor.submit(_search, speaker_2_user_id, q))
                    for f in as_completed(futures):
                        uid, mems = f.result()
                        for m in mems:
                            if uid == speaker_1_user_id:
                                _add_to_map(a_map, m)
                            else:
                                _add_to_map(b_map, m)

            # ---------- 第三阶段：Reranking + 时间加权 ----------
            reranker = self.reranker_model  # 使用全局单例
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
                        rerank_scores = reranker.predict(pairs)
                    except Exception as e:
                        print(f"⚠️ Reranking failed: {e}. Fallback to base score.")
                
                # 计算最终分数：重排序分数 + 时间衰减分数
                final_scores = {}
                for i, m in enumerate(cands):
                    mem_text = m.get("memory", "")
                    rr = float(rerank_scores[i]) if rerank_scores is not None else base_scores[i]
                    ts = _parse_timestamp(m.get("timestamp"))
                    rec = _recency_score(ts, now_ts, half_life_days=HALF_LIFE_DAYS)
                    final_scores[mem_text] = RERANK_WEIGHT * rr + RECENCY_WEIGHT * rec
                    m["rerank_score"] = rr
                    m["final_score"] = final_scores[mem_text]
                return sorted(cands, key=lambda x: x.get("final_score", 0.0), reverse=True), final_scores
                

            # 并发重排序两个说话人的记忆
            a_candidates, b_candidates = list(a_map.values()), list(b_map.values())
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_a = executor.submit(_score_after_rerank, a_candidates)
                future_b = executor.submit(_score_after_rerank, b_candidates)
                a_sorted, a_scores = future_a.result()
                b_sorted, b_scores = future_b.result()

            # ---------- 第四阶段：MMR 多样化选择 ----------
            a_top = _mmr_select(a_sorted, a_scores, k=top_k_rerank, lambda_div=LAMBDA_DIV) if a_sorted else []
            b_top = _mmr_select(b_sorted, b_scores, k=top_k_rerank, lambda_div=LAMBDA_DIV) if b_sorted else []

            # ---------- 格式化输出 ----------
            search_1_memory = [f"{m['timestamp']}: {m['memory']}" for m in a_top]
            search_2_memory = [f"{m['timestamp']}: {m['memory']}" for m in b_top]
            return search_1_memory, search_2_memory
          
        else:
            # ========== 默认搜索方法 ==========
            speaker_1_memories, speaker_1_graph_memories, speaker_1_memory_time = self.search_memory(
                speaker_1_user_id, question, pbar=pbar
            )
            speaker_2_memories, speaker_2_graph_memories, speaker_2_memory_time = self.search_memory(
                speaker_2_user_id, question, pbar=pbar
            )
            search_1_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_1_memories]
            search_2_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_2_memories]
            return search_1_memory, search_2_memory



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
        # 初始化返回变量，避免未定义错误
        speaker_1_memory_time = 0.0
        speaker_2_memory_time = 0.0
        speaker_1_graph_memories = None
        speaker_2_graph_memories = None
        response_time = 0.0
        
        # 执行搜索
        t1 = time.time()
        search_1_memory, search_2_memory = self.Search(
            speaker_1_user_id, speaker_2_user_id, question, self.search_method, pbar=pbar
        )
        t2 = time.time()
        search_1_memory_time = t2 - t1
        search_2_memory_time = t2 - t1


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
                    model=os.getenv("MODEL", "Qwen/Qwen3-14B"), 
                    messages=[{"role": "system", "content": answer_prompt}], 
                    temperature=0.0
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
        )

    def process_question(self, val, speaker_a_user_id, speaker_b_user_id, idx, pbar=None, lock=None):
        """
        处理单个问答对，包括搜索记忆和生成答案
        
        Args:
            val: 问答对数据字典
            speaker_a_user_id: 说话者A的用户ID
            speaker_b_user_id: 说话者B的用户ID
            idx: 对话索引
            pbar: 进度条对象
            lock: 线程锁（用于并发写入）
            
        Returns:
            dict: 包含问题、答案、记忆等信息的结果字典
        """
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
        """
        处理整个数据文件，并发处理所有问答对
        
        Args:
            file_path: 数据文件路径
            max_workers: 最大并发工作线程数
        """
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