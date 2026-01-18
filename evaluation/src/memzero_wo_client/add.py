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
from typing import Optional

from dotenv import load_dotenv
from tqdm import tqdm
from src.utils import compute_dataset_stats, stream_normalized_dataset
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
        
        # Initialize spacy model for entity extraction (lazy loading)
        self._nlp_model = None
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
        self._dataset_stats = None
        self.data = _StreamingDataset(self) if data_path else None
        self.is_graph = is_graph
        self.figure_view = kwargs.get("figure_view", False)
        self.fact_extraction_mode = self._normalize_mode(kwargs.get("fact_extraction_mode", "0"))
        self.memory_decision_mode = self._normalize_mode(kwargs.get("memory_decision_mode", "0"))
        self.fact_abstract_mode = self._normalize_mode(kwargs.get("fact_abstract_mode", "0"))
        self.add_mode = self._normalize_mode(kwargs.get("add_mode", "0"))
        self.long_term_profile_mode = self._normalize_mode(kwargs.get("long_term_profile_mode", "0"))
        if self.long_term_profile_mode == "0":
            self.long_term_profile_mode = False
        else:
            self.long_term_profile_mode = True
        self.llm_model = llm_model

        qdrant_path = config["vector_store"]["config"]["path"]
        os.makedirs(qdrant_path, exist_ok=True)
        self.collection_name = kwargs.get("collection_name") or self._derive_collection_name(qdrant_path)
        config["vector_store"]["config"]["collection_name"] = self.collection_name
        self.logger.info(
            f"Using Qdrant path '{qdrant_path}' with collection '{self.collection_name}' for memory writes."
        )

        if self.fact_extraction_mode == "0":
            from mem0.configs.prompts import FACT_RETRIEVAL_PROMPT

            config["custom_fact_extraction_prompt"] = FACT_RETRIEVAL_PROMPT
        elif self.fact_extraction_mode == "1":
            from mem0.configs.prompts import FACT_RETRIEVAL_PROMPT_1

            config["custom_fact_extraction_prompt"] = FACT_RETRIEVAL_PROMPT_1
        elif self.fact_extraction_mode == "2":
            from mem0.configs.prompts import FACT_RETRIEVAL_PROMPT_2

            config["custom_fact_extraction_prompt"] = FACT_RETRIEVAL_PROMPT_2
        elif self.fact_extraction_mode == "3":
            from mem0.configs.prompts import FACT_RETRIEVAL_PROMPT_3

            config["custom_fact_extraction_prompt"] = FACT_RETRIEVAL_PROMPT_3
        elif self.fact_extraction_mode == "5":
            from mem0.configs.prompts import FACT_RETRIEVAL_PROMPT_5
            config["custom_fact_extraction_prompt"] = FACT_RETRIEVAL_PROMPT_5
        elif self.fact_extraction_mode == "10":
            from mem0.configs.prompts import FACT_RETRIEVAL_PROMPT_10
            config["custom_fact_extraction_prompt"] = FACT_RETRIEVAL_PROMPT_10
        elif self.fact_extraction_mode == "12":
            from mem0.configs.prompts import FACT_RETRIEVAL_PROMPT_12
            config["custom_fact_extraction_prompt"] = FACT_RETRIEVAL_PROMPT_12
        elif self.fact_extraction_mode == "13":
            from mem0.configs.prompts import FACT_RETRIEVAL_PROMPT_13
            config["custom_fact_extraction_prompt"] = FACT_RETRIEVAL_PROMPT_13
        elif self.fact_extraction_mode == "3c":
            from mem0.configs.prompts import FACT_RETRIEVAL_PROMPT_3c
            config["custom_fact_extraction_prompt"] = FACT_RETRIEVAL_PROMPT_3c
        elif self.fact_extraction_mode == "14":
            from mem0.configs.prompts import FACT_RETRIEVAL_PROMPT_14
            config["custom_fact_extraction_prompt"] = FACT_RETRIEVAL_PROMPT_14
        elif self.fact_extraction_mode == "14.6":
            from mem0.configs.prompts import FACT_RETRIEVAL_PROMPT_14_6
            config["custom_fact_extraction_prompt"] = FACT_RETRIEVAL_PROMPT_14_6
        elif self.fact_extraction_mode == "14.7":
            from mem0.configs.prompts import FACT_RETRIEVAL_PROMPT_14_7
            config["custom_fact_extraction_prompt"] = FACT_RETRIEVAL_PROMPT_14_7
        elif self.fact_extraction_mode == "14.8":
            from mem0.configs.prompts import FACT_RETRIEVAL_PROMPT_14_8
            config["custom_fact_extraction_prompt"] = FACT_RETRIEVAL_PROMPT_14_8
        elif self.fact_extraction_mode == "14.9":
            from mem0.configs.prompts import FACT_RETRIEVAL_PROMPT_14_9
            config["custom_fact_extraction_prompt"] = FACT_RETRIEVAL_PROMPT_14_9
        elif self.fact_extraction_mode == "15":
            from mem0.configs.prompts import FACT_RETRIEVAL_PROMPT_15
            config["custom_fact_extraction_prompt"] = FACT_RETRIEVAL_PROMPT_15

        if self.memory_decision_mode == "0":
            from mem0.configs.prompts import DEFAULT_UPDATE_MEMORY_PROMPT
            config["custom_update_memory_prompt"] = DEFAULT_UPDATE_MEMORY_PROMPT
        elif self.memory_decision_mode == "1":
            from mem0.configs.prompts import UPDATE_MEMORY_PROMPT_1
            config["custom_update_memory_prompt"] = UPDATE_MEMORY_PROMPT_1
        elif self.memory_decision_mode == "2":
            from mem0.configs.prompts import UPDATE_MEMORY_PROMPT_2
            config["custom_update_memory_prompt"] = UPDATE_MEMORY_PROMPT_2
        elif self.memory_decision_mode == "2agg":
            from mem0.configs.prompts import UPDATE_MEMORY_PROMPT_2_agg
            config["custom_update_memory_prompt"] = UPDATE_MEMORY_PROMPT_2_agg
        elif self.memory_decision_mode == "2con":
            from mem0.configs.prompts import UPDATE_MEMORY_PROMPT_2_con
            config["custom_update_memory_prompt"] = UPDATE_MEMORY_PROMPT_2_con
        elif self.memory_decision_mode == "10":
            from mem0.configs.prompts import UPDATE_MEMORY_PROMPT_10
            config["custom_update_memory_prompt"] = UPDATE_MEMORY_PROMPT_10
        elif self.memory_decision_mode == "11":
            from mem0.configs.prompts import UPDATE_MEMORY_PROMPT_11
            config["custom_update_memory_prompt"] = UPDATE_MEMORY_PROMPT_11
        elif self.memory_decision_mode == "5":
            from mem0.configs.prompts import UPDATE_MEMORY_PROMPT_5
            config["custom_update_memory_prompt"] = UPDATE_MEMORY_PROMPT_5
        elif self.memory_decision_mode == "12":
            from mem0.configs.prompts import UPDATE_MEMORY_PROMPT_12
            config["custom_update_memory_prompt"] = UPDATE_MEMORY_PROMPT_12
        elif self.memory_decision_mode == "13":
            from mem0.configs.prompts import UPDATE_MEMORY_PROMPT_13
            config["custom_update_memory_prompt"] = UPDATE_MEMORY_PROMPT_13
        elif self.memory_decision_mode == "0c":
            from mem0.configs.prompts import UPDATE_MEMORY_PROMPT_0c
            config["custom_update_memory_prompt"] = UPDATE_MEMORY_PROMPT_0c
        elif self.memory_decision_mode == "14":
            from mem0.configs.prompts import UPDATE_MEMORY_PROMPT_14
            config["custom_update_memory_prompt"] = UPDATE_MEMORY_PROMPT_14
        elif self.memory_decision_mode == "14.5":
            from mem0.configs.prompts import UPDATE_MEMORY_PROMPT_14_5
            config["custom_update_memory_prompt"] = UPDATE_MEMORY_PROMPT_14_5
        elif self.memory_decision_mode == "14.6":
            from mem0.configs.prompts import UPDATE_MEMORY_PROMPT_14_6
            config["custom_update_memory_prompt"] = UPDATE_MEMORY_PROMPT_14_6
        elif self.memory_decision_mode == "14.7":
            from mem0.configs.prompts import UPDATE_MEMORY_PROMPT_14_7
            config["custom_update_memory_prompt"] = UPDATE_MEMORY_PROMPT_14_7
        elif self.memory_decision_mode == "14.9":
            from mem0.configs.prompts import UPDATE_MEMORY_PROMPT_14_9
            config["custom_update_memory_prompt"] = UPDATE_MEMORY_PROMPT_14_9
        # please modify the prompt in mem0/configs/prompts.py if you want to change the memory decision prompt

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
        # Then, set the logger attribute on the created instance
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
        """
        Ensure mode values are normalized strings for downstream prompt selection.
        """
        normalized = "0" if value is None else str(value).strip()
        if not normalized:
            normalized = "0"
        return normalized

    def _build_timestamp_metadata(self, timestamp_value):
        """
        Assemble consistent timestamp metadata for storage.
        """
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

    def _summarize_session_memory(self, user_id: str, messages: list, timestamp: Optional[str]) -> None:
        """
        Generate and store a session-level summary memory for a given user.
        """
        if not messages:
            return

        conversation_text = "\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in messages)
        system_prompt = (
            "You are a concise session summarizer. Read the conversation and produce a short summary that captures key "
            "facts, decisions, preferences, and important events. Keep it crisp (3-5 bullet lines max), avoid "
            "repetition, and retain concrete details like names, dates, and numbers when present. Output plain text."
        )
        user_prompt = f"Session timestamp: {timestamp or 'unknown'}\nConversation:\n{conversation_text}"

        try:
            summary = self.memory.llm.generate_response(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
        except Exception as exc:
            self.logger.warning("Session summary generation failed for %s: %s", user_id, exc)
            return

        if not summary:
            return

        metadata = self._build_timestamp_metadata(timestamp)
        metadata["session_level"] = True
        try:
            with self._memory_semaphore:
                self.memory.add(summary, user_id=user_id, metadata=metadata, infer=False)
        except Exception as exc:
            self.logger.warning("Failed to store session summary for %s: %s", user_id, exc)

    def _ensure_spacy_model(self):
        """Lazy load spacy model for entity extraction."""
        if self._nlp_model is None:
            try:
                import spacy
                # disable参数禁用了不需要的组件，能提升速度
                # en_core_web_trf
                # en_core_web_sm
                self._nlp_model = spacy.load("en_core_web_trf", disable=["parser", "textcat"])
                self.logger.info("✅ en_core_web_trf 模型加载成功！")
            except OSError:
                self.logger.warning("未找到 SpaCy 模型，正在尝试下载...")
                try:
                    from spacy.cli import download
                    download("en_core_web_trf")
                    import spacy
                    self._nlp_model = spacy.load("en_core_web_trf", disable=["parser", "textcat"])
                    self.logger.info("✅ en_core_web_trf 模型下载并加载成功！")
                except Exception as e:
                    self.logger.warning("SpaCy 模型加载失败: %s，将跳过实体提取", e)
                    self._nlp_model = False  # Mark as failed to avoid retrying
        return self._nlp_model if self._nlp_model is not False else None
    
    def _extract_entities_from_text(self, text: str) -> list:
        """
        从文本中提取实体（人名、组织名等）
        
        Args:
            text: 要提取实体的文本
            
        Returns:
            list: 提取到的实体列表（去重）
        """
        nlp_model = self._ensure_spacy_model()
        if not nlp_model:
            return []
        
        try:
            doc = nlp_model(text)
            # 提取人名(PERSON)、组织(ORG)、地理位置(GPE)等实体
            entities = []
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE"]:  # 人名、组织、地理位置
                    entities.append(ent.text.strip())
            
            # 去重并保持顺序
            seen = set()
            unique_entities = []
            for ent in entities:
                if ent.lower() not in seen:
                    seen.add(ent.lower())
                    unique_entities.append(ent)
            
            return unique_entities
        except Exception as e:
            self.logger.warning("实体提取失败: %s", e)
            return []
    
    def _generate_long_term_profiles(self, user_id: str, timestamp: Optional[str]) -> None:
        """
        Generate or update long-term profile memories (L3) for key entities after a session ends.
        
        Logic:
        1. If no long-term profiles exist for this user -> Generate new profiles based on current session
        2. If long-term profiles already exist -> Update them based on new session memories
        """
        try:
            import json
            import re
            
            # Get all memories for this user
            all_memories_result = self.memory.get_all(user_id=user_id, limit=1000)
            
            if not all_memories_result or not all_memories_result.get("results"):
                self.logger.debug("No memories found for user %s, skipping long-term profile processing.", user_id)
                return
            
            memories = all_memories_result["results"]
            # Separate long-term profiles from other memories
            existing_profiles = []
            session_memories = []
            
            for mem in memories:
                memory_text = mem.get("memory", "")
                memory_metadata = mem.get("metadata", {})
                
                # Check if it's a long-term profile
                if memory_text.startswith("[Long-term Profile]:") or memory_metadata.get("level") == "L3":
                    existing_profiles.append({
                        "id": mem.get("id"),
                        "text": memory_text
                    })
                else:
                    # Collect non-profile memories for processing
                    session_memories.append(memory_text)
            
            if not session_memories:
                self.logger.debug("No non-profile memories found for user %s, skipping long-term profile processing.", user_id)
                return
            
            # Prepare memories text for LLM
            memories_text = "\n".join([f"- {mem}" for mem in session_memories])
            
            # Extract entities from memories text to help LLM generate more comprehensive profiles
            extracted_entities = self._extract_entities_from_text(memories_text)
            
            # Case 1: No existing profiles -> Generate new ones
            if not existing_profiles:
                self.logger.info("No existing long-term profiles found for user %s, generating new profiles.", user_id)
                if extracted_entities:
                    self.logger.info("Extracted %d entities: %s", len(extracted_entities), ", ".join(extracted_entities[:5]))
                self._generate_new_profiles(user_id, timestamp, memories_text, extracted_entities)
            
            # Case 2: Existing profiles -> Update them
            else:
                self.logger.info("Found %d existing long-term profiles for user %s, updating them.", len(existing_profiles), user_id)
                if extracted_entities:
                    self.logger.info("Extracted %d entities: %s", len(extracted_entities), ", ".join(extracted_entities[:5]))
                self._update_existing_profiles(user_id, timestamp, existing_profiles, memories_text, extracted_entities)
                
        except Exception as exc:
            self.logger.warning("Long-term profile processing failed for %s: %s", user_id, exc)
    
    def _generate_new_profiles(self, user_id: str, timestamp: Optional[str], memories_text: str, extracted_entities: list = None) -> None:
        """Generate new long-term profiles when none exist."""
        try:
            import json
            import re
            
            from mem0.configs.prompts import LONG_TERM_PROFILE_GENERATION_PROMPT
            
            # Prepare entities information for prompt
            entities_info = ""
            if extracted_entities:
                entities_info = f"\n\n## Key Entities Identified\nBased on entity extraction, the following key entities were found in the memories:\n" + "\n".join([f"- {ent}" for ent in extracted_entities[:20]]) + "\n\nPlease ensure you create profiles for these entities if they have sufficient information. This list is provided to help you identify all relevant entities and generate more comprehensive profiles."
            
            # Generate long-term profiles using LLM
            prompt = LONG_TERM_PROFILE_GENERATION_PROMPT.format(memories=memories_text) + entities_info
            
            try:
                response = self.memory.llm.generate_response(
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
            except Exception as exc:
                self.logger.warning("Long-term profile generation failed for %s: %s", user_id, exc)
                return
            
            if not response:
                return
            
            # Parse the response
            try:
                # Remove code blocks if present
                response_clean = re.sub(r'```json\s*', '', response)
                response_clean = re.sub(r'```\s*', '', response_clean).strip()
                
                profile_data = json.loads(response_clean)
                profiles = profile_data.get("profiles", [])
                
                if not profiles:
                    self.logger.debug("No long-term profiles generated for user %s", user_id)
                    return
                
                # Store each profile as a separate memory
                metadata = self._build_timestamp_metadata(timestamp)
                metadata["long_term_profile"] = True
                metadata["level"] = "L3"
                
                for profile_text in profiles:
                    if not profile_text or not profile_text.strip():
                        continue
                    
                    # Ensure the profile starts with the correct prefix
                    if not profile_text.startswith("[Long-term Profile]:"):
                        profile_text = f"[Long-term Profile]: {profile_text}"
                    
                    try:
                        with self._memory_semaphore:
                            # Use infer=False to directly add without going through fact extraction
                            self.memory.add(
                                profile_text,
                                user_id=user_id,
                                metadata=metadata,
                                infer=False
                            )
                        self.logger.info("Stored new long-term profile for user %s: %s", user_id, profile_text[:100])
                    except Exception as exc:
                        self.logger.warning("Failed to store long-term profile '%s' for %s: %s", 
                                          profile_text[:50], user_id, exc)
                        
            except json.JSONDecodeError as e:
                self.logger.warning("Failed to parse long-term profile response for %s: %s. Response: %s", 
                                  user_id, e, response[:200])
            except Exception as exc:
                self.logger.warning("Error processing new long-term profiles for %s: %s", user_id, exc)
                
        except Exception as exc:
            self.logger.warning("Failed to generate new long-term profiles for %s: %s", user_id, exc)
    
    def _update_existing_profiles(self, user_id: str, timestamp: Optional[str], 
                                  existing_profiles: list, new_memories_text: str, extracted_entities: list = None) -> None:
        """Update existing long-term profiles based on new session memories."""
        try:
            import json
            import re
            
            from mem0.configs.prompts import LONG_TERM_PROFILE_UPDATE_PROMPT
            
            # Prepare existing profiles text
            existing_profiles_text = "\n".join([f"- {p['text']}" for p in existing_profiles])
            
            # Prepare entities information for prompt
            entities_info = ""
            if extracted_entities:
                entities_info = f"\n\n## Key Entities Identified\nBased on entity extraction from new memories, the following key entities were found:\n" + "\n".join([f"- {ent}" for ent in extracted_entities[:20]]) + "\n\nPlease ensure you check if these entities need new profiles or updates to existing profiles. This list is provided to help you identify all relevant entities and generate more comprehensive updates."
            
            # Generate update prompt
            prompt = LONG_TERM_PROFILE_UPDATE_PROMPT.format(
                existing_profiles=existing_profiles_text,
                new_memories=new_memories_text
            ) + entities_info
            
            try:
                response = self.memory.llm.generate_response(
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
            except Exception as exc:
                self.logger.warning("Long-term profile update failed for %s: %s", user_id, exc)
                return
            
            if not response:
                return
            
            # Parse the response
            try:
                # Remove code blocks if present
                response_clean = re.sub(r'```json\s*', '', response)
                response_clean = re.sub(r'```\s*', '', response_clean).strip()
                
                update_data = json.loads(response_clean)
                updates = update_data.get("updates", [])
                
                if not updates:
                    self.logger.debug("No updates needed for long-term profiles of user %s", user_id)
                    return
                
                # Create a mapping from profile text to memory ID
                profile_id_map = {p['text']: p['id'] for p in existing_profiles}
                
                # Update each profile that needs updating
                metadata = self._build_timestamp_metadata(timestamp)
                metadata["long_term_profile"] = True
                metadata["level"] = "L3"
                
                for update_item in updates:
                    original_text = update_item.get("id", "")
                    updated_text = update_item.get("updated_text", "")
                    should_update = update_item.get("should_update", False)
                    
                    if not should_update or not updated_text:
                        continue
                    
                    # Find the memory ID
                    memory_id = profile_id_map.get(original_text)
                    if not memory_id:
                        self.logger.warning("Could not find memory ID for profile: %s", original_text[:50])
                        continue
                    
                    # Ensure the updated text has the correct prefix
                    if not updated_text.startswith("[Long-term Profile]:"):
                        updated_text = f"[Long-term Profile]: {updated_text}"
                    
                    try:
                        with self._memory_semaphore:
                            # Update the memory directly
                            self.memory.update(memory_id=memory_id, data=updated_text)
                        self.logger.info("Updated long-term profile for user %s: %s -> %s", 
                                        user_id, original_text[:50], updated_text[:50])
                    except Exception as exc:
                        self.logger.warning("Failed to update long-term profile '%s' for %s: %s", 
                                          original_text[:50], user_id, exc)
                        
            except json.JSONDecodeError as e:
                self.logger.warning("Failed to parse long-term profile update response for %s: %s. Response: %s", 
                                  user_id, e, response[:200])
            except Exception as exc:
                self.logger.warning("Error processing long-term profile updates for %s: %s", user_id, exc)
                
        except Exception as exc:
            self.logger.warning("Failed to update existing long-term profiles for %s: %s", user_id, exc)

    def add_memories_for_speaker(
        self,
        speaker,
        messages,
        timestamp,
        message_pbar=None,
        update_progress=True,
    ):
        overlap_mode = self.add_mode == "1"
        for i in range(0, len(messages), self.batch_size):
            start_index = i
            if overlap_mode and i > 0:
                start_index = max(0, i - 1)
            end_index = min(len(messages), i + self.batch_size)
            batch_messages = messages[start_index:end_index]
            metadata = self._build_timestamp_metadata(timestamp)
            self.add_memory(speaker, batch_messages, metadata=metadata or None)
            if message_pbar and update_progress:
                increment = len(batch_messages)
                if overlap_mode and i > 0:
                    increment = max(0, increment - 1)
                with self._pbar_lock:
                    message_pbar.update(increment)

        if self.fact_abstract_mode == "1":
            self._summarize_session_memory(speaker, messages, timestamp)

        # Generate long-term profiles (L3 memory) after session ends
        if self.long_term_profile_mode:
            self._generate_long_term_profiles(speaker, timestamp)
        else:
            self.logger.info("Long-term profile mode is disabled, skipping long-term profile generation.")
                
    def process_conversation(self, item, idx, session_pbar=None, message_pbar=None):

        max_retries = 2  # 定义最大重试次数 (总共尝试 1 + 2 = 3 次)

        for attempt in range(max_retries + 1):
            total_session_count = 0
            total_dialogue_count = 0
            sessions_processed = 0
            dialogues_processed = 0
            try:
                conversation = item.get("conversation") or {}
                speaker_a = conversation.get("speaker_a")
                speaker_b = conversation.get("speaker_b")

                if not speaker_a or not speaker_b:
                    raise ValueError(f"Conversation {idx} 缺少必要的说话者信息。")

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
                total_dialogue_count = 0
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

                    self.add_memories_for_speaker(
                        speaker_a_user_id,
                        messages,
                        timestamp,
                        message_pbar,
                        update_progress=True,
                    )
                    self.add_memories_for_speaker(
                        speaker_b_user_id,
                        messages_reverse,
                        timestamp,
                        message_pbar,
                        update_progress=False,
                    )

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
        if not self.data_path:
            raise ValueError("No data path configured. 请先设置 data_path 再调用 process_all_conversations().")

        if not self._dataset_stats:
            self.load_data()
        stats = self._dataset_stats or {}
        total_conversations = stats.get("total_conversations", 0)
        total_sessions = stats.get("total_sessions", 0)
        total_dialogues = stats.get("total_dialogues", 0)

        resolved_workers = self._resolve_max_workers(max_workers)
        print(f"--- 总共需要处理 {total_conversations} 组对话 ---")
        if total_sessions:
            print(f"--- 覆盖 {total_sessions} 个会话，{total_dialogues} 条对话 ---")
        if resolved_workers != max_workers:
            print(f"⚙️ 调整 max_workers: 从 {max_workers} -> {resolved_workers}")
        else:
            print(f"⚙️ 使用 max_workers = {resolved_workers}")

        if total_conversations == 0:
            print("📭 数据集为空，无需执行导入。")
            return

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

        futures = {}
        drain_threshold = max(resolved_workers, 1) * 4

        def consume_one():
            nonlocal successful_count, failed_count
            if not futures:
                return
            done, _ = wait(tuple(futures.keys()), return_when=FIRST_COMPLETED)
            for finished in done:
                conversation_id = futures.pop(finished)
                try:
                    finished.result()
                    successful_count += 1
                except Exception as exc:
                    failed_count += 1
                    error_details = "".join(
                        traceback.format_exception(type(exc), exc, exc.__traceback__)
                    )
                    conversation_pbar.write(
                        f"\n---[Error {failed_count}] ❌ Error processing {conversation_id}: {exc} ---\n"
                    )
                    conversation_pbar.write(f"{error_details}\n")
                finally:
                    conversation_pbar.update(1)

        try:
            with ThreadPoolExecutor(max_workers=resolved_workers, thread_name_prefix="mem-add") as executor:
                for idx, item in enumerate(stream_normalized_dataset(self.data_path)):
                    future = executor.submit(self.process_conversation, item, idx, session_pbar, message_pbar)
                    futures[future] = f"Conversation {idx}"
                    if len(futures) >= drain_threshold:
                        consume_one()

                while futures:
                    consume_one()
        except Exception as exc:
            raise RuntimeError("Failed during threaded memory ingestion.") from exc
        finally:
            conversation_pbar.close()
            if session_pbar:
                session_pbar.close()
            if message_pbar:
                message_pbar.close()

        print(f"\n✅ All conversations processed. Success: {successful_count}, Failed: {failed_count}")