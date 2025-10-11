import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import json
import logging
import uuid 

from dotenv import load_dotenv
from tqdm import tqdm
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
import math
load_dotenv()

model_name = os.getenv("BASE_MODEL", "Qwen/Qwen3-14B")



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



import random   

class MemoryADD:
    def __init__(self, data_path=None, batch_size=6, is_graph=False, logger=None, **kwargs):
        config = {
            "llm": {
            "provider": "openai",
            "config": {
                "model": model_name,
                "openai_base_url": os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1"),
                "temperature": 0.1,
                "max_tokens": 2000,
                # "prompts": {
                #     "memory_creation": custom_instructions
                # }
            }
            },
            "embedder": {
            "provider": "openai",
            "config": {
                "model": kwargs.get("embedder_model", "BAAI/bge-m3"),
                "openai_base_url": os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1"),
            }
            },
            "vector_store": {
            "provider": "qdrant",
            "config": {
                "path": kwargs.get("qdrant_path", "./qdrant_data/tmp"),
                "on_disk": True,
                "embedding_model_dims": 1024
            }
            },
            "version": "v1.1",
        }
        self.batch_size = batch_size
        self.data_path = data_path
        self.data = None
        self.is_graph = is_graph
        self.figure_view = kwargs.get("figure_view", False)
        self.fact_extraction_mode = int(kwargs.get("fact_extraction_mode", "0"))
        self.memory_decision_mode = int(kwargs.get("memory_decision_mode", "0"))
        if self.fact_extraction_mode == 0:
            from mem0.configs.prompts import FACT_RETRIEVAL_PROMPT 
            config["custom_fact_extraction_prompt"] = FACT_RETRIEVAL_PROMPT
        elif self.fact_extraction_mode == 1:
            from mem0.configs.prompts import FACT_RETRIEVAL_PROMPT_1
            config["custom_fact_extraction_prompt"] = FACT_RETRIEVAL_PROMPT_1

        if self.memory_decision_mode == 0:
            from mem0.configs.prompts import DEFAULT_UPDATE_MEMORY_PROMPT
            config["custom_memory_decision_prompt"] = DEFAULT_UPDATE_MEMORY_PROMPT
        # please modify the prompt in mem0/configs/prompts.py if you want to change the memory decision prompt

        if data_path:
            self.load_data()
        # Create the memory object first
        self.logger = logger if logger else logging.getLogger(__name__)
        self.memory = Memory.from_config(config)
        # Then, set the logger attribute on the created instance
        self.memory.logger = self.logger

    def load_data(self):
        with open(self.data_path, "r") as f:
            self.data = json.load(f)
        return self.data

    def add_memory(self, user_id, message, metadata, retries=2):
        request_id = f"add-mem-{uuid.uuid4()}"
        _ = self.memory.add( message, user_id=user_id, metadata=metadata, fact_extraction_mode=self.fact_extraction_mode, memory_decision_mode=self.memory_decision_mode)
        return

        # for attempt in range(retries):
        #     try:
        #         _ = self.memory.add(
        #             message, user_id=user_id, metadata=metadata
        #         )
        #         return
        #     except Exception as e:
        #         if attempt < retries - 1:
        #             self.logger.warning(f"Request ID [{request_id}] - Retrying...{attempt+1}/{retries}\t{str(e)}")
        #             continue
        #         else:
        #             self.logger.error(f"Request ID [{request_id}] - Failed to add memory after retries.", str(e))
        #             raise e

    def add_memories_for_speaker(self, speaker, messages, timestamp, desc, pbar=None):
        # for i in range(0, len(messages), self.batch_size):
        for i in tqdm(range(0, len(messages), self.batch_size), desc=desc):
            batch_messages = messages[i : i + self.batch_size]
            self.add_memory(speaker, batch_messages, metadata={"timestamp": timestamp})
            if pbar:
                pbar.update(1)

    def process_conversation(self, item, idx, pbar=None):

        max_retries = 2  # 定义最大重试次数 (总共尝试 1 + 2 = 3 次)

        for attempt in range(max_retries + 1):
            try:
                conversation = item["conversation"]
                speaker_a = conversation["speaker_a"]
                speaker_b = conversation["speaker_b"]

                speaker_a_user_id = f"{speaker_a}_{idx}"
                speaker_b_user_id = f"{speaker_b}_{idx}"

                # delete all memories for the two users
                self.memory.delete_all(user_id=speaker_a_user_id)
                self.memory.delete_all(user_id=speaker_b_user_id)

                for key in conversation.keys():
                    if key in ["speaker_a", "speaker_b"] or "date" in key or "timestamp" in key:
                        continue

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

                    # add memories for the two users on different threads
                    # thread_a = threading.Thread(
                    #     target=self.add_memories_for_speaker,
                    #     args=(speaker_a_user_id, messages, timestamp, "Adding Memories for Speaker A", pbar),
                    # )
                    # thread_b = threading.Thread(
                    #     target=self.add_memories_for_speaker,
                    #     args=(speaker_b_user_id, messages_reverse, timestamp, "Adding Memories for Speaker B", pbar),
                    # )

                    # thread_a.start()
                    # thread_b.start()
                    # thread_a.join()
                    # thread_b.join()
                
                    self.add_memories_for_speaker(speaker_a_user_id, messages, timestamp, "Adding Memories for Speaker A", pbar)
                    self.add_memories_for_speaker(speaker_b_user_id, messages_reverse, timestamp, "Adding Memories for Speaker B", pbar)
                # --- 如果代码成功执行到这里，说明没有错误 ---
                self.logger.info(f"Conversation {idx} processed successfully on attempt {attempt + 1}.")
                return  # 成功后直接退出函数，不再重试

            except Exception as e:
                # --- 如果 try 块中任何地方发生错误，都会进入这里 ---
                self.logger.warning(f"An error occurred on attempt {attempt + 1}/{max_retries + 1} for conversation {idx}. Error: {e}")
                
                if attempt < max_retries:
                    # 如果还未达到最大重试次数，则等待一小段时间后重试
                    self.logger.info(f"Retrying conversation {idx}...")
                    time.sleep(1)  # 等待1秒，避免因瞬时问题立即重试导致连续失败
                else:
                    # 如果已经达到最大重试次数，记录严重错误并放弃
                    self.logger.error(f"Failed to process conversation {idx} after {max_retries + 1} attempts.")
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
        total_batches = 0
        for item in self.data:
            conversation = item["conversation"]
            for key in conversation.keys():
                if key in ["speaker_a", "speaker_b"] or "date" in key or "timestamp" in key:
                    continue
                
                num_messages = len(conversation[key])
                # 每个对话轮次，A和B都要处理一次，所以计算两次
                batches_for_speaker_a = math.ceil(num_messages / self.batch_size)
                batches_for_speaker_b = math.ceil(num_messages / self.batch_size)
                total_batches += batches_for_speaker_a + batches_for_speaker_b
        
        print(f"--- 预计总共需要处理 {total_batches} 个批次 ---")

        print("🔧 Pre-initializing vector store collection...")
        try:
            # 执行一个临时的、无害的操作来触发 collection 的创建
            init_user_id = f"system_init_{uuid.uuid4()}"
            self.memory.add("init", user_id=init_user_id)
            self.memory.delete_all(user_id=init_user_id)
            print("✅ Collection pre-initialization successful.")
        except Exception as e:
            print(f"🔥 Error during pre-initialization, this might be okay if collection already exists: {e}")
            pass

        successful_count = 0
        failed_count = 0
        
        with tqdm(total=total_batches, desc="💡Total Batch Progress") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # futures = {
                #     executor.submit(self.process_conversation, item, idx, pbar): f"Conversation {idx}" 
                #     for idx, item in enumerate(self.data)
                # }
                futures = {}
                for idx, item in enumerate(self.data):
                    futures[executor.submit(self.process_conversation, item, idx, pbar)] = f"Conversation {idx}"
                    time.sleep(2 / (idx + 1))

                for future in as_completed(futures):
                    conversation_id = futures[future]
                    try:
                        future.result()
                        successful_count += 1
                    except Exception as e:
                        failed_count += 1
                        pbar.write(f"\n---[retry{failed_count}] ❌ Error processing {conversation_id}: {e} ---\n")
                        # time.sleep(random.randint(5, 10))

        print(f"\n✅ All conversations processed. Success: {successful_count}, Failed: {failed_count}")