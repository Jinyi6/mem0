import json
import os
import random
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import uuid

from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI
from tqdm import tqdm
from src.utils import normalize_dataset_records

# --- Step 1: Define the Prompt Template for the Full Context approach ---
# This prompt is designed to take the entire conversation history directly.
ANSWER_PROMPT_FULL_CONTEXT = """
You are an intelligent assistant. Your task is to answer a question based on the provided conversation history.

# CONTEXT:
You have access to the complete conversation history between two speakers. This history contains all the information needed to answer the question.

# INSTRUCTIONS:
1.  Carefully read the entire conversation history to understand the context.
2.  Analyze the user's question and locate the relevant parts of the conversation that contain the answer.
3.  Pay close attention to timestamps and the order of messages to resolve any conflicting information, prioritizing the most recent messages if necessary.
4.  Your answer must be derived directly from the text in the history. Do not infer information or use external knowledge.
5.  Keep your answer concise and to the point, ideally less than 5-6 words, unless more detail is explicitly required by the question.
6.  If the question involves relative time references (e.g., "last year"), use the timestamps in the conversation to calculate the specific date or year.

# APPROACH (Think step by step):
1.  Identify the key entities and concepts in the question.
2.  Scan the full conversation history to find all mentions related to these keys.
3.  Synthesize the information from the relevant messages to formulate a precise answer.
4.  Double-check that your answer directly addresses the question and is supported by the provided text.

--- CONVERSATION HISTORY ---
{{conversation_history}}
--- END OF HISTORY ---

Question: {{question}}

Answer:
"""
DEFAULT_LLM_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen3-14B")
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"

class FullContextManager:
    """
    Handles the logic for answering questions using the full conversation context.
    This class reads a dataset, processes each question against its full conversation,
    and saves the LLM-generated answers.
    """
    def __init__(self, output_path, logger=None, figure_view=False, llm_config=None):
        load_dotenv()
        self.output_path = output_path
        llm_config = llm_config or {}
        llm_model = llm_config.get("model") or DEFAULT_LLM_MODEL
        llm_base_url = llm_config.get("base_url") or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL
        llm_api_key = llm_config.get("api_key") or os.getenv("OPENAI_API_KEY")

        openai_client_kwargs = {}
        if llm_base_url:
            openai_client_kwargs["base_url"] = llm_base_url
        if llm_api_key:
            openai_client_kwargs["api_key"] = llm_api_key
        self.openai_client = OpenAI(**openai_client_kwargs)
        os.environ["MODEL"] = llm_model
        self.model_name = llm_model
        self.logger = logger if logger else logging.getLogger(__name__)
        self.results = defaultdict(list)
        self.lock = threading.Lock()
        self.template = Template(ANSWER_PROMPT_FULL_CONTEXT)
        self.figure_view = figure_view

    def _log_llm_call(self, request_id, attempt, max_retries, prompt_components, full_prompt, response_content, status):
        """Formats and logs the complete LLM interaction."""
        log_message = f"""
========================= LLM Call Start (Full Context) =========================
--------------------------- INPUT ----------------------------
{full_prompt}
--------------------------- OUTPUT ---------------------------
{response_content if response_content else 'N/A'}
========================== SUMMARY ==========================
Request ID: {request_id}
Attempt: {attempt}/{max_retries}
Status: {status}
========================== LLM Call End ==========================
"""
        if "Success" in status:
            self.logger.info(log_message)
        else:
            self.logger.error(log_message)

    def _format_conversation(self, conversation_item):
        """Formats the conversation dictionary into a readable string."""
        history = []
        speaker_a = conversation_item['speaker_a']
        speaker_b = conversation_item['speaker_b']

        conversation_keys = [
            k for k in conversation_item 
            if k not in ["speaker_a", "speaker_b"] and not k.endswith(("_date_time", "_timestamp"))
        ]

        for key in conversation_keys:
            # 获取对应的时间戳
            timestamp = conversation_item.get(f"{key}_date_time", "Unknown time")
            history.append(f"\n--- Turn started at {timestamp} ---")
            
            # 检查'key'对应的值是否是列表
            chats = conversation_item.get(key, [])
            if isinstance(chats, list):
                for chat in chats:
                    # 确保 chat 是一个字典
                    if isinstance(chat, dict) and 'speaker' in chat and 'text' in chat:
                        speaker_name = speaker_a if chat['speaker'] == speaker_a else speaker_b
                        text_content = chat['text']
                        if self.figure_view:
                            if "img_url" in chat and "blip_caption" in chat:
                                text_content += f" [Image: {chat.get('img_url')}] with caption: {chat.get('blip_caption')}"
                        history.append(f"{speaker_name}: {text_content}")

        return "\n".join(history)

    
    def _answer_question_with_full_context(self, conversation_history, question, max_retries=5):
        """Calls the LLM with the full context to get an answer."""
        request_id = f"full-context-q-{uuid.uuid4()}"
        response_content = None
        start_time = time.time()
        max_context_exceeded = 0
        max_trim_attempts = 5
        trim_attempts = 0

        conversation_lines = conversation_history.splitlines()
        total_lines = len(conversation_lines)
        current_length = total_lines if total_lines > 0 else 0

        def build_prompt(length):
            history_slice = "\n".join(conversation_lines[:length]) if length else ""
            prompt_components_local = {
                "conversation_history": history_slice,
                "question": question,
            }
            rendered_prompt = self.template.render(prompt_components_local)
            return history_slice, prompt_components_local, rendered_prompt

        current_history_str, prompt_components, answer_prompt = build_prompt(current_length)

        attempt = 0
        while attempt < max_retries:
            attempt += 1
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "system", "content": answer_prompt}],
                    temperature=0.0
                )
                response_content = response.choices[0].message.content
                self._log_llm_call(
                    request_id,
                    attempt,
                    max_retries,
                    prompt_components,
                    answer_prompt,
                    response_content,
                    "Success",
                )
                break  # Success, exit loop
            except Exception as e:
                error_message = f"LLM API call failed. Error: {e}"
                error_lower = str(e).lower()
                is_context_error = "maximum context length" in error_lower
                status_c = "Failed Attempt"

                if is_context_error:
                    max_context_exceeded = 1
                if (
                    is_context_error
                    and trim_attempts < max_trim_attempts
                    and current_length > 0
                ):
                    trim_attempts += 1
                    trim_size = max(1, int(current_length * 0.1))
                    current_length = max(0, current_length - trim_size)
                    current_history_str, prompt_components, answer_prompt = build_prompt(current_length)
                    trim_message = (
                        f"{error_message} | Reducing conversation history by 10% "
                        f"(trim attempt {trim_attempts}/{max_trim_attempts})."
                    )
                    self._log_llm_call(
                        request_id,
                        attempt,
                        max_retries,
                        prompt_components,
                        "",  # Avoid logging large prompt repeatedly
                        trim_message,
                        status_c,
                    )
                    # Retry immediately without additional wait.
                    continue

                if attempt >= max_retries:
                    status_c = "Failed Attempt (END)"
                    error_message = (
                        f"Request ID [{request_id}] - LLM call failed permanently after {max_retries} attempts."
                    )
                    response_content = "Error: Failed to get response from LLM."
                    self._log_llm_call(
                        request_id,
                        attempt,
                        max_retries,
                        prompt_components,
                        "",
                        error_message,
                        status_c,
                    )
                    break

                sleep_time = random.uniform(45, 75)
                status_c += f", retrying in {sleep_time:.2f} seconds..."
                self._log_llm_call(
                    request_id,
                    attempt,
                    max_retries,
                    prompt_components,
                    "",
                    error_message,
                    status_c,
                )
                time.sleep(sleep_time)

        response_time = time.time() - start_time
        return response_content, response_time, answer_prompt, max_context_exceeded

    def _process_single_question(self, conversation_item, question_item, idx, pbar):
        """Worker function to process one question using its conversation context."""
        question = question_item.get("question", "")
        answer = question_item.get("answer", "")
        category = question_item.get("category", -1)

        conversation_history = self._format_conversation(conversation_item["conversation"])
        response, response_time, answer_prompt, max_context_flag = self._answer_question_with_full_context(
            conversation_history, question
        )

        result = {
            "question": question,
            "answer": answer,
            "category": category,
            "response": response,
            "response_time": response_time,
            "answer_prompt": answer_prompt,
            "max_context_exceeded": max_context_flag,
        }

        # The lock is now only held for a very short time to do a quick memory update.
        with self.lock:
            self.results[idx].append(result)

        pbar.update(1)
        return result

    def process_data_file(self, file_path, max_workers=10):
        """
        Main method to orchestrate the processing of the dataset.
        It uses a thread pool to handle questions in parallel.
        """
        with open(file_path, "r") as f:
            raw_data = json.load(f)
        data = normalize_dataset_records(raw_data)

        total_questions = sum(len(item.get("qa", [])) for item in data)
        if total_questions == 0:
            print("No questions found to process.")
            return

        print(f"--- Starting Full Context Evaluation: {total_questions} total questions to process ---")

        with tqdm(total=total_questions, desc="💡 Full Context Progress") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for idx, item in enumerate(data):
                    for question_item in item.get("qa", []):
                        future = executor.submit(
                            self._process_single_question,
                            item, question_item, idx, pbar
                        )
                        futures.append(future)

                # Wait for all futures to complete and handle potential exceptions
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        import traceback
                        error_details = traceback.format_exc()
                        self.logger.error(f"A task failed in the thread pool: {e}")
                        self.logger.error(f"Full error traceback: {error_details}")

        # --- THIS IS THE CORRECT PLACE TO SAVE THE FILE ---
        # All threads are done, now write the final result to the file once.
        print("\nAll threads finished. Saving final results to disk...")
        with open(self.output_path, "w") as f:
            # Note: defaultdict needs to be converted to a regular dict for clean JSON output if keys are integers
            final_results = {str(k): v for k, v in self.results.items()}
            json.dump(final_results, f, indent=4)

        print(f"✅ Full context evaluation complete. Results saved to {self.output_path}")
