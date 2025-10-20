import json
import os
import re
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Optional


def load_json_file(path: str) -> Dict[str, Any]:
    """Load JSON content from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(path: str, data: Dict[str, Any]) -> None:
    """Persist JSON data to disk with pretty formatting."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def _normalise_key(key: Any) -> str:
    """Ensure dictionary keys are treated consistently as strings."""
    return str(key)


def merge_memory_and_scores(
    memory_results: Dict[str, List[Dict[str, Any]]],
    evaluation_results: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Merge memory retrieval outputs with evaluation scores using the question text as the join key.

    Args:
        memory_results: Dict keyed by conversation identifier containing memory pipeline outputs.
        evaluation_results: Dict keyed by conversation identifier containing evaluation metrics (e.g., llm_score).

    Returns:
        A dict keyed by conversation identifier where each entry contains the merged information.
    """
    combined: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    # Prepare evaluation lookup while preserving per-conversation ordering.
    eval_lookup: Dict[str, List[Dict[str, Any]]] = {
        _normalise_key(conv_key): [deepcopy(item) for item in conv_items]
        for conv_key, conv_items in evaluation_results.items()
    }

    for conv_key, memory_items in memory_results.items():
        norm_key = _normalise_key(conv_key)
        eval_items = eval_lookup.get(norm_key, [])

        for mem_item in memory_items:
            merged_entry = deepcopy(mem_item)
            matched_eval: Optional[Dict[str, Any]] = None

            question = mem_item.get("question")
            # Match by exact question text to avoid cross-conversation collisions.
            for idx, candidate in enumerate(eval_items):
                if candidate.get("question") == question:
                    matched_eval = eval_items.pop(idx)
                    break

            if matched_eval:
                for field, value in matched_eval.items():
                    if field in {"question", "answer", "response", "category"}:
                        # These fields already exist in mem_item; keep the memory version.
                        continue
                    merged_entry[field] = value
            else:
                merged_entry.setdefault("_merge_warnings", []).append("evaluation_missing")

            combined[norm_key].append(merged_entry)

    # Handle evaluation items that never matched a memory entry.
    for conv_key, remaining_items in eval_lookup.items():
        if not remaining_items:
            continue
        for eval_item in remaining_items:
            placeholder = {
                "question": eval_item.get("question"),
                "answer": eval_item.get("answer"),
                "response": eval_item.get("response"),
                "category": eval_item.get("category"),
            }
            for field, value in eval_item.items():
                if field not in placeholder:
                    placeholder[field] = value
            placeholder.setdefault("_merge_warnings", []).append("memory_missing")
            combined[conv_key].append(placeholder)

    return dict(combined)


def find_matching_result_file(workspace_dir: str, evaluation_file: str) -> Optional[str]:
    """
    Attempt to locate the result JSON file that corresponds to a given evaluation file.

    Strategy:
        1. If the evaluation filename contains a timestamp, prefer files that contain the same token.
        2. Otherwise, fall back to the most recently modified mem0/full_context result file.

    Args:
        workspace_dir: Directory containing experiment artefacts.
        evaluation_file: Path to the evaluation metrics file.

    Returns:
        Path to the best-matching result file, or None if nothing reasonable was found.
    """
    if not os.path.isdir(workspace_dir):
        raise NotADirectoryError(f"Workspace directory does not exist: {workspace_dir}")

    evaluation_name = os.path.basename(evaluation_file)
    timestamp_match = re.search(r"(\\d{8}_\\d{6})", evaluation_name)
    timestamp = timestamp_match.group(1) if timestamp_match else None

    json_files = [
        os.path.join(workspace_dir, name)
        for name in os.listdir(workspace_dir)
        if name.endswith(".json")
    ]

    def _is_candidate(path: str) -> bool:
        base = os.path.basename(path)
        if base.startswith("evaluation_metrics"):
            return False
        if base.endswith("_combined.json"):
            return False
        if base.startswith("evaluation_") and base.endswith(".json"):
            return False
        return True

    candidates = [path for path in json_files if _is_candidate(path)]
    if not candidates:
        return None

    if timestamp:
        timestamp_matches = [path for path in candidates if timestamp in os.path.basename(path)]
        if timestamp_matches:
            return max(timestamp_matches, key=os.path.getmtime)

    # Prefer mem0 outputs, then full_context, then anything else.
    def _priority(path: str) -> int:
        base = os.path.basename(path)
        if base.startswith("mem0_"):
            return 0
        if base.startswith("full_context_"):
            return 1
        return 2

    return min(
        candidates,
        key=lambda path: (_priority(path), -os.path.getmtime(path)),
    )
