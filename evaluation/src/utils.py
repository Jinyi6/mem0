from copy import deepcopy
from typing import Any, Dict, Iterable, List, Sequence

TECHNIQUES = ["mem0", "rag", "langmem", "zep", "openai", "full_context"]

METHODS = ["add", "search"]

MODES = ["client", "no_client", "no_client_async"]


def _coerce_numeric_key(value: Any) -> Any:
    """
    Helper to convert stringified numeric keys to integers so ordering is stable
    between list-based and dict-based QA payloads.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def normalize_qa_section(qa_section: Any) -> List[Dict[str, Any]]:
    """
    Normalize the QA section of a dataset entry so downstream code can assume it is a list.
    Supports both the legacy list-of-dicts format and the new dict-of-dicts format.
    """
    if isinstance(qa_section, dict):
        ordered_keys = sorted(qa_section.keys(), key=_coerce_numeric_key)
        return [deepcopy(qa_section[key]) for key in ordered_keys]
    if isinstance(qa_section, list):
        return [deepcopy(item) for item in qa_section]
    return []


def normalize_dataset_records(data: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize an entire dataset so each entry contains a list-based QA section.
    """
    normalized_records: List[Dict[str, Any]] = []
    for item in data or []:
        if not isinstance(item, dict):
            continue
        item_copy = deepcopy(item)
        item_copy["qa"] = normalize_qa_section(item_copy.get("qa", []))
        normalized_records.append(item_copy)
    return normalized_records
