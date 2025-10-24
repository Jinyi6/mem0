import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Union

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


def normalize_dataset_record(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a single dataset record while avoiding the cost of copying the
    entire collection into memory. Only the QA section is deep-copied so callers
    can safely mutate it.
    """
    if not isinstance(item, dict):
        return {}
    item_copy: Dict[str, Any] = dict(item)
    item_copy["qa"] = normalize_qa_section(item_copy.get("qa", []))
    return item_copy


def _stream_json_array_fast(path: Path, chunk_size: int) -> Iterator[Any]:
    """
    Efficiently stream JSON objects from an on-disk array without loading the
    entire payload into memory.
    """
    decoder = json.JSONDecoder()
    buffer = ""
    inside_array = False

    with path.open("r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            buffer += chunk

            while True:
                stripped = buffer.lstrip()
                if not stripped:
                    buffer = ""
                    break

                if not inside_array:
                    lead = stripped[0]
                    if lead in "\ufeff":
                        buffer = stripped[1:]
                        continue
                    if lead == "[":
                        inside_array = True
                        buffer = stripped[1:]
                        continue
                    raise ValueError("Top-level JSON value is not an array.")

                head = stripped[0]
                if head == "]":
                    return
                if head == ",":
                    buffer = stripped[1:]
                    continue

                try:
                    obj, offset = decoder.raw_decode(stripped)
                except json.JSONDecodeError:
                    # Need more data from disk.
                    buffer = stripped
                    break

                yield obj
                buffer = stripped[offset:]

        # Handle any trailing content after the final read.
        stripped = buffer.lstrip()
        if not stripped:
            return
        if not inside_array:
            raise ValueError("Top-level JSON value is not an array.")
        if stripped[0] == "]":
            return
        if stripped[0] == ",":
            stripped = stripped[1:].lstrip()
        if stripped and stripped[0] != "]":
            raise ValueError("Unexpected trailing content in JSON array.")


def stream_json_array(path: Union[str, Path], chunk_size: int = 65_536) -> Iterator[Any]:
    """
    Stream JSON records from a file whose top-level value is an array. Falls back
    to loading the full payload when streaming is not possible (e.g. dict input).
    """
    json_path = Path(path)
    try:
        yield from _stream_json_array_fast(json_path, chunk_size)
        return
    except ValueError:
        pass

    data = json.loads(json_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in sorted(data.keys(), key=_coerce_numeric_key):
            yield data[key]
    elif isinstance(data, list):
        for item in data:
            yield item


def stream_normalized_dataset(path: Union[str, Path], chunk_size: int = 65_536) -> Iterator[Dict[str, Any]]:
    """
    Stream and normalize dataset records from disk. Each iteration yields a
    single normalized item so downstream callers can process records lazily.
    """
    for raw_record in stream_json_array(path, chunk_size=chunk_size):
        normalized = normalize_dataset_record(raw_record)
        if normalized:
            yield normalized


def compute_dataset_stats(path: Union[str, Path], chunk_size: int = 65_536) -> Dict[str, Any]:
    """
    Collect aggregate statistics for a dataset without retaining every record in
    memory. Returns counts for conversations, sessions, dialog turns, and QA
    items, as well as the QA counts per conversation for downstream bookkeeping.
    """
    totals = {
        "total_conversations": 0,
        "total_sessions": 0,
        "total_dialogues": 0,
        "total_questions": 0,
        "qa_per_conversation": [],
    }

    for record in stream_normalized_dataset(path, chunk_size=chunk_size):
        totals["total_conversations"] += 1

        conversation = record.get("conversation") or {}
        session_keys = [
            key for key in conversation.keys() if key.startswith("session_") and not key.endswith("_date_time")
        ]
        totals["total_sessions"] += len(session_keys)

        dialogue_count = 0
        for key in session_keys:
            chats = conversation.get(key, [])
            if isinstance(chats, list):
                dialogue_count += len(chats)
        totals["total_dialogues"] += dialogue_count

        qa_count = len(record.get("qa", []))
        totals["total_questions"] += qa_count
        totals["qa_per_conversation"].append(qa_count)

    return totals
