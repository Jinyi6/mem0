#!/usr/bin/env python3
"""
Filter CL-bench locomo JSONL by conversation context token length.

Definition of context length:
  Concatenate all `text` fields from every `session_*` in `conversation`,
  then count tokens on the concatenated string.

Outputs a JSONL subset containing samples with token length in [min, max].
"""

import argparse
import json
import os
import re
from typing import Iterable, List, Tuple


def _iter_session_texts(conversation) -> Iterable[str]:
    if conversation is None:
        return []

    # Common format: dict with session_1, session_2, ...
    if isinstance(conversation, dict):
        session_items: List[Tuple[int, object]] = []
        for key, value in conversation.items():
            if not key.startswith("session_"):
                continue
            if key.endswith("_date_time"):
                continue
            match = re.match(r"session_(\d+)$", key)
            if not match:
                continue
            session_index = int(match.group(1))
            session_items.append((session_index, value))

        if session_items:
            for _, session in sorted(session_items, key=lambda item: item[0]):
                for text in _iter_session_texts(session):
                    if text:
                        yield text
            return

    # Session can be a list of turns or nested sessions.
    if isinstance(conversation, list):
        for item in conversation:
            for text in _iter_session_texts(item):
                if text:
                    yield text
        return

    # A single turn dict with text.
    if isinstance(conversation, dict) and "text" in conversation:
        text = conversation.get("text")
        if isinstance(text, str):
            yield text


def _build_token_counter(tokenizer: str, encoding_name: str):
    if tokenizer == "tiktoken":
        try:
            import tiktoken  # type: ignore
        except Exception:
            print("⚠️  tiktoken not installed; falling back to whitespace tokenization.")
            tokenizer = "whitespace"
        else:
            enc = tiktoken.get_encoding(encoding_name)

            def _count_tiktoken(text: str) -> int:
                return len(enc.encode(text))

            return _count_tiktoken

    if tokenizer == "whitespace":
        def _count_ws(text: str) -> int:
            return len(re.findall(r"\\S+", text))

        return _count_ws

    if tokenizer == "chars":
        def _count_chars(text: str) -> int:
            return len(text)

        return _count_chars

    raise ValueError(f"Unknown tokenizer: {tokenizer}")


def count_context_tokens(record: dict, counter) -> int:
    conversation = record.get("conversation")
    parts = list(_iter_session_texts(conversation))
    joined = "\n".join(parts)
    return counter(joined)


def filter_jsonl(input_path: str, output_path: str, min_tokens: int, max_tokens: int,
                 tokenizer: str, encoding_name: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    counter = _build_token_counter(tokenizer, encoding_name)

    total = 0
    kept = 0
    min_seen = None
    max_seen = None

    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            record = json.loads(line)
            token_count = count_context_tokens(record, counter)

            if min_seen is None or token_count < min_seen:
                min_seen = token_count
            if max_seen is None or token_count > max_seen:
                max_seen = token_count

            if min_tokens <= token_count <= max_tokens:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1

    print("=" * 60)
    print("Done.")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Tokenizer: {tokenizer} (encoding={encoding_name})")
    print(f"Total: {total}")
    print(f"Kept:  {kept}")
    if min_seen is not None:
        print(f"Min tokens seen: {min_seen}")
    if max_seen is not None:
        print(f"Max tokens seen: {max_seen}")
    print("=" * 60)


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Filter CL-bench locomo JSONL by conversation token length."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join(script_dir, "CL-bench_locomo.jsonl"),
        help="Input JSONL file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(script_dir, "CL-bench_locomo_0_0.1k.jsonl"),
        help="Output JSONL file",
    )
    parser.add_argument("--min_tokens", type=int, default=0, help="Minimum token length (inclusive)")
    parser.add_argument("--max_tokens", type=int, default=102, help="Maximum token length (inclusive)")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="tiktoken",
        choices=["tiktoken", "whitespace", "chars"],
        help="Tokenizer to use for length counting",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="cl100k_base",
        help="tiktoken encoding name (used when tokenizer=tiktoken)",
    )
    args = parser.parse_args()

    filter_jsonl(
        args.input,
        args.output,
        args.min_tokens,
        args.max_tokens,
        args.tokenizer,
        args.encoding,
    )


if __name__ == "__main__":
    main()
