#!/usr/bin/env python3
"""Aggregate evaluation combined JSON files and highlight differing outcomes."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


COMBINED_PATTERN = re.compile(
    r"evaluation_metrics_(?P<timestamp>\d{8}_\d{6})_combined\.json$"
)


@dataclass
class MethodInfo:
    """Metadata resolved for a given combined file."""

    method_id: str
    timestamp: str
    combined_file: Path
    config_file: Optional[Path]


def load_combined_entries(path: Path) -> List[dict]:
    """Load entries from a combined JSON file."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON from {path}: {exc}") from exc

    entries: List[dict] = []
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                entries.extend(item for item in value if isinstance(item, dict))
    elif isinstance(data, list):
        entries = [item for item in data if isinstance(item, dict)]

    return entries


def resolve_method_info(directory: Path, combined_path: Path) -> MethodInfo:
    """Derive the method identifier from the matching config file, when possible."""
    match = COMBINED_PATTERN.search(combined_path.name)
    if not match:
        fallback_key = combined_path.stem
        return MethodInfo(
            method_id=fallback_key,
            timestamp="",
            combined_file=combined_path,
            config_file=None,
        )

    timestamp = match.group("timestamp")
    date_part, time_part = timestamp.split("_")
    method_id = timestamp.replace("_", "-")
    chosen_config: Optional[Path] = None
    closest_diff: Optional[int] = None

    for candidate in sorted(directory.glob(f"config_{date_part}_*.json")):
        suffix = candidate.stem.split("_")[-1]
        if not suffix.isdigit():
            continue
        # Prefer configs that share the HHMM prefix.
        if suffix[:4] != time_part[:4]:
            continue
        diff = abs(int(suffix) - int(time_part))
        if closest_diff is None or diff < closest_diff:
            chosen_config = candidate
            closest_diff = diff

    if chosen_config is None:
        for candidate in sorted(directory.glob(f"config_{date_part}_*.json")):
            suffix = candidate.stem.split("_")[-1]
            if not suffix.isdigit():
                continue
            diff = abs(int(suffix) - int(time_part))
            if closest_diff is None or diff < closest_diff:
                chosen_config = candidate
                closest_diff = diff

    if chosen_config:
        try:
            with chosen_config.open("r", encoding="utf-8") as handle:
                config = json.load(handle)
            exp_params = config.get("exp_params", {})
            modes = [
                exp_params.get("fact_extraction_mode"),
                exp_params.get("memory_decision_mode"),
                exp_params.get("search_mode"),
                exp_params.get("answer_mode"),
            ]
            if all(isinstance(mode, str) for mode in modes):
                method_id = "-".join(modes)  # e.g. "5-11-6-5"
        except (json.JSONDecodeError, OSError):
            chosen_config = None

    return MethodInfo(
        method_id=method_id,
        timestamp=timestamp,
        combined_file=combined_path,
        config_file=chosen_config,
    )


def aggregate_directory(directory: Path) -> Dict[str, object]:
    """Aggregate combined files within a directory."""
    combined_files = sorted(directory.glob("evaluation_metrics_*_combined.json"))
    question_metadata: Dict[str, dict] = {}
    question_methods: Dict[str, Dict[str, dict]] = {}
    method_details: Dict[str, dict] = {}

    for combined_path in combined_files:
        entries = load_combined_entries(combined_path)
        if not entries:
            continue

        method_info = resolve_method_info(directory, combined_path)
        method_id = method_info.method_id
        method_details.setdefault(
            method_id,
            {
                "combined_file": str(combined_path),
                "timestamp": method_info.timestamp,
                "config_file": str(method_info.config_file)
                if method_info.config_file
                else None,
            },
        )

        for entry in entries:
            category = entry.get("category")
            if category is not None and str(category) == "5":
                continue

            question = entry.get("question")
            if not question:
                continue

            meta = question_metadata.setdefault(
                question,
                {
                    "question": question,
                    "answer": entry.get("answer"),
                    "category": entry.get("category"),
                    "answer_fixed": entry.get("answer_fixed"),
                },
            )

            # In case multiple files disagree on metadata, prefer the first seen value.
            for key in ("answer", "category", "answer_fixed"):
                if meta.get(key) is None and entry.get(key) is not None:
                    meta[key] = entry.get(key)

            question_methods.setdefault(question, {})
            question_methods[question][method_id] = {
                "speaker_1_memories": entry.get("speaker_1_memories", []),
                "speaker_2_memories": entry.get("speaker_2_memories", []),
                "response": entry.get("response"),
                "llm_score": entry.get("llm_score"),
                "answer_prompt": entry.get("answer_prompt"),
                "source_file": combined_path.name,
            }

    all_correct = 0
    all_wrong = 0
    differing_questions: List[dict] = []

    for question, methods in question_methods.items():
        scores = [
            details.get("llm_score")
            for details in methods.values()
            if details.get("llm_score") is not None
        ]
        method_count = len(methods)

        if scores and len(scores) == method_count:
            if all(score == 1 for score in scores):
                all_correct += 1
                continue
            if all(score == 0 for score in scores):
                all_wrong += 1
                continue

        metadata = question_metadata.get(question, {"question": question})
        differing_questions.append(
            {
                "question": metadata.get("question"),
                "answer": metadata.get("answer"),
                "category": metadata.get("category"),
                "answer_fixed": metadata.get("answer_fixed"),
                "methods": methods,
            }
        )

    summary = {
        "source_directory": str(directory),
        "total_questions": len(question_methods),
        "all_correct_count": all_correct,
        "all_wrong_count": all_wrong,
        "differing_question_count": len(differing_questions),
        "methods": method_details,
        "questions": differing_questions,
    }

    return summary


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge evaluation *_combined.json files within one or more directories "
            "and retain only questions whose outcomes differ across methods."
        )
    )
    parser.add_argument(
        "directories",
        nargs="+",
        type=Path,
        help="Directories that contain evaluation_metrics_*_combined.json files.",
    )
    parser.add_argument(
        "--output-name",
        default="evaluation_metrics_differences_summary.json",
        help="Filename (within each directory) to write the aggregated summary.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    exit_code = 0

    for directory in args.directories:
        if not directory.exists():
            print(f"[WARN] Directory not found: {directory}", file=sys.stderr)
            exit_code = 1
            continue
        if not directory.is_dir():
            print(f"[WARN] Not a directory: {directory}", file=sys.stderr)
            exit_code = 1
            continue

        summary = aggregate_directory(directory)
        output_path = directory / args.output_name
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")

        print(
            f"[{directory}] differing: {summary['differing_question_count']}, "
            f"all-correct: {summary['all_correct_count']}, "
            f"all-wrong: {summary['all_wrong_count']}"
        )
        print(f"  wrote: {output_path}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
