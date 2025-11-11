#!/usr/bin/env python3
"""Aggregate evaluation combined JSON files and highlight differing outcomes."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Exactly one of DIRECTORY_TO_ANALYZE or COMBINED_JSON_FILES should be set.
# Example: DIRECTORY_TO_ANALYZE = Path("/path/to/eval/runs")
DIRECTORY_TO_ANALYZE: Optional[Path] = None

# Example: COMBINED_JSON_FILES = [Path("run_a.json"), Path("run_b.json")]
COMBINED_JSON_FILES: List[Path] = []

# Defaults to DIRECTORY_TO_ANALYZE (or the first file's parent when using a list).
OUTPUT_DIRECTORY: Optional[Path] = None

# Base filename for the summary. A timestamp and .json extension will be appended.
OUTPUT_BASENAME = "evaluation_metrics_differences_summary"

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


def aggregate_combined_files(
    combined_files: Iterable[Path], source_label: str
) -> Dict[str, object]:
    """Aggregate one or more combined JSON files."""
    question_metadata: Dict[str, dict] = {}
    question_methods: Dict[str, Dict[str, dict]] = {}
    method_details: Dict[str, dict] = {}
    processed_files = 0

    for combined_path in combined_files:
        if not combined_path.exists():
            print(f"[WARN] Combined file not found: {combined_path}", file=sys.stderr)
            continue

        entries = load_combined_entries(combined_path)
        if not entries:
            continue

        processed_files += 1
        method_info = resolve_method_info(combined_path.parent, combined_path)
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
        "source": source_label,
        "files_processed": processed_files,
        "total_questions": len(question_methods),
        "all_correct_count": all_correct,
        "all_wrong_count": all_wrong,
        "differing_question_count": len(differing_questions),
        "methods": method_details,
        "questions": differing_questions,
    }

    return summary


def normalize_path(path: Path) -> Path:
    """Expand user/home references and return an absolute Path."""
    return Path(path).expanduser().resolve()


def derive_output_filename(base_name: str, timestamp: str) -> str:
    """Append a timestamp and ensure the filename ends with .json."""
    sanitized = base_name.strip()
    if sanitized.lower().endswith(".json"):
        sanitized = sanitized[: -len(".json")]
    if not sanitized:
        sanitized = "evaluation_metrics_differences_summary"
    return f"{sanitized}_{timestamp}.json"


def build_output_path(
    output_dir: Optional[Path], fallback_dir: Path, base_name: str, timestamp: str
) -> Path:
    """Select the directory for the summary and ensure it exists."""
    target_dir = normalize_path(output_dir) if output_dir else normalize_path(fallback_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = derive_output_filename(base_name, timestamp)
    return target_dir / filename


def resolve_combined_sources() -> Tuple[List[Path], str, Path]:
    """Determine which combined files to analyze based on the configuration."""
    if DIRECTORY_TO_ANALYZE and COMBINED_JSON_FILES:
        raise ValueError("Configure either DIRECTORY_TO_ANALYZE or COMBINED_JSON_FILES, not both.")

    if DIRECTORY_TO_ANALYZE:
        directory = normalize_path(DIRECTORY_TO_ANALYZE)
        if not directory.exists():
            raise ValueError(f"Directory not found: {directory}")
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")
        combined_files = sorted(directory.glob("evaluation_metrics_*_combined.json"))
        if not combined_files:
            raise ValueError(f"No combined JSON files found under {directory}")
        return combined_files, str(directory), directory

    if COMBINED_JSON_FILES:
        normalized_files: List[Path] = []
        for combined_file in COMBINED_JSON_FILES:
            normalized = normalize_path(combined_file)
            if not normalized.exists():
                print(f"[WARN] Combined file not found: {normalized}", file=sys.stderr)
                continue
            if not normalized.is_file():
                print(f"[WARN] Not a file: {normalized}", file=sys.stderr)
                continue
            normalized_files.append(normalized)
        if not normalized_files:
            raise ValueError("No valid combined JSON files to analyze.")
        return normalized_files, f"explicit-file-list ({len(normalized_files)} files)", normalized_files[0].parent

    raise ValueError("Please set DIRECTORY_TO_ANALYZE or COMBINED_JSON_FILES in this script.")


def main() -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        combined_paths, source_label, default_output_dir = resolve_combined_sources()
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    summary = aggregate_combined_files(combined_paths, source_label)
    summary["generated_at"] = timestamp
    output_path = build_output_path(OUTPUT_DIRECTORY, default_output_dir, OUTPUT_BASENAME, timestamp)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(
        f"[Aggregation] differing: {summary['differing_question_count']}, "
        f"all-correct: {summary['all_correct_count']}, "
        f"all-wrong: {summary['all_wrong_count']}"
    )
    print(f"  processed files: {summary['files_processed']}")
    print(f"  wrote summary: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
