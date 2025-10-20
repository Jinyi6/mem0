#!/usr/bin/env python3

import argparse
import csv
import json
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple


def load_records(path: str) -> Dict[str, Dict[str, Any]]:
    """Load evaluation metrics file and index records by question text."""
    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, List[Dict[str, Any]]] = json.load(f)

    records: Dict[str, Dict[str, Any]] = {}
    duplicates: Dict[str, List[Tuple[str, int]]] = defaultdict(list)

    for group_key, items in raw.items():
        if not isinstance(items, list):
            raise ValueError(f"Expected list for key {group_key!r} in {path}")

        for idx, item in enumerate(items):
            question = item.get("question")
            if not question:
                raise ValueError(
                    f"Missing question text for key {group_key!r} index {idx} in {path}"
                )

            if question in records:
                duplicates[question].append((group_key, idx))
            else:
                records[question] = dict(item, _group_key=group_key, _index=idx)

    if duplicates:
        for question, locations in duplicates.items():
            formatted = ", ".join(f"{key}[{idx}]" for key, idx in locations)
            print(
                f"Warning: duplicate question {question!r} in {path} at {formatted}",
                file=sys.stderr,
            )

    return records


def classify(a_score: int, b_score: int) -> str:
    """Return a label describing the relative correctness."""
    if a_score == 1 and b_score == 0:
        return "A only correct"
    if a_score == 0 and b_score == 1:
        return "B only correct"
    if a_score == 1 and b_score == 1:
        return "Both correct"
    if a_score == 0 and b_score == 0:
        return "Both wrong"
    return "Incomplete data"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare evaluation metrics from two runs and print a table showing "
            "correctness differences."
        )
    )
    parser.add_argument("file_a", help="Reference file (assumed superset).")
    parser.add_argument("file_b", help="Second file (assumed subset).")
    parser.add_argument(
        "--only-common",
        action="store_true",
        help="Restrict output to questions that appear in both files.",
    )
    parser.add_argument(
        "--output-csv",
        help="Write detailed comparison rows to the given CSV file.",
    )
    args = parser.parse_args()

    records_a = load_records(args.file_a)
    records_b = load_records(args.file_b)

    if args.only_common:
        all_questions = sorted(set(records_a) & set(records_b))
    else:
        all_questions = sorted(set(records_a) | set(records_b))

    headers = [
        "Question",
        "Category A",
        "Category B",
        "Score A",
        "Score B",
        "Classification",
        "Answer",
        "Response A",
        "Response B",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" for _ in headers) + "|")

    stats: Dict[str, int] = defaultdict(int)
    rows: List[Dict[str, Any]] = []

    def normalize(value: Any) -> str:
        if value is None:
            return ""
        return str(value).replace("\n", "\\n").strip()

    for question in all_questions:
        rec_a = records_a.get(question)
        rec_b = records_b.get(question)

        score_a = rec_a.get("llm_score") if rec_a else None
        score_b = rec_b.get("llm_score") if rec_b else None

        if score_a is None or score_b is None:
            classification = "Only in A" if score_b is None else "Only in B"
        else:
            classification = classify(int(score_a), int(score_b))

        stats[classification] += 1

        category_a = rec_a.get("category") if rec_a else ""
        category_b = rec_b.get("category") if rec_b else ""
        answer = rec_a.get("answer") if rec_a else rec_b.get("answer") if rec_b else ""
        response_a = rec_a.get("response") if rec_a else ""
        response_b = rec_b.get("response") if rec_b else ""

        row = {
            "Question": question,
            "Category A": category_a,
            "Category B": category_b,
            "Score A": score_a if score_a is not None else "",
            "Score B": score_b if score_b is not None else "",
            "Classification": classification,
            "Answer": answer,
            "Response A": response_a,
            "Response B": response_b,
        }
        rows.append(row)

        print(
            "| "
            + " | ".join(
                normalize(value)
                for value in (
                    row[h] for h in headers
                )
            )
            + " |"
        )

    if args.output_csv:
        with open(args.output_csv, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: normalize(v) for k, v in row.items()})

    if stats:
        print("\nSummary")
        for label, count in sorted(stats.items()):
            print(f"{label}: {count}")


if __name__ == "__main__":
    main()
