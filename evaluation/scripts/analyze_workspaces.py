import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openai import OpenAI

from utils.result_merger import (
    find_matching_result_file,
    load_json_file,
    merge_memory_and_scores,
    save_json_file,
)


def _prepare_workspace_data(workspace_path: str) -> Tuple[str, Dict[str, Any]]:
    """Ensure a combined JSON exists for the workspace and return its path and content."""
    if not os.path.isdir(workspace_path):
        raise NotADirectoryError(f"Workspace directory does not exist: {workspace_path}")

    json_files = [
        os.path.join(workspace_path, name)
        for name in os.listdir(workspace_path)
        if name.endswith(".json")
    ]

    combined_candidates = [
        path for path in json_files if path.endswith("_combined.json")
    ]
    if combined_candidates:
        combined_path = max(combined_candidates, key=os.path.getmtime)
        return combined_path, load_json_file(combined_path)

    evaluation_candidates = [
        path for path in json_files if os.path.basename(path).startswith("evaluation_metrics")
    ]
    if not evaluation_candidates:
        raise FileNotFoundError(f"No evaluation_metrics*.json file found in {workspace_path}")

    evaluation_path = max(evaluation_candidates, key=os.path.getmtime)
    result_path = find_matching_result_file(workspace_path, evaluation_path)
    if not result_path:
        raise FileNotFoundError(
            f"Unable to infer the search result file for evaluation file {evaluation_path}"
        )

    memory_results = load_json_file(result_path)
    evaluation_results = load_json_file(evaluation_path)
    combined_data = merge_memory_and_scores(memory_results, evaluation_results)

    combined_path = os.path.join(
        workspace_path,
        f"{os.path.splitext(os.path.basename(evaluation_path))[0]}_combined.json",
    )
    save_json_file(combined_path, combined_data)
    return combined_path, combined_data


def _flatten_results(results: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Flatten conversation-keyed results into a list while preserving the source conversation id."""
    flattened = []
    for conv_key, entries in results.items():
        for item in entries:
            record = dict(item)
            record["_conversation_key"] = conv_key
            flattened.append(record)
    return flattened


def _build_index(entries: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    """Index entries by (conversation key, question)."""
    index: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for item in entries:
        question = item.get("question")
        conv = item.get("_conversation_key")
        if question is None or conv is None:
            continue
        index[(str(conv), question)].append(item)
    return index


def _normalise_score(score: Any) -> Any:
    if score in {None, "", "None"}:
        return None
    if isinstance(score, str):
        score = score.strip()
        if score.isdigit():
            score = int(score)
        else:
            try:
                score = float(score)
            except ValueError:
                return None
    if isinstance(score, float) and score.is_integer():
        score = int(score)
    if score in {0, 1}:
        return score
    return None


def _classification_label(a_score: Any, b_score: Any) -> str:
    """Return the requested classification string based on judge scores."""
    a_val = _normalise_score(a_score)
    b_val = _normalise_score(b_score)

    if a_val is None or b_val is None:
        return "unknown"
    if a_val == 1 and b_val == 1:
        return "both correct"
    if a_val == 0 and b_val == 0:
        return "both wrong"
    if a_val == 1 and b_val == 0:
        return "A only correct"
    if a_val == 0 and b_val == 1:
        return "B only correct"
    return "unknown"


def _write_summary_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    """Persist comparison rows to CSV."""
    fieldnames = [
        "question",
        "answer",
        "category",
        "response_a",
        "judge_a",
        "response_b",
        "judge_b",
        "classification",
    ]
    with open(path, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _format_source_payload(
    entry_a: Dict[str, Any],
    entry_b: Dict[str, Any],
) -> Dict[str, Any]:
    """Collect the relevant fields we want to expose to the analysis model."""
    def _extract(entry: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "response": entry.get("response"),
            "judge_score": entry.get("llm_score"),
            "memories_speaker_1": entry.get("speaker_1_memories"),
            "memories_speaker_2": entry.get("speaker_2_memories"),
            "answer_prompt": entry.get("answer_prompt"),
        }

    return {
        "question": entry_a.get("question") or entry_b.get("question"),
        "gold_answer": entry_a.get("answer") or entry_b.get("answer"),
        "category": entry_a.get("category") or entry_b.get("category"),
        "system_a": _extract(entry_a),
        "system_b": _extract(entry_b),
    }


def _run_llm_analysis(
    client: OpenAI,
    model_name: str,
    scenario: str,
    pairs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Run LLM-based diagnostics for each pair of responses."""
    analyses = []
    for payload in pairs:
        data_blob = _format_source_payload(payload["entry_a"], payload["entry_b"])
        prompt = (
            "You are auditing two question-answering systems.\n"
            f"Scenario: {scenario}.\n"
            "Analyse the responses, explain the observed outcome, and highlight actionable guidance.\n"
            "Focus on:\n"
            "1. Whether the retrieved context or prompt content explains the success/failure.\n"
            "2. Specific mistakes or omissions that led to an incorrect answer.\n"
            "3. Concrete suggestions for the under-performing system.\n"
            "Provide a concise analysis.\n"
            "Here is the structured data:\n"
            f"{json.dumps(data_blob, indent=2)}"
        )

        analysis_record = {
            "question": data_blob["question"],
            "classification": scenario,
            "analysis_prompt": prompt,
            "model": model_name,
            "source_data": data_blob,
        }

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": prompt}],
                temperature=0.2,
            )
            analysis_record["analysis_response"] = response.choices[0].message.content
        except Exception as exc:
            analysis_record["analysis_error"] = str(exc)

        analyses.append(analysis_record)

    return analyses


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two experiment workspaces and analyse differences.")
    parser.add_argument("workspace_a", type=str, help="Path to workspace A.")
    parser.add_argument("workspace_b", type=str, help="Path to workspace B.")
    parser.add_argument(
        "--label_a",
        type=str,
        default="A",
        help="Friendly label for workspace A in the output.",
    )
    parser.add_argument(
        "--label_b",
        type=str,
        default="B",
        help="Friendly label for workspace B in the output.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory where summary and analysis reports will be saved.",
    )
    parser.add_argument(
        "--analysis_model",
        type=str,
        default=os.getenv("MODEL", "Qwen/Qwen3-14B"),
        help="Model used for automated diagnostics.",
    )
    args = parser.parse_args()

    combined_a_path, combined_a = _prepare_workspace_data(args.workspace_a)
    combined_b_path, combined_b = _prepare_workspace_data(args.workspace_b)

    entries_a = _flatten_results(combined_a)
    entries_b = _flatten_results(combined_b)

    index_a = _build_index(entries_a)
    index_b = _build_index(entries_b)

    shared_keys = sorted(set(index_a.keys()) & set(index_b.keys()))
    if not shared_keys:
        raise ValueError("No overlapping questions found between the two workspaces.")

    comparison_rows: List[Dict[str, Any]] = []
    llm_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for key in shared_keys:
        a_entry = index_a[key][0]
        b_entry = index_b[key][0]
        classification = _classification_label(a_entry.get("llm_score"), b_entry.get("llm_score"))

        comparison_rows.append(
            {
                "question": a_entry.get("question"),
                "answer": a_entry.get("answer"),
                "category": a_entry.get("category"),
                "response_a": a_entry.get("response"),
                "judge_a": a_entry.get("llm_score"),
                "response_b": b_entry.get("response"),
                "judge_b": b_entry.get("llm_score"),
                "classification": classification,
            }
        )

        if classification in {"both wrong", "A only correct", "B only correct"}:
            llm_groups[classification].append(
                {"entry_a": a_entry, "entry_b": b_entry}
            )

    output_dir = args.output_dir or os.path.join(
        os.getcwd(),
        f"comparison_reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(output_dir, exist_ok=True)

    summary_csv_path = os.path.join(output_dir, "comparison_summary.csv")
    _write_summary_csv(summary_csv_path, comparison_rows)

    client = OpenAI()
    analysis_results: Dict[str, List[Dict[str, Any]]] = {}
    for scenario, payload in llm_groups.items():
        analysis_results[scenario] = _run_llm_analysis(client, args.analysis_model, scenario, payload)

    report_payload = {
        "workspace_a": {"path": args.workspace_a, "combined_file": combined_a_path, "label": args.label_a},
        "workspace_b": {"path": args.workspace_b, "combined_file": combined_b_path, "label": args.label_b},
        "summary_csv": summary_csv_path,
        "analysis": analysis_results,
    }
    report_json_path = os.path.join(output_dir, "analysis_report.json")
    save_json_file(report_json_path, report_payload)

    print(f"Summary table saved to: {summary_csv_path}")
    print(f"Detailed analysis saved to: {report_json_path}")


if __name__ == "__main__":
    main()
