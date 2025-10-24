"""
Utility script to re-run evaluation over historical result files.

Each command mirrors the invocation shown at the bottom of this file previously:
python -u ./evals.py --input_file ... --output_file ... --max_workers 10
    --evaluator_model Qwen/Qwen3-14B --evaluator_base_url https://api.siliconflow.cn/v1
    --evaluator_api_key sk-...

The output filename replaces the original timestamp segment (the last portion
that starts with 2025 and precedes .json) with the required
evaluation_metrics_{timestamp}.json.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path
from typing import Iterable, List


# Historical input files that need to be re-evaluated.
INPUT_FILES: List[str] = [
    "/Users/jinyi/Documents/code/memory/mem0/evaluation/exp_data/locomo10_0/locomo10_0_top_k_30_filter_False_graph_False_3_1_5_0_20251014_234426/mem0_locomo10_0_results_top_30_filter_False_graph_False_20251014_234426_3_1_5_0.json",
    "/Users/jinyi/Documents/code/memory/mem0/evaluation/exp_data/locomo10_1/locomo10_1_top_k_30_filter_False_graph_False_3_1_5_0_20251015_035044/mem0_locomo10_1_results_top_30_filter_False_graph_False_20251015_035044_3_1_5_0.json",
    "/Users/jinyi/Documents/code/memory/mem0/evaluation/exp_data/locomo10_9/locomo10_9_top_k_30_filter_False_graph_False_3_1_5_0_20251016_134035/mem0_locomo10_9_results_top_30_filter_False_graph_False_20251016_134035_3_1_5_0.json",
    "/Users/jinyi/Documents/code/memory/mem0/evaluation/exp_data/locomo10_8/locomo10_8_top_k_30_filter_False_graph_False_3_1_5_0_20251016_080228/mem0_locomo10_8_results_top_30_filter_False_graph_False_20251016_080228_3_1_5_0.json",
    "/Users/jinyi/Documents/code/memory/mem0/evaluation/exp_data/locomo10_7/locomo10_7_top_k_30_filter_False_graph_False_3_1_5_0_20251015_200201/mem0_locomo10_7_results_top_30_filter_False_graph_False_20251015_200201_3_1_5_0.json",
    "/Users/jinyi/Documents/code/memory/mem0/evaluation/exp_data/locomo10_2/locomo10_2_top_k_0_filter_False_graph_False_na_na_na_na_20251022_004331/full_context_locomo10_2_results_20251022_004331.json",
    "/Users/jinyi/Documents/code/memory/mem0/evaluation/exp_data/locomo10_2/locomo10_2_top_k_0_filter_False_graph_False_na_na_na_na_20251022_131839/full_context_locomo10_2_results_20251022_131839.json",
    "/Users/jinyi/Documents/code/memory/mem0/evaluation/exp_data/locomo10_2/locomo10_2_top_k_30_filter_False_graph_False_3_1_5_0_20251015_060725/mem0_locomo10_2_results_top_30_filter_False_graph_False_20251015_060725_3_1_5_0.json",
    "/Users/jinyi/Documents/code/memory/mem0/evaluation/exp_data/locomo10_2/locomo10_2_top_k_30_filter_False_graph_False_3_1_5_0_20251022_004002/mem0_locomo10_2_results_top_30_filter_False_graph_False_20251022_105615_3_1_5_0.json",
    "/Users/jinyi/Documents/code/memory/mem0/evaluation/exp_data/locomo10_2/locomo10_2_top_k_30_filter_False_graph_False_3_1_5_0_20251022_004002/mem0_locomo10_2_results_top_30_filter_False_graph_False_20251022_131142_3_1_5_7.json",
    "/Users/jinyi/Documents/code/memory/mem0/evaluation/exp_data/locomo10_2/locomo10_2_top_k_30_filter_False_graph_False_3_1_5_5_20251020_001617/mem0_locomo10_2_results_top_30_filter_False_graph_False_20251020_001617_3_1_5_5.json",
    "/Users/jinyi/Documents/code/memory/mem0/evaluation/exp_data/locomo10_2/locomo10_2_top_k_30_filter_False_graph_False_3_1_5_6_20251020_232546/mem0_locomo10_2_results_top_30_filter_False_graph_False_20251020_232546_3_1_5_6.json",

    "/Users/jinyi/Documents/code/memory/mem0/evaluation/exp_data/locomo10_2/locomo10_2_top_k_30_filter_False_graph_False_3_1_5_7_20251022_011323/mem0_locomo10_2_results_top_30_filter_False_graph_False_20251022_011323_3_1_5_7.json",
    "/Users/jinyi/Documents/code/memory/mem0/evaluation/exp_data/locomo10_2/locomo10_2_top_k_30_filter_False_graph_False_3_1_5_7_20251022_011453/mem0_locomo10_2_results_top_30_filter_False_graph_False_20251022_011453_3_1_5_7.json",
    "/Users/jinyi/Documents/code/memory/mem0/evaluation/exp_data/locomo10_2/locomo10_2_top_k_30_filter_False_graph_False_3_1_5_7_20251022_140025/mem0_locomo10_2_results_top_30_filter_False_graph_False_20251022_140025_3_1_5_7.json",
    "/Users/jinyi/Documents/code/memory/mem0/evaluation/exp_data/locomo10_2/locomo10_2_top_k_30_filter_False_graph_False_3_1_5_6_20251020_232546/mem0_locomo10_2_results_top_30_filter_False_graph_False.json",
    "/Users/jinyi/Documents/code/memory/mem0/evaluation/exp_data/locomo10_2/locomo10_2_top_k_30_filter_False_graph_False_3_1_5_7_20251022_212343/mem0_locomo10_2_results_top_30_filter_False_graph_False.json",

    # "",
    # "",
    # "",
    # "",
]

EVALUATION_DIR = Path(__file__).resolve().parent.parent
EVALS_SCRIPT = EVALUATION_DIR / "evals.py"

FILENAME_TIMESTAMP_PATTERN = re.compile(r"(2025[^/]*?)(?:\.json)?$")

# Static command arguments reused for each run.
METRICS: List[str] = ["llm"]  # Extend with "f1"/"bleu" to enable additional metrics.

BASE_COMMAND = [
    "python",
    "-u",
    "./evals_v0.py",
    "--max_workers",
    "20",
    "--evaluator_model",
    "Qwen/Qwen3-14B",
    "--evaluator_base_url",
    "https://api.siliconflow.cn/v1",
    "--evaluator_api_key",
    "sk-vyvftxtwuiznrwrfvayhfitxgpdpsykrdnukzfdtdwtjgqvo",
]

if METRICS:
    BASE_COMMAND.extend(["--metrics", *METRICS])


def extract_timestamp(input_path: Path) -> str:
    """Try to extract the timestamp fragment from the path.

    Returns an empty string when no timestamp is found so the caller can still
    construct an output filename.
    """
    # Prefer grabbing the last timestamp from the filename itself.
    filename_match = FILENAME_TIMESTAMP_PATTERN.search(input_path.name)
    if filename_match:
        return filename_match.group(1)

    # Fall back to scanning parent directories for a timestamp fragment.
    for part in reversed(input_path.parts):
        if "2025" not in part:
            continue
        part_match = re.search(r"2025[^/]*", part)
        if part_match:
            return part_match.group(0)

    return ""


def derive_output_file(input_path: Path) -> Path:
    """Return the evaluation_metrics output path derived from the input file."""
    timestamp = extract_timestamp(input_path)
    if not timestamp:
        print(
            f"Unable to derive timestamp from input file name: {input_path}",
            flush=True,
        )
        print("Falling back to empty timestamp fragment.\n", flush=True)

    return input_path.parent / f"evaluation_metrics_{timestamp}_v5.json"


def build_command(input_file: Path, output_file: Path) -> List[str]:
    """Compose the command that will be executed."""
    return [
        *BASE_COMMAND,
        "--input_file",
        str(input_file),
        "--output_file",
        str(output_file),
    ]


def run_command(command: Iterable[str]) -> None:
    """Run a shell command while streaming output to stdout."""
    print("#" * 80, flush=True)
    print("Running command:", flush=True)
    printable_cmd = " ".join(shlex.quote(str(part)) for part in command)
    print(printable_cmd, flush=True)
    print("#" * 80, flush=True)

    process = subprocess.Popen(
        list(command),
        cwd=str(EVALUATION_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert process.stdout is not None  # for type-checkers
    for line in process.stdout:
        print(line, end="")

    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, list(command))


def main() -> None:
    for raw_input in INPUT_FILES:
        input_path = Path(raw_input).resolve()
        if not input_path.exists():
            print(f"Skipping missing input file: {input_path}", flush=True)
            continue

        output_path = derive_output_file(input_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = build_command(input_path, output_path)
        run_command(cmd)


if __name__ == "__main__":
    main()
