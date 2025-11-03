#!/usr/bin/env python3
"""Run multiple `run_pipeline` experiments with preset mode combinations."""

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Sequence

# Path to the base configuration file.
CONFIG_PATH = "config/mem0_qwen.json"

# List of mode configurations to run. Each entry must contain either 2 values
#   [search_mode, answer_mode] or 4 values
#   [fact_extraction_mode, memory_decision_mode, search_mode, answer_mode].
# Populate this list before running the script.
CONFIGS: list[list[str]] = []

# Mapping for the two supported configuration shapes.
_MODE_KEYS = {
    2: ("search_mode", "answer_mode"),
    4: (
        "fact_extraction_mode",
        "memory_decision_mode",
        "search_mode",
        "answer_mode",
    ),
}


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def dump_config(config_path: Path, payload: dict) -> None:
    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def apply_modes(base_config: dict, values: Sequence[str]) -> dict:
    keys = _MODE_KEYS.get(len(values))
    if not keys:
        raise ValueError(
            f"Unsupported config length {len(values)}. Expected 2 or 4 values."
        )

    updated = deepcopy(base_config)
    exp_params = updated.get("exp_params")
    if not isinstance(exp_params, dict):
        raise KeyError("`exp_params` section missing from configuration file.")

    for key, value in zip(keys, values):
        if key not in exp_params:
            raise KeyError(f"`{key}` not found inside `exp_params`.")
        exp_params[key] = value

    return updated


def iter_configs(configs: Iterable[Sequence[str]]) -> Iterable[tuple[int, Sequence[str]]]:
    for idx, values in enumerate(configs, start=1):
        if len(values) not in _MODE_KEYS:
            raise ValueError(
                f"Config #{idx} has invalid length {len(values)}. Expected 2 or 4 values."
            )
        yield idx, values


def build_command(values: Sequence[str], config_path: Path) -> list[str]:
    command = ["python", "run_pipeline.py", "--config", str(config_path)]
    if len(values) == 2:
        command.extend(["--start_from_step", "2"])
    return command


def run_command(command: Sequence[str], workdir: Path) -> None:
    cmd_display = " ".join(command)
    print(f"Executing: {cmd_display}")
    subprocess.run(command, cwd=str(workdir), check=True)


def main() -> int:
    project_dir = Path(__file__).resolve().parent
    config_path = (project_dir / CONFIG_PATH).resolve()

    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        return 1

    if not CONFIGS:
        print("CONFIGS is empty. Populate it with the desired mode combinations.")
        return 0

    original_config = load_config(config_path)

    try:
        for idx, values in iter_configs(CONFIGS):
            mode_keys = _MODE_KEYS[len(values)]
            mode_repr = ", ".join(f"{k}={v}" for k, v in zip(mode_keys, values))
            print(f"\n[{idx}/{len(CONFIGS)}] Running with {mode_repr}")

            updated_config = apply_modes(original_config, values)
            dump_config(config_path, updated_config)
            command = build_command(values, config_path)
            run_command(command, project_dir)
    except subprocess.CalledProcessError as err:
        print(f"Command failed with exit code {err.returncode}", file=sys.stderr)
        return err.returncode
    finally:
        dump_config(config_path, original_config)

    print("\nAll configurations processed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
