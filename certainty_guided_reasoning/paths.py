"""Run-folder layout (single source of truth). Change MODEL_ID / TASK to retarget every script."""

from pathlib import Path

MODEL_ID = "Qwen/Qwen3.5-9B"
TASK = "aime2025"

RUN_NAME = f"{MODEL_ID.split('/')[-1]}_{TASK}"
CLOUD_RUN = f"{MODEL_ID.replace('/', '--')}_{TASK}"

RUN_ROOT = Path("outputs") / RUN_NAME
TRACES_DIR = RUN_ROOT / "traces"
PROBES_DIR = RUN_ROOT / "probes"
FIGS_DIR = RUN_ROOT / "figs"
GRID_SUMMARY = RUN_ROOT / "grid_summary.md"
GRID_RECORDS_CSV = RUN_ROOT / "grid_records.csv"
