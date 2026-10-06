# Certainty-Guided Reasoning (CGR)

A model-agnostic adaptive inference procedure that periodically probes whether
the current reasoning trace supports a confident final answer, and terminates
early once a target certainty threshold is reached. Certainty is the **minimum
predicted probability** across the decoded answer tokens (paper Eq. 3). CGR
preserves baseline accuracy while reducing token usage.

## Layout

- `certainty_guided_reasoning/` — core package:
  - `capture.py` — one trace per (seed, problem) with per-token logprobs.
  - `grid_runner.py` — Modal app that runs the capture grid cloud-side.
  - `modal_serve.py` — SGLang OpenAI-compatible endpoint on Modal (env-configurable).
  - `probe_sweep.py` — post-hoc checkpoint probes against the live endpoint.
  - `probe_sweep_modal.py` — detached Modal wrapper for the probe sweep.
  - `sync_traces.py` — pull traces from the Modal volume.
  - `paths.py` (run-folder layout) and `grader.py` (task registry + answer equality).
- `analysis/` — offline analysis over `outputs/{RUN}/probes/` and `grid_records.csv`:
  - `replay.py` — shared post-hoc CGR replay + calibration math.
  - `pipeline.py`, `theta_sweep.py`, `calibration.py`, `probe_analysis.py`,
    `answers_over_time.py`, `probe_mode_diff.py`, `exit_separation.py`,
    `holdout_seeds.py`, `recompute_clean65.py`, `seedfilter.py`.

## Setup

```bash
uv sync   # creates .venv with the deps in pyproject.toml
```

`sglang[all]` and `fastapi[standard]` are installed inside the Modal images, not locally.

## End-to-end (model of size N)

1. **Serve** — `uv run modal deploy certainty_guided_reasoning/modal_serve.py`
   (35B-A3B: `GPU=H200 MIN_CONTAINERS=4 MAX_CONTAINERS=4 MODEL_ID=Qwen/Qwen3.5-35B-A3B uv run modal deploy certainty_guided_reasoning/modal_serve.py`)
2. **Capture** — deploy `grid_runner.py`, then trigger a run with `--model`/`--seeds` (see its docstring).
3. **Sync** — `uv run python -m certainty_guided_reasoning.sync_traces --model <id> --task aime2025`
4. **Probe sweep** — `uv run python -m certainty_guided_reasoning.probe_sweep --endpoint <url>`
5. **Analyze** — see the map below.

## Reproducing the Qwen3.5 AIME2025 results

Run from repo root; `{RUN}` is `Qwen3.5-9B_aime2025` or `Qwen3.5-35B-A3B_aime2025`.
Seeds are 0–64 (65 seeds).

| Script | Output |
| --- | --- |
| `uv run python -m analysis.pipeline outputs/{RUN}/traces` | `grid_summary.md`, `grid_records.csv` |
| `uv run python -m analysis.recompute_clean65 {RUN}` | `figs_clean65/theta_sweep_min.txt`, `calib_report.md` |
| `uv run python -m analysis.theta_sweep outputs/{RUN}/probes --max-seed 64 --out outputs/{RUN}/figs_clean65` | `figc_theta_sweep.png`, `figc_grade_vs_theta.png` |
| `uv run python -m analysis.calibration outputs/{RUN}/probes --max-seed 64 --out outputs/{RUN}/figs_clean65` | `calib_reliability_*.png`, `calib_report.md` |
| `uv run python -m analysis.probe_analysis outputs/{RUN}/probes --max-seed 64 --out outputs/{RUN}/figs_clean65` | `figc_certainty_vs_step.png`, `figc_cumulative_answers.png`, `figc_cgr_sim.png`, `figc_grade_vs_budget.png` |
| `uv run python -m analysis.answers_over_time outputs/{RUN}/grid_records.csv --max-seed 64 --out outputs/{RUN}/figs_clean65 --name <stem>` | `<stem>_answers_over_time.png` |
| `uv run python -m analysis.probe_mode_diff {RUN}` | `greedy_vs_sampled.csv`, `greedy_vs_sampled.md` |
| `uv run python -m analysis.exit_separation {RUN}` | `figs_clean65/exit_separation.md` |
| `uv run python -m analysis.holdout_seeds {RUN}` | `figs_clean65/theta_holdout.md` |

The `outputs/` directory (traces, probes, figures) is gitignored and not shipped with the repo.
