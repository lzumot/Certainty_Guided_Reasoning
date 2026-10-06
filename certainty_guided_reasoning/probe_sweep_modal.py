"""Detached Modal probe sweep (same logic as probe_sweep.py).

    uv run modal deploy certainty_guided_reasoning/probe_sweep_modal.py
"""

from __future__ import annotations

import modal
from pathlib import Path

MINUTES = 60
TRACE_MOUNT = "/traces"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("requests", "fastapi[standard]")
    .add_local_python_source("certainty_guided_reasoning")
)

trace_vol = modal.Volume.from_name("cgr-traces", create_if_missing=True)

app = modal.App("cgr-probe-sweep")


@app.function(
    image=image,
    volumes={TRACE_MOUNT: trace_vol},
    # must cover the full sweep (~50k probe calls).
    timeout=10 * 60 * MINUTES,
    scaledown_window=1 * MINUTES,
)
def run_probe_sweep(
    endpoint: str,
    model: str,
    run_slug: str,
    workers: int = 8,
):
    """Sweep every trace on the volume for this run; write probes back to volume."""
    from certainty_guided_reasoning.probe_sweep import sweep_trace
    from concurrent.futures import ThreadPoolExecutor, as_completed

    base = f"{model.replace('/', '--')}_{run_slug}"
    trace_dir = Path(TRACE_MOUNT) / base
    out_dir = Path(TRACE_MOUNT) / f"{base}-probes"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(trace_dir.glob("*.json"))
    print(f"[probe] {len(paths)} traces x <=32 checkpoints "
          f"-> {out_dir}", flush=True)

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(sweep_trace, endpoint, model, p, out_dir,
                          task=run_slug): p
                for p in paths}
        for fut in as_completed(futs):
            msg = fut.result()
            if msg.startswith("[done]"):
                done += 1
            if done % 25 == 0:
                print(f"[progress] done={done}", flush=True)
    print(f"[complete] done={done}/{len(paths)}", flush=True)


@app.local_entrypoint()
def main(
    endpoint: str,
    model: str = "Qwen/Qwen3.5-9B",
    run_slug: str = "aime2025",
    workers: int = 8,
):
    """Attached run (streams logs; dies with the laptop). Prefer the launch URL."""
    modal.enable_output()
    run_probe_sweep.remote(
        endpoint=endpoint, model=model, run_slug=run_slug, workers=workers,
    )


# Detached launcher: deploy, then POST a spec.
@app.function(image=image, volumes={TRACE_MOUNT: trace_vol})
@modal.fastapi_endpoint(method="POST", label="cgr-probe-launch")
def launch(spec: dict):
    handle = run_probe_sweep.spawn(
        endpoint=spec["endpoint"],
        model=spec["model"],
        run_slug=spec["run_slug"],
        workers=spec.get("workers", 8),
    )
    return {"started": True, "call_id": handle.object_id}
