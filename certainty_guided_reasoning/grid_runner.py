"""Cloud-side grid runner: capture traces entirely on Modal.

Survives laptop shutdown — the capture loop runs as a Modal function writing
to a Modal Volume. Monitor/collect with `modal volume ls` or the download
command in README.

Usage:
    uv run modal deploy certainty_guided_reasoning/grid_runner.py
    # attached run (streams logs; dies with laptop):
    uv run modal run certainty_guided_reasoning/grid_runner.py \
        --endpoint <url> --model Qwen/Qwen3.5-9B --task aime2025 --seeds 0-64
    # detached run: POST to the deployed launch URL (see launch() below).
"""

from __future__ import annotations

import modal

MINUTES = 60
CACHE_MOUNT = "/cache"
OUT_MOUNT = "/traces"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "requests",
        "datasets",
        "huggingface_hub",
        "fastapi[standard]",
    )
    .env({"HF_HOME": f"{CACHE_MOUNT}/huggingface"})
    # Bake the package into the image so the remote function can import it.
    .add_local_python_source("certainty_guided_reasoning")
)

trace_vol = modal.Volume.from_name("cgr-traces", create_if_missing=True)
hf_cache = modal.Volume.from_name("cgr-hf-cache", create_if_missing=True)

app = modal.App("cgr-grid-runner")


@app.function(
    image=image,
    volumes={OUT_MOUNT: trace_vol, CACHE_MOUNT: hf_cache},
    secrets=[modal.Secret.from_name("hf-token")],
    # Whole-container timeout: must exceed the longest full-grid run
    # (24h covers 1950 jobs with retry margin).
    timeout=24 * 60 * MINUTES,
    scaledown_window=1 * MINUTES,
)
def run_grid(
    endpoint: str,
    model: str,
    task: str,
    seeds_spec: str,
    max_problems: int | None,
    workers: int,
    tag: str = "",
    max_tokens: int = 32768,
    temperature: float = 0.6,
    presence_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
    min_p: float = 0.0,
):
    """Run the capture loop cloud-side. Imports the same logic as local runs."""
    from pathlib import Path

    # The image bakes `certainty_guided_reasoning` as a top-level package, so
    # import directly (NOT via `scripts.` — that fails in the container with
    # "No module named 'scripts'").
    from certainty_guided_reasoning.capture import (  # noqa: E402
        cache_key,
        count_thinking_tokens,
        create_prompt,
        extract_answer,
        generate_trace,
        get_certainty,
        get_certainty_greedy,
        is_done,
        parse_seeds,
        write_record,
        DATASETS,
    )
    from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: E402
    from datasets import load_dataset  # noqa: E402

    out_dir = Path(OUT_MOUNT) / f"{model.replace('/', '--')}_{task}{'_' + tag if tag else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_name, process_answer = DATASETS[task]
    ds = load_dataset(dataset_name)["train"]
    problems = (
        ds["problem"][:max_problems] if max_problems else ds["problem"]
    )
    answers = [process_answer(a) for a in ds["answer"][: len(problems)]]
    seeds = parse_seeds(seeds_spec)

    jobs = [(s, i) for s in seeds for i in range(len(problems))]
    print(f"[grid] {len(jobs)} jobs, {workers} workers -> {out_dir}", flush=True)
    import time
    t0 = time.monotonic()

    def one(seed: int, i: int) -> str:
        pid = f"{task}_{i:04d}"
        key = cache_key(model, seed, pid)
        if is_done(out_dir, key):
            return f"[skip] {key}"
        prompt = create_prompt(problems[i])
        try:
            trace = generate_trace(
                endpoint, model, prompt, seed,
                max_tokens=max_tokens, temperature=temperature,
                presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty, min_p=min_p,
            )
        except Exception as e:  # noqa: BLE001
            return f"[FAIL] {key}: {e}"
        try:
            cert = get_certainty(trace)
            cert_g = get_certainty_greedy(trace)
        except ValueError:
            cert = cert_g = None
        record = {
            "key": key, "model": model, "seed": seed, "problem_id": pid,
            "task": task, "prompt": prompt, **trace,
            "certainty": cert, "certainty_greedy": cert_g,
            "num_thinking_tokens": count_thinking_tokens(trace["tokens"]),
            "num_total_tokens": len(trace["tokens"]),
            "extracted_answer": extract_answer(trace["tokens"]),
            "ground_truth": answers[i],
        }
        write_record(out_dir, key, record)
        return f"[done] {key} certainty={cert}"

    n_done = n_fail = n_skip = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, s, i) for s, i in jobs]
        for fut in as_completed(futs):
            msg = fut.result()
            if "[FAIL]" in msg:
                n_fail += 1
            elif "[skip]" in msg:
                n_skip += 1
            else:
                n_done += 1
            if (n_done + n_fail + n_skip) % 25 == 0:
                print(f"[progress] done={n_done} fail={n_fail} skip={n_skip}",
                      flush=True)
    print(f"[complete] done={n_done} fail={n_fail} skip={n_skip} "
          f"elapsed={time.monotonic() - t0:.1f}s", flush=True)


@app.local_entrypoint()
def main(
    endpoint: str,
    model: str = "Qwen/Qwen3.5-9B",
    task: str = "aime2025",
    seeds: str = "0-64",
    problems: int | None = None,
    workers: int = 16,
    tag: str = "",
    max_tokens: int = 32768,
    temperature: float = 0.6,
    presence_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
    min_p: float = 0.0,
):
    """Attached run: streams logs to this terminal. If the terminal dies,
    the run dies — but .done markers make rerunning resume cleanly."""
    modal.enable_output()
    run_grid.remote(
        endpoint=endpoint, model=model, task=task,
        seeds_spec=seeds, max_problems=problems, workers=workers,
        tag=tag, max_tokens=max_tokens, temperature=temperature,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty, min_p=min_p,
    )


# Detached launcher (shutdown-proof): deploy, then POST a spec to the launch
# URL to spawn a run that outlives the laptop.
@app.function(image=image, volumes={OUT_MOUNT: trace_vol})
@modal.fastapi_endpoint(method="POST", label="cgr-launch")
def launch(spec: dict):
    handle = run_grid.spawn(
        endpoint=spec["endpoint"],
        model=spec["model"],
        task=spec["task"],
        seeds_spec=spec["seeds"],
        max_problems=spec.get("problems"),
        workers=spec.get("workers", 16),
        tag=spec.get("tag", ""),
        max_tokens=spec.get("max_tokens", 32768),
        temperature=spec.get("temperature", 0.6),
        presence_penalty=spec.get("presence_penalty", 0.0),
        repetition_penalty=spec.get("repetition_penalty", 1.0),
        min_p=spec.get("min_p", 0.0),
    )
    return {"started": True, "call_id": handle.object_id}
