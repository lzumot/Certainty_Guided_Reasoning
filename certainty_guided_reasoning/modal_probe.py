"""Cloud-side probe sweep: probe_sweep.sweep_trace on Modal.

Traces never leave the cloud — this reads them off a volume and writes probe
records to another. 8 shard containers x 256 workers = 2048 in-flight requests,
which is what makes the serve app provision 8 H100 replicas (Modal sizes
containers as in-flight / max_inputs).

    .venv/bin/python -m modal run certainty_guided_reasoning/modal_probe.py
"""

from __future__ import annotations

import io
import re
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import modal

MINUTES = 60
IN_MOUNT = "/traces"
SLIM_MOUNT = "/slim"
OUT_MOUNT = "/probes"

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
TASK = "gsm8k"

RUN_NAME = f"{MODEL_ID.split('/')[-1]}_{TASK}"
CLOUD_RUN = f"{MODEL_ID.replace('/', '--')}_{TASK}"

DEFAULT_ENDPOINT = "https://laithzumot--cgr-sglang-r1-serve.modal.run"

# Underscores make this exact: "_seed0_" cannot match "_seed10_". ".done"
# markers are excluded by the .json glob.
SEED_RE = re.compile(r"_seed(\d+)_")

# Same deps as grid_runner: the probe path imports grader.py.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "requests",
        "datasets",
        "huggingface_hub",
    )
    .add_local_python_source("certainty_guided_reasoning")
)

trace_vol = modal.Volume.from_name("cgr-traces")
# Slim (gzipped-source) session; "traces" is the default source below.
slim_vol = modal.Volume.from_name("cgr-slim", create_if_missing=True)
probe_vol = modal.Volume.from_name("cgr-probes", create_if_missing=True)

app = modal.App("cgr-cloud-probe")


@app.function(
    image=image,
    volumes={IN_MOUNT: trace_vol, SLIM_MOUNT: slim_vol, OUT_MOUNT: probe_vol},
    timeout=8 * 60 * MINUTES,
    scaledown_window=1 * MINUTES,
    # 256 threads; probe responses are tiny (max_new_tokens=64,
    # top_logprobs_num=1) so memory is dominated by the parsed traces.
    cpu=8,
    memory=16384,
)
def probe_shard(
    seed_lo: int,
    seed_hi: int,
    endpoint: str,
    model: str,
    task: str,
    interval: int,
    max_budget: int,
    workers: int,
    source: str,
    logprob_start_len: int,
) -> dict:
    """Probe one seed range. sweep_trace is used exactly as the local CLI does."""
    from certainty_guided_reasoning.grader import TASKS  # noqa: E402
    from certainty_guided_reasoning.probe_sweep import sweep_trace  # noqa: E402

    src = Path(SLIM_MOUNT if source == "slim" else IN_MOUNT) / CLOUD_RUN
    out = Path(OUT_MOUNT) / CLOUD_RUN
    out.mkdir(parents=True, exist_ok=True)

    max_ans = TASKS[task].probe_answer_tokens
    budgets = list(range(interval, max_budget + 1, interval))

    # Globbed per seed so the archive post-pass has the same split.
    paths = []
    for seed in range(seed_lo, seed_hi + 1):
        paths.extend(sorted(src.glob(f"*_seed{seed}_*.json")))

    print(f"[probe] seeds {seed_lo}-{seed_hi}: {len(paths)} traces, "
          f"{workers} workers, interval {interval} -> {out}", flush=True)
    t0 = time.monotonic()

    done = fail = skip = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(
                sweep_trace, endpoint, model, p, out, False, None, 0.0,
                task=task, max_ans_tokens=max_ans, budgets=budgets,
                logprob_start_len=logprob_start_len,
            )
            for p in paths
        ]
        for fut in as_completed(futs):
            # One bad trace must not kill the shard: sweep_trace raises only on
            # a tokenize failure. The error text MUST be logged — counting
            # failures silently made a 145-trace failure undiagnosable.
            try:
                msg = fut.result()
            except Exception as e:  # noqa: BLE001
                fail += 1
                print(f"[FAIL] {type(e).__name__}: {e}", flush=True)
            else:
                if msg.startswith("[skip]"):
                    skip += 1
                elif msg.startswith("[FAIL]"):
                    fail += 1
                    print(msg, flush=True)
                else:
                    done += 1
            if (done + fail + skip) % 25 == 0:
                print(f"[progress] done={done} fail={fail} skip={skip}",
                      flush=True)

    probe_vol.commit()
    return {
        "shard": f"seed{seed_lo:02d}-{seed_hi:02d}",
        "traces": len(paths),
        "done": done,
        "fail": fail,
        "skip": skip,
        "secs": time.monotonic() - t0,
    }


@app.function(
    image=image,
    volumes={OUT_MOUNT: probe_vol},
    timeout=4 * 60 * MINUTES,
    cpu=8,
    memory=8192,
)
def pack_shard(seed_lo: int, seed_hi: int, workers: int = 32) -> dict:
    """Archive one shard's probe records.

    A directory of 84k small files is slow to fetch one at a time, so the set
    ships as 8 tarballs instead. CPU-only, so it costs no GPU time. Threaded
    for the same reason slim_traces is: serial reads are ~149 ms/file against
    ~1 ms across 32 threads.
    """
    d = Path(OUT_MOUNT) / CLOUD_RUN
    tar_path = d / f"probes_seed{seed_lo:02d}-{seed_hi:02d}.tar.gz"
    t0 = time.monotonic()

    keep = range(seed_lo, seed_hi + 1)
    want = [
        p for p in d.glob("*.json")
        if (m := SEED_RE.search(p.name)) and int(m.group(1)) in keep
    ]

    lock = threading.Lock()
    n = 0
    with tarfile.open(tar_path, "w:gz") as tar:

        def one(p: Path) -> None:
            nonlocal n
            blob = p.read_bytes()
            # One tar stream: addfile must serialise.
            with lock:
                info = tarfile.TarInfo(p.name)
                info.size = len(blob)
                info.mtime = int(time.time())
                tar.addfile(info, io.BytesIO(blob))
                n += 1

        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(one, want))

    probe_vol.commit()
    return {
        "shard": f"seed{seed_lo:02d}-{seed_hi:02d}",
        "files": n,
        "mib": tar_path.stat().st_size / 1024**2,
        "secs": time.monotonic() - t0,
    }


@app.local_entrypoint()
def main(
    endpoint: str = DEFAULT_ENDPOINT,
    model: str = MODEL_ID,
    task: str = TASK,
    seeds: int = 64,
    shards: int = 8,
    workers: int = 256,
    interval: int = 200,
    max_budget: int = 150000,
    source: str = "traces",
    logprob_start_len: int = -1,
    pack: bool = True,
):
    # Balanced split so any shard count works (64/5 -> 13,13,13,13,12). A hard
    # cap on total containers means probers and serve replicas compete:
    # replicas = probers * workers / 256, so both come out of one budget.
    base, extra = divmod(seeds, shards)
    if base < 1:
        raise SystemExit(f"shards ({shards}) cannot exceed seeds ({seeds})")
    ranges, start = [], 0
    for i in range(shards):
        n = base + (1 if i < extra else 0)
        ranges.append((start, start + n - 1))
        start += n
    per = f"{base}-{base + 1}" if extra else str(base)

    inflight = shards * workers
    replicas = -(-inflight // 256)
    print(f"[probe] source={source} interval={interval} "
          f"{shards} shards x {per} seeds x {workers} workers "
          f"= {inflight} in-flight -> ~{replicas} serve replicas "
          f"= {shards + replicas} containers TOTAL")

    jobs = [
        (lo, hi, endpoint, model, task, interval, max_budget, workers,
         source, logprob_start_len)
        for lo, hi in ranges
    ]
    results = list(probe_shard.starmap(jobs))

    print(f"\n{'shard':<14} {'traces':>7} {'done':>7} {'fail':>6} "
          f"{'skip':>6} {'secs':>8}")
    for r in results:
        print(f"{r['shard']:<14} {r['traces']:>7} {r['done']:>7} {r['fail']:>6} "
              f"{r['skip']:>6} {r['secs']:>8.0f}")
    total = sum(r["done"] for r in results)
    print(f"{'TOTAL':<14} {sum(r['traces'] for r in results):>7} {total:>7} "
          f"{sum(r['fail'] for r in results):>6} "
          f"{sum(r['skip'] for r in results):>6}")

    if pack:
        print("\n[pack] archiving probe records...", flush=True)
        packed = list(pack_shard.starmap(ranges))
        print(f"{'shard':<14} {'files':>7} {'MiB':>8} {'secs':>8}")
        for r in packed:
            print(f"{r['shard']:<14} {r['files']:>7} {r['mib']:>8.1f} "
                  f"{r['secs']:>8.0f}")
        print(f"{'TOTAL':<14} {sum(r['files'] for r in packed):>7} "
              f"{sum(r['mib'] for r in packed):>8.1f}")

    print("\nDownload (one stream per shard):")
    print("  for s in 00-07 08-15 16-23 24-31 32-39 40-47 48-55 56-63; do")
    print("    .venv/bin/python -m modal volume get cgr-probes \\")
    print(f'      "{CLOUD_RUN}/probes_seed$s.tar.gz" \\')
    print(f'      "outputs/{RUN_NAME}/probes_seed$s.tar.gz"')
    print("  done")
