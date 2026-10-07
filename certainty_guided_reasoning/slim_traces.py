"""Probe-only slim traces: keep the 12 fields probe_sweep reads.

`top_logprobs` is 91% of every trace file, so stripping it shrinks 66 GiB to
~2.1 GiB — small enough to hold locally. Volume-to-volume: the fat traces are
read cloud-side and never cross the wire.

    .venv/bin/python -m modal volume create cgr-slim
    .venv/bin/python -m modal run certainty_guided_reasoning/slim_traces.py
"""

from __future__ import annotations

import io
import json
import re
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

MINUTES = 60
IN_MOUNT = "/traces"
OUT_MOUNT = "/slim"

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
TASK = "gsm8k"

RUN_NAME = f"{MODEL_ID.split('/')[-1]}_{TASK}"
CLOUD_RUN = f"{MODEL_ID.replace('/', '--')}_{TASK}"

# Exactly the fields probe_sweep.py reads, plus two free diagnostics so the
# slim traces stay self-describing for verification.
KEEP = (
    "key", "model", "seed", "problem_id", "task",
    "prompt", "token_ids", "tokens", "text",
    "ground_truth", "extracted_answer", "finish_reason",
)

# Underscores make this exact: "_seed0_" cannot match "_seed10_". ".done"
# markers are excluded by the .json glob.
SEED_RE = re.compile(r"_seed(\d+)_")

# Stdlib only (json/tarfile): no pip installs, fast cold start.
image = modal.Image.debian_slim(python_version="3.12")

in_vol = modal.Volume.from_name("cgr-traces", create_if_missing=True)
out_vol = modal.Volume.from_name("cgr-slim", create_if_missing=True)

app = modal.App("cgr-slim-traces")


@app.function(
    image=image,
    volumes={IN_MOUNT: in_vol, OUT_MOUNT: out_vol},
    timeout=30 * MINUTES,
    cpu=8,
    memory=16384,
)
def slim_shard(seed_lo: int, seed_hi: int, workers: int = 32) -> dict:
    """Rewrite one seed range: fat traces in, slim JSON + one .tar.gz out.

    Threaded because the serial loop was latency-bound: ~149 ms/file single
    threaded vs ~1.1 ms/file across 32 threads on this volume. JSON parsing is
    the floor (~33 ms/file) and is GIL-bound, so it scales only with shards.
    """
    src = Path(IN_MOUNT) / CLOUD_RUN
    dst = Path(OUT_MOUNT) / CLOUD_RUN
    dst.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    keep = range(seed_lo, seed_hi + 1)
    # One directory scan, then partition in Python: globbing per seed would
    # re-walk a 168k-entry directory once per seed (~3s each).
    want = [
        p for p in src.glob("*.json")
        if (m := SEED_RE.search(p.name)) and int(m.group(1)) in keep
    ]

    tar_path = dst / f"seed{seed_lo:02d}-{seed_hi:02d}.tar.gz"
    lock = threading.Lock()
    stats = {"n": 0, "raw": 0, "slim": 0, "reused": 0}

    with tarfile.open(tar_path, "w:gz") as tar:

        def one(p: Path) -> None:
            out_path = dst / p.name
            reused = out_path.exists()
            if reused:
                # Already built (e.g. this build was restarted): the slim file
                # is all the tar needs, so skip the fat read and the parse.
                blob = out_path.read_bytes()
                raw = 0
            else:
                raw = p.stat().st_size
                rec = json.loads(p.read_bytes())
                blob = json.dumps({k: rec[k] for k in KEEP if k in rec},
                                  separators=(",", ":")).encode()
                out_path.write_bytes(blob)
            # One tar stream, so addfile must serialise. Feeding it from memory
            # avoids re-reading the file we just wrote.
            with lock:
                info = tarfile.TarInfo(p.name)
                info.size = len(blob)
                info.mtime = int(time.time())
                tar.addfile(info, io.BytesIO(blob))
                stats["n"] += 1
                stats["slim"] += len(blob)
                stats["reused"] += reused
                stats["raw"] += raw

        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(one, want))

    out_vol.commit()
    return {
        "shard": f"seed{seed_lo:02d}-{seed_hi:02d}",
        "files": stats["n"],
        "reused": stats["reused"],
        "raw_gib": stats["raw"] / 1024**3,
        "slim_gib": stats["slim"] / 1024**3,
        "tar_mib": tar_path.stat().st_size / 1024**2,
        "secs": time.monotonic() - t0,
    }


@app.local_entrypoint()
def main(shards: int = 16, seeds: int = 64, workers: int = 32):
    base, extra = divmod(seeds, shards)
    if base < 1:
        raise SystemExit(f"shards ({shards}) cannot exceed seeds ({seeds})")
    ranges, start = [], 0
    for i in range(shards):
        n = base + (1 if i < extra else 0)
        ranges.append((start, start + n - 1))
        start += n
    per = f"{base}-{base + 1}" if extra else str(base)

    print(f"[slim] {shards} shards x {per} seeds, {workers} threads each "
          f"-> {CLOUD_RUN}", flush=True)
    results = list(slim_shard.starmap(
        [(lo, hi, workers) for lo, hi in ranges]))

    files = sum(r["files"] for r in results)
    raw = sum(r["raw_gib"] for r in results)
    slim = sum(r["slim_gib"] for r in results)
    reused = sum(r["reused"] for r in results)
    slowest = max(r["secs"] for r in results)
    print(f"\n{'shard':<14} {'files':>7} {'reused':>7} {'raw GiB':>9} "
          f"{'slim GiB':>9} {'tar MiB':>9} {'secs':>7}")
    for r in results:
        print(f"{r['shard']:<14} {r['files']:>7} {r['reused']:>7} "
              f"{r['raw_gib']:>9.2f} {r['slim_gib']:>9.2f} "
              f"{r['tar_mib']:>9.1f} {r['secs']:>7.1f}")
    print(f"{'TOTAL':<14} {files:>7} {reused:>7} {raw:>9.2f} {slim:>9.2f} "
          f"   reduction {raw / slim:.1f}x   wall {slowest:.0f}s")

    print("\nDownload (one stream per shard, then unpack):")
    print("  for s in 00-03 04-07 08-11 12-15 16-19 20-23 24-27 28-31 \\")
    print("           32-35 36-39 40-43 44-47 48-51 52-55 56-59 60-63; do")
    print("    .venv/bin/python -m modal volume get cgr-slim \\")
    print(f'      "{CLOUD_RUN}/seed$s.tar.gz" \\')
    print(f'      "outputs/{RUN_NAME}/traces/seed$s.tar.gz"')
    print("  done")
    print(f"  cd outputs/{RUN_NAME}/traces && "
          "for s in *.tar.gz; do tar -xzf \"$s\"; done && rm -f *.tar.gz")
