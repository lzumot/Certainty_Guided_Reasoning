"""Sync grid traces from the Modal volume to the local run dir.

    uv run python -m certainty_guided_reasoning.sync_traces [--watch 300]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

from certainty_guided_reasoning.paths import MODEL_ID, TASK

VOLUME = "cgr-traces"


def _resolve(model: str, task: str, tag: str = "") -> tuple[str, str]:
    """Remote volume dir + local traces dir for a (model, task[, tag]) run."""
    remote = f"{model.replace('/', '--')}_{task}{'_' + tag if tag else ''}"
    local = f"outputs/{model.split('/')[-1]}_{task}{'_' + tag if tag else ''}/traces"
    return remote, local


REMOTE_DIR, LOCAL_DIR = _resolve(MODEL_ID, TASK)


def _modal() -> list[str]:
    """Modal CLI as a command list.

    Falls back to `<this python> -m modal`: a venv's script carries a shebang
    with the path it was created under, which breaks if the repo is moved.
    """
    exe = shutil.which("modal")
    return [exe] if exe else [sys.executable, "-m", "modal"]


def list_remote() -> list[str]:
    out = subprocess.run(
        [*_modal(), "volume", "ls", VOLUME, REMOTE_DIR],
        capture_output=True, text=True, check=True,
    ).stdout
    return [
        line.strip().split("/")[-1]
        for line in out.splitlines()
        if line.strip().endswith(".json")
    ]


def fetch(name: str) -> bool:
    r = subprocess.run(
        [*_modal(), "volume", "get", VOLUME,
         f"{REMOTE_DIR}/{name}", LOCAL_DIR + "/"],
        capture_output=True, text=True,
    )
    return r.returncode == 0


def sync() -> tuple[int, int]:
    from pathlib import Path
    Path(LOCAL_DIR).mkdir(parents=True, exist_ok=True)
    local = {p.name for p in Path(LOCAL_DIR).glob("*.json")}
    remote = list_remote()
    new = [n for n in remote if n not in local]
    ok = sum(1 for n in new if fetch(n))
    return ok, len(new) - ok


def main() -> None:
    global REMOTE_DIR, LOCAL_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=int, default=0,
                    help="poll interval in seconds; 0 = one-shot")
    ap.add_argument("--model", default=MODEL_ID,
                    help="HF model id, e.g. Qwen/Qwen3.5-35B-A3B "
                         "(default: paths.py MODEL_ID)")
    ap.add_argument("--task", default=TASK,
                    help="task name, e.g. aime2025 (default: paths.py TASK)")
    ap.add_argument("--tag", default="",
                    help="optional run tag appended to remote/local dir names")
    args = ap.parse_args()

    REMOTE_DIR, LOCAL_DIR = _resolve(args.model, args.task, args.tag)
    print(f"[sync] {REMOTE_DIR} -> {LOCAL_DIR}", flush=True)

    while True:
        got, failed = sync()
        print(f"[sync] downloaded={got} failed={failed}", flush=True)
        if not args.watch:
            sys.exit(0 if failed == 0 else 1)
        import time
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
