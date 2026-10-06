"""Generate one reasoning trace with per-token logprobs via SGLang /generate.

OpenAI-compatible HTTP client against the Modal SGLang endpoint
(`certainty_guided_reasoning/modal_serve.py`).

Usage:
    python -m certainty_guided_reasoning.capture \
        --endpoint http://localhost:8000 --model Qwen/Qwen3.5-9B \
        --task aime2025 --seeds 1-64 --out outputs/Qwen3.5-9B_aime2025/traces
"""

from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from certainty_guided_reasoning.paths import TRACES_DIR
from certainty_guided_reasoning.grader import TASKS, process_answer_for


def _load_dotenv() -> None:
    """Minimal .env loader (no dependency). Reads repo-root .env if present."""
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()

# Backwards-compatible view of grader.TASKS: task -> (dataset_id, loader).
DATASETS = {
    key: (cfg.dataset_id, lambda a, key=key: process_answer_for(key, a))
    for key, cfg in TASKS.items()
}

PROMPT_SUFFIX = (
    " Please reason step by step, and put your final answer within \\boxed{}."
)

INVALID_CERTAINTY = -1.0

# Bare assistant turn, no system block (prefilled '<think>' is ignored).
# An earlier draft had 'nn' instead of '\\n\\n'; never reached baseline.
def create_prompt(question: str) -> str:
    return (
        f"<|im_start|>user\n{question}{PROMPT_SUFFIX}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def parse_seeds(spec: str) -> list[int]:
    """'1-64' -> [1..64] inclusive. Avoids the range() off-by-one defect."""
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",")]


# ---------------------------------------------------------------------------
# Generation via OpenAI-compatible endpoint
# ---------------------------------------------------------------------------
def generate_trace(
    endpoint: str,
    model: str,
    prompt: str,
    seed: int,
    max_tokens: int = 32768,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    presence_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
    min_p: float = 0.0,
    retries: int = 3,
) -> dict:
    """POST SGLang native /generate; return {text, tokens, logprobs}."""
    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            # SGLang: 'sampling_seed' ('seed'/'random_seed' are rejected).
            "sampling_seed": seed,
            "presence_penalty": presence_penalty,
            "repetition_penalty": repetition_penalty,
            "min_p": min_p,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "top_logprobs_num": 10,
        "return_text_in_logprobs": True,
    }
    last_err = None
    headers = {}  # endpoint runs without --api-key
    for attempt in range(retries):
        try:
            r = requests.post(
                f"{endpoint.rstrip('/')}/generate",
                json=payload,
                headers=headers,
                timeout=3600,  # full 32k-token generation
            )
            r.raise_for_status()
            data = r.json()
            meta = data.get("meta_info") or {}
            # output_token_logprobs: [logprob, token_id, decoded_text]
            out_lps = meta.get("output_token_logprobs") or []
            tokens = [t[2] if len(t) > 2 else "" for t in out_lps]
            logprobs = [t[0] for t in out_lps]
            # Top-k: [logprob, token_id, decoded_text], sorted descending.
            top_logprobs = meta.get("output_top_logprobs") or []
            return {
                "text": data.get("text", ""),
                "tokens": tokens,
                "token_ids": [t[1] for t in out_lps],
                "logprobs": logprobs,
                "top_logprobs": top_logprobs,
                "finish_reason": meta.get("finish_reason", {}).get("type")
                if isinstance(meta.get("finish_reason"), dict)
                else meta.get("finish_reason"),
            }
        except Exception as e:  # noqa: BLE001 — retry any transient failure
            last_err = e
            time.sleep(2**attempt * 5)
    raise RuntimeError(f"generation failed after {retries} attempts: {last_err}")


# Certainty (paper Eq. 3): exp(min logprob over the boxed answer window), token-space.
def _boxed_open_from(tokens: list[str], start: int) -> int | None:
    """Index of the '{' opening the \\boxed{...} whose 'boxed' token is at
    index `start`, or None if malformed. Split-token safe: the tokenizer may
    emit \\boxed{ as one token ('boxed{'), or split it ('boxed', '{') or
    ('\\boxed', '{')."""
    j = start if "{" in tokens[start] else None
    k = start
    while j is None and k < len(tokens) - 1:
        k += 1
        if "{" in tokens[k]:
            j = k
            break
        if "}" in tokens[k] or "boxed" in tokens[k]:
            break  # malformed; treat as no window
    return j


def _boxed_window_from(tokens: list[str], start: int) -> tuple[int, int] | None:
    """Brace-depth-aware (open_idx, close_idx) of the \\boxed{...} whose
    'boxed' token is at index `start`. Tracks brace nesting so an answer with
    nested braces is captured in full rather than cut at the first '}'.
    None if the window never closes."""
    j = _boxed_open_from(tokens, start)
    if j is None:
        return None
    depth = 0
    for idx in range(j, len(tokens)):
        depth += tokens[idx].count("{") - tokens[idx].count("}")
        if idx == j and depth <= 0:
            return None  # closed within the opening token -> not extractable
        if depth <= 0:
            return j, idx
    return None


def _boxed_bounds(tokens: list[str]) -> tuple[int, int] | None:
    """Brace-depth-aware (open, close) of the FIRST complete \\boxed{...}."""
    for i, t in enumerate(tokens):
        if "boxed" in t:
            b = _boxed_window_from(tokens, i)
            if b is not None:
                return b
    return None


def select_answer_window(
    tokens: list[str], logprobs: list[float]
) -> tuple[list[str], list[float]]:
    """Return (window_tokens, window_logprobs); raises ValueError on failure."""
    if not tokens or not logprobs or len(tokens) != len(logprobs):
        raise ValueError("empty or misaligned tokens/logprobs")
    bounds = _boxed_bounds(tokens)
    if bounds is None:
        raise ValueError("no complete 'boxed{' window in trace tokens")
    start, end = bounds
    if end <= start:
        raise ValueError("empty answer window")
    return list(tokens[start + 1 : end]), list(logprobs[start + 1 : end])


def min_token_prob(window_logprobs: list[float]) -> float:
    if not window_logprobs:
        raise ValueError("empty answer window — cannot compute certainty")
    try:
        return math.exp(min(window_logprobs))
    except (ValueError, OverflowError):
        return INVALID_CERTAINTY


def get_certainty(trace: dict) -> float:
    """Certainty via the answer window's min sampled-token logprob."""
    _, window_lps = select_answer_window(trace["tokens"], trace["logprobs"])
    return min_token_prob(window_lps)


# Two certainties per record: `certainty` (sampled-token) and
# `certainty_greedy` (top-k argmax; the paper's Eq. 3, headline metric).
def get_certainty_greedy(trace: dict) -> float:
    """exp(min over answer window of log P(argmax token)) via top-k logprobs."""
    # Locate window start the same way select_answer_window does (split-token
    # safe). Returns (start_index_of_open_brace, end_index_of_close_brace).
    top = trace.get("top_logprobs") or []
    if len(top) != len(trace["tokens"]):
        raise ValueError("top_logprobs missing or misaligned with tokens")
    bounds = _boxed_bounds(trace["tokens"])
    if bounds is None:
        raise ValueError("no complete 'boxed{' window in trace tokens")
    start, end = bounds
    if end <= start:
        raise ValueError("empty answer window")
    max_lps = []
    for i in range(start + 1, end):
        entries = top[i] or []
        if not entries:
            raise ValueError(f"empty top_logprobs at position {i}")
        # entries: [logprob, token_id, decoded_text]; sorted descending
        max_lps.append(entries[0][0])
    if not max_lps:
        raise ValueError("empty answer window — cannot compute certainty")
    try:
        return math.exp(min(max_lps))
    except (ValueError, OverflowError):
        return INVALID_CERTAINTY


def count_thinking_tokens(tokens: list[str]) -> int:
    """Tokens before the closing   </think> (baseline thinking-token count).

    If   </think> never appears (loop trace), all tokens count as thinking.
    """
    for i, t in enumerate(tokens):
        if "    </think> " in t:
            return i + 1
    return len(tokens)


def extract_answer(tokens: list[str]) -> str | None:
    """Extract the last \\boxed{...} window (brace-aware, split-token safe)."""
    boxed_idx = [i for i, t in enumerate(tokens) if "boxed" in t]
    for start in reversed(boxed_idx):
        b = _boxed_window_from(tokens, start)
        if b is None:
            continue
        open_idx, close_idx = b
        if close_idx <= open_idx:
            continue
        body = "".join(tokens[open_idx + 1 : close_idx])
        return body.strip()
    return None


# ---------------------------------------------------------------------------
# Resumable writer: atomic write + .done marker per (model, seed, problem)
# ---------------------------------------------------------------------------
def cache_key(model: str, seed: int, problem_id: str) -> str:
    safe_model = model.replace("/", "--")
    return f"{safe_model}_seed{seed}_{problem_id}"


def write_record(out_dir: Path, key: str, record: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f".{key}.json.tmp"
    final = out_dir / f"{key}.json"
    tmp.write_text(json.dumps(record))
    os.replace(tmp, final)  # atomic
    (out_dir / f"{key}.done").write_text("")


def is_done(out_dir: Path, key: str) -> bool:
    return (out_dir / f"{key}.done").exists()


# ---------------------------------------------------------------------------
# Main capture loop (concurrent: thread pool over (seed, problem) pairs)
# ---------------------------------------------------------------------------
_print_lock = threading.Lock()


def _safe_print(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def capture_one(args, out_dir: Path, seed: int, i: int, problem: str,
                answer: str) -> str:
    """Generate + save one trace. Returns a status string; never raises
    (failures are recorded and printed so the grid keeps running)."""
    pid = f"{args.task}_{i:04d}"
    key = cache_key(args.model, seed, pid)
    if is_done(out_dir, key):
        return f"[skip] {key}"
    prompt = create_prompt(problem)
    try:
        trace = generate_trace(
            args.endpoint, args.model, prompt, seed,
            max_tokens=args.max_tokens, temperature=args.temperature,
            top_p=args.top_p, top_k=args.top_k,
            presence_penalty=args.presence_penalty,
        )
    except Exception as e:  # noqa: BLE001 — record failure, keep grid alive
        return f"[FAIL] {key}: {e}"

    # Loop-trace accounting: traces that hit max_tokens without emitting
    # \boxed{} get certainty=None — never crash the grid run. These are
    # CGR's best-case savings; count them.
    try:
        certainty = get_certainty(trace)
        certainty_greedy = get_certainty_greedy(trace)
    except ValueError:
        certainty = None
        certainty_greedy = None
    record = {
        "key": key,
        "model": args.model,
        "seed": seed,
        "problem_id": pid,
        "task": args.task,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "presence_penalty": args.presence_penalty,
        "prompt": prompt,
        **trace,
        "certainty": certainty,
        "certainty_greedy": certainty_greedy,
        "num_thinking_tokens": count_thinking_tokens(trace["tokens"]),
        "num_total_tokens": len(trace["tokens"]),
        "extracted_answer": extract_answer(trace["tokens"]),
        "ground_truth": answer,
    }
    write_record(out_dir, key, record)
    cert_str = (
        f"certainty={record['certainty']:.4f} "
        f"greedy={record['certainty_greedy']:.4f}"
        if record["certainty"] is not None
        else "certainty=None (no boxed answer)"
    )
    return (f"[done] {key} {cert_str} "
            f"answer={record['extracted_answer']} truth={answer}")


def main() -> None:
    p = argparse.ArgumentParser(description="Capture reasoning traces via Modal SGLang")
    p.add_argument("--endpoint", required=True)
    p.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--task", default="aime2025", choices=sorted(DATASETS))
    p.add_argument("--seeds", default="0-64", help="'0-64' inclusive (65 runs) or '1,2,3'")
    p.add_argument("--problems", type=int, default=None, help="limit to first N problems")
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--temperature", type=float, default=0.6,
                   help="Qwen3 thinking-mode recommended: 0.6")
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    # >0 reduces thinking loops but can cause language mixing.
    p.add_argument("--presence-penalty", type=float, default=0.0)
    # Keep workers <= 32 (endpoint @modal.concurrent cap); reduce for larger models.
    p.add_argument("--workers", type=int, default=4,
                   help="parallel requests in flight (endpoint caps at 32)")
    p.add_argument("--out", default=str(TRACES_DIR))
    args = p.parse_args()

    from datasets import load_dataset  # local import: only needed at run time

    dataset_name, process_answer = DATASETS[args.task]
    ds = load_dataset(dataset_name)["train"]
    problems = ds["problem"][: args.problems] if args.problems else ds["problem"]
    answers = [process_answer(a) for a in ds["answer"][: len(problems)]]

    seeds = parse_seeds(args.seeds)
    out_dir = Path(args.out)

    jobs = [(seed, i) for seed in seeds for i in range(len(problems))]
    print(f"[grid] {len(jobs)} jobs, {args.workers} workers", flush=True)

    n_done = n_fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(capture_one, args, out_dir, seed, i, problems[i], answers[i]): (seed, i)
            for seed, i in jobs
        }
        for fut in as_completed(futures):
            msg = fut.result()  # capture_one never raises
            if "[FAIL]" in msg:
                n_fail += 1
            elif "[done]" in msg:
                n_done += 1
            _safe_print(f"{msg}  ({n_done} done, {n_fail} failed)")


if __name__ == "__main__":
    main()
