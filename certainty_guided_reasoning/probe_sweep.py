"""Post-hoc checkpoint probes: cut each saved trace at interval-token budgets,
decode the answer under the probe prefix, record (answer, certainty) per budget.

    uv run python -m certainty_guided_reasoning.probe_sweep \
        --endpoint <url> --trace-dir outputs/Qwen3.5-9B_aime2025/traces --limit 2
    # GSM8K: --interval 200
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from certainty_guided_reasoning.paths import PROBES_DIR, TRACES_DIR
from certainty_guided_reasoning.grader import TASKS, infer_task
from certainty_guided_reasoning.capture import think_end_index

PROBE_PREFIX = "\n\nFinal Answer: \\boxed{"  # legacy: Qwen3.5 (no </think>)
BUDGETS = [1000 * i for i in range(1, 34)]  # 1k..33k (AIME grid)
OUT_DIR = PROBES_DIR


def probe_prefix_for(model: str) -> str:
    """Probe prefix for a model family.

    R1-Distill and QwQ prefill '<think>', so a probe cut mid-thinking must
    close the block before asking for an answer.
    """
    m = (model or "").lower()
    if "deepseek-r1" in m or "qwq" in m:
        return "</think>\n\nFinal answer: \\boxed{"
    return PROBE_PREFIX


def _probe_window(tokens: list[str]) -> int:
    """Index closing the probe answer window (brace-aware)."""
    depth = 1  # the '\\boxed{' from PROBE_PREFIX is already open
    for i, t in enumerate(tokens):
        depth += t.count("{") - t.count("}")
        if depth <= 0:
            return i  # exclude the closing '}'
    return len(tokens)


def sampling_block(args, max_ans_tokens: int) -> dict:
    """Sampling params for all probe calls (greedy if temperature=0)."""
    block = {
        "max_new_tokens": max_ans_tokens,
        "temperature": args.temperature,
        "skip_special_tokens": False,
    }
    if args.temperature > 0:
        block["top_p"] = args.top_p
        if args.sampling_seed is not None:
            # SGLang names the per-request seed 'sampling_seed'.
            block["sampling_seed"] = args.sampling_seed
    return block


def probe_once(endpoint: str, model: str, prompt_text: str,
               max_ans_tokens: int = 8) -> dict | None:
    """One probe call: greedy decode after prefix; return answer + certainty."""
    payload = {
        "text": prompt_text,
        "sampling_params": {
            "max_new_tokens": max_ans_tokens,
            "temperature": 0.0,  # greedy (paper: greedy decode under prefix)
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "top_logprobs_num": 1,
        "return_text_in_logprobs": True,
    }
    for attempt in range(3):
        try:
            r = requests.post(f"{endpoint.rstrip('/')}/generate", json=payload,
                              timeout=600)
            r.raise_for_status()
            data = r.json()
            meta = data.get("meta_info") or {}
            out_lps = meta.get("output_token_logprobs") or []
            # [logprob, token_id, decoded_text]
            tokens = [t[2] if len(t) > 2 else "" for t in out_lps]
            lps = [t[0] for t in out_lps]
            # window = tokens before the outer closing '}' (brace-aware)
            k = _probe_window(tokens)
            if not lps or k == 0:
                return {"answer": "".join(tokens).split("}")[0],
                        "certainty": None}
            answer = "".join(tokens[:k])
            import math
            return {"answer": answer, "certainty": math.exp(min(lps[:k]))}
        except Exception:
            time.sleep(2 ** attempt * 5)
    return None


def sweep_trace(endpoint: str, model: str, path: Path, out_dir: Path,
                force: bool = False, sampling_params: dict | None = None,
                temperature: float = 0.0, *,
                task: str | None = None,
                max_ans_tokens: int | None = None,
                budgets: list[int] | None = None,
                logprob_start_len: int = -1) -> str:
    rec = json.loads(path.read_text())
    key = rec["key"]
    if task is None:
        task = rec.get("task") or infer_task(
            rec.get("problem_id", "") or str(path)
        )
    if max_ans_tokens is None:
        max_ans_tokens = TASKS[task].probe_answer_tokens
    out_path = out_dir / f"{key}.json"
    if out_path.exists() and not force:
        return f"[skip] {key}"
    if sampling_params is None:
        sampling_params = {
            "max_new_tokens": max_ans_tokens,
            "temperature": 0.0,
            "skip_special_tokens": False,
        }

    def tokenize(text: str) -> list[int]:
        # Retried: one 408 from /tokenize on a long prompt must not abort the sweep.
        last_err = None
        for attempt in range(3):
            try:
                r = requests.post(f"{endpoint.rstrip('/')}/tokenize",
                                  json={"prompt": text}, timeout=(10, 120))
                r.raise_for_status()
                return r.json()["tokens"]
            except Exception as e:  # noqa: BLE001 — retry any transient failure
                last_err = e
                time.sleep(2 ** attempt * 5)
        raise RuntimeError(f"/tokenize failed after 3 attempts: {last_err}")

    probe_ids = tokenize(probe_prefix_for(rec["model"]))
    probes = []
    # Replay the same sequence the model saw: rendered prompt + trace ids, so
    # cutting the trace does not drop the question.
    prompt_ids = tokenize(rec["prompt"]) if rec.get("prompt") else []
    stored_ids = rec.get("token_ids")
    try:
        if stored_ids:
            all_ids = prompt_ids + list(stored_ids)
        else:
            all_ids = prompt_ids + tokenize(rec["text"])
    except Exception as e:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"key": key,
                                        "error": f"tokenize failed: {e}"}))
        return f"[FAIL] {key}: tokenize failed"
    import math
    budgets = BUDGETS if budgets is None else budgets
    stored_toks = rec.get("tokens") or []
    think_end = think_end_index(stored_toks) if stored_toks else len(all_ids)
    # /tokenize prepends BOS; all_ids[0] is already the prompt's BOS.
    if probe_ids and all_ids and probe_ids[0] == all_ids[0]:
        probe_ids = probe_ids[1:]
    # Probe only inside the thinking region, plus one final checkpoint at
    # think_end (cutting past </think> would feed the model its own answer).
    steps = [b for b in budgets if b < think_end] + [think_end]
    for b in steps:
        gen_payload = {
            "input_ids": all_ids[:b] + probe_ids,
            "sampling_params": dict(sampling_params),
            "return_logprob": True,
            # -1: output-token logprobs only, so SGLang can reuse the cached
            # prefix instead of forcing a full prefill per probe.
            "logprob_start_len": logprob_start_len,
            "top_logprobs_num": 1,
            "return_text_in_logprobs": True,
        }
        for attempt in range(3):
            try:
                r = requests.post(f"{endpoint.rstrip('/')}/generate",
                                  json=gen_payload, timeout=(15, 600))
                r.raise_for_status()
                data = r.json()
                meta = data.get("meta_info") or {}
                out_lps = meta.get("output_token_logprobs") or []
                toks = [t[2] if len(t) > 2 else "" for t in out_lps]
                lps = [t[0] for t in out_lps]
                k = _probe_window(toks)
                ans = "".join(toks[:k]) if toks else ""
                cert = math.exp(min(lps[:k])) if (lps and k > 0) else None
                cert_mean = (math.exp(sum(lps[:k]) / k)
                             if (lps and k > 0) else None)
                probes.append({"budget": b, "answer": ans,
                               "certainty": cert, "certainty_mean": cert_mean,
                               "is_final": b == think_end})
                break
            except Exception:
                time.sleep(2 ** attempt * 5)
        else:
            probes.append({"budget": b, "answer": None, "certainty": None,
                           "error": "generate failed", "is_final": b == think_end})
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"key": key, "model": rec["model"],
                               "seed": rec["seed"],
                               "problem_id": rec["problem_id"],
                               "ground_truth": rec["ground_truth"],
                               "task": task,
                               "probe_temperature": temperature,
                               "probe_mode": "greedy" if temperature == 0
                               else "sampled",
                               "probes": probes}))
    tmp.rename(out_path)
    return f"[done] {key}: {len(probes)} probes"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--trace-dir", default=str(TRACES_DIR))
    ap.add_argument("--task", default=None,
                    help="task key, e.g. aime2025 "
                         "(default: inferred from trace-dir path)")
    ap.add_argument("--out", default=None,
                    help="probes dir (default: <trace_dir>/../probes)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--interval", type=int, default=1000,
                    help="probe grid step in thinking tokens; use 200 for GSM8K")
    ap.add_argument("--max-budget", type=int, default=150000)
    ap.add_argument("--logprob-start-len", type=int, default=-1,
                    help="-1 (default) = logprobs for generated tokens only, so "
                         "SGLang can reuse the cached prefix; 0 = full prefill")
    # temperature=0 -> greedy; >0 -> sampled (matches generation protocol).
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="probe decode temperature (0 = greedy default; "
                         "e.g. 0.6 to match generation protocol)")
    ap.add_argument("--top-p", type=float, default=0.95,
                    help="top_p for sampled probes (ignored when greedy)")
    ap.add_argument("--sampling-seed", type=int, default=None,
                    help="per-request sampling_seed for reproducible "
                         "sampled probes (optional)")
    args = ap.parse_args()

    root = Path(args.trace_dir)
    task = args.task or infer_task(str(root))
    max_ans_tokens = TASKS[task].probe_answer_tokens
    # greedy -> probes/, sampled -> probes_sampled/ (mode-named, not temperature).
    if args.out:
        out = Path(args.out)
    elif args.temperature == 0:
        out = root.parent / "probes"
    else:
        out = root.parent / "probes_sampled"
    paths = sorted(root.glob("*.json"))
    if args.limit:
        paths = paths[:args.limit]
    mode = "greedy" if args.temperature == 0 else "sampled"
    print(f"[sweep] task={task} mode={mode} T={args.temperature} "
          f"interval={args.interval} {len(paths)} traces -> {out}")

    budgets = list(range(args.interval, args.max_budget + 1, args.interval))
    sblock = sampling_block(args, max_ans_tokens)
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(sweep_trace, args.endpoint, args.model, p,
                          out, args.force, sblock,
                          args.temperature, task=task,
                          max_ans_tokens=max_ans_tokens,
                          budgets=budgets,
                          logprob_start_len=args.logprob_start_len): p
                for p in paths}
        for fut in as_completed(futs):
            # A single failing trace must not abort the sweep.
            try:
                msg = fut.result()
            except Exception as e:  # noqa: BLE001
                msg = f"[FAIL] {futs[fut]}: {e}"
            print(msg, flush=True)
            if msg.startswith("[done]"):
                done += 1
    print(f"[complete] {done}/{len(paths)}")


if __name__ == "__main__":
    main()
