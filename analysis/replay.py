"""Shared offline replay of the post-hoc CGR protocol over saved probes.

Reads `probes/*.json` written by probe_sweep; no GPU or live endpoint needed.
Holds the replay (exit at the first checkpoint with certainty >= theta) and
calibration math shared by theta_sweep, calibration, recompute_clean65,
exit_separation, and holdout_seeds.
"""

from __future__ import annotations

import json
from pathlib import Path

from certainty_guided_reasoning.grader import infer_task, probe_answer_equal

THETAS = [0.90, 0.95, 0.96, 0.97, 0.98, 0.99]
MAX_SEED = 64
BINS = 15
PROBE_PREFIX = "\n\nFinal Answer: \\boxed{"  # mirrors probe_sweep.PROBE_PREFIX


def count_chars(text: str) -> int:
    """Character count: an upper bound on tokens for any tokenizer."""
    return len(text)


def make_counter(spec: str | None):
    """Return (count_tokens, prefix_tokens, label) for probe-cost accounting."""
    if not spec:
        n = count_chars(PROBE_PREFIX)
        return count_chars, n, "chars (upper bound on tokens)"
    from tokenizers import Tokenizer  # exact mode only

    path = Path(spec)
    if path.is_dir():
        path = path / "tokenizer.json"
    tok = Tokenizer.from_file(str(path))

    def count(text: str) -> int:
        return len(tok.encode(text).ids)

    return count, count(PROBE_PREFIX), f"tokenizer {path.name}"


def answer_cost(answer, count_tokens, prefix_tokens: int) -> int:
    """Cost of one probe answer: prefix, plus content and closing brace."""
    if answer is None:
        return prefix_tokens
    return prefix_tokens + count_tokens(answer) + 1


def load_probes(root: Path) -> list[dict]:
    recs = []
    for p in sorted(root.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if d.get("probes"):
            recs.append(d)
    return recs


def correct(rec: dict, ans) -> bool:
    task = infer_task(rec.get("problem_id", ""))
    return probe_answer_equal(ans, rec.get("ground_truth"), task)


def answered(rec: dict) -> list[dict]:
    return [p for p in rec["probes"]
            if p.get("answer") is not None and p.get("certainty") is not None]


def simulate(recs: list[dict], theta: float, count_tokens=count_chars,
             prefix_tokens: int | None = None) -> dict:
    """Post-hoc CGR replay, with and without probe cost.

    `saved_pct` counts thinking tokens only; `saved_pct_net` adds the cost of
    every probe that ran (prefix plus decoded answer) and one final decode.
    """
    prefix_cost = count_chars(PROBE_PREFIX) if prefix_tokens is None else prefix_tokens
    n = acc_base = acc_cgr = 0
    tok_full = tok_exit = tok_exit_net = overhead_tok = 0
    exits = never = 0
    for r in recs:
        ans = answered(r)
        if not ans:
            continue
        last = ans[-1]
        base_ok = correct(r, last["answer"])
        n += 1
        acc_base += base_ok
        tok_full += last["budget"]
        hit = next((p for p in ans if p["certainty"] >= theta), None)
        exit_budget = hit["budget"] if hit else last["budget"]
        exit_answer = hit["answer"] if hit else last["answer"]
        tok_exit += exit_budget
        ran = [p for p in r["probes"] if p["budget"] <= exit_budget]
        overhead = sum(answer_cost(p.get("answer"), count_tokens, prefix_cost)
                       for p in ran)
        overhead += answer_cost(exit_answer, count_tokens, prefix_cost)
        overhead_tok += overhead
        tok_exit_net += exit_budget + overhead
        if hit:
            acc_cgr += correct(r, hit["answer"])
            exits += 1
        else:
            acc_cgr += base_ok
            never += 1
    return {
        "n": n,
        "acc_base": acc_base / n if n else float("nan"),
        "acc_cgr": acc_cgr / n if n else float("nan"),
        "saved_pct": 1 - tok_exit / tok_full if tok_full else float("nan"),
        "saved_pct_net": 1 - tok_exit_net / tok_full if tok_full else float("nan"),
        "overhead_pct": overhead_tok / tok_full if tok_full else float("nan"),
        "exits": exits,
        "never": never,
    }


def ece_brier(pairs: list[tuple[float, bool]]) -> tuple[float, float, float, float]:
    """Equal-width ECE (count-weighted), Brier, accuracy, mean certainty."""
    n = len(pairs)
    if n == 0:
        return (float("nan"),) * 4
    edges = [i / BINS for i in range(BINS + 1)]
    ece = 0.0
    for b in range(BINS):
        lo, hi = edges[b], edges[b + 1]
        cell = [(c, k) for c, k in pairs
                if (lo <= c < hi) or (b == BINS - 1 and c == 1.0)]
        if not cell:
            continue
        conf = sum(c for c, _ in cell) / len(cell)
        acc = sum(k for _, k in cell) / len(cell)
        ece += len(cell) / n * abs(acc - conf)
    brier = sum((c - float(k)) ** 2 for c, k in pairs) / n
    acc = sum(k for _, k in pairs) / n
    conf = sum(c for c, _ in pairs) / n
    return ece, brier, acc, conf
