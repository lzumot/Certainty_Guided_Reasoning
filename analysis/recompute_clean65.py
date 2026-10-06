"""Recompute clean-grid replay numbers for one run -> figs_clean65/theta_sweep_min.txt.

    uv run python -m analysis.recompute_clean65 Qwen3.5-9B_aime2025 [--tokenizer PATH]
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from analysis.replay import MAX_SEED, THETAS, load_probes, make_counter, simulate


def main() -> None:
    run = sys.argv[1] if len(sys.argv) > 1 else "Qwen3.5-9B_aime2025"
    tok_spec = None
    if "--tokenizer" in sys.argv:
        tok_spec = sys.argv[sys.argv.index("--tokenizer") + 1]
    count_tokens, prefix_tokens, counter_label = make_counter(tok_spec)
    print(f"probe cost: {counter_label}; answer prefix = {prefix_tokens} tokens")
    root = Path("outputs") / run
    probes = load_probes(root / "probes")
    clean = [r for r in probes if r.get("seed", -1) <= MAX_SEED]
    dropped = len(probes) - len(clean)

    rows = list(csv.DictReader((root / "grid_records.csv").open()))
    rows_clean = [r for r in rows if int(r["seed"]) <= MAX_SEED]
    acc = sum(r["correct"] == "True" for r in rows_clean) / len(rows_clean)
    toks = [int(r["num_thinking_tokens"] or 0) for r in rows_clean]
    length_fin = sum(r["finish_reason"] == "length" for r in rows_clean)

    out = root / "figs_clean65"
    out.mkdir(parents=True, exist_ok=True)
    lines = [f"# Clean-grid recompute (seeds 0-{MAX_SEED}) — {run}", ""]
    lines.append(f"- probe records: {len(probes)} total, {len(clean)} clean "
                 f"({dropped} dev-run dropped)")
    lines.append(f"- grid records: {len(rows)} total, {len(rows_clean)} clean")
    lines.append(f"- pass@1 (trace-level, clean) = {acc:.4f}")
    lines.append(f"- mean thinking tokens = {sum(toks) / len(toks):.1f}"
                 f"  |  length-finish = {length_fin / len(rows_clean):.3%}")
    lines.append("")

    lines.append("## Probe-checkpoint replay (min stat), clean grid")
    lines.append(f"# probe cost: {counter_label}; prefix = {prefix_tokens} tokens")
    lines.append(f"{'theta':>6} {'acc_base':>9} {'acc_cgr':>8} {'delta':>8} "
                 f"{'think%':>7} {'overhd%':>7} {'net%':>7} {'exits':>6} "
                 f"{'neverconf':>9}")
    sweep = []
    for t in THETAS:
        r = simulate(clean, t, count_tokens, prefix_tokens)
        sweep.append((t, r))
        lines.append(f"{t:>6.2f} {r['acc_base']:>9.4f} {r['acc_cgr']:>8.4f} "
                     f"{r['acc_cgr'] - r['acc_base']:>+8.4f} "
                     f"{r['saved_pct'] * 100:>6.1f}% "
                     f"{r['overhead_pct'] * 100:>6.1f}% "
                     f"{r['saved_pct_net'] * 100:>6.1f}% {r['exits']:>6} "
                     f"{r['never']:>9}")
    (out / "theta_sweep_min.txt").write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
