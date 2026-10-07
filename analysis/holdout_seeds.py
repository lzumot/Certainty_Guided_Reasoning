"""Held-out theta selection: pick theta on half the seeds, score on the other.

    uv run python -m analysis.holdout_seeds Qwen3.5-9B_aime2025
"""

from __future__ import annotations

import sys
from pathlib import Path

from analysis.replay import (MAX_SEED, THETAS, load_probes, make_counter,
                             probe_prefix_for, simulate)

SPLIT = 32  # seeds 0..SPLIT on one side, SPLIT+1..MAX_SEED on the other


def pick_theta(select: list[dict], count, prefix) -> tuple[float, dict]:
    """Theta with best CGR accuracy on the selection half."""
    table = {t: simulate(select, t, count, prefix) for t in THETAS}
    best = max(THETAS, key=lambda t: (table[t]["acc_cgr"], t))
    return best, table


def main() -> None:
    run = sys.argv[1] if len(sys.argv) > 1 else "Qwen3.5-9B_aime2025"
    tok = None
    if "--tokenizer" in sys.argv:
        tok = sys.argv[sys.argv.index("--tokenizer") + 1]
    root = Path("outputs") / run
    recs = [r for r in load_probes(root / "probes")
            if r.get("seed", -1) <= MAX_SEED]
    # Probe cost depends on the model family's answer prefix.
    count, prefix, label = make_counter(
        tok, probe_prefix_for(recs[0].get("model", "") if recs else ""))
    low = [r for r in recs if r["seed"] <= SPLIT]
    high = [r for r in recs if r["seed"] > SPLIT]

    lines = [f"# Held-out theta selection — {run}", "",
             f"- probe cost: {label}; answer prefix = {prefix} tokens",
             f"- select half 0-{SPLIT}: {len(low)} traces; "
             f"held-out half {SPLIT + 1}-{MAX_SEED}: {len(high)} traces", ""]

    print(f"probe cost: {label}; prefix = {prefix} tokens")
    for name, sel, ev in ((f"select 0-{SPLIT} → evaluate {SPLIT + 1}-{MAX_SEED}", low, high),
                          (f"select {SPLIT + 1}-{MAX_SEED} → evaluate 0-{SPLIT}", high, low)):
        theta_star, table = pick_theta(sel, count, prefix)
        on_sel = table[theta_star]
        on_ev = simulate(ev, theta_star, count, prefix)
        header = (f"{'theta':>6} {'sel Δ':>8} {'sel net%':>9} "
                  f"{'held Δ':>9} {'held net%':>10}")
        print(f"\n{name}: chosen theta = {theta_star:g}")
        print(header)
        print("-" * len(header))
        rows = []
        for t in THETAS:
            r_sel = table[t]
            r_ev = simulate(ev, t, count, prefix)
            rows.append((t, r_sel, r_ev))
            mark = " <-" if t == theta_star else ""
            print(f"{t:>6.2f} {r_sel['acc_cgr'] - r_sel['acc_base']:>+8.4f} "
                  f"{r_sel['saved_pct_net'] * 100:>8.1f}% "
                  f"{r_ev['acc_cgr'] - r_ev['acc_base']:>+9.4f} "
                  f"{r_ev['saved_pct_net'] * 100:>9.1f}%{mark}")
        lines += [f"## {name}", "",
                  f"- reference accuracy on the held-out half: "
                  f"{on_ev['acc_base']:.4f} (n = {on_ev['n']})",
                  f"- chosen theta = {theta_star:g}",
                  f"- on the selection half: Δ = "
                  f"{on_sel['acc_cgr'] - on_sel['acc_base']:+.4f}, net savings = "
                  f"{on_sel['saved_pct_net'] * 100:.1f}%",
                  f"- on the held-out half: Δ = "
                  f"{on_ev['acc_cgr'] - on_ev['acc_base']:+.4f}, net savings = "
                  f"{on_ev['saved_pct_net'] * 100:.1f}%", "",
                  f"| theta | sel Δ | sel net % | held Δ | held net % |",
                  "|---|---|---|---|---|"]
        for t, r_sel, r_ev in rows:
            mark = " (chosen)" if t == theta_star else ""
            lines.append(
                f"| {t:g}{mark} | {r_sel['acc_cgr'] - r_sel['acc_base']:+.4f} | "
                f"{r_sel['saved_pct_net'] * 100:.1f} | "
                f"{r_ev['acc_cgr'] - r_ev['acc_base']:+.4f} | "
                f"{r_ev['saved_pct_net'] * 100:.1f} |")
        lines.append("")

    out = root / "figs_clean65"
    out.mkdir(parents=True, exist_ok=True)
    (out / "theta_holdout.md").write_text("\n".join(lines) + "\n")
    print(f"\n[written] {out / 'theta_holdout.md'}")


if __name__ == "__main__":
    main()
