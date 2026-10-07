"""Difficulty-stratified certainty curves — the "CGR as RL-training probe" test.

Tests the mechanistic claim (AI/RESULTS_OVERVIEW.md § Interpretation):

  * Easy problems   -> certainty spikes EARLY (recognition/retrieval; high
                       savings potential).  Model knows it fast -> exit cheap.
  * Hard problems   -> certainty RISES monotonically with budget (multi-step
                       search converging).  Model buys evidence -> ~0 savings.
  * Unsolved        -> certainty stays FLAT/LOW (working beyond competence;
                       CGR should keep going / abstain).  Anything high here
                       = miscalibration / confidently-wrong.

Difficulty proxy: empirical per-problem solve rate across all seeds
(#correct traces / #traces for that problem). Tier assignment:

  solved   : solve_rate >= 0.5
  mixed    : 0.0 < solve_rate < 0.5   (hard but solvable sometimes)
  unsolved : solve_rate == 0.0

Outputs (per probe dir):
  figs/curves_certainty_vs_budget.png  — mean certainty per budget, one line/tier
  figs/curves_accuracy_vs_budget.png   — per-budget accuracy, one line/tier
  figs/curves_report.md                — per-tier stats: slope of certainty,
                                         time-to-0.9, savings at theta

Usage:
    .venv/bin/python -m analysis.certainty_curves [PROBE_DIR] [--metric min|mean]
    # AIME (well-populated solve rates):
    .venv/bin/python -m analysis.certainty_curves outputs/Qwen3.5-9B_aime2025/probes
    # arxivmath ma192k seed-0 (solve rate is 0/1 per problem -> 2 tiers):
    .venv/bin/python -m analysis.certainty_curves outputs/Qwen3.5-9B_arxivmath-0626_ma192k/probes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from certainty_guided_reasoning.paths import PROBES_DIR
from certainty_guided_reasoning.grader import probe_answer_equal, infer_task

THETA = 0.9  # reference threshold for "time-to-confidence"


def load(root: Path) -> list[dict]:
    recs = []
    for p in sorted(root.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if "probes" in d and d["probes"]:
            recs.append(d)
    return recs


def is_correct(rec: dict, ans) -> bool:
    task = infer_task(rec.get("problem_id", ""))
    return probe_answer_equal(ans, rec.get("ground_truth"), task)


def tier_of(solve_rate: float) -> str:
    if solve_rate >= 0.5:
        return "solved"
    if solve_rate > 0.0:
        return "mixed"
    return "unsolved"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("probe_dir", nargs="?", default=str(PROBES_DIR))
    ap.add_argument("--out", default=None, help="figs dir")
    ap.add_argument("--metric", choices=("min", "mean"), default="min",
                    help="certainty variant: min (Eq. 3) or mean token-prob")
    ap.add_argument("--theta", type=float, default=THETA)
    args = ap.parse_args()

    cert_key = "certainty" if args.metric == "min" else "certainty_mean"
    metric_label = "min token-prob" if args.metric == "min" else "mean token-prob"

    root = Path(args.probe_dir)
    default_figs = ("figs_sampled" if root.name == "probes_sampled"
                    else "figs")
    out = Path(args.out) if args.out else root.parent / default_figs
    out.mkdir(parents=True, exist_ok=True)

    recs = load(root)
    if not recs:
        print(f"no probe records in {args.probe_dir}")
        return

    # --- per-problem solve rate (difficulty proxy) -----------------------
    by_problem: dict[str, list[dict]] = {}
    for r in recs:
        by_problem.setdefault(r["problem_id"], []).append(r)

    solve_rate = {}
    for pid, rs in by_problem.items():
        n_ok = 0
        for r in rs:
            answered = [p["answer"] for p in r["probes"]
                        if p.get("answer") is not None]
            # A trace can have zero answered probes (a failed probe run);
            # score it as unsolved instead of indexing an empty list.
            if answered and is_correct(r, answered[-1]):
                n_ok += 1
        solve_rate[pid] = n_ok / len(rs)

    # --- stratify --------------------------------------------------------
    tiers = {t: [] for t in ("solved", "mixed", "unsolved")}
    for r in recs:
        tiers[tier_of(solve_rate[r["problem_id"]])].append(r)
    for t, rs in tiers.items():
        if rs:
            rates = [solve_rate[r["problem_id"]] for r in rs]
            print(f"[{t:>9}] problems={len({r['problem_id'] for r in rs})} "
                  f"traces={len(rs)} solve_rate {min(rates):.2f}..{max(rates):.2f}")

    # --- certainty + accuracy vs budget per tier -------------------------
    # Regular interval checkpoints only. A trace's final checkpoint sits at its
    # own stopping budget, so including those puts ~1 trace into a bucket that
    # otherwise holds tens of thousands — which both biases the mean high (a
    # trace stops when it is confident) and makes the curve a sawtooth.
    all_budgets = sorted({p["budget"] for r in recs for p in r["probes"]
                          if not p.get("is_final")})
    cert_by: dict[str, dict[int, list[float]]] = {
        t: {b: [] for b in all_budgets} for t in tiers}
    acc_by: dict[str, dict[int, list[bool]]] = {
        t: {b: [] for b in all_budgets} for t in tiers}
    for t, rs in tiers.items():
        for r in rs:
            for p in r["probes"]:
                if p.get("is_final"):
                    continue
                if p.get(cert_key) is not None:
                    cert_by[t][p["budget"]].append(p[cert_key])
                if p.get("answer") is not None:
                    acc_by[t][p["budget"]].append(is_correct(r, p["answer"]))

    bs = all_budgets
    tcolors = {"solved": "#27ae60", "mixed": "#e67e22", "unsolved": "#c0392b"}

    # Fig 1: mean certainty vs budget per tier
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for t in ("solved", "mixed", "unsolved"):
        ys = []
        for b in bs:
            xs = cert_by[t][b]
            ys.append(sum(xs) / len(xs) if xs else float("nan"))
        ax.plot(bs, ys, "o-", color=tcolors[t], ms=3, label=t)
    ax.axhline(args.theta, color="gray", ls="--", lw=1,
               label=f"θ={args.theta}")
    ax.set_xlabel("thinking budget (tokens)")
    ax.set_ylabel(f"mean certainty ({metric_label})")
    ax.set_title(f"Certainty vs budget by difficulty tier — {root.parent.name}\n"
                 "(easy: early spike · hard: rising · unsolved: flat/low)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "curves_certainty_vs_budget.png", dpi=150)
    plt.close(fig)

    # Fig 2: accuracy vs budget per tier
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for t in ("solved", "mixed", "unsolved"):
        ys = []
        for b in bs:
            xs = acc_by[t][b]
            ys.append(sum(xs) / len(xs) if xs else float("nan"))
        ax.plot(bs, ys, "o-", color=tcolors[t], ms=3, label=t)
    ax.set_xlabel("thinking budget (tokens)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Per-budget accuracy by difficulty tier — {root.parent.name}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "curves_accuracy_vs_budget.png", dpi=150)
    plt.close(fig)

    # Fig 3: sample size per budget (survivor-bias check)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for t in ("solved", "mixed", "unsolved"):
        ys = [len(cert_by[t][b]) for b in bs]
        ax.plot(bs, ys, "o-", color=tcolors[t], ms=3, label=t)
    ax.set_xlabel("thinking budget (tokens)")
    ax.set_ylabel("traces with a probe at budget")
    ax.set_title(f"Sample size per budget by difficulty tier — {root.parent.name}\n"
                 "(fewer traces at high budget = less reliable mean)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "curves_n_vs_budget.png", dpi=150)
    plt.close(fig)

    # --- report ----------------------------------------------------------
    lines = [f"# Certainty curves by difficulty — {root.parent.name} "
             f"({metric_label})", ""]
    lines.append(f"- probe dir: `{root}`")
    lines.append(f"- metric: `{metric_label}` | θ_ref = {args.theta}")
    lines.append(f"- problems: {len(by_problem)} | traces: {len(recs)}")
    lines.append("")
    lines.append("## Per-tier summary")
    lines.append("")
    lines.append("| tier | problems | traces | certainty@1k | certainty@max | "
                 "Δ(cert) | time-to-θ |")
    lines.append("|---|---|---|---|---|---|---|")
    for t in ("solved", "mixed", "unsolved"):
        rs = tiers[t]
        if not rs:
            continue
        # first and last meaningful budget
        first_b = next((b for b in bs if cert_by[t][b]), None)
        last_b = next((b for b in reversed(bs) if cert_by[t][b]), None)
        c_first = (sum(cert_by[t][first_b]) / len(cert_by[t][first_b])
                   if first_b is not None else float("nan"))
        c_last = (sum(cert_by[t][last_b]) / len(cert_by[t][last_b])
                  if last_b is not None else float("nan"))
        # time to theta: first budget where mean certainty >= theta
        ttt = next((b for b in bs
                    if cert_by[t][b]
                    and sum(cert_by[t][b]) / len(cert_by[t][b]) >= args.theta),
                   None)
        n_prob = len({r["problem_id"] for r in rs})
        lines.append(
            f"| {t} | {n_prob} | {len(rs)} | {c_first:.3f} | {c_last:.3f} | "
            f"{c_last - c_first:+.3f} | {ttt if ttt else 'never'} |")
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    lines.append("- **solved** should spike early (high certainty at low budget) "
                 "→ CGR saves tokens.")
    lines.append("- **mixed** should rise monotonically (search converging) "
                 "→ little savings available.")
    lines.append("- **unsolved** should stay flat/low → model should keep going "
                 "or abstain; a high curve here = miscalibration.")
    lines.append("")
    lines.append("Sample size shrinks with budget (traces end early), so the "
                 "high-budget tail of each curve is a smaller, biased sample; "
                 "see curves_n_vs_budget.png.")
    lines.append("")
    lines.append("The separation between tiers is the evidence that certainty "
                 "tracks genuine search progress (the 'CGR measures RL-training "
                 "quality' claim), not mere verbosity.")

    (out / "curves_report.md").write_text("\n".join(lines))
    print(f"[written] {out}/curves_certainty_vs_budget.png")
    print(f"[written] {out}/curves_accuracy_vs_budget.png")
    print(f"[written] {out}/curves_n_vs_budget.png")
    print(f"[written] {out}/curves_report.md")


if __name__ == "__main__":
    main()
