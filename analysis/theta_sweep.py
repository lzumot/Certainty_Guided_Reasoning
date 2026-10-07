"""CGR accuracy/savings sweep over certainty thresholds, from saved probes.

    uv run python -m analysis.theta_sweep [PROBE_DIR] [--tokenizer PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from certainty_guided_reasoning.paths import PROBES_DIR

from analysis.replay import correct as is_correct, load_probes, make_counter, simulate
from analysis.seedfilter import add_seed_args, filter_seeds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("probe_dir", nargs="?", default=str(PROBES_DIR))
    ap.add_argument("--out", default=None,
                    help="figs dir (default: <probe_dir>/../figs)")
    ap.add_argument("--thetas", default="0.90,0.95,0.96,0.97,0.98,0.99",
                    help="comma-separated certainty thresholds")
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer.json (or a directory holding one) for exact "
                         "probe cost; without it the character count is used, "
                         "which is an upper bound on tokens")
    ap.add_argument("--metric", choices=("min", "mean"), default="min",
                    help="certainty variant: min (Eq. 3) or mean token-prob")
    ap.add_argument("--prefix-tokens", type=int, default=None,
                    help="override the measured answer-prefix token count")
    ap.add_argument("--dpi", type=int, default=150,
                    help="raster resolution; ignored for pdf and svg")
    ap.add_argument("--fmt", default="png", choices=("png", "pdf", "svg"),
                    help="figure format; pdf and svg are vector and preferred "
                         "for submission")
    add_seed_args(ap)
    args = ap.parse_args()
    cert_key = "certainty" if args.metric == "min" else "certainty_mean"
    probe_root = Path(args.probe_dir)
    run_name = probe_root.parent.name
    # Sampled probes (probes_sampled/) get their own figs dir so greedy and
    # sampled analyses never overwrite each other's figures.
    default_figs = ("figs_sampled" if probe_root.name == "probes_sampled"
                    else "figs")
    out = Path(args.out) if args.out else probe_root.parent / default_figs
    recs = load_probes(probe_root)
    recs = filter_seeds(recs, args.max_seed, args.min_seed)
    if not recs:
        print(f"no probe records in {args.probe_dir}")
        return
    if args.max_seed is not None or args.min_seed is not None:
        print(f"[seed filter] max={args.max_seed} min={args.min_seed} "
              f"-> {len(recs)} records")

    thetas = [float(t) for t in args.thetas.split(",")]
    count_tokens, prefix_tokens, counter_label = make_counter(args.tokenizer)
    if args.prefix_tokens is not None:
        prefix_tokens = args.prefix_tokens
    print(f"probe cost: {counter_label}; answer prefix = {prefix_tokens} tokens")
    results = []
    print(f"{'theta':>6} {'acc_base':>9} {'acc_cgr':>8} {'delta':>7} "
          f"{'think%':>7} {'overhd%':>7} {'net%':>7} {'exits':>6} "
          f"{'neverconf':>10}")
    print("-" * 76)
    for t in thetas:
        r = simulate(recs, t, count_tokens, prefix_tokens, cert_key)
        results.append((t, r))
        print(f"{t:>6.2f} {r['acc_base']:>9.4f} {r['acc_cgr']:>8.4f} "
              f"{r['acc_cgr']-r['acc_base']:>+7.4f} "
              f"{r['saved_pct']*100:>6.1f}% {r['overhead_pct']*100:>6.1f}% "
              f"{r['saved_pct_net']*100:>6.1f}% {r['exits']:>6} "
              f"{r['never']:>10}")

    # figure: accuracy vs savings trade-off
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    xs = [t for t, _ in results]
    ax1.plot(xs, [r["acc_cgr"] for _, r in results], "o-",
             color="#27ae60", label="CGR accuracy")
    ax1.plot(xs, [r["acc_base"] for _, r in results], "--",
             color="#2980b9", label="baseline accuracy")
    ax1.set_xlabel("certainty threshold θ")
    ax1.set_ylabel("accuracy", color="#27ae60")
    ax1.tick_params(axis="y", labelcolor="#27ae60")
    ax2 = ax1.twinx()
    ax2.plot(xs, [r["saved_pct"] * 100 for _, r in results], "s-",
             color="#e67e22", label="saved %, thinking only")
    ax2.plot(xs, [r["saved_pct_net"] * 100 for _, r in results], "^--",
             color="#c0392b", label="saved %, incl. probe cost")
    ax2.set_ylabel("token savings (%)", color="#e67e22")
    ax2.tick_params(axis="y", labelcolor="#e67e22")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left",
               fontsize=9)
    ax1.set_title(f"CGR θ sweep — {run_name} [{args.metric}] "
                  f"(n={results[0][1]['n']} traces)")
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"figc_theta_sweep.{args.fmt}", dpi=args.dpi)
    plt.close(fig)
    print(f"\n[written] {out}/figc_theta_sweep.{args.fmt}")

    # ---------------- Grade table per theta (paper Table 4/5 style) ------
    penalties = (0.0, 0.25, 0.5, 1.0)
    print(f"\nGrade per problem (n={len(recs)} traces, "
          f"{len({r['problem_id'] for r in recs})} problems), "
          f"abstain when certainty < theta:")
    header = f"{'theta':>6}" + "".join(f" {'p='+str(p):>8}" for p in penalties) \
        + f" {'base(p=0)':>10}"
    print(header)
    print("-" * len(header))
    grade_rows = []
    for t in thetas:
        r = simulate(recs, t)
        # recompute with abstention: never-confident traces abstain (score 0)
        nc = ni = na = 0
        for rec in recs:
            answered = [p for p in rec["probes"] if p.get("answer") is not None]
            if not answered:
                continue
            hit = next((p for p in rec["probes"]
                        if p.get(cert_key) is not None
                        and p[cert_key] >= t), None)
            if hit:
                if is_correct(rec, hit["answer"]):
                    nc += 1
                else:
                    ni += 1
            else:
                na += 1  # abstain
        n_prob = len({rec["problem_id"] for rec in recs})
        grades = [(nc - p * ni) / n_prob for p in penalties]
        # baseline: answer everything at final checkpoint
        bnc = bni = 0
        for rec in recs:
            answered = [p for p in rec["probes"] if p.get("answer") is not None]
            if not answered:
                continue
            if is_correct(rec, answered[-1]["answer"]):
                bnc += 1
            else:
                bni += 1
        base_g0 = bnc / n_prob
        grade_rows.append((t, grades))
        print(f"{t:>6.2f}" + "".join(f" {g:>8.2f}" for g in grades)
              + f" {base_g0:>10.2f}")

    # figure: Grade vs theta per penalty
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ("#bdc3c7", "#f39c12", "#e67e22", "#c0392b")
    xs = [t for t, _ in grade_rows]
    for pi, (p, color) in enumerate(zip(penalties, colors)):
        ys = [g[pi] for _, g in grade_rows]
        ax.plot(xs, ys, "o-", color=color, label=f"CGR Grade p={p}")
    # baseline grade lines (flat)
    n_prob = len({r["problem_id"] for r in recs})
    bnc = sum(1 for r in recs
              if (a := [p for p in r["probes"] if p.get("answer") is not None])
              and is_correct(r, a[-1]["answer"]))
    bni = len([r for r in recs
               if (a := [p for p in r["probes"] if p.get("answer") is not None])
               and not is_correct(r, a[-1]["answer"])]) 
    for p, color in zip(penalties, colors):
        ax.axhline((bnc - p * bni) / n_prob, color=color, ls=":", lw=1,
                   alpha=0.6)
    ax.set_xlabel("certainty threshold θ")
    ax.set_ylabel("Grade per problem")
    ax.set_title("Grade vs θ with abstention (dotted = baseline)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"figc_grade_vs_theta.{args.fmt}", dpi=args.dpi)
    plt.close(fig)
    print(f"[written] {out}/figc_grade_vs_theta.{args.fmt}")


if __name__ == "__main__":
    main()
