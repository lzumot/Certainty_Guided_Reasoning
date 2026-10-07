"""Combined MIN vs MEAN CGR theta sweep (single figure, shared y-axis).

Reads one probe dir, replays the post-hoc CGR simulation for both certainty
statistics, and plots CGR accuracy vs theta on a single standardized y-axis
(0.10–0.15) so the two statistics are directly comparable. Token savings is
shown on a secondary axis.

Usage:
    .venv/bin/python -m analysis.combined_theta_sweep [PROBE_DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.replay import load_probes as load, simulate
from certainty_guided_reasoning.paths import PROBES_DIR

THETAS = "0.90,0.95,0.96,0.97,0.98,0.99"
ACC_YLIM = (0.10, 0.16)  # standardized accuracy axis; includes baseline (~0.155)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("probe_dir", nargs="?", default=str(PROBES_DIR))
    ap.add_argument("--out", default=None,
                    help="figs dir (default: <probe_dir>/../figs)")
    ap.add_argument("--thetas", default=THETAS)
    args = ap.parse_args()

    root = Path(args.probe_dir)
    run_name = root.parent.name
    default_figs = ("figs_sampled" if root.name == "probes_sampled"
                    else "figs")
    out = Path(args.out) if args.out else root.parent / default_figs
    out.mkdir(parents=True, exist_ok=True)

    recs = load(root)
    if not recs:
        print(f"no probe records in {root}")
        return

    thetas = [float(t) for t in args.thetas.split(",")]
    results = {
        metric: [simulate(recs, t, cert_key=key)
                 for t in thetas]
        for metric, key in (("min", "certainty"),
                            ("mean", "certainty_mean"))
    }

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    colors = {"min": "#27ae60", "mean": "#e67e22"}
    for metric in ("min", "mean"):
        ax1.plot(thetas, [r["acc_cgr"] for r in results[metric]], "o-",
                 color=colors[metric], label=f"CGR accuracy ({metric})")
    ax1.plot(thetas, [r["acc_base"] for r in results["min"]], "--",
             color="#2980b9", label="baseline accuracy")
    ax1.set_xlabel("certainty threshold θ")
    ax1.set_ylabel("accuracy", color="#27ae60")
    ax1.set_ylim(*ACC_YLIM)
    ax1.tick_params(axis="y", labelcolor="#27ae60")

    ax2 = ax1.twinx()
    for metric in ("min", "mean"):
        ax2.plot(thetas, [r["saved_pct"] * 100 for r in results[metric]], "s--",
                 color=colors[metric], alpha=0.55,
                 label=f"token savings % ({metric})")
    ax2.set_ylabel("token savings (%)", color="#e67e22")
    ax2.tick_params(axis="y", labelcolor="#e67e22")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=8)
    ax1.set_title(f"CGR θ sweep: MIN vs MEAN certainty — {run_name} "
                  f"(n={len(recs)} traces)")

    fig.tight_layout()
    fig.savefig(out / "figc_theta_sweep_combined.png", dpi=150)
    plt.close(fig)
    print(f"[written] {out}/figc_theta_sweep_combined.png")


if __name__ == "__main__":
    main()
