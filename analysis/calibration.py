"""Calibration (ECE / Brier / reliability diagram) from saved probes.

    uv run python -m analysis.calibration [PROBE_DIR] [--bins N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from certainty_guided_reasoning.paths import PROBES_DIR

from analysis.replay import correct as is_correct, ece_brier, load_probes
from analysis.seedfilter import add_seed_args, filter_seeds

N_BINS = 15


def reliability_data(conf: np.ndarray, correct: np.ndarray,
                     n_bins: int = N_BINS):
    """Return (bin_centers, bin_acc, bin_conf, bin_counts, bin_edges)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.searchsorted(bins[1:-1], conf, side="left"), 0,
                  n_bins - 1)
    centers, accs, confs, counts = [], [], [], []
    for b in range(n_bins):
        m = idx == b
        nb = int(m.sum())
        centers.append((bins[b] + bins[b + 1]) / 2)
        accs.append(correct[m].mean() if nb else np.nan)
        confs.append(conf[m].mean() if nb else np.nan)
        counts.append(nb)
    return (np.array(centers), np.array(accs), np.array(confs),
            np.array(counts), bins)


def collect(recs: list[dict], mode: str, cert_key: str = "certainty"):
    """Return (conf, correct) arrays for the requested aggregation mode."""
    confs, corrects = [], []
    for r in recs:
        ans = [p for p in r["probes"] if p.get("answer") is not None
               and p.get(cert_key) is not None]
        if not ans:
            continue
        if mode == "pooled":
            for p in ans:
                confs.append(p[cert_key])
                corrects.append(is_correct(r, p["answer"]))
        else:  # final
            last = ans[-1]
            confs.append(last[cert_key])
            corrects.append(is_correct(r, last["answer"]))
    return np.array(confs, dtype=float), np.array(corrects, dtype=bool)


def _fmt_ci(a: np.ndarray) -> str:
    return f"{a.mean():.4f} ± {a.std(ddof=1) / np.sqrt(len(a)):.4f}" if len(a) else "n/a"


def reliability_plot(out: Path, conf: np.ndarray, correct: np.ndarray,
                     title: str, metric_label: str, dpi: int = 150) -> None:
    centers, accs, confs, counts, _ = reliability_data(conf, correct)
    valid = ~np.isnan(accs)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot([0, 1], [0, 1], "--", color="#7f8c8d", lw=1,
            label="Perfect calibration")
    ax.plot(confs[valid], accs[valid], "o-", color="#2980b9", ms=5,
            label="Observed accuracy")
    ax2 = ax.twinx()
    ax2.bar(centers, counts, width=1 / N_BINS * 0.9, color="#bdc3c7",
            alpha=0.35, label="count")
    ax2.set_ylabel("sample count")
    ax.set_xlabel(f"mean predicted certainty ({metric_label})")
    ax.set_ylabel("observed accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("probe_dir", nargs="?", default=str(PROBES_DIR))
    ap.add_argument("--out", default=None, help="figs dir")
    ap.add_argument("--metric", choices=("min", "mean"), default="min",
                    help="certainty variant: min (Eq. 3) or mean token-prob")
    ap.add_argument("--bins", type=int, default=15)
    ap.add_argument("--dpi", type=int, default=150,
                    help="raster resolution; ignored for pdf and svg")
    ap.add_argument("--fmt", default="png", choices=("png", "pdf", "svg"),
                    help="figure format; pdf and svg are vector and preferred "
                         "for submission")
    add_seed_args(ap)
    args = ap.parse_args()

    global N_BINS
    N_BINS = args.bins
    cert_key = "certainty" if args.metric == "min" else "certainty_mean"
    metric_label = "min token-prob" if args.metric == "min" else "mean token-prob"

    root = Path(args.probe_dir)
    default_figs = ("figs_sampled" if root.name == "probes_sampled"
                    else "figs")
    out = Path(args.out) if args.out else root.parent / default_figs
    out.mkdir(parents=True, exist_ok=True)

    recs = load_probes(root)
    recs = filter_seeds(recs, args.max_seed, args.min_seed)
    if not recs:
        print(f"no probe records in {args.probe_dir}")
        return
    if args.max_seed is not None or args.min_seed is not None:
        print(f"[seed filter] max={args.max_seed} min={args.min_seed} "
              f"-> {len(recs)} records")

    print(f"[calib] {root}  metric={metric_label}  n_traces={len(recs)}")
    print("-" * 64)

    report: list[str] = []
    report.append(f"# Calibration report — {root.parent.name} "
                  f"({metric_label})")
    report.append("")
    report.append(f"- probe dir: `{root}`")
    report.append(f"- metric: `{metric_label}`")
    report.append(f"- traces: {len(recs)}")
    report.append(f"- bins: {N_BINS}")
    report.append("")

    for mode in ("final", "pooled"):
        conf, correct = collect(recs, mode, cert_key)
        if len(conf) == 0:
            print(f"[{mode}] no samples")
            continue
        ece, brier, acc, conf_mean = ece_brier(
            list(zip(conf.tolist(), correct.tolist())))
        print(f"[{mode}] n={len(conf):>6}  acc={acc:>7.4f}  "
              f"ECE={ece:>7.4f}  Brier={brier:>7.4f}")
        print(f"         conf {_fmt_ci(conf)}  (mean ± se)")
        title = (f"Reliability ({mode}) — {root.parent.name}\n"
                 f"ECE={ece:.4f}  Brier={brier:.4f}  n={len(conf)}")
        reliability_plot(out / f"calib_reliability_{mode}.{args.fmt}",
                         conf, correct, title, metric_label, args.dpi)
        report.append(f"## {mode} aggregation")
        report.append(f"- n = {len(conf)}")
        report.append(f"- accuracy = {acc:.4f}")
        report.append(f"- mean certainty = {conf_mean:.4f} "
                      f"(±{conf.std(ddof=1) / np.sqrt(len(conf)):.4f})")
        report.append(f"- **ECE = {ece:.4f}**")
        report.append(f"- **Brier = {brier:.4f}**")
        report.append("")
        print(f"         [written] {out}/calib_reliability_{mode}.{args.fmt}")

    (out / "calib_report.md").write_text("\n".join(report))
    print(f"[written] {out}/calib_report.md")


if __name__ == "__main__":
    main()
