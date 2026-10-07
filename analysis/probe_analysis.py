"""Probe-cache figures: certainty vs step, cumulative answers, CGR sim, grade vs budget.

    uv run python -m analysis.probe_analysis [PROBE_DIR] [--theta 0.99]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from certainty_guided_reasoning.paths import PROBES_DIR

from analysis.replay import correct as is_correct, load_probes
from analysis.seedfilter import add_seed_args, filter_seeds


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("probe_dir", nargs="?", default=str(PROBES_DIR))
    ap.add_argument("--out", default=None,
                    help="figs dir (default: <probe_dir>/../figs)")
    ap.add_argument("--theta", type=float, default=0.99)
    ap.add_argument("--metric", choices=("min", "mean"), default="min",
                    help="certainty variant: min (Eq. 3) or mean token-prob")
    ap.add_argument("--dpi", type=int, default=150,
                    help="raster resolution; ignored for pdf and svg")
    ap.add_argument("--fmt", default="png", choices=("png", "pdf", "svg"),
                    help="figure format; pdf and svg are vector and preferred "
                         "for submission")
    add_seed_args(ap)
    args = ap.parse_args()

    cert_key = "certainty" if args.metric == "min" else "certainty_mean"
    root = Path(args.probe_dir)
    # Sampled probes (probes_sampled/) get their own figs dir so greedy and
    # sampled analyses never overwrite each other's figures.
    default_figs = ("figs_sampled" if root.name == "probes_sampled"
                    else "figs")
    out = Path(args.out) if args.out else root.parent / default_figs
    out.mkdir(parents=True, exist_ok=True)
    recs = load_probes(root)
    recs = filter_seeds(recs, args.max_seed, args.min_seed)
    if not recs:
        print(f"no probe records in {root}")
        return
    if args.max_seed is not None or args.min_seed is not None:
        print(f"[seed filter] max={args.max_seed} min={args.min_seed} "
              f"-> {len(recs)} records")
    theta = args.theta

    all_budgets = sorted({p["budget"] for r in recs for p in r["probes"]})
    # Regular interval checkpoints versus each trace's own final checkpoint.
    # The latter sits at a near-unique budget, so averaging per budget mixes a
    # handful of traces against the tens of thousands behind a real checkpoint
    # — and it biases high, since a trace stops when it is already confident.
    cert_by_b: dict[int, list[float]] = defaultdict(list)
    final_b: list[int] = []
    final_c: list[float] = []
    for r in recs:
        for p in r["probes"]:
            cert = p.get(cert_key)
            if cert is None:
                continue
            if p.get("is_final"):
                final_b.append(p["budget"])
                final_c.append(cert)
            else:
                cert_by_b[p["budget"]].append(cert)

    # Cumulative: a trace joins the count at its own final checkpoint — the
    # budget it actually stopped at. Keying on the first *answered* checkpoint
    # instead saturates at the first budget probes exist for (every trace is
    # forced to answer there), so the curves never move.
    stop: list[tuple[int, bool]] = []
    for r in recs:
        last = next((p for p in reversed(r["probes"]) if p.get("is_final")),
                    None)
        if last is None or last.get("answer") is None:
            continue
        stop.append((last["budget"], is_correct(r, last["answer"])))
    stop.sort()
    n_total = len(recs)
    cum_corr, cum_incorr, cum_unans = [], [], []
    c = i = k = 0
    for b in all_budgets:
        while k < len(stop) and stop[k][0] <= b:
            if stop[k][1]:
                c += 1
            else:
                i += 1
            k += 1
        cum_corr.append(c)
        cum_incorr.append(i)
        cum_unans.append(n_total - c - i)

    # ---------------- Fig: certainty vs step -----------------------------
    means, q1s, q3s = [], [], []
    bs = sorted(cert_by_b)  # regular checkpoints only
    for b in bs:
        xs = sorted(cert_by_b[b])
        means.append(sum(xs) / len(xs))
        q1s.append(xs[len(xs) // 4])
        q3s.append(xs[(3 * len(xs)) // 4])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(final_b, final_c, s=2, color="#95a5a6", alpha=0.15, zorder=1,
               label="per-trace final checkpoint")
    ax.plot(bs, means, "o-", color="#2980b9", ms=3, zorder=3,
            label="mean certainty (regular checkpoints)")
    ax.fill_between(bs, q1s, q3s, alpha=0.25, color="#2980b9", zorder=2,
                    label="IQR across traces")
    ax.axhline(theta, color="#c0392b", ls="--", lw=1, label=f"θ={theta}")
    ax.set_xlabel("thinking budget (tokens)")
    ax.set_ylabel("probe certainty (Eq. 3)")
    ax.set_title(f"Certainty vs thinking step ({len(recs)} traces)\n"
                 "grey = per-trace final checkpoint (paper Fig 4)")
    ax.legend()
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(out / f"figc_certainty_vs_step.{args.fmt}", dpi=args.dpi)
    plt.close(fig)

    # ---------------- Fig: cumulative answers over budget ----------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(bs, [cum_corr[all_budgets.index(b)] for b in bs], "o-",
            color="#27ae60", ms=3, label="correct")
    ax.plot(bs, [cum_incorr[all_budgets.index(b)] for b in bs], "s-",
            color="#c0392b", ms=3, label="incorrect")
    ax.plot(bs, [cum_unans[all_budgets.index(b)] for b in bs], "^-",
            color="#bdc3c7", ms=3, label="still thinking")
    ax.set_xlabel("thinking budget (tokens)")
    ax.set_ylabel(f"traces (n={n_total})")
    ax.set_title("Cumulative traces by their final checkpoint\n"
                 "(correct / incorrect / still thinking)")
    ax.legend()
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(out / f"figc_cumulative_answers.{args.fmt}", dpi=args.dpi)
    plt.close(fig)

    # ---------------- Fig: TRUE post-hoc CGR sim -------------------------
    exit_tokens = []
    exits_ok = []
    full_tokens = []
    base_ok = 0
    for r in recs:
        last_b = max((p["budget"] for p in r["probes"]
                      if p.get("answer") is not None), default=None)
        if last_b is None:
            continue
        full_tokens.append(last_b)
        base_ok += is_correct(r, next(
            p["answer"] for p in reversed(r["probes"])
            if p.get("answer") is not None))
        hit = next((p for p in r["probes"]
                    if p.get(cert_key) is not None
                    and p[cert_key] >= theta), None)
        if hit:
            exit_tokens.append(hit["budget"])
            exits_ok.append(is_correct(r, hit["answer"]))
        else:
            exit_tokens.append(last_b)
            exits_ok.append(is_correct(r, next(
                p["answer"] for p in reversed(r["probes"])
                if p.get("answer") is not None)))
    n = len(exit_tokens)
    tot_full, tot_exit = sum(full_tokens), sum(exit_tokens)
    acc_base = base_ok / n if n else float("nan")
    acc_cgr = sum(exits_ok) / n if n else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(["baseline", f"CGR θ={theta}"],
                [acc_base, acc_cgr], color=["#95a5a6", "#27ae60"])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("accuracy")
    axes[0].set_title(f"Accuracy (n={n})")
    axes[1].bar(["baseline", f"CGR θ={theta}"],
                [tot_full, tot_exit], color=["#95a5a6", "#27ae60"])
    axes[1].set_ylabel("total thinking tokens")
    axes[1].set_title(f"Token usage — saved {tot_full - tot_exit:,} "
                      f"({(tot_full - tot_exit) / tot_full:.1%}), "
                      f"thinking tokens only"
                      if tot_full else "no data")
    for ax in axes:
        style_ax(ax)
    fig.suptitle("Post-hoc CGR simulation from real probe points",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(out / f"figc_cgr_sim.{args.fmt}", dpi=args.dpi,
                bbox_inches="tight")
    plt.close(fig)

    # ---------------- Fig: Grade vs budget with abstention ---------------
    penalties = (0.0, 0.25, 0.5, 1.0)
    n_problems = len({r["problem_id"] for r in recs})
    # The penalty enters only through (nc, ni), and a trace's correctness at a
    # budget is fixed, so tally each trace/budget pair once. Re-scanning and
    # re-grading inside the penalty loop costs 4 x |bs| x |recs| (about two
    # hours at 84k traces).
    tally: dict[int, list[int]] = {}
    for r in recs:
        for q in r["probes"]:
            if q.get("answer") is None:
                continue
            t = tally.setdefault(q["budget"], [0, 0, 0])
            cert = q.get(cert_key)
            if cert is not None and cert < theta:
                t[0] += 1  # abstain
            elif is_correct(r, q["answer"]):
                t[1] += 1
            else:
                t[2] += 1
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for p, color in zip(penalties, ("#bdc3c7", "#f39c12", "#e67e22", "#c0392b")):
        grades = []
        for b in bs:
            na, nc, ni = tally.get(b, (0, 0, 0))
            grades.append((nc - p * ni) / n_problems)
        ax.plot(bs, grades, "o-", ms=3, color=color,
                label=f"p={p} (abstain below θ={theta})")
    ax.set_xlabel("thinking budget (tokens)")
    ax.set_ylabel("Grade per problem")
    ax.set_title("Grade vs budget under certainty-based abstention\n"
                 "(thesis Figs 7–8 family)")
    ax.legend(fontsize=8)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(out / f"figc_grade_vs_budget.{args.fmt}", dpi=args.dpi)
    plt.close(fig)

    print(f"[written] {out}/figc_certainty_vs_step.{args.fmt}")
    print(f"[written] {out}/figc_cumulative_answers.{args.fmt}")
    print(f"[written] {out}/figc_cgr_sim.{args.fmt}")
    print(f"[written] {out}/figc_grade_vs_budget.{args.fmt}")
    print(f"\nn={n} traces scored | θ={theta}")
    print(f"accuracy: baseline={acc_base:.4f} cgr={acc_cgr:.4f}")
    if tot_full:
        print(f"tokens: {tot_full:,} -> {tot_exit:,} "
              f"(saved {(tot_full - tot_exit) / tot_full:.1%}; thinking tokens "
              f"only, probe cost excluded)")


if __name__ == "__main__":
    main()
