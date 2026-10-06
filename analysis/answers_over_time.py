"""Cumulative correct/incorrect answers vs thinking budget, from grid_records.csv.

    uv run python -m analysis.answers_over_time outputs/{RUN}/grid_records.csv \
        --max-seed 64 --out outputs/{RUN}/figs_clean65 --name <stem>
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.seedfilter import add_seed_args

BUDGETS = list(range(1000, 32001, 1000))


def load(path: Path, max_seed: int | None, min_seed: int | None) -> list[dict]:
    rows = []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            seed = int(row["seed"])
            if max_seed is not None and seed > max_seed:
                continue
            if min_seed is not None and seed < min_seed:
                continue
            rows.append(row)
    return rows


def curves(rows: list[dict], budgets: list[int]):
    """Return (correct, incorrect) curves, each averaged over seeds."""
    by_seed: dict[int, dict[str, tuple[int | None, bool]]] = defaultdict(dict)
    late = 0
    for row in rows:
        answered = bool((row.get("extracted_answer") or "").strip())
        if answered:
            budget = int(row["num_thinking_tokens"])
            if budget > budgets[-1]:
                budget = budgets[-1]  # late completion -> final bar
                late += 1
        else:
            budget = None
        by_seed[int(row["seed"])][row["problem_id"]] = (
            budget, row["correct"].strip() == "True")
    n_seeds = len(by_seed)
    correct, incorrect = [], []
    for b in budgets:
        tot_c = tot_i = 0
        for per_problem in by_seed.values():
            for budget, ok in per_problem.values():
                if budget is not None and budget <= b:
                    if ok:
                        tot_c += 1
                    else:
                        tot_i += 1
        correct.append(tot_c / n_seeds if n_seeds else 0.0)
        incorrect.append(tot_i / n_seeds if n_seeds else 0.0)
    return correct, incorrect, n_seeds, late


def style_ax(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="grid_records.csv for one run")
    ap.add_argument("--out", default=None, help="figs dir (default: csv dir)")
    ap.add_argument("--name", default="answers_over_time",
                    help="output filename stem")
    ap.add_argument("--title", default="", help="run label for the title")
    ap.add_argument("--n-problems", type=int, default=30,
                    help="AIME2025 has 30 problems; sets the stack height")
    ap.add_argument("--dpi", type=int, default=150,
                    help="raster resolution; ignored for pdf and svg")
    ap.add_argument("--fmt", default="png", choices=("png", "pdf", "svg"),
                    help="figure format; pdf and svg are vector and preferred "
                         "for submission")
    add_seed_args(ap)
    args = ap.parse_args()

    path = Path(args.csv_path)
    out = Path(args.out) if args.out else path.parent
    out.mkdir(parents=True, exist_ok=True)

    rows = load(path, args.max_seed, args.min_seed)
    if not rows:
        print(f"no rows in {path} after the seed filter")
        return
    if args.max_seed is not None or args.min_seed is not None:
        print(f"[seed filter] max={args.max_seed} min={args.min_seed} "
              f"-> {len(rows)} records")

    labels = [f"{b // 1000}k" for b in BUDGETS]
    correct, incorrect, n_seeds, late = curves(rows, BUDGETS)
    unanswered = [args.n_problems - c - i
                  for c, i in zip(correct, incorrect)]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(labels, correct, color="#2ecc71", label="Cumulative Average Correct")
    ax.bar(labels, incorrect, bottom=correct, color="#e74c3c",
           label="Cumulative Average Incorrect")
    ax.bar(labels, unanswered,
           bottom=[c + i for c, i in zip(correct, incorrect)],
           color="#95a5a6", label="Cumulative Average Unanswered")
    ax.set_ylim(0, args.n_problems)
    ax.set_xlabel("Thinking Token Budget")
    ax.set_ylabel(f"Cumulative Average Predictions (of {args.n_problems})")
    label = args.title or path.parent.name
    ax.set_title(f"Cumulative Average Predictions over thinking budget — {label}")
    ax.tick_params(axis="x", labelsize=7)
    ax.legend(fontsize=8)
    style_ax(ax)
    fig.tight_layout()
    target = out / f"{args.name}.{args.fmt}"
    fig.savefig(target, dpi=args.dpi)
    plt.close(fig)

    print(f"[written] {target}")
    print(f"seeds={n_seeds}  rows={len(rows)}  late completions clamped to "
          f"{BUDGETS[-1] // 1000}k={late}")
    print(f"final bar: correct={correct[-1]:.2f} incorrect={incorrect[-1]:.2f} "
          f"unanswered={unanswered[-1]:.2f} of {args.n_problems}")


if __name__ == "__main__":
    main()
