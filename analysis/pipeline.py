"""Grid analysis: traces -> grid_summary.md + grid_records.csv.

    uv run python -m analysis.pipeline [TRACE_DIR] [--out outputs/{RUN_NAME}]
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from certainty_guided_reasoning.paths import TRACES_DIR
from certainty_guided_reasoning.grader import answers_equal, infer_task


@dataclass
class Rec:
    key: str
    seed: int
    problem_id: str
    extracted_answer: str | None
    ground_truth: str
    certainty: float | None
    certainty_greedy: float | None
    num_thinking_tokens: int
    finish_reason: str | None
    task: str = ""
    correct: bool = field(init=False)

    def __post_init__(self):
        if not self.task:
            self.task = infer_task(self.problem_id)
        self.correct = answers_equal(
            self.extracted_answer, self.ground_truth, self.task
        )


def load_records(root: Path) -> list[Rec]:
    recs = []
    for p in sorted(root.glob("*.json")):
        try:
            r = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            print(f"[warn] corrupt {p.name}: {e}")
            continue
        recs.append(Rec(
            key=r["key"], seed=r["seed"], problem_id=r["problem_id"],
            extracted_answer=r.get("extracted_answer"),
            ground_truth=r.get("ground_truth") or "",
            certainty=r.get("certainty")
            if isinstance(r.get("certainty"), (int, float)) else None,
            certainty_greedy=r.get("certainty_greedy")
            if isinstance(r.get("certainty_greedy"), (int, float)) else None,
            num_thinking_tokens=r.get("num_thinking_tokens") or 0,
            finish_reason=r.get("finish_reason"),
            task=r.get("task") or infer_task(r.get("problem_id", "")),
        ))
    return recs


def fmt(xs: list[float]) -> str:
    if not xs:
        return "n/a"
    xs = sorted(xs)
    mean = sum(xs) / len(xs)
    med = xs[len(xs) // 2]
    return f"mean={mean:.4f} median={med:.4f} min={xs[0]:.4f} max={xs[-1]:.4f}"


def majority_vote(recs: list[Rec]) -> tuple[str | None, int]:
    """Most common extracted answer among seeds for one problem."""
    counts: dict[str, int] = defaultdict(int)
    for r in recs:
        if r.extracted_answer:
            counts[str(r.extracted_answer).strip()] += 1
    if not counts:
        return None, 0
    ans, n = max(counts.items(), key=lambda kv: kv[1])
    return ans, n


def theta_sweep(recs: list[Rec], attr: str, thetas) -> list[dict]:
    rows = []
    for theta in thetas:
        conf = [r for r in recs if getattr(r, attr) is not None
                and getattr(r, attr) >= theta]
        cov = len(conf) / len(recs) if recs else 0.0
        acc = (sum(r.correct for r in conf) / len(conf)) if conf else float("nan")
        rows.append({"theta": theta, "coverage": cov, "n": len(conf),
                     "acc_in_confident": acc})
    return rows


def build_report(recs: list[Rec], run_name: str) -> str:
    L: list[str] = []
    add = L.append
    total = len(recs)
    n_prob = len({r.problem_id for r in recs})
    seeds_per_problem = total / n_prob if n_prob else 0

    add(f"# CGR Grid Analysis — {run_name}\n")
    add(f"- records: **{total}** across **{n_prob} problems** "
        f"(~{seeds_per_problem:.1f} seeds/problem so far)")
    add(f"- baseline pass@1: "
        f"**{sum(r.correct for r in recs)/total:.4f}**\n")

    # --- finish reasons ---
    fr = defaultdict(int)
    for r in recs:
        fr[r.finish_reason] += 1
    add("## Finish reasons\n")
    add("| finish_reason | n | share |")
    add("|---|---|---|")
    for k, v in sorted(fr.items(), key=lambda kv: -kv[1]):
        add(f"| {k} | {v} | {v/total:.1%} |")
    add("")

    # --- accuracy by finish reason ---
    add("## Accuracy by finish reason\n")
    add("| finish_reason | n | accuracy | mean tokens |")
    add("|---|---|---|---|")
    for k in sorted(fr):
        rs = [r for r in recs if r.finish_reason == k]
        acc = sum(r.correct for r in rs) / len(rs)
        tok = sum(r.num_thinking_tokens for r in rs) / len(rs)
        add(f"| {k} | {len(rs)} | {acc:.3f} | {tok:.0f} |")
    add("")

    # --- certainty by correctness x finish ---
    add("## Certainty by correctness (split by finish reason)\n")
    for attr, label in (("certainty", "sampled"), ("certainty_greedy", "greedy")):
        add(f"### {label} (`{attr}`)\n")
        add("| subset | n | stats |")
        add("|---|---|---|")
        subsets = [
            ("correct, stop", [r for r in recs if r.correct and r.finish_reason == "stop"]),
            ("correct, length", [r for r in recs if r.correct and r.finish_reason == "length"]),
            ("wrong, stop", [r for r in recs if not r.correct and r.finish_reason == "stop"]),
            ("wrong, length", [r for r in recs if not r.correct and r.finish_reason == "length"]),
        ]
        for name, rs in subsets:
            vals = [getattr(r, attr) for r in rs if getattr(r, attr) is not None]
            invalid = len(rs) - len(vals)
            extra = f" (+{invalid} invalid)" if invalid else ""
            add(f"| {name} | {len(vals)}{extra} | {fmt(vals)} |")
        add("")

    # --- per-problem table ---
    by_problem: dict[str, list[Rec]] = defaultdict(list)
    for r in recs:
        by_problem[r.problem_id].append(r)
    add("## Per-problem results\n")
    add("| problem | n_seeds | pass@1 | maj@seeds | maj_answer | truth |")
    add("|---|---|---|---|---|---|")
    for pid in sorted(by_problem):
        rs = by_problem[pid]
        acc = sum(r.correct for r in rs) / len(rs)
        maj_ans, _ = majority_vote(rs)
        maj_ok = answers_equal(maj_ans, rs[0].ground_truth, rs[0].task)
        add(f"| {pid} | {len(rs)} | {acc:.2f} | "
            f"{'✓' if maj_ok else '✗'} | {maj_ans} | {rs[0].ground_truth} |")
    n_maj = sum(
        1 for pid, rs in by_problem.items()
        if answers_equal(majority_vote(rs)[0], rs[0].ground_truth, rs[0].task)
    )
    add(f"\nMajority-vote (self-consistency) accuracy: "
        f"**{n_maj}/{n_prob} = {n_maj/n_prob:.4f}**\n")

    # --- theta sweeps ---
    thetas = (0.5, 0.8, 0.9, 0.95, 0.99)
    for attr, label in (("certainty", "sampled"), ("certainty_greedy", "greedy")):
        add(f"## Theta sweep — {label}\n")
        add("Coverage = share of all traces with certainty ≥ θ. "
            "`acc_in_confident` = accuracy among those traces "
            "(how well θ separates correct from wrong).\n")
        add("| θ | coverage | n | acc in confident set |")
        add("|---|---|---|---|")
        for row in theta_sweep(recs, attr, thetas):
            add(f"| {row['theta']:.2f} | {row['coverage']:.2%} | "
                f"{row['n']} | {row['acc_in_confident']:.4f} |")
        add("")
    return "\n".join(L)


def write_csv(recs: list[Rec], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "seed", "problem_id", "extracted_answer",
                    "ground_truth", "correct", "certainty", "certainty_greedy",
                    "num_thinking_tokens", "finish_reason"])
        for r in recs:
            w.writerow([r.key, r.seed, r.problem_id, r.extracted_answer,
                        r.ground_truth, r.correct, r.certainty,
                        r.certainty_greedy, r.num_thinking_tokens,
                        r.finish_reason])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir", nargs="?", default=str(TRACES_DIR))
    ap.add_argument("--out", default=None,
                    help="output run dir (default: parent of trace_dir)")
    args = ap.parse_args()

    root = Path(args.trace_dir)
    # Default output to THIS run's root (parent of the traces dir) so each
    # model's report/CSV always land in that model's own folder.
    out = Path(args.out) if args.out else root.parent
    recs = load_records(root)
    if not recs:
        print(f"no records in {root}")
        return

    out.mkdir(parents=True, exist_ok=True)

    report = build_report(recs, out.name)
    (out / "grid_summary.md").write_text(report)
    write_csv(recs, out / "grid_records.csv")

    print(report)
    print(f"\n[written] {out/'grid_summary.md'}")
    print(f"[written] {out/'grid_records.csv'}")


if __name__ == "__main__":
    main()
