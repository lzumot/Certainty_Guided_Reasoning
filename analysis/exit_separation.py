"""Exited vs held-back traces at each theta, from saved probes.

    uv run python -m analysis.exit_separation Qwen3.5-9B_aime2025
"""

from __future__ import annotations

import sys
from pathlib import Path

from analysis.replay import MAX_SEED, THETAS, correct, load_probes


def split(recs: list[dict], theta: float) -> dict:
    """Count and score the exited against the held-back traces."""
    out = {
        "n": 0,
        "exit_n": 0, "exit_ok": 0, "exit_noanswer": 0,
        "held_n": 0, "held_ok": 0,
    }
    for r in recs:
        answered = [p for p in r["probes"] if p.get("answer") is not None]
        if not answered:
            continue
        last = answered[-1]
        out["n"] += 1
        hit = next((p for p in r["probes"]
                    if p.get("certainty") is not None
                    and p["certainty"] >= theta), None)
        if hit is None:
            out["held_n"] += 1
            out["held_ok"] += correct(r, last["answer"])
        else:
            out["exit_n"] += 1
            if hit.get("answer") is None:
                out["exit_noanswer"] += 1
            else:
                out["exit_ok"] += correct(r, hit["answer"])
    return out


def coverage(recs: list[dict], theta: float) -> dict:
    """Per-problem breakdown of the exited group."""
    seen: dict[str, list[bool]] = {}
    for r in recs:
        hit = next((p for p in r["probes"]
                    if p.get("certainty") is not None
                    and p["certainty"] >= theta), None)
        if hit is None or hit.get("answer") is None:
            continue
        pid = r.get("problem_id", "?")
        seen.setdefault(pid, []).append(correct(r, hit["answer"]))
    return {
        "problems_covered": len(seen),
        "problems_with_wrong": sum(1 for v in seen.values() if not all(v)),
        "answers": sum(len(v) for v in seen.values()),
    }


def main() -> None:
    run = sys.argv[1] if len(sys.argv) > 1 else "Qwen3.5-9B_aime2025"
    root = Path("outputs") / run
    recs = [r for r in load_probes(root / "probes")
            if r.get("seed", -1) <= MAX_SEED]

    lines = [f"# Exit separation — {run}", "",
             f"- clean grid, seeds 0-{MAX_SEED}, min statistic, greedy probe",
             f"- traces with at least one decoded probe answer: "
             f"{split(recs, 0.99)['n']}", ""]
    print(f"{run}: {split(recs, 0.99)['n']} traces on the clean grid")

    header = (f"{'theta':>6} {'exited':>7} {'exited ok':>10} {'held back':>10} "
              f"{'held ok':>8}")
    print(header)
    print("-" * len(header))

    rows = []
    for t in THETAS:
        s = split(recs, t)
        rows.append((t, s))
        print(f"{t:>6.2f} {s['exit_n']:>7} "
              f"{s['exit_ok'] / s['exit_n']:>10.4f} {s['held_n']:>10} "
              f"{s['held_ok'] / s['held_n']:>8.4f}")

    lines += ["| theta | exited | correct when exited | held back | correct when held back |",
              "|---|---|---|---|---|"]
    for t, s in rows:
        lines.append(
            f"| {t:g} | {s['exit_n']} | {s['exit_ok'] / s['exit_n']:.4f} | "
            f"{s['held_n']} | {s['held_ok'] / s['held_n']:.4f} |")

    t0, s0 = rows[-1]
    cov = coverage(recs, t0)
    lines += ["", f"## theta = {t0:g}, the operating point", "",
              f"- exited: {s0['exit_n']} of {s0['n']} traces, "
              f"{s0['exit_ok']} correct "
              f"({s0['exit_ok'] / s0['exit_n'] * 100:.1f} %). "
              f"{s0['exit_noanswer']} of the exits carry no decoded answer.",
              f"- held back: {s0['held_n']} traces, {s0['held_ok']} correct when "
              f"forced to answer at the last probe point "
              f"({s0['held_ok'] / s0['held_n'] * 100:.1f} %).",
              f"- so {s0['held_ok']} correct answers sit below the threshold, "
              f"against {s0['exit_n'] - s0['exit_ok']} incorrect answers above it.",
              f"- the exits cover {cov['problems_covered']} of the 30 problems "
              f"over {cov['answers']} answers, and "
              f"{cov['problems_with_wrong']} of those "
              f"{cov['problems_covered']} problems carry at least one wrong "
              f"exited answer.",
              "", f"[written] figs_clean65/exit_separation.md"]

    out = root / "figs_clean65"
    out.mkdir(parents=True, exist_ok=True)
    (out / "exit_separation.md").write_text("\n".join(lines) + "\n")
    print(f"\n[written] {out / 'exit_separation.md'}")


if __name__ == "__main__":
    main()
