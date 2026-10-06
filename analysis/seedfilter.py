"""Shared --max-seed/--min-seed filtering (excludes 9B dev leftovers at seeds 90-91)."""

from __future__ import annotations

import argparse
import re

_SEED_IN_KEY = re.compile(r"seed(\d+)")


def record_seed(rec: dict) -> int:
    """Seed of a probe or trace record, from its field or from its key."""
    seed = rec.get("seed")
    if seed is not None:
        return int(seed)
    match = _SEED_IN_KEY.search(str(rec.get("key", "")))
    return int(match.group(1)) if match else -1


def filter_seeds(recs: list[dict], max_seed: int | None = None,
                 min_seed: int | None = None) -> list[dict]:
    """Keep records with min_seed <= seed <= max_seed; None means unbounded."""
    return [r for r in recs
            if (max_seed is None or record_seed(r) <= max_seed)
            and (min_seed is None or record_seed(r) >= min_seed)]


def add_seed_args(ap: argparse.ArgumentParser) -> None:
    """Add the shared seed-range flags to a figure script's parser."""
    ap.add_argument("--max-seed", type=int, default=None,
                    help="drop records with a larger seed (the 9B run holds "
                         "dev-run leftovers at seeds 90-91; use 64 for the "
                         "clean grid)")
    ap.add_argument("--min-seed", type=int, default=None,
                    help="drop records with a smaller seed (for held-out "
                         "splits)")
