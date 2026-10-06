"""Task registry and answer equality — single source of truth for grading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

@dataclass(frozen=True)
class TaskConfig:
    """Static configuration that differs per benchmark task."""

    key: str
    dataset_id: str  # HuggingFace dataset name
    probe_answer_tokens: int = 8  # max_new_tokens for probe decode
    process_answer: Callable[[str], str] | None = None  # dataset-load normalizer
    description: str = ""


def _aime_norm(a: str) -> str:
    a = str(a or "").strip()
    return a.lstrip("0") or "0" if a else ""


def _identity(a: str) -> str:
    return str(a or "").strip()


TASKS: dict[str, TaskConfig] = {
    "aime2024": TaskConfig(
        key="aime2024",
        dataset_id="HuggingFaceH4/aime_2024",
        probe_answer_tokens=8,
        process_answer=lambda a: a.lstrip("0") or "0",
        description="AIME 2024 (integers 0-999, two-digit zero-padded)",
    ),
    "aime2025": TaskConfig(
        key="aime2025",
        dataset_id="MathArena/aime_2025",
        probe_answer_tokens=8,
        process_answer=_identity,
        description="AIME 2025 (integers 0-999)",
    ),
}


def resolve_task(task: str) -> TaskConfig:
    try:
        return TASKS[task]
    except KeyError:
        raise KeyError(f"unknown task {task!r}; known: {sorted(TASKS)}") from None


def process_answer_for(task: str, answer) -> str:
    cfg = resolve_task(task)
    if cfg.process_answer is None:
        return clean_answer(answer)
    return cfg.process_answer(answer)


def clean_answer(s: str) -> str:
    """Trim and unwrap a single outer ``\\boxed{...}`` / ``{...}`` / ``$...$``."""
    s = str(s or "").strip()
    s = s.replace("$", "")
    s = s.strip()
    if s.startswith(r"\boxed{"):
        s = s[len(r"\boxed{") :]
        if s.endswith("}"):
            s = s[:-1]
    elif s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    return s.strip()


def answers_equal(extracted, ground_truth, task: str) -> bool:
    resolve_task(task)
    return _aime_norm(extracted) == _aime_norm(ground_truth)


def infer_task(s, default: str = "aime2025") -> str:
    """Infer a task key from a problem_id or run-folder name."""
    s = str(s or "")
    for key in sorted(TASKS, key=len, reverse=True):
        if key in s:
            return key
    return default


def probe_answer_equal(answer, ground_truth, task: str) -> bool:
    """Probe-answer equality (strips ``}``/``\\``, then the leading-zero rule)."""
    resolve_task(task)

    def _probe_norm(a):
        a = str(a or "").strip().replace("}", "").replace("\\", "").strip()
        return a.lstrip("0") or "0" if a else ""

    return _probe_norm(answer) == _probe_norm(ground_truth)
