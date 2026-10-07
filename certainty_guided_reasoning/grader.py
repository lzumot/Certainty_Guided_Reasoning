"""Task registry and answer equality — single source of truth for grading."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class TaskConfig:
    """Static configuration that differs per benchmark task."""

    key: str
    dataset_id: str  # HuggingFace dataset name
    mode: str = "exact"  # "exact" | "numeric"
    probe_answer_tokens: int = 8  # max_new_tokens for probe decode
    process_answer: Callable[[str], str] | None = None  # dataset-load normalizer
    description: str = ""
    # Dataset-load metadata; the defaults reproduce the AIME layout.
    dataset_config: str | None = None  # HF config name; None = default config
    split: str = "train"
    problem_field: str = "problem"
    answer_field: str = "answer"


def _aime_norm(a: str) -> str:
    a = str(a or "").strip()
    return a.lstrip("0") or "0" if a else ""


def _identity(a: str) -> str:
    return str(a or "").strip()


def _gsm8k_gold_answer(rationale: str) -> str:
    """GSM8K gold: the number after the last ``####`` (thousands stripped)."""
    text = str(rationale or "")
    if "####" in text:
        text = text.rsplit("####", 1)[1]
    return text.strip().replace(",", "")


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
    "gsm8k": TaskConfig(
        key="gsm8k",
        dataset_id="openai/gsm8k",
        mode="numeric",
        dataset_config="main",
        split="test",
        problem_field="question",
        answer_field="answer",
        probe_answer_tokens=64,
        process_answer=_gsm8k_gold_answer,
        description="GSM8K (grade-school math; gold after '####', graded numerically)",
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
    cfg = resolve_task(task)
    if cfg.mode == "numeric":
        return _numeric_equal(extracted, ground_truth)
    return _aime_norm(extracted) == _aime_norm(ground_truth)


def infer_task(s, default: str = "aime2025") -> str:
    """Infer a task key from a problem_id or run-folder name."""
    s = str(s or "")
    for key in sorted(TASKS, key=len, reverse=True):
        if key in s:
            return key
    return default


def probe_answer_equal(answer, ground_truth, task: str) -> bool:
    """Probe-answer equality: numeric defers to answers_equal; exact strips
    ``}``/``\\`` then applies the leading-zero rule."""
    cfg = resolve_task(task)
    if cfg.mode != "exact":
        return answers_equal(answer, ground_truth, task)

    def _probe_norm(a):
        a = str(a or "").strip().replace("}", "").replace("\\", "").strip()
        return a.lstrip("0") or "0" if a else ""

    return _probe_norm(answer) == _probe_norm(ground_truth)


# ---------------------------------------------------------------------------
# Numeric grading (GSM8K): one number in the box, units/currency/% stripped.
# ---------------------------------------------------------------------------
_TEXT_COMMAND = re.compile(r"\\(?:text|textbf|textit|mathrm|mathbf|mbox)\s*\{([^{}]*)\}")
_THOUSANDS_SEP = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_FRACTION_LATEX = re.compile(r"(-?)\\frac\{(\d+)\}\{(\d+)\}")
_FRACTION_SLASH = re.compile(r"(-?)()(\d+)/(\d+)")


def _clean_prediction(string) -> str:
    """Strip units, currency / percent signs and thousands separators."""
    string = str(string)
    previous = None
    while previous != string:
        previous = string
        string = _TEXT_COMMAND.sub(r" \1 ", string)
    for token in ("\\$", "$", "\\%", "%"):
        string = string.replace(token, "")
    for token in ("\\!", "\\,", "\\;", "\\ "):
        string = string.replace(token, " " if token == "\\ " else "")
    string = string.replace("dfrac", "frac").replace("tfrac", "frac")
    string = _THOUSANDS_SEP.sub("", string)
    if "=" in string:
        string = string.split("=")[-1]
    return string.strip()


def _to_number(string):
    """The single number written in a boxed answer, or None."""
    cleaned = _clean_prediction(string)
    compact = cleaned.replace(" ", "")
    m = _FRACTION_LATEX.fullmatch(compact) or _FRACTION_SLASH.fullmatch(compact)
    if m:
        numerator, denominator = int(m.groups()[-2]), int(m.groups()[-1])
        if denominator == 0:
            return None
        return (-1 if m.group(1) else 1) * numerator / denominator
    numbers = _NUMBER.findall(cleaned)
    if len(numbers) != 1:
        return None
    return float(numbers[0])


def _numeric_equal(predicted, true) -> bool:
    """Number-only equality with the GSM8K tolerances (rel 1e-9, abs 1e-6)."""
    if predicted is None or true is None:
        return False
    p, t = _to_number(predicted), _to_number(true)
    if p is None or t is None:
        return False
    return math.isclose(p, t, rel_tol=1e-9, abs_tol=1e-6)
