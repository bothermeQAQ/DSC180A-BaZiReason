"""Phase 6 round 2: the short system prompt and which models were trained with it.

A model always receives the system prompt it was trained with. Round-2 training sets record
theirs in ``data/phase6/train/<recipe>/manifest.json`` (``"system": "short"``); every other
model uses the Phase 2 prompt. Only the main system prompt is swapped: the strict-format
calendar tasks and the multiple-choice task keep theirs, because they define another format.
"""

import json
from functools import cache
from pathlib import Path

from bazireason.p2_prompts import P2_SYSTEM

SHORT_SYSTEM = (
    "你是 BaZiReason，可核验的八字历法与规则推理助手。"
    "先写依据，最后一行以“结论JSON:”开头给出 JSON 对象。"
)
TRAIN = Path("data/phase6/train")


def recipe_of(label: str) -> str | None:
    """``p6u2`` -> ``u2``; other labels (base, p2, p4a1, checkpoints of a panel) -> None."""
    if label.startswith("panel-"):
        label = label.split("-")[1]
    return label[2:] if label.startswith("p6") else None


@cache
def manifest(recipe: str) -> dict:
    path = TRAIN / recipe / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def system_for(label: str) -> str:
    recipe = recipe_of(label)
    if recipe and manifest(recipe).get("system") == "short":
        return SHORT_SYSTEM
    return P2_SYSTEM


def swap(messages: list, system: str) -> list:
    """The same conversation with the main system prompt replaced (other prompts untouched)."""
    if system == P2_SYSTEM:
        return messages
    return [
        {"role": "system", "content": system}
        if m["role"] == "system" and m["content"] == P2_SYSTEM
        else m
        for m in messages
    ]


def apply(label: str, conversations: list) -> list:
    system = system_for(label)
    if system == P2_SYSTEM:
        return conversations
    return [swap(c, system) for c in conversations]


def lineage(recipe: str) -> list:
    """A recipe and its ancestors (continuations name their parent in the manifest)."""
    out = []
    while recipe and recipe not in out:
        out.append(recipe)
        recipe = manifest(recipe).get("parent")
    return out
