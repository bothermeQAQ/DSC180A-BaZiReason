"""Version of the worked-answer wording used when generating training text.

Version 1 is the wording of ``p2_train_v1`` and stays the default so that
dataset regenerates byte for byte. Version 2 (``p2_train_v2``) changes two
sentences after the first training round's error analysis: the day-number sum is
written as single additions, and each stem's polarity is stated before the Ten
God is named. Version 3 (``p3_train_v2``) keeps those and, in the Phase 3 task
builders, writes the Ten-God table of the day master before any Ten God is read
from it. Version 4 (``p3_train_v3``) also drops the name of a classical book from
answer text, after the round-2 adapter started citing (and inventing) books.
Evaluation never depends on this setting; only reference answers do.
"""

CURRENT = {"version": 1}


def set_version(version: int) -> None:
    if version not in (1, 2, 3, 4):
        raise ValueError("Unknown wording version")
    CURRENT["version"] = version


def version() -> int:
    return CURRENT["version"]
