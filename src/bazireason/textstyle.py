"""Version of the worked-answer wording used when generating training text.

Version 1 is the wording of ``p2_train_v1`` and stays the default so that
dataset regenerates byte for byte. Version 2 (``p2_train_v2``) changes two
sentences after the first training round's error analysis: the day-number sum is
written as single additions, and each stem's polarity is stated before the Ten
God is named. Evaluation never depends on this setting; only reference answers do.
"""

CURRENT = {"version": 1}


def set_version(version: int) -> None:
    if version not in (1, 2):
        raise ValueError("Unknown wording version")
    CURRENT["version"] = version


def version() -> int:
    return CURRENT["version"]
