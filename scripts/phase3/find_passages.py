"""Print passages of the downloaded classics that contain a phrase (research helper)."""

import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
files = sys.argv[2].split(",")
for phrase in sys.argv[3:]:
    for name in files:
        text = (root / name).read_text(encoding="utf-8")
        for match in list(re.finditer(re.escape(phrase), text))[:2]:
            start, end = max(0, match.start() - 60), min(len(text), match.end() + 110)
            snippet = text[start:end].replace("\n", "⏎")
            print(f"[{phrase}] {name}@{match.start()}: {snippet}")
