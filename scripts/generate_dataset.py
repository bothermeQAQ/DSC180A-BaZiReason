"""Convenience wrapper; see bazireason --help."""

import sys

from bazireason.cli import main

if __name__ == "__main__":
    sys.argv.insert(1, "generate")
    main()
