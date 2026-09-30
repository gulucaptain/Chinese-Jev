"""`python -m chinese_jev` — same entry point as the `chinese-jev` console script."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
