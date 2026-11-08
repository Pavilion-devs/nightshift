"""Copy the repo README into the plugin folder, where the Hermes catalog page reads it (``subdir: nightshift``).

Relative links and images are rewritten one level up so they still resolve from ``nightshift/README.md``.

    python scripts/sync_readme.py          # write nightshift/README.md
    python scripts/sync_readme.py --check  # exit 1 if it is out of date
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "README.md"
TARGET = ROOT / "nightshift" / "README.md"
HEADER = "<!-- Generated from ../README.md by scripts/sync_readme.py. Edit that file instead. -->\n\n"
_LINK = re.compile(r"(\]\()([^)\s]+)(\))")


def _relink(match: re.Match[str]) -> str:
    target = match.group(2)
    if re.match(r"^([a-z][a-z0-9+.-]*:|#|/)", target, re.I):
        return match.group(0)
    return f"{match.group(1)}../{target}{match.group(3)}"


def render(text: str) -> str:
    return HEADER + _LINK.sub(_relink, text)


def main(argv: list[str]) -> int:
    want = render(SOURCE.read_text(encoding="utf-8"))
    if "--check" in argv:
        have = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""
        if have != want:
            print("nightshift/README.md is out of date: run python scripts/sync_readme.py", file=sys.stderr)
            return 1
        return 0
    TARGET.write_text(want, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
