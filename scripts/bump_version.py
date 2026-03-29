#!/usr/bin/env python3
"""
Bump the version in pyproject.toml.

Usage:
    python scripts/bump_version.py           # bump patch (default)
    python scripts/bump_version.py --patch   # bump patch
    python scripts/bump_version.py --minor   # bump minor, reset patch to 0
    python scripts/bump_version.py --major   # bump major, reset minor+patch to 0
    python scripts/bump_version.py --set 1.2.3  # set an explicit version
"""

import argparse
import re
import sys
from pathlib import Path

PYPROJECT = Path(__file__).parent.parent / "pyproject.toml"
VERSION_RE = re.compile(r'^(version\s*=\s*")(\d+)\.(\d+)\.(\d+)(")', re.MULTILINE)


def read_version(text: str) -> tuple[int, int, int]:
    m = VERSION_RE.search(text)
    if not m:
        sys.exit("Could not find version in pyproject.toml")
    return int(m.group(2)), int(m.group(3)), int(m.group(4))


def write_version(text: str, major: int, minor: int, patch: int) -> str:
    new_ver = f"{major}.{minor}.{patch}"
    return VERSION_RE.sub(lambda m: f"{m.group(1)}{new_ver}{m.group(5)}", text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bump pyproject.toml version")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--patch", action="store_true", help="Bump patch (default)")
    group.add_argument("--minor", action="store_true", help="Bump minor, reset patch")
    group.add_argument("--major", action="store_true", help="Bump major, reset minor+patch")
    group.add_argument("--set", metavar="VERSION", help="Set an explicit version")
    args = parser.parse_args()

    text = PYPROJECT.read_text(encoding="utf-8")
    major, minor, patch = read_version(text)
    old_ver = f"{major}.{minor}.{patch}"

    if args.set:
        parts = args.set.strip().lstrip("v").split(".")
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            sys.exit(f"Invalid version format: {args.set!r} — expected MAJOR.MINOR.PATCH")
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
    elif args.minor:
        minor += 1
        patch = 0
    elif args.major:
        major += 1
        minor = 0
        patch = 0
    else:
        # --patch is the default
        patch += 1

    new_ver = f"{major}.{minor}.{patch}"
    PYPROJECT.write_text(write_version(text, major, minor, patch), encoding="utf-8")
    print(f"Bumped {old_ver} → {new_ver}")


if __name__ == "__main__":
    main()
