"""Verify the archived notebook, original results, and supplied abstract PDF."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    originals = manifest["original_project_files"] + [manifest["supplied_abstract"]]
    supplements = manifest.get("derived_supplements", [])
    entries = originals + supplements
    failures = []
    for entry in entries:
        relative = entry["repository_path"]
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            failures.append(f"Invalid archive path: {relative}")
        elif not path.is_file():
            failures.append(f"Missing: {relative}")
        elif path.stat().st_size != entry["bytes"]:
            failures.append(f"Size mismatch: {relative}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            failures.append(f"SHA-256 mismatch: {relative}")
    if failures:
        print("Archive verification FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print(f"Verified {len(originals)} original artifacts and {len(supplements)} derived supplements against SOURCE_MANIFEST.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
