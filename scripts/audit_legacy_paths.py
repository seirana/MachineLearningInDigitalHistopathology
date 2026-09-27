from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


HOME_PATH_PATTERN = re.compile(r"/home/[A-Za-z0-9_.-]+/")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit archived legacy scripts for machine-specific paths."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.root.resolve()
    legacy_dir = root / "legacy"

    findings = []
    for path in sorted(legacy_dir.glob("*.py")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        matches = HOME_PATH_PATTERN.findall(text)
        if matches:
            findings.append(
                {
                    "file": str(path.relative_to(root)),
                    "hard_coded_home_path_occurrences": len(matches),
                }
            )

    result = {
        "scope": "historical Python scripts under legacy/",
        "files_with_machine_specific_home_paths": len(findings),
        "findings": findings,
        "note": (
            "These files are retained as historical research artifacts. "
            "The maintained src/ package is path-configurable."
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
