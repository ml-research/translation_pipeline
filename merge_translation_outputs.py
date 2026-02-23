#!/usr/bin/env python3
"""
Merge retry translations into a base output directory by row id.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Iterable


ROW_ID_KEYS = ("id", "_id", "qid", "doc_id", "task_id", "uuid")


def iter_jsonl(path: Path) -> Iterable[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def find_id_key(sample: dict, override: str | None) -> str:
    if override:
        if override not in sample:
            raise ValueError(f"Provided id key '{override}' not present in sample row")
        return override
    for key in ROW_ID_KEYS:
        if key in sample:
            return key
    raise ValueError("Could not infer row id key; pass --id-key explicitly")


def collect_files(output_dir: Path) -> list[Path]:
    files = sorted(output_dir.glob("*.jsonl*"))
    if not files:
        raise FileNotFoundError(f"No JSONL outputs found in {output_dir}")
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge retry translations into base outputs")
    parser.add_argument("--base-dir", required=True, type=str)
    parser.add_argument("--retry-dir", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--id-key", type=str, default=None, help="Row id key (optional)")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    retry_dir = Path(args.retry_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_files = collect_files(base_dir)
    retry_files = collect_files(retry_dir)

    sample_row = next(iter_jsonl(base_files[0]))
    id_key = find_id_key(sample_row, args.id_key)

    retry_map: dict[str, dict] = {}
    for path in retry_files:
        for row in iter_jsonl(path):
            row_id = row.get(id_key)
            if row_id is None:
                raise ValueError(f"Missing id key '{id_key}' in retry row from {path}")
            retry_map[str(row_id)] = row

    merged_path = output_dir / "merged.jsonl.gz"
    with gzip.open(merged_path, "wt", encoding="utf-8") as handle:
        replaced = 0
        total = 0
        for path in base_files:
            for row in iter_jsonl(path):
                total += 1
                row_id = row.get(id_key)
                if row_id is None:
                    raise ValueError(f"Missing id key '{id_key}' in base row from {path}")
                row_id = str(row_id)
                if row_id in retry_map:
                    row = retry_map.pop(row_id)
                    replaced += 1
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        if retry_map:
            for row in retry_map.values():
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Merged file: {merged_path}")
    print(f"Base rows: {total}")
    print(f"Replaced rows: {replaced}")
    if retry_map:
        print(f"Extra retry rows appended: {len(retry_map)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
