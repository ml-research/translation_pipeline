#!/usr/bin/env python3
"""
Lightweight validation for Datatrove translation outputs.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


def iter_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def collect_files(output_dir: Path) -> list[Path]:
    files = sorted(output_dir.glob("*.jsonl*"))
    if not files:
        raise FileNotFoundError(f"No JSONL outputs found in {output_dir}")
    return files


def value_length(value) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, list):
        return sum(len(item) for item in value if isinstance(item, str))
    return 0


def is_empty(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, list):
        return all(is_empty(item) for item in value)
    return False


def collect_stats(output_dir: Path, fields: list[str], language: str, use_suffix: bool):
    files = collect_files(output_dir)
    total = 0
    missing = 0
    empty = 0
    failed = 0
    length_sums = {field: 0 for field in fields}
    length_counts = {field: 0 for field in fields}

    for path in files:
        for row in iter_jsonl(path):
            total += 1
            if row.get("_translation_failed") is True:
                failed += 1
            for field in fields:
                key = f"{field}_{language}" if use_suffix else field
                if key not in row:
                    missing += 1
                    continue
                value = row.get(key)
                if is_empty(value):
                    empty += 1
                    continue
                length_sums[field] += value_length(value)
                length_counts[field] += 1

    return {
        "files": len(files),
        "rows": total,
        "missing": missing,
        "empty": empty,
        "failed": failed,
        "length_sums": length_sums,
        "length_counts": length_counts,
    }


def validate(
    output_dir: Path,
    fields: list[str],
    language: str,
    sample: int,
    use_suffix: bool,
    compare_dir: Path | None,
    length_ratio_min: float,
    length_ratio_max: float,
) -> int:
    files = collect_files(output_dir)
    total = 0
    missing = 0
    empty = 0
    failed = 0
    samples = []

    for path in files:
        for row in iter_jsonl(path):
            total += 1
            if row.get("_translation_failed") is True:
                failed += 1
            for field in fields:
                key = f"{field}_{language}" if use_suffix else field
                if key not in row:
                    missing += 1
                    if len(samples) < sample:
                        samples.append((key, row))
                    continue
                value = row.get(key)
                if is_empty(value):
                    empty += 1
                    if len(samples) < sample:
                        samples.append((key, row))

    print(f"Files: {len(files)}")
    print(f"Rows: {total}")
    print(f"Missing translated fields: {missing}")
    print(f"Empty translated fields: {empty}")
    print(f"_translation_failed True: {failed}")
    if failed:
        print("! Found rows with _translation_failed == True")

    if samples:
        print("\nSample issues:")
        for key, row in samples:
            print(f"- {key} missing/empty; keys: {sorted(row.keys())}")

    if compare_dir:
        base_stats = collect_stats(compare_dir, fields, "en", use_suffix)
        target_stats = collect_stats(output_dir, fields, language, use_suffix)
        print("\nComparison vs English:")
        print(f"- English rows: {base_stats['rows']}")
        print(f"- Target rows: {target_stats['rows']}")
        if base_stats["rows"] != target_stats["rows"]:
            print("! Row count mismatch")
        for field in fields:
            en_count = base_stats["length_counts"][field]
            tgt_count = target_stats["length_counts"][field]
            en_mean = base_stats["length_sums"][field] / max(1, en_count)
            tgt_mean = target_stats["length_sums"][field] / max(1, tgt_count)
            ratio = tgt_mean / en_mean if en_mean else 0.0
            status = "OK"
            if ratio < length_ratio_min or ratio > length_ratio_max:
                status = "OUT_OF_RANGE"
            print(
                f"- {field}: en_mean={en_mean:.1f} tgt_mean={tgt_mean:.1f} "
                f"ratio={ratio:.2f} {status}"
            )

    return 0


def parse_comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Datatrove translation outputs")
    parser.add_argument("--output-dir", required=True, type=str, help="Output config directory")
    parser.add_argument("--language", required=True, type=str, help="Target language code")
    parser.add_argument(
        "--fields",
        type=str,
        required=True,
        help="Comma-separated list of base fields to check (e.g., input,context,answers)",
    )
    parser.add_argument("--sample", type=int, default=3, help="Number of issue samples to print")
    parser.add_argument("--no-suffix", action="store_true", help="Validate base fields without _{lang} suffix")
    parser.add_argument("--compare-dir", type=str, default=None, help="English output config directory for comparison")
    parser.add_argument("--length-ratio-min", type=float, default=0.5)
    parser.add_argument("--length-ratio-max", type=float, default=2.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    fields = parse_comma_list(args.fields)
    compare_dir = Path(args.compare_dir) if args.compare_dir else None
    return validate(
        output_dir,
        fields,
        args.language,
        args.sample,
        use_suffix=not args.no_suffix,
        compare_dir=compare_dir,
        length_ratio_min=args.length_ratio_min,
        length_ratio_max=args.length_ratio_max,
    )


if __name__ == "__main__":
    raise SystemExit(main())
