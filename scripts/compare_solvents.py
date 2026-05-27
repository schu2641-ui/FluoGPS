from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CSV_A = ROOT_DIR / "examples" / "inputs" / "long_tztz_experiment.csv"
DEFAULT_CSV_B = ROOT_DIR / "examples" / "inputs" / "long_tztz.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Compare solvent values in two CSV files.")
    parser.add_argument("csv_a", nargs="?", type=Path, default=DEFAULT_CSV_A, help="First CSV path.")
    parser.add_argument("csv_b", nargs="?", type=Path, default=DEFAULT_CSV_B, help="Second CSV path.")
    parser.add_argument("--column", default="solvent", help="Column name to compare.")
    return parser.parse_args()


def read_values(csv_path, column):
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if csv_path.stat().st_size == 0:
        raise ValueError(f"CSV is empty: {csv_path}")

    counter = Counter()
    row_count = 0
    empty_count = 0

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        if column not in reader.fieldnames:
            columns = ", ".join(reader.fieldnames)
            raise ValueError(f"Column '{column}' not found in {csv_path}. Available columns: {columns}")

        for row in reader:
            row_count += 1
            value = (row.get(column) or "").strip()
            if value:
                counter[value] += 1
            else:
                empty_count += 1

    return counter, row_count, empty_count


def print_section(title, values, counts_a, counts_b=None):
    print(f"\n{title} ({len(values)})")
    if not values:
        print("  <none>")
        return

    for value in sorted(values):
        if counts_b is None:
            print(f"  {value}\tcount={counts_a[value]}")
        else:
            print(f"  {value}\ta_count={counts_a[value]}\tb_count={counts_b[value]}")


def main():
    args = parse_args()
    try:
        counts_a, rows_a, empty_a = read_values(args.csv_a, args.column)
        counts_b, rows_b, empty_b = read_values(args.csv_b, args.column)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    solvents_a = set(counts_a)
    solvents_b = set(counts_b)

    common = solvents_a & solvents_b
    only_a = solvents_a - solvents_b
    only_b = solvents_b - solvents_a

    print(f"CSV A: {args.csv_a}")
    print(f"  rows={rows_a}, unique_{args.column}={len(solvents_a)}, empty_{args.column}={empty_a}")
    print(f"CSV B: {args.csv_b}")
    print(f"  rows={rows_b}, unique_{args.column}={len(solvents_b)}, empty_{args.column}={empty_b}")

    print_section("Common solvents", common, counts_a, counts_b)
    print_section("Only in CSV A", only_a, counts_a)
    print_section("Only in CSV B", only_b, counts_b)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
