#!/usr/bin/env python3
import argparse
import csv
import math
import statistics
from pathlib import Path


def percentile(values, fraction):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def load_guide_samples(csv_path):
    samples = []
    with csv_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "guide_infer_ms" not in reader.fieldnames:
            raise ValueError(f"{csv_path} has no guide_infer_ms column")
        for row in reader:
            try:
                value = float(row["guide_infer_ms"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                samples.append(value)
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Summarize synchronized full-GUIDE inference latency."
    )
    parser.add_argument(
        "perf_csv",
        type=Path,
        help="One rl_real_perf_*.csv run.",
    )
    parser.add_argument(
        "--discard-first",
        type=int,
        default=0,
        help="Discard this many finite GUIDE samples (default: 0).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Summary CSV path (default: <input>_guide_summary.csv).",
    )
    args = parser.parse_args()

    if args.discard_first < 0:
        parser.error("--discard-first must be non-negative")

    samples = load_guide_samples(args.perf_csv)[args.discard_first :]
    if not samples:
        raise SystemExit(
            f"No finite guide_infer_ms samples remain in {args.perf_csv} after filtering."
        )

    result = {
        "source_csv": str(args.perf_csv.resolve()),
        "discarded_samples": args.discard_first,
        "sample_count": len(samples),
        "mean_ms": statistics.fmean(samples),
        "std_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "p95_ms": percentile(samples, 0.95),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }

    output_path = args.output
    if output_path is None:
        output_path = args.perf_csv.with_name(
            f"{args.perf_csv.stem}_guide_summary.csv"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=result.keys())
        writer.writeheader()
        writer.writerow(result)

    print(f"samples: {result['sample_count']}")
    print(f"GUIDE latency: {result['mean_ms']:.3f} +/- {result['std_ms']:.3f} ms")
    print(f"P95 latency: {result['p95_ms']:.3f} ms")
    print(f"range: [{result['min_ms']:.3f}, {result['max_ms']:.3f}] ms")
    print(f"summary_csv: {output_path}")


if __name__ == "__main__":
    main()
