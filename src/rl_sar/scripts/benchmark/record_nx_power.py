#!/usr/bin/env python3
import argparse
import csv
import math
import re
import selectors
import statistics
import subprocess
import time
from pathlib import Path


POWER_PATTERNS = (
    re.compile(r"\bVDD_IN\s+(\d+(?:\.\d+)?)mW/"),
    re.compile(r"\bPOM_5V_IN\s+(\d+(?:\.\d+)?)/"),
)


def parse_power_w(line):
    for pattern in POWER_PATTERNS:
        match = pattern.search(line)
        if match:
            return float(match.group(1)) / 1000.0
    return None


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


def main():
    parser = argparse.ArgumentParser(
        description="Record Jetson NX input power from tegrastats."
    )
    parser.add_argument("--condition", default="guide", help="Condition label.")
    parser.add_argument("--duration", type=float, default=60.0, help="Capture seconds.")
    parser.add_argument("--interval-ms", type=int, default=100, help="tegrastats interval.")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results"))
    args = parser.parse_args()
    if args.duration <= 0.0:
        parser.error("--duration must be positive")
    if args.interval_ms <= 0:
        parser.error("--interval-ms must be positive")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"{args.condition}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    raw_path = run_dir / "tegrastats.log"
    samples_path = run_dir / "power_samples.csv"
    summary_path = run_dir / "power_summary.csv"

    command = ["tegrastats", "--interval", str(args.interval_ms)]
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError as error:
        raise SystemExit("tegrastats was not found on this machine") from error

    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    samples = []
    started = time.monotonic()
    deadline = started + args.duration
    try:
        with raw_path.open("w") as raw_stream:
            while time.monotonic() < deadline:
                timeout = min(1.0, max(0.0, deadline - time.monotonic()))
                events = selector.select(timeout)
                for key, _ in events:
                    line = key.fileobj.readline()
                    if not line:
                        if process.poll() is not None:
                            raise RuntimeError(
                                f"tegrastats exited early with code {process.returncode}"
                            )
                        continue
                    elapsed_s = time.monotonic() - started
                    raw_stream.write(line)
                    raw_stream.flush()
                    power_w = parse_power_w(line)
                    if power_w is not None:
                        samples.append((elapsed_s, power_w))
    finally:
        selector.close()
        process.terminate()
        try:
            process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3.0)

    if not samples:
        raise SystemExit(
            f"No VDD_IN or POM_5V_IN samples found; inspect {raw_path}"
        )

    with samples_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("elapsed_s", "power_w"))
        writer.writerows(samples)

    values = [power_w for _, power_w in samples]
    result = {
        "condition": args.condition,
        "duration_s": args.duration,
        "interval_ms": args.interval_ms,
        "sample_count": len(values),
        "mean_power_w": statistics.fmean(values),
        "std_power_w": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p95_power_w": percentile(values, 0.95),
        "peak_power_w": max(values),
    }
    with summary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=result.keys())
        writer.writeheader()
        writer.writerow(result)

    print(
        f"NX power: {result['mean_power_w']:.3f} +/- "
        f"{result['std_power_w']:.3f} W"
    )
    print(f"P95 power: {result['p95_power_w']:.3f} W")
    print(f"samples_csv: {samples_path}")
    print(f"summary_csv: {summary_path}")


if __name__ == "__main__":
    main()
