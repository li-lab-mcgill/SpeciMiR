#!/usr/bin/env python3
"""
Benchmark RNAhybrid inference time and memory usage against increasing RNA sequence lengths.

Usage:
    python benchmark_rnahybrid.py \
        --input inference_dataset.tsv \
        --output rnahybrid_benchmark_results.tsv \
        --bin /path/to/RNAhybrid \
        --num_runs 18 \
        --warmup_runs 2

RNAhybrid CLI:
    RNAhybrid [options] <target_seq> <query_seq>
    - target = mRNA (long), query = miRNA (short)
    - -c  compact (colon-separated) output
    - -b  number of hits per target (default 1)
    - -m  max target length (MUST be >= longest mRNA)
    - -n  max query length
    - -s  3utr_human|3utr_worm|3utr_fly for p-value EVD params
"""

import argparse
import subprocess
import time
import sys
import statistics
import csv
from collections import defaultdict

RNAHYBRID_BIN = "path/to/rnahybrid/bin/RNAhybrid"


def run_rnahybrid(mirna_seq: str, mrna_seq: str, rnahybrid_bin: str,
                  max_target_len: int, timeout: int = 3600) -> dict | None:
    """
    Run RNAhybrid on a miRNA (query) & mRNA (target) pair.
    Measures wall-clock time and peak RSS via GNU /usr/bin/time -v.

    Returns dict with time_s and mem_MB, or None on failure.
    """
    # RNAhybrid <target> <query>  — target=mRNA(long), query=miRNA(short)
    rnahybrid_cmd = [
        rnahybrid_bin,
        "-c",                           # compact output
        "-b", "1",                       # 1 hit per target
        "-m", str(max_target_len),       # max target length
        mrna_seq,                        # target (long mRNA)
        mirna_seq,                       # query  (short miRNA)
    ]

    # Wrap with GNU /usr/bin/time -v for memory measurement
    time_cmd = ["/usr/bin/time", "-v"] + rnahybrid_cmd

    start = time.perf_counter()
    try:
        result = subprocess.run(
            time_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.perf_counter() - start
    except FileNotFoundError:
        # /usr/bin/time not available — fallback without memory tracking
        start = time.perf_counter()
        try:
            result = subprocess.run(
                rnahybrid_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            elapsed = time.perf_counter() - start
        except subprocess.TimeoutExpired:
            print(f"  [WARN] RNAhybrid timed out after {timeout}s", file=sys.stderr)
            return None
        return {"time_s": elapsed, "mem_MB": None}
    except subprocess.TimeoutExpired:
        print(f"  [WARN] RNAhybrid timed out after {timeout}s", file=sys.stderr)
        return None

    if result.returncode != 0:
        # RNAhybrid writes its own output to stdout; errors/time go to stderr
        print(f"  [WARN] RNAhybrid returned exit code {result.returncode}", file=sys.stderr)
        stderr_preview = result.stderr[:500] if result.stderr else "(empty)"
        print(f"         stderr: {stderr_preview}", file=sys.stderr)
        return None

    # Parse peak RSS from GNU time stderr output
    mem_mb = None
    for line in result.stderr.splitlines():
        if "Maximum resident set size" in line:
            try:
                mem_kb = int(line.strip().split(":")[-1].strip())
                mem_mb = mem_kb / 1024.0
            except ValueError:
                pass
            break

    return {"time_s": elapsed, "mem_MB": mem_mb}


def load_dataset(input_path: str) -> list[dict]:
    """Load the TSV dataset and return list of records sorted by mrna_length."""
    records = []
    with open(input_path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            records.append({
                "id": row["id"],
                "mirna_seq": row["mirna_seq"],
                "mrna_seq": row["mrna_seq"],
                "mrna_length": int(row["mrna_length"]),
            })
    records.sort(key=lambda r: r["mrna_length"])
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark RNAhybrid inference time vs. RNA sequence length."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to input TSV with columns: id, mirna_seq, mrna_seq, mrna_length",
    )
    parser.add_argument(
        "--output", "-o", default="rnahybrid_benchmark_results.tsv",
        help="Path to output TSV results (default: rnahybrid_benchmark_results.tsv)",
    )
    parser.add_argument(
        "--num_runs", "-n", type=int, default=18,
        help="Number of timed runs per sequence length (default: 18)",
    )
    parser.add_argument(
        "--warmup_runs", "-w", type=int, default=2,
        help="Number of warmup runs before timed runs (default: 2)",
    )
    parser.add_argument(
        "--timeout", "-t", type=int, default=3600,
        help="Timeout per RNAhybrid call in seconds (default: 3600)",
    )
    parser.add_argument(
        "--label", "-l", default="RNAhybrid",
        help="Label for the attention_mode column (default: RNAhybrid)",
    )
    args = parser.parse_args()

    # --- Check RNAhybrid is available ---
    try:
        ver = subprocess.run(
            [RNAHYBRID_BIN, "-h"],
            capture_output=True, text=True, timeout=10,
        )
        # RNAhybrid prints help/version to stdout or stderr depending on version
        output = ver.stdout.strip() or ver.stderr.strip()
        first_line = output.splitlines()[0] if output else "unknown"
        print(f"Found RNAhybrid: {first_line}")
        print(f"Binary: {RNAHYBRID_BIN}")
    except FileNotFoundError:
        print(f"ERROR: RNAhybrid not found at '{RNAHYBRID_BIN}'.", file=sys.stderr)
        print("  Install: conda install -c bioconda rnahybrid", file=sys.stderr)
        print("  Or specify path: --bin /path/to/RNAhybrid", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"  [WARNING] RNAhybrid check failed: {e}", file=sys.stderr)
        raise e

    # --- Load dataset ---
    records = load_dataset(args.input)
    # Group by mrna_length
    by_length = defaultdict(list)
    for rec in records:
        by_length[rec["mrna_length"]].append(rec)

    lengths = sorted(by_length.keys())
    max_target_len = max(lengths)  # for -m flag

    print(f"Loaded {len(records)} sequence pairs across {len(lengths)} lengths: {lengths}")
    print(f"Max target length for -m flag: {max_target_len}")
    print(f"Will run {args.warmup_runs} warmup + {args.num_runs} timed runs per pair\n")

    # --- Benchmark ---
    results = []

    for length in lengths:
        pairs = by_length[length]
        pair = pairs[0]  # use first pair for this length
        mirna = pair["mirna_seq"]
        mrna = pair["mrna_seq"]

        print(f"--- mRNA length {length} nt (id: {pair['id']}) ---")

        # Warmup runs
        for w in range(args.warmup_runs):
            print(f"  Warmup {w + 1}/{args.warmup_runs}...", end="", flush=True)
            res = run_rnahybrid(mirna, mrna, rnahybrid_bin=RNAHYBRID_BIN,
                                max_target_len=max_target_len, timeout=args.timeout)
            if res is None:
                print(" FAILED")
            else:
                print(f" {res['time_s']:.4f}s")

        # Timed runs
        times = []
        mems = []
        for r in range(args.num_runs):
            print(f"  Run {r + 1}/{args.num_runs}...", end="", flush=True)
            res = run_rnahybrid(mirna, mrna, rnahybrid_bin=RNAHYBRID_BIN,
                                max_target_len=max_target_len, timeout=args.timeout)
            if res is None:
                print(" FAILED")
                continue
            times.append(res["time_s"])
            if res["mem_MB"] is not None:
                mems.append(res["mem_MB"])
            print(f" {res['time_s']:.4f}s, {res.get('mem_MB', 'N/A')} MB")

        if not times:
            print(f"  [SKIP] No successful runs for length {length}\n")
            continue

        median_time = round(statistics.median(times), 4)
        std_time = round(statistics.stdev(times), 4) if len(times) > 1 else 0.0
        median_mem = round(statistics.median(mems), 1) if mems else 0.0
        std_mem = round(statistics.stdev(mems), 1) if len(mems) > 1 else 0.0

        results.append({
            "attention_mode": args.label,
            "mrna_length": length,
            "median_time_s": median_time,
            "std_time_s": std_time,
            "median_mem_MB": median_mem,
            "std_mem_MB": std_mem,
            "num_runs": len(times),
        })

        print(f"  => median={median_time}s, std={std_time}s, "
              f"mem={median_mem}MB ± {std_mem}MB ({len(times)} runs)\n")

    # --- Write output ---
    fieldnames = [
        "attention_mode", "mrna_length", "median_time_s", "std_time_s",
        "median_mem_MB", "std_mem_MB", "num_runs",
    ]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(results)

    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
