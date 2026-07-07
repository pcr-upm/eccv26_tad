#!/usr/bin/env python
"""
Run benchmark_inference.py with different combinations of adapter_use_attn and n_landmarks.

Usage:
    python tools/run_benchmark_sweep.py
"""

import subprocess
import itertools
import os

# Configuration
CONFIG_FILE = "configs/vitsparse/thumos/e2e_thumos_videomae_b_768x1_160_sparse_adapter.py"
OUTPUT_CSV = "benchmark_sweep_results.csv"

# Parameter sweep values
ADAPTER_USE_ATTN_VALUES = [3]
N_LANDMARKS_VALUES = [0]
KEEP_RATE_VALUES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def main():
    # Generate all combinations
    combinations = list(
        itertools.product(ADAPTER_USE_ATTN_VALUES, N_LANDMARKS_VALUES, KEEP_RATE_VALUES)
    )
    total = len(combinations)

    print(f"Running {total} benchmark combinations")
    print(f"adapter_use_attn: {ADAPTER_USE_ATTN_VALUES}")
    print(f"n_landmarks: {N_LANDMARKS_VALUES}")
    print(f"keep_rate: {KEEP_RATE_VALUES}")
    print(f"Output CSV: {OUTPUT_CSV}")
    print("=" * 60)

    for i, (adapter_use_attn, n_landmarks, keep_rate) in enumerate(combinations, 1):
        print(
            f"\n[{i}/{total}] Running: adapter_use_attn={adapter_use_attn}, n_landmarks={n_landmarks}, keep_rate={keep_rate}"
        )
        print("-" * 60)

        cmd = [
            "python",
            "tools/benchmarks/benchmark_inference.py",
            CONFIG_FILE,
            "--use-amp",
            # "--measure-adapter",
            "--output-csv",
            OUTPUT_CSV,
            "--cfg-options",
            f"model.backbone.backbone.adapter_use_attn={adapter_use_attn}",
            f"model.backbone.backbone.n_landmarks={n_landmarks}",
            f"model.backbone.backbone.keep_rate={keep_rate}",
        ]

        print(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, check=True)
            print(f"[{i}/{total}] Completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"[{i}/{total}] Failed with return code {e.returncode}")
            continue

    print("\n" + "=" * 60)
    print(f"Sweep complete! Results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
