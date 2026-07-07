# Benchmarking Scripts

All scripts must be run from the **workspace root** (`/path/to/SVTAD`).

---

### `benchmark_inference.py` — End-to-end inference benchmark (primary tool)

Benchmarks a full detector loaded from a config file. Measures latency with warmup, reports mean/std/p95/p99/throughput, optionally measures GFLOPs, per-adapter timing, and per-block timing. Saves results to CSV for tracking across runs.

```bash
python tools/benchmarks/benchmark_inference.py \
    configs/vitsparse/thumos/e2e_thumos_videomae_b_768x1_160_sparse_adapter.py \
    --use-amp \
    --output-csv results.csv

# With adapter/block-level timing breakdown
python tools/benchmarks/benchmark_inference.py <config> \
    --measure-adapter --measure-blocks --use-amp

# Override config options at runtime
python tools/benchmarks/benchmark_inference.py <config> \
    --cfg-options model.backbone.backbone.keep_rate=0.5
```

**Key flags:** `--checkpoint`, `--use-amp`, `--warmup-iters` (default 10), `--benchmark-iters` (default 50), `--measure-adapter`, `--measure-blocks`, `--output-csv`

---

### `benchmark_ncu.py` — NCU kernel profiler target

Minimal script designed to be launched under NVIDIA Nsight Compute (`ncu`) to profile the sparse vs dense conv CUDA kernels with realistic blob-patterned sparse masks. Uses NVTX ranges for kernel-level attribution.

```bash
# Profile sparse kernel only
ncu --set full -k sparse_fwd_tiled_v2 -o profile_sparse \
    $(which python) tools/benchmarks/benchmark_ncu.py

# Profile sparse vs dense side-by-side
ncu --set full --nvtx --nvtx-include "regex:(SparseConv2d|DenseConv2d)/" \
    -o profile_comparison -f \
    $(which python) tools/benchmarks/benchmark_ncu.py --measure-dense
```

---

### `plot_speed_vs_keeprate.py` — Sparse vs dense speed/memory vs keep rate

Sweeps keep rate from 5% to 80% and plots `SparseConv2d` vs dense `Conv2d` latency and peak memory. Saves PNG plots to the working directory.

```bash
python tools/benchmarks/plot_speed_vs_keeprate.py --dtype bfloat16
python tools/benchmarks/plot_speed_vs_keeprate.py --dtype bfloat16 --chunk-size 8192
```

Outputs: `speed_vs_keeprate_<dtype>.png`, `memory_vs_keeprate_<dtype>.png`

---

### `plot_speed_vs_resolution.py` — Sparse vs dense speed/memory vs resolution

Benchmarks sparse vs dense Conv2d across temporal resolution (number of frames) and spatial resolution (input image size). Uses VideoMAE-style chunking (16 frames/chunk, tubelet=2) for realistic effective batch sizes.

```bash
python tools/benchmarks/plot_speed_vs_resolution.py \
    --dtype bfloat16 --keep-rate 0.3 --batch-size 1
```

Outputs: `speed_vs_resolution_<dtype>.png` and individual plots per axis/metric.

---

### `run_benchmark_sweep.py` — Parameter sweep over `benchmark_inference.py`

Runs `benchmark_inference.py` in a subprocess for every combination of `adapter_use_attn`, `n_landmarks`, and `keep_rate` values. Edit the constants at the top of the file to define the sweep. Results accumulate in a single CSV.

```bash
python tools/benchmarks/run_benchmark_sweep.py
```

> Edit `ADAPTER_USE_ATTN_VALUES`, `N_LANDMARKS_VALUES`, `KEEP_RATE_VALUES`, and `CONFIG_FILE` at the top of the file to define the sweep.
