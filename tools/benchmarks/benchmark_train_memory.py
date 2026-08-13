"""
Benchmark peak GPU memory of one full training step (forward + backward +
optimizer.step), e.g. to compare ViTSparse vs AdaTAD adapters on THUMOS.

Single config:
    python tools/benchmarks/benchmark_train_memory.py <config.py>

Side-by-side comparison (the main use case):
    python tools/benchmarks/benchmark_train_memory.py \\
        configs/vitsparse/thumos/e2e_thumos_videomae_b_768x1_160_sparse_adapter.py \\
        configs/adatad/thumos/e2e_thumos_videomae_b_768x1_160_adapter.py

Each config is benchmarked with its own batch_size, AMP setting etc. taken
from the config file (matching how it's actually trained), unless overridden
with --batch-size / --amp. Results append to --output-csv.
"""

import gc
import os
import sys
import csv
import argparse
import logging
from datetime import datetime

sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "..", "..")
if path not in sys.path:
    sys.path.insert(0, path)

import torch
from mmengine.config import Config, DictAction

from benchmark_inference import extract_model_config  # noqa: E402  (sibling script)

from opentad.models import build_detector
from opentad.datasets import build_dataset
from opentad.datasets.builder import collate
from opentad.cores import build_optimizer
from opentad.utils import set_seed, ModelEma


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark peak GPU memory of a training step"
    )
    parser.add_argument(
        "configs", metavar="FILE", type=str, nargs="+", help="one or more config files"
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--warmup-iters", type=int, default=3, help="train steps to discard (cudnn autotune, alloc warmup)"
    )
    parser.add_argument(
        "--benchmark-iters", type=int, default=5, help="train steps over which peak memory is measured"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="override solver.train.batch_size"
    )
    parser.add_argument(
        "--amp", choices=["auto", "on", "off"], default="auto",
        help="auto = follow the config's solver.amp (default)",
    )
    parser.add_argument(
        "--amp-dtype", choices=["bfloat16", "float16"], default="bfloat16",
        help="matches the dtype used by train_one_epoch's autocast",
    )
    parser.add_argument(
        "--ema", action="store_true",
        help="also allocate an EMA shadow model, as real training does when solver.ema=True",
    )
    parser.add_argument(
        "--load-pretrained", action="store_true",
        help="load the backbone's pretrained checkpoint (custom.pretrain). Off by default since "
        "weight values don't affect peak memory, and not every config's checkpoint is downloaded locally",
    )
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    parser.add_argument("--output-csv", type=str, default="train_memory_results.csv")
    return parser.parse_args()


def get_logger():
    logger = logging.getLogger("benchmark_train_memory")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    return logger


def build_train_batch(dataset, batch_size, device):
    n = len(dataset)
    samples = [dataset[i % n] for i in range(batch_size)]
    batch = collate(samples)
    if isinstance(batch["inputs"], torch.Tensor):
        batch["inputs"] = batch["inputs"].to(device)
    if isinstance(batch["masks"], torch.Tensor):
        batch["masks"] = batch["masks"].to(device)
    for key in ("gt_segments", "gt_labels"):
        batch[key] = [v.to(device) if isinstance(v, torch.Tensor) else v for v in batch[key]]
    return batch


def benchmark_config(config_path, args, logger):
    cfg = Config.fromfile(config_path)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    if args.batch_size is not None:
        cfg.solver.train["batch_size"] = args.batch_size
    batch_size = cfg.solver.train["batch_size"]

    if not args.load_pretrained:
        cfg.model.backbone.custom.pretrain = None

    use_amp = cfg.solver.get("amp", False) if args.amp == "auto" else (args.amp == "on")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    clip_grad_norm = cfg.solver.get("clip_grad_norm", -1)
    use_ema = args.ema and cfg.solver.get("ema", False)

    device = args.device
    set_seed(args.seed)

    print(f"\n{'=' * 70}")
    print(f"Config: {config_path}")
    print(f"batch_size={batch_size}  use_amp={use_amp} ({args.amp_dtype})  "
          f"clip_grad_norm={clip_grad_norm}  ema={use_ema}")
    print(f"{'=' * 70}")

    train_dataset = build_dataset(cfg.dataset.train, default_args=dict(logger=logger))
    batch = build_train_batch(train_dataset, batch_size, device)
    print(f"Input shape: {list(batch['inputs'].shape)}")

    model = build_detector(cfg.model).to(device)
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}   Trainable: {trainable_params:,}")

    optimizer = build_optimizer(cfg.optimizer.copy(), model, logger)
    model_ema = ModelEma(model) if use_ema else None

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            losses = model(**batch, return_loss=True)
        losses["cost"].backward()
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        if model_ema is not None:
            model_ema.update(model)
        return losses["cost"].item()

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    oom = False
    try:
        print(f"Warming up ({args.warmup_iters} iters)...")
        for _ in range(args.warmup_iters):
            loss = train_step()
        torch.cuda.synchronize(device)
        print(f"  last warmup loss: {loss:.4f}")

        # discard warmup allocations (cudnn autotune, lazy buffer growth) from the peak reading
        torch.cuda.reset_peak_memory_stats(device)
        print(f"Benchmarking ({args.benchmark_iters} iters)...")
        for _ in range(args.benchmark_iters):
            loss = train_step()
        torch.cuda.synchronize(device)
        print(f"  last benchmark loss: {loss:.4f}")
    except torch.cuda.OutOfMemoryError as e:
        oom = True
        print(f"\nOUT OF MEMORY at batch_size={batch_size}: {e}")

    peak_alloc_gb = torch.cuda.max_memory_allocated(device) / 1e9
    peak_reserved_gb = torch.cuda.max_memory_reserved(device) / 1e9
    stats = torch.cuda.memory_stats(device)
    alloc_retries = stats.get("num_alloc_retries", 0)

    if not oom:
        print(f"\nPeak allocated:  {peak_alloc_gb:.3f} GB")
        print(f"Peak reserved:   {peak_reserved_gb:.3f} GB")
        if alloc_retries > 0:
            print(f"  Alloc retries: {alloc_retries} (high = fragmentation)")

    config_info = extract_model_config(cfg)
    result = dict(
        config_file=os.path.basename(config_path),
        model_size=config_info["model_size"],
        keep_rate=config_info["keep_rate"],
        adapter_conv_type=config_info["adapter_conv_type"],
        batch_size=batch_size,
        use_amp=use_amp,
        ema=use_ema,
        total_params=total_params,
        trainable_params=trainable_params,
        oom=oom,
        peak_alloc_gb=peak_alloc_gb,
        peak_reserved_gb=peak_reserved_gb,
        alloc_retries=alloc_retries,
    )

    # Free everything before the next config is benchmarked in this same process.
    del model, optimizer, model_ema, batch, train_dataset
    gc.collect()
    torch.cuda.empty_cache()

    return result


def save_to_csv(csv_path, row):
    row = dict(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **row)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def print_comparison(results):
    print(f"\n{'=' * 90}")
    print("TRAINING PEAK GPU MEMORY COMPARISON")
    print(f"{'=' * 90}")
    print(f"{'config':<55} {'bs':>4} {'peak_alloc':>12} {'peak_reserved':>14}")
    print("-" * 90)
    for r in results:
        alloc = "OOM" if r["oom"] else f"{r['peak_alloc_gb']:.3f}GB"
        reserved = "OOM" if r["oom"] else f"{r['peak_reserved_gb']:.3f}GB"
        print(f"{r['config_file']:<55} {r['batch_size']:>4} {alloc:>12} {reserved:>14}")
    print("=" * 90)


def main():
    args = parse_args()
    logger = get_logger()

    results = [benchmark_config(config_path, args, logger) for config_path in args.configs]
    for result in results:
        save_to_csv(args.output_csv, result)

    if len(results) > 1:
        print_comparison(results)

    print(f"\nResults appended to: {args.output_csv}")


if __name__ == "__main__":
    main()
