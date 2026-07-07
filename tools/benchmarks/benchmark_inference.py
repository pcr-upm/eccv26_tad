"""
Benchmark script to measure inference time of a model on a single test example.
Usage:
    python tools/benchmark_inference.py configs/path/to/config.py --checkpoint path/to/checkpoint.pth
"""

import gc
import os
import sys
import time
import argparse
import csv
from datetime import datetime
from collections import Counter

sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)

import torch
import numpy as np
from mmengine.config import Config, DictAction
from opentad.models import build_detector
from opentad.datasets import build_dataset
from opentad.utils import set_seed

try:
    from torch.utils.flop_counter import FlopCounterMode

    FLOP_COUNTER_AVAILABLE = True
except ImportError:
    FLOP_COUNTER_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark inference time of a Temporal Action Detector"
    )
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="the checkpoint path"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="device to use (e.g., cuda:0, cpu)"
    )
    parser.add_argument(
        "--warmup-iters", type=int, default=10, help="number of warmup iterations"
    )
    parser.add_argument(
        "--benchmark-iters",
        type=int,
        default=50,
        help="number of benchmark iterations",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="batch size for inference"
    )
    parser.add_argument(
        "--use-amp", action="store_true", help="use automatic mixed precision"
    )
    parser.add_argument(
        "--measure-adapter",
        action="store_true",
        help="measure adapter module timing separately",
    )
    parser.add_argument(
        "--measure-blocks",
        action="store_true",
        help="measure per-block timing separately",
    )
    parser.add_argument(
        "--cfg-options", nargs="+", action=DictAction, help="override settings"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="benchmark_results.csv",
        help="path to output CSV file (default: benchmark_results.csv)",
    )
    args = parser.parse_args()
    return args


def get_majority_conv_type(adapter_conv_types):
    """Get the majority adapter conv type from a list."""
    if not adapter_conv_types or adapter_conv_types == "N/A":
        return "N/A"
    counts = Counter(adapter_conv_types)
    return counts.most_common(1)[0][0]


def get_model_size_name(embed_dims, depth):
    """Determine model size name based on embed_dims and depth."""
    # Common VideoMAE configurations
    if embed_dims == 384 and depth == 12:
        return "small"
    elif embed_dims == 768 and depth == 12:
        return "base"
    elif embed_dims == 1024 and depth == 24:
        return "large"
    elif embed_dims == 1280 and depth == 32:
        return "huge"
    elif embed_dims == 1408 and depth == 40:
        return "giant"
    else:
        return f"{embed_dims}d_{depth}l"


def extract_model_config(cfg):
    """Extract model configuration from config object."""
    config_info = {
        "embed_dims": "N/A",
        "depth": "N/A",
        "num_heads": "N/A",
        "keep_rate": "N/A",
        "adapter_conv_types": "N/A",
        "adapter_conv_type": "N/A",
        "adapter_use_attn": "N/A",
        "n_landmarks": "N/A",
        "model_size": "N/A",
    }

    if hasattr(cfg.model, "backbone") and hasattr(cfg.model.backbone, "backbone"):
        backbone_cfg = cfg.model.backbone.backbone
        config_info["embed_dims"] = backbone_cfg.get("embed_dims", "N/A")
        config_info["depth"] = backbone_cfg.get("depth", "N/A")
        config_info["num_heads"] = backbone_cfg.get("num_heads", "N/A")
        config_info["keep_rate"] = backbone_cfg.get("keep_rate", "N/A")
        config_info["adapter_conv_types"] = backbone_cfg.get(
            "adapter_conv_types", "N/A"
        )
        config_info["adapter_use_attn"] = backbone_cfg.get("adapter_use_attn", "N/A")
        config_info["n_landmarks"] = backbone_cfg.get("n_landmarks", "N/A")

        # Get majority conv type
        config_info["adapter_conv_type"] = get_majority_conv_type(
            config_info["adapter_conv_types"]
        )

        # Get model size name
        if config_info["embed_dims"] != "N/A" and config_info["depth"] != "N/A":
            config_info["model_size"] = get_model_size_name(
                config_info["embed_dims"], config_info["depth"]
            )

    return config_info


def calculate_gflops(model, batch, device, use_amp=False):
    """Calculate GFLOPs for the model using torch.utils.flop_counter.

    FlopCounterMode only intercepts standard aten ops. SparseConv2dFunction is a
    custom CUDA kernel and is invisible to it (counted as 0). We patch that by
    attaching forward hooks to all SparseConv2d modules that accumulate the
    theoretical FLOPs (N_kept * 9 * C_in * C_out * 2) and add them to the total.
    """
    if not FLOP_COUNTER_AVAILABLE:
        print(
            "Warning: torch.utils.flop_counter not available (requires PyTorch >= 2.1)"
        )
        return None

    model.eval()

    inputs = batch.get("inputs", None)
    if inputs is None:
        print("Warning: Could not find 'inputs' in batch for FLOP calculation")
        return None

    # Accumulate sparse conv FLOPs via forward hooks
    sparse_flops = [0]
    hooks = []
    try:
        from opentad.models.bricks.sparse_conv_layer import SparseConv2d as _SparseConv2d

        def _sparse_hook(module, inp, _out):
            x = inp[0]  # (N_kept, C_in) or (B, N, C_in)
            n_tokens = x.numel() // module.in_channels
            sparse_flops[0] += n_tokens * 9 * module.in_channels * module.out_channels * 2

        for m in model.modules():
            if isinstance(m, _SparseConv2d):
                hooks.append(m.register_forward_hook(_sparse_hook))
    except ImportError:
        pass

    try:
        flop_counter = FlopCounterMode(model, depth=3)
        with flop_counter:
            with torch.no_grad():
                with torch.amp.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=use_amp
                ):
                    _ = model.backbone(inputs)
        flops = flop_counter.get_total_flops() + sparse_flops[0]
        if sparse_flops[0] > 0:
            print(f"  Sparse conv FLOPs added: {sparse_flops[0] / 1e9:.2f} GFLOPs")
        return flops / 1e9
    except Exception as e:
        print(f"Warning: Could not calculate FLOPs: {e}")
        return None
    finally:
        for h in hooks:
            h.remove()


def save_to_csv(
    csv_path,
    config_info,
    times,
    batch_size,
    adapter_time_ms,
    benchmark_iters,
    peak_memory_gb,
    config_file,
    use_amp,
    gflops=None,
):
    """Save benchmark results to CSV file (create or append)."""
    times_ms = np.array(times) * 1000  # Convert to milliseconds

    # Calculate statistics
    mean_time = np.mean(times_ms)
    std_time = np.std(times_ms)
    min_time = np.min(times_ms)
    max_time = np.max(times_ms)
    median_time = np.median(times_ms)
    p95_time = np.percentile(times_ms, 95)
    p99_time = np.percentile(times_ms, 99)
    throughput = 1000 / mean_time * batch_size

    # Calculate adapter statistics
    adapter_time_per_iter = (
        adapter_time_ms / benchmark_iters if adapter_time_ms else None
    )
    adapter_percentage = (
        (adapter_time_per_iter / mean_time) * 100 if adapter_time_ms else None
    )
    non_adapter_time = mean_time - adapter_time_per_iter if adapter_time_ms else None

    # Prepare row data
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config_file": os.path.basename(config_file),
        "model_size": config_info["model_size"],
        "embed_dims": config_info["embed_dims"],
        "depth": config_info["depth"],
        "num_heads": config_info["num_heads"],
        "keep_rate": config_info["keep_rate"],
        "adapter_conv_type": config_info["adapter_conv_type"],
        "adapter_use_attn": config_info["adapter_use_attn"],
        "n_landmarks": config_info["n_landmarks"],
        "use_amp": use_amp,
        "batch_size": batch_size,
        "benchmark_iters": benchmark_iters,
        "mean_ms": f"{mean_time:.2f}",
        "std_ms": f"{std_time:.2f}",
        "min_ms": f"{min_time:.2f}",
        "max_ms": f"{max_time:.2f}",
        "median_ms": f"{median_time:.2f}",
        "p95_ms": f"{p95_time:.2f}",
        "p99_ms": f"{p99_time:.2f}",
        "throughput_samples_per_s": f"{throughput:.2f}",
        "adapter_time_ms": (
            f"{adapter_time_per_iter:.2f}" if adapter_time_per_iter else "N/A"
        ),
        "adapter_percentage": (
            f"{adapter_percentage:.2f}" if adapter_percentage else "N/A"
        ),
        "non_adapter_time_ms": f"{non_adapter_time:.2f}" if non_adapter_time else "N/A",
        "peak_memory_gb": f"{peak_memory_gb:.2f}" if peak_memory_gb else "N/A",
        "gflops": f"{gflops:.2f}" if gflops else "N/A",
    }

    # Check if file exists to determine if we need to write header
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"\nResults saved to: {csv_path}")


def get_single_sample(dataset, device):
    """Get a single sample from the dataset and prepare it for inference."""
    sample = dataset[0]

    # Move tensors to device and add batch dimension
    batch = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.unsqueeze(0).to(device)
        elif isinstance(value, list):
            batch[key] = value
        elif isinstance(value, dict):
            # metas should be a list of dicts (one per batch item)
            batch[key] = [value]
        else:
            batch[key] = value

    return batch


def measure_inference_time(
    model,
    batch,
    cfg,
    warmup_iters,
    benchmark_iters,
    use_amp,
    device,
    ext_cls,
    measure_adapter=False,
    measure_blocks=False,
):
    """Measure inference time with warmup and multiple iterations."""

    cfg.post_processing.sliding_window = False  # Single sample, not sliding window

    model.eval()

    # Get backbone for adapter/block timing
    backbone = None
    if measure_adapter or measure_blocks:
        try:
            backbone = model.backbone.model.backbone
            if measure_adapter:
                if hasattr(backbone, "set_measure_adapter_time"):
                    backbone.set_measure_adapter_time(True)
                    backbone.reset_adapter_time()
                    print("Adapter timing enabled")
                else:
                    print("Warning: Backbone does not support adapter timing")
            if measure_blocks:
                if hasattr(backbone, "set_measure_block_time"):
                    backbone.set_measure_block_time(True)
                    backbone.reset_block_time()
                    print("Per-block timing enabled")
                else:
                    print("Warning: Backbone does not support per-block timing")
                    if not measure_adapter:
                        backbone = None
        except AttributeError:
            print("Warning: Could not access backbone for timing")
            backbone = None

    # Warmup
    print(f"Running {warmup_iters} warmup iterations...")
    with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
        with torch.no_grad():
            for _ in range(warmup_iters):
                _ = model(
                    **batch,
                    return_loss=False,
                    infer_cfg=cfg.inference,
                    post_cfg=cfg.post_processing,
                    ext_cls=ext_cls,
                )

    # Reset adapter/block timing after warmup
    if backbone is not None:
        if measure_adapter and hasattr(backbone, "reset_adapter_time"):
            backbone.reset_adapter_time()
        if measure_blocks and hasattr(backbone, "reset_block_time"):
            backbone.reset_block_time()

    # Synchronize before timing
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    # Benchmark
    print(f"Running {benchmark_iters} benchmark iterations...")
    times = []

    with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
        with torch.no_grad():
            for i in range(benchmark_iters):
                if device.startswith("cuda"):
                    torch.cuda.synchronize()

                start_time = time.perf_counter()

                _ = model(
                    **batch,
                    return_loss=False,
                    infer_cfg=cfg.inference,
                    post_cfg=cfg.post_processing,
                    ext_cls=ext_cls,
                )

                if device.startswith("cuda"):
                    torch.cuda.synchronize()

                end_time = time.perf_counter()
                times.append(end_time - start_time)

    # Get adapter time if measured
    adapter_time_ms = None
    if backbone is not None and measure_adapter:
        adapter_time_ms = backbone.get_adapter_time_ms()
        backbone.set_measure_adapter_time(False)

    # Get per-block times if measured
    block_times_ms = None
    if backbone is not None and measure_blocks:
        block_times_ms = backbone.get_block_times_ms()
        backbone.set_measure_block_time(False)

    return times, adapter_time_ms, block_times_ms


def print_statistics(
    times,
    batch_size,
    adapter_time_ms=None,
    benchmark_iters=100,
    gflops=None,
    block_times_ms=None,
):
    """Print timing statistics."""
    times = np.array(times) * 1000  # Convert to milliseconds

    print("\n" + "=" * 60)
    print("INFERENCE TIME STATISTICS")
    print("=" * 60)
    print(f"Batch size: {batch_size}")
    print(f"Number of iterations: {len(times)}")
    print("-" * 60)
    print(f"Mean:       {np.mean(times):>10.2f} ms")
    print(f"Std:        {np.std(times):>10.2f} ms")
    print(f"Min:        {np.min(times):>10.2f} ms")
    print(f"Max:        {np.max(times):>10.2f} ms")
    print(f"Median:     {np.median(times):>10.2f} ms")
    print(f"P95:        {np.percentile(times, 95):>10.2f} ms")
    print(f"P99:        {np.percentile(times, 99):>10.2f} ms")
    print("-" * 60)
    print(f"Throughput: {1000 / np.mean(times) * batch_size:>10.2f} samples/s")
    if gflops is not None:
        print(f"GFLOPs:     {gflops:>10.2f}")
    print("=" * 60)

    # Print adapter timing statistics if available
    if adapter_time_ms is not None:
        adapter_per_iter = adapter_time_ms / benchmark_iters
        adapter_percentage = (adapter_per_iter / np.mean(times)) * 100
        print("\n" + "=" * 60)
        print("ADAPTER MODULE TIMING")
        print("=" * 60)
        print(f"Total adapter time:     {adapter_time_ms:>10.2f} ms")
        print(f"Adapter time per iter:  {adapter_per_iter:>10.2f} ms")
        print(f"Adapter % of total:     {adapter_percentage:>10.2f} %")
        print(f"Non-adapter time:       {np.mean(times) - adapter_per_iter:>10.2f} ms")
        print("=" * 60)

    # Print per-block timing statistics if available
    if block_times_ms is not None:
        print("\n" + "=" * 60)
        print("PER-BLOCK TIMING")
        print("=" * 60)
        total_block_time = sum(block_times_ms)
        total_per_iter = total_block_time / benchmark_iters
        print(
            f"{'Block':<10} {'Total (ms)':<12} {'Per Iter (ms)':<14} {'% of Total':<12}"
        )
        print("-" * 60)
        for i, block_time in enumerate(block_times_ms):
            per_iter = block_time / benchmark_iters
            percentage = (per_iter / np.mean(times)) * 100
            print(f"{i:<10} {block_time:<12.2f} {per_iter:<14.2f} {percentage:<12.2f}")
        print("-" * 60)
        print(
            f"{'Sum':<10} {total_block_time:<12.2f} {total_per_iter:<14.2f} {(total_per_iter / np.mean(times)) * 100:<12.2f}"
        )
        print("=" * 60)


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Set device
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU")
        device = "cpu"

    print(f"Using device: {device}")
    print(f"Config: {args.config}")

    # Build dataset
    print("Building test dataset...")
    test_dataset = build_dataset(cfg.dataset.test, default_args=dict(logger=None))
    print(f"Test dataset size: {len(test_dataset)}")

    # Build model
    print("Building model...")
    model = build_detector(cfg.model)
    model = model.to(device)

    # Load checkpoint if provided
    if args.checkpoint is not None:
        print(f"Loading checkpoint from: {args.checkpoint}")
        # Load to CPU first to avoid doubling GPU memory usage
        checkpoint = torch.load(args.checkpoint, map_location="cpu")

        # Handle Model EMA
        use_ema = getattr(cfg.solver, "ema", False)
        if use_ema and "state_dict_ema" in checkpoint:
            state_dict = checkpoint["state_dict_ema"]
            print("Loaded EMA weights")
        else:
            state_dict = checkpoint["state_dict"]

        # Strip "module." prefix if checkpoint was saved with DDP
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v  # Remove "module." prefix
            else:
                new_state_dict[k] = v

        # Free original checkpoint before loading to model
        epoch = checkpoint.get("epoch", "unknown")
        del checkpoint, state_dict

        model.load_state_dict(new_state_dict)
        print(f"Checkpoint epoch: {epoch}")

        # Free the state dict
        del new_state_dict
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    else:
        print("No checkpoint provided, using random weights")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Print model configuration from config file
    print("\nModel configuration (from config):")
    if hasattr(cfg.model, "backbone") and hasattr(cfg.model.backbone, "backbone"):
        backbone_cfg = cfg.model.backbone.backbone
        print(f"  embed_dims: {backbone_cfg.get('embed_dims', 'N/A')}")
        print(f"  depth: {backbone_cfg.get('depth', 'N/A')}")
        print(f"  num_heads: {backbone_cfg.get('num_heads', 'N/A')}")
        print(f"  keep_rate: {backbone_cfg.get('keep_rate', 'N/A')}")
        print(f"  adapter_index: {backbone_cfg.get('adapter_index', 'N/A')}")
        print(f"  adapter_conv_types: {backbone_cfg.get('adapter_conv_types', 'N/A')}")
        print(f"  adapter_use_attn: {backbone_cfg.get('adapter_use_attn', 'N/A')}")
        print(f"  n_landmarks: {backbone_cfg.get('n_landmarks', 'N/A')}")
        print(f"  Keep_rate: {backbone_cfg.get('keep_rate', 'N/A')}")

    # Print loaded backbone attributes (from actual model)
    print("\nBackbone attributes (from loaded model):")
    try:
        backbone = model.backbone.model.backbone
        if hasattr(backbone, "keep_rate"):
            print(f"  keep_rate: {backbone.keep_rate}")
        if hasattr(backbone, "adapter_conv_types"):
            print(f"  adapter_conv_types: {backbone.adapter_conv_types}")
        if hasattr(backbone, "embed_dims"):
            print(f"  embed_dims: {backbone.embed_dims}")
        if hasattr(backbone, "num_layers"):
            print(f"  num_layers: {backbone.num_layers}")
        print(backbone.keep_rate)
    except AttributeError as e:
        print(f"  Could not access backbone attributes: {e}")

    # Get single sample
    print("Preparing single test sample...")
    batch = get_single_sample(test_dataset, device)

    # Print input shapes
    print("\nInput shapes:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")

    # Build external classifier (use dataset class_map as fallback)
    from opentad.models.utils.post_processing import build_classifier

    if (
        "external_cls" in cfg.post_processing
        and cfg.post_processing.external_cls is not None
    ):
        ext_cls = build_classifier(cfg.post_processing.external_cls)
    else:
        ext_cls = test_dataset.class_map

    # Clean up before benchmarking
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # Calculate GFLOPs
    print("\nCalculating GFLOPs...")
    gflops = calculate_gflops(model, batch, device, use_amp=args.use_amp)
    if gflops is not None:
        print(f"Model GFLOPs: {gflops:.2f}")
    else:
        print("GFLOPs calculation skipped or failed")

    # Reset peak memory stats after GFLOPs calculation to get accurate benchmark memory
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    # Run benchmark
    print(f"\nAMP enabled: {args.use_amp}")
    print(f"Measure adapter: {args.measure_adapter}")
    print(f"Measure blocks: {args.measure_blocks}")
    times, adapter_time_ms, block_times_ms = measure_inference_time(
        model,
        batch,
        cfg,
        warmup_iters=args.warmup_iters,
        benchmark_iters=args.benchmark_iters,
        use_amp=args.use_amp,
        device=device,
        ext_cls=ext_cls,
        measure_adapter=args.measure_adapter,
        measure_blocks=args.measure_blocks,
    )

    # Print statistics
    print_statistics(
        times,
        args.batch_size,
        adapter_time_ms,
        args.benchmark_iters,
        gflops,
        block_times_ms,
    )

    # Print memory usage
    peak_memory_gb = None
    if device.startswith("cuda"):
        peak_memory_gb = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"\nPeak GPU memory: {peak_memory_gb:.2f} GB")

        # Memory stats
        stats = torch.cuda.memory_stats(device)
        alloc_retries = stats.get("num_alloc_retries", 0)
        if alloc_retries > 0:
            print(f"  Alloc retries: {alloc_retries} (high = fragmentation)")

    # Extract model config and save to CSV
    config_info = extract_model_config(cfg)
    save_to_csv(
        csv_path=args.output_csv,
        config_info=config_info,
        times=times,
        batch_size=args.batch_size,
        adapter_time_ms=adapter_time_ms,
        benchmark_iters=args.benchmark_iters,
        peak_memory_gb=peak_memory_gb,
        config_file=args.config,
        use_amp=args.use_amp,
        gflops=gflops,
    )


if __name__ == "__main__":
    main()


"""
python tools/benchmark_inference.py configs/vitsparse/thumos/e2e_thumos_videomae_b_768x1_160_sparse_adapter.py --checkpoint /media/ricardo/data/datasets/best_thumos_b.pth --use-amp
python tools/benchmark_inference.py configs/vitsparse/thumos/e2e_thumos_videomae_l_768x1_160_sparse_adapter.py --checkpoint /media/ricardo/data/datasets/best_thumos_l.pth --amp
"""
