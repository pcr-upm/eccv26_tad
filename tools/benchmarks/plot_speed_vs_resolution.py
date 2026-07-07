#!/usr/bin/env python
"""
Benchmark sparse vs dense convolution speed as a function of input resolution.
Tests both temporal (batch size = B*T) and spatial (H x W) resolution scaling at a fixed keep rate.

In the ViT sparse adapter:
- Each frame is processed independently (shape: [B*T, N_kept, C])
- Temporal scaling = more frames = larger effective batch (B*T)
- Spatial scaling = higher resolution per frame = larger H, W
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Add workspace root to path
sys.path.append(os.getcwd())

try:
    from opentad.models.bricks.sparse_conv_layer import SparseConv2d
except ImportError:
    print(
        "Error: Could not import SparseConv2d. Make sure you are running this script from the workspace root."
    )
    sys.exit(1)


def prepare_data_vit(
    num_frames,
    C_in,
    C_out,
    H,
    W,
    keep_rate,
    dtype,
    device,
    batch_size=4,
    frames_per_chunk=16,
    tubelet_size=2,
):
    """
    Prepare data mimicking ViT sparse adapter batching (VideoMAE-style).

    VideoMAE processing:
    - Input: num_frames total (e.g., 768)
    - Split into chunks of frames_per_chunk (e.g., 16)
    - chunk_num = num_frames // frames_per_chunk = 768 // 16 = 48
    - Each chunk is processed independently
    - Tubelet compression: T = frames_per_chunk // tubelet_size = 16 // 2 = 8
    - For 224x224 with patch=16: H=W=14, spatial tokens = 196
    - Tokens per chunk = T * H * W = 8 * 196 = 1568

    With batch_size=4:
    - Effective B = batch_size * chunk_num = 4 * 48 = 192
    - Shape: [192, 1568, C]

    After keep_rate pruning (across T*H*W, not per-frame):
    - N_kept = int(T * H * W * keep_rate)
    - Shape: [B, N_kept, C]

    Args:
        num_frames: Total number of frames (e.g., 768)
        H, W: Spatial patch grid (e.g., 14x14 for 224x224 with patch 16)
        keep_rate: Fraction of tokens to keep (applied across T*H*W)
        batch_size: Number of videos in batch (default: 4)
        frames_per_chunk: Frames per ViT input (default: 16 for VideoMAE)
        tubelet_size: Temporal compression (default: 2)

    Returns:
        x_batched: [B, N_kept, C] - kept tokens
        indices_local: [B, N_kept, 9] - neighbor indices for sparse conv
        kept_indices: [B, N_kept] - original indices in T*H*W space
        N_full: T*H*W (total tokens per chunk)
        T: temporal patches per chunk
    """
    T = frames_per_chunk // tubelet_size  # Temporal patches per chunk (8 for VideoMAE)
    N_full = T * H * W  # Total tokens per chunk (e.g., 8*196=1568)
    N_kept = int(N_full * keep_rate)  # Kept tokens (pruning across T*H*W)
    if N_kept == 0:
        N_kept = 1

    # Calculate effective batch size
    chunk_num = num_frames // frames_per_chunk
    B = batch_size * chunk_num  # e.g., 4 * 48 = 192

    # Generate per-sample masks/indices (Clustered/Block-wise for realism)
    # Prune across T*H*W space, not per-frame
    kept_indices_list = []
    for _ in range(B):
        # Create spatially coherent blobs per frame
        full_mask_list = []
        for t in range(T):
            num_blobs = 3
            centers_y = torch.randint(0, H, (num_blobs,), device=device)
            centers_x = torch.randint(0, W, (num_blobs,), device=device)

            y_grid, x_grid = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing="ij",
            )

            dist_mask = torch.zeros(H, W, device=device)
            for i in range(num_blobs):
                dist = (y_grid - centers_y[i]) ** 2 + (x_grid - centers_x[i]) ** 2
                dist_mask = torch.maximum(dist_mask, (dist < 5).float())
            full_mask_list.append(dist_mask)

        # Flatten across T*H*W and select
        flat_mask = torch.stack(full_mask_list).flatten()
        indices = torch.nonzero(flat_mask).squeeze(-1)

        if indices.numel() < N_kept:
            remaining = N_kept - indices.numel()
            extra = torch.randperm(N_full, device=device)[:remaining]
            indices = torch.cat([indices, extra])
        elif indices.numel() > N_kept:
            indices = indices[:N_kept]

        kept_indices_list.append(indices.sort().values)

    kept_indices = torch.stack(kept_indices_list)  # [B, N_kept]

    # Create lookup table: [B, N_full] -> Compressed Index (or -1)
    lookup = torch.full((B, N_full), -1, dtype=torch.long, device=device)
    lookup.scatter_(
        1,
        kept_indices,
        torch.arange(N_kept, device=device).unsqueeze(0).expand(B, N_kept),
    )

    # Grid coordinates (T*H*W space)
    t_grid, y_grid, x_grid = torch.meshgrid(
        torch.arange(T, device=device),
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )
    coords = torch.stack(
        (t_grid.flatten(), y_grid.flatten(), x_grid.flatten()), dim=1
    )  # [N_full, 3]

    # Neighbor offsets (spatial only, same temporal frame)
    offsets_2d = torch.tensor(
        [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]],
        device=device,
    )
    offsets = torch.zeros((9, 3), device=device, dtype=torch.long)
    offsets[:, 1:] = offsets_2d

    # Get neighbors for kept tokens
    kept_coords = coords[kept_indices.view(-1)].view(B, N_kept, 3)  # [B, N_kept, 3]
    neighbor_coords = kept_coords.unsqueeze(2) + offsets.unsqueeze(0).unsqueeze(0)

    nt = neighbor_coords[..., 0]
    ny = neighbor_coords[..., 1]
    nx = neighbor_coords[..., 2]
    valid = (nt >= 0) & (nt < T) & (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)

    # Convert neighbor coords to full indices
    neighbor_full_indices = nt * (H * W) + ny * W + nx
    neighbor_full_indices[~valid] = 0  # Dummy safe index

    # Map Full Indices -> Compressed Indices using lookup
    neighbor_compressed = torch.gather(lookup, 1, neighbor_full_indices.view(B, -1))
    neighbor_compressed = neighbor_compressed.view(B, N_kept, 9)
    neighbor_compressed[~valid] = -1  # Mask out invalid spatial neighbors

    indices_local = neighbor_compressed.int()

    # Input Data: [B, N_kept, C]
    x_batched = torch.randn(B, N_kept, C_in, device=device, dtype=dtype)

    return x_batched, indices_local, kept_indices, N_full, T


def measure(func, *args, warmup=10, iters=100):
    # Warmup (with no_grad to avoid memory buildup)
    with torch.no_grad():
        for _ in range(warmup):
            func(*args)
    torch.cuda.synchronize()

    # Clear cache and reset memory tracking
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Measure baseline memory (input tensors already allocated)
    baseline_memory = torch.cuda.memory_allocated()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        start.record()
        for _ in range(iters):
            out = func(*args)
            del out
        end.record()
    torch.cuda.synchronize()

    avg_time = start.elapsed_time(end) / iters
    peak_memory_mb = (torch.cuda.max_memory_allocated() - baseline_memory) / (
        1024 * 1024
    )
    return avg_time, peak_memory_mb


def run_temporal_experiment(
    dtype_str="bfloat16", chunk_size=0, keep_rate=0.5, batch_size=4
):
    """
    Benchmark speed vs temporal resolution (number of frames).
    Uses VideoMAE-style chunking: num_frames -> chunk_num chunks of 16 frames each.
    Effective batch = batch_size * chunk_num.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        return

    device = torch.device("cuda:0")
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(dtype_str, torch.bfloat16)

    # Fixed spatial resolution (224x224 input with patch 16 = 14x14)
    H = 14
    W = 14
    C_in = 256
    C_out = 256
    frames_per_chunk = 16  # VideoMAE input
    tubelet_size = 2  # VideoMAE tubelet compression
    T = frames_per_chunk // tubelet_size  # 8 temporal patches per chunk

    # Vary total number of frames (multiple videos or longer videos)
    frame_counts = [768, 768 * 2, 768 * 4, 768 * 8]

    dense_fwd_times = []
    sparse_fwd_times = []
    dense_memory = []
    sparse_memory = []
    token_counts = []
    segment_counts = []

    chunk_str = f", chunk_size={chunk_size}" if chunk_size > 0 else ""
    print(f"\n=== Temporal Resolution Benchmark (VideoMAE chunking) ===")
    print(f"Running on {device} with dtype {dtype}{chunk_str}")
    print(
        f"Config: H={H}, W={W}, frames_per_chunk={frames_per_chunk}, "
        f"batch_size={batch_size}, keep_rate={keep_rate}"
    )
    header = "Frames   | EffBatch   | Tokens       | Dense (ms)   | Dense Mem    | Sparse (ms)  | Sparse Mem"
    print(header)
    print("-" * len(header))

    for num_frames in frame_counts:
        try:
            x_batched, indices_local, kept_indices, N_full, T = prepare_data_vit(
                num_frames,
                C_in,
                C_out,
                H,
                W,
                keep_rate,
                dtype,
                device,
                batch_size=batch_size,
                frames_per_chunk=frames_per_chunk,
                tubelet_size=tubelet_size,
            )
            B = x_batched.shape[0]
            N_kept = kept_indices.shape[1]  # [B, N_kept]
            total_tokens = B * N_kept
            token_counts.append(total_tokens)
            segment_counts.append(B)

            print(
                f"  Shape: x={list(x_batched.shape)}, idx={list(indices_local.shape)}"
            )

            # --- Dense Path ---
            conv2d_bench = (
                nn.Conv2d(C_in, C_out, kernel_size=3, padding=1, bias=True)
                .to(device)
                .to(dtype)
            )

            # kept_indices: [B, N_kept] - indices into T*H*W space
            scatter_indices = kept_indices.unsqueeze(2).expand(B, N_kept, C_in)

            def run_d_fwd():
                # 1. Create Dense Tensor [B, N_full, C] where N_full = T*H*W
                x_dense_flat = torch.zeros(B, N_full, C_in, device=device, dtype=dtype)
                # 2. Scatter kept tokens
                x_dense_flat.scatter_(1, scatter_indices, x_batched)
                # 3. Reshape to images: [B, T*H*W, C] -> [B*T, H, W, C] -> [B*T, C, H, W]
                x_img = (
                    x_dense_flat.view(B * T, H, W, C_in)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                # 4. Apply Conv2d
                out_img = conv2d_bench(x_img)
                # 5. Reshape back: [B*T, C, H, W] -> [B, T*H*W, C]
                out_dense_flat = out_img.permute(0, 2, 3, 1).reshape(B, N_full, C_out)
                # 6. Gather kept tokens
                gather_indices = kept_indices.unsqueeze(2).expand(B, N_kept, C_out)
                out = torch.gather(out_dense_flat, 1, gather_indices)
                return out

            df_t, df_mem = measure(run_d_fwd)
            dense_fwd_times.append(df_t)
            dense_memory.append(df_mem)

            # --- Sparse Path ---
            layer = (
                SparseConv2d(C_in, C_out, chunk_size=chunk_size).to(device).to(dtype)
            )

            def run_s_fwd():
                return layer(x_batched, indices_local)

            sf_t, sf_mem = measure(run_s_fwd)
            sparse_fwd_times.append(sf_t)
            sparse_memory.append(sf_mem)

            print(
                f"{num_frames:<8} | {B:<10} | {total_tokens:<12} | "
                f"{df_t:<12.3f} | {df_mem:<12.1f} | {sf_t:<12.3f} | {sf_mem:<12.1f}"
            )

            del x_batched, indices_local, kept_indices, conv2d_bench, layer
            torch.cuda.empty_cache()

        except RuntimeError as e:
            print(f"{num_frames:<8} | OOM or error: {e}")
            break

    return (
        frame_counts[: len(dense_fwd_times)],
        dense_fwd_times,
        sparse_fwd_times,
        dense_memory,
        sparse_memory,
        token_counts,
    )


def run_spatial_experiment(
    dtype_str="bfloat16", chunk_size=0, keep_rate=0.5, batch_size=4
):
    """
    Benchmark speed vs spatial resolution.
    Uses VideoMAE-style chunking: num_frames -> chunk_num chunks of 16 frames each.
    Effective batch = batch_size * chunk_num.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        return

    device = torch.device("cuda:0")
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(dtype_str, torch.bfloat16)

    # Fixed temporal
    num_frames = 768  # Fixed number of frames
    frames_per_chunk = 16  # VideoMAE input
    tubelet_size = 2  # VideoMAE tubelet compression
    T = frames_per_chunk // tubelet_size  # 8 temporal patches per chunk
    C_in = 256
    C_out = 256
    patch_size = 16  # VideoMAE patch size

    # Vary spatial resolution (input image sizes, patch_size=16 -> grid = input/16)
    # 112x112 -> 7x7, 224x224 -> 14x14, 256x256 -> 16x16, etc.
    input_resolutions = [112, 224, 256, 384, 448, 512, 768, 896]

    dense_fwd_times = []
    sparse_fwd_times = []
    dense_memory = []
    sparse_memory = []
    token_counts = []

    chunk_str = f", chunk_size={chunk_size}" if chunk_size > 0 else ""
    print(f"\n=== Spatial Resolution Benchmark (VideoMAE chunking) ===")
    print(f"Running on {device} with dtype {dtype}{chunk_str}")
    print(
        f"Config: num_frames={num_frames}, frames_per_chunk={frames_per_chunk}, "
        f"patch_size={patch_size}, batch_size={batch_size}, keep_rate={keep_rate}"
    )
    header = "Input Res  | EffBatch   | Tokens       | Dense (ms)   | Dense Mem    | Sparse (ms)  | Sparse Mem"
    print(header)
    print("-" * len(header))

    for input_res in input_resolutions:
        H = W = input_res // patch_size
        try:
            x_batched, indices_local, kept_indices, N_full, T = prepare_data_vit(
                num_frames,
                C_in,
                C_out,
                H,
                W,
                keep_rate,
                dtype,
                device,
                batch_size=batch_size,
                frames_per_chunk=frames_per_chunk,
                tubelet_size=tubelet_size,
            )
            B = x_batched.shape[0]
            N_kept = kept_indices.shape[1]  # [B, N_kept]
            total_tokens = B * N_kept
            token_counts.append(total_tokens)

            print(
                f"  Shape: x={list(x_batched.shape)}, indices={list(indices_local.shape)}"
            )

            # --- Dense Path ---
            conv2d_bench = (
                nn.Conv2d(C_in, C_out, kernel_size=3, padding=1, bias=True)
                .to(device)
                .to(dtype)
            )

            # kept_indices: [B, N_kept] - indices into T*H*W space
            scatter_indices = kept_indices.unsqueeze(2).expand(B, N_kept, C_in)

            def run_d_fwd():
                # 1. Create Dense Tensor [B, N_full, C] where N_full = T*H*W
                x_dense_flat = torch.zeros(B, N_full, C_in, device=device, dtype=dtype)
                # 2. Scatter kept tokens
                x_dense_flat.scatter_(1, scatter_indices, x_batched)
                # 3. Reshape to images: [B, T*H*W, C] -> [B*T, H, W, C] -> [B*T, C, H, W]
                x_img = (
                    x_dense_flat.view(B * T, H, W, C_in)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                # 4. Apply Conv2d
                out_img = conv2d_bench(x_img)
                # 5. Reshape back: [B*T, C, H, W] -> [B, T*H*W, C]
                out_dense_flat = out_img.permute(0, 2, 3, 1).reshape(B, N_full, C_out)
                # 6. Gather kept tokens
                gather_indices = kept_indices.unsqueeze(2).expand(B, N_kept, C_out)
                out = torch.gather(out_dense_flat, 1, gather_indices)
                return out

            df_t, df_mem = measure(run_d_fwd)
            dense_fwd_times.append(df_t)
            dense_memory.append(df_mem)

            # --- Sparse Path ---
            layer = (
                SparseConv2d(C_in, C_out, chunk_size=chunk_size).to(device).to(dtype)
            )

            def run_s_fwd():
                return layer(x_batched, indices_local)

            sf_t, sf_mem = measure(run_s_fwd)
            sparse_fwd_times.append(sf_t)
            sparse_memory.append(sf_mem)

            print(
                f"{input_res}x{input_res:<4} | {B:<10} | {total_tokens:<12} | "
                f"{df_t:<12.3f} | {df_mem:<12.1f} | {sf_t:<12.3f} | {sf_mem:<12.1f}"
            )

            del x_batched, indices_local, kept_indices, conv2d_bench, layer
            torch.cuda.empty_cache()

        except RuntimeError as e:
            print(f"{input_res}x{input_res:<4} | OOM or error: {e}")
            break

    return (
        input_resolutions[: len(dense_fwd_times)],
        dense_fwd_times,
        sparse_fwd_times,
        dense_memory,
        sparse_memory,
        token_counts,
    )


def run_experiment(dtype_str="bfloat16", chunk_size=0, keep_rate=0.5, batch_size=4):
    """Run both temporal and spatial resolution benchmarks using VideoMAE-style chunking."""

    # Run temporal benchmark
    frame_counts, t_dense_times, t_sparse_times, t_dense_mem, t_sparse_mem, t_tokens = (
        run_temporal_experiment(dtype_str, chunk_size, keep_rate, batch_size)
    )

    # Run spatial benchmark
    (
        spatial_sizes,
        s_dense_times,
        s_sparse_times,
        s_dense_mem,
        s_sparse_mem,
        s_tokens,
    ) = run_spatial_experiment(dtype_str, chunk_size, keep_rate, batch_size)

    # Plotting
    chunk_suffix = f"_chunk{chunk_size}" if chunk_size > 0 else ""

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Temporal Resolution Plots ---
    # Speed vs Frames
    ax1 = axes[0, 0]
    ax1.plot(frame_counts, t_dense_times, label="Dense Conv2d", marker="o")
    ax1.plot(frame_counts, t_sparse_times, label="Sparse Conv2d", marker="x")
    ax1.set_title(f"Speed vs Temporal Resolution (keep_rate={keep_rate})")
    ax1.set_xlabel("Number of Frames")
    ax1.set_ylabel("Time (ms)")
    ax1.grid(True)
    ax1.legend()

    # Add secondary x-axis with token counts
    ax1_twin = ax1.twiny()
    ax1_twin.set_xlim(ax1.get_xlim())
    ax1_twin.set_xticks(frame_counts)
    ax1_twin.set_xticklabels(
        [f"{int(t/1000)}k" for t in t_tokens], rotation=45, fontsize=8
    )
    ax1_twin.set_xlabel("Kept Tokens")

    # Memory vs Frames
    ax2 = axes[0, 1]
    ax2.plot(frame_counts, t_dense_mem, label="Dense Conv2d", marker="o")
    ax2.plot(frame_counts, t_sparse_mem, label="Sparse Conv2d", marker="x")
    ax2.set_title(f"Memory vs Temporal Resolution (keep_rate={keep_rate})")
    ax2.set_xlabel("Number of Frames")
    ax2.set_ylabel("Peak Memory (MB)")
    ax2.grid(True)
    ax2.legend()

    # --- Spatial Resolution Plots ---
    spatial_labels = [f"{s}x{s}" for s in spatial_sizes]
    x_pos = np.arange(len(spatial_sizes))

    # Speed vs Spatial
    ax3 = axes[1, 0]
    ax3.plot(x_pos, s_dense_times, label="Dense Conv2d", marker="o")
    ax3.plot(x_pos, s_sparse_times, label="Sparse Conv2d", marker="x")
    ax3.set_title(f"Speed vs Spatial Resolution (keep_rate={keep_rate})")
    ax3.set_xlabel("Input Resolution")
    ax3.set_ylabel("Time (ms)")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(spatial_labels, rotation=45, ha="right")
    ax3.grid(True)
    ax3.legend()

    # Add secondary x-axis with token counts
    ax3_twin = ax3.twiny()
    ax3_twin.set_xlim(ax3.get_xlim())
    ax3_twin.set_xticks(x_pos)
    ax3_twin.set_xticklabels(
        [f"{int(t/1000)}k" for t in s_tokens], rotation=45, fontsize=8
    )
    ax3_twin.set_xlabel("Kept Tokens")

    # Memory vs Spatial
    ax4 = axes[1, 1]
    ax4.plot(x_pos, s_dense_mem, label="Dense Conv2d", marker="o")
    ax4.plot(x_pos, s_sparse_mem, label="Sparse Conv2d", marker="x")
    ax4.set_title(f"Memory vs Spatial Resolution (keep_rate={keep_rate})")
    ax4.set_xlabel("Input Resolution")
    ax4.set_ylabel("Peak Memory (MB)")
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(spatial_labels, rotation=45, ha="right")
    ax4.grid(True)
    ax4.legend()

    fig.suptitle(
        f"Sparse vs Dense Conv2d: Resolution Scaling (dtype={dtype_str})", fontsize=14
    )
    fig.tight_layout()

    output_file = f"speed_vs_resolution_{dtype_str}{chunk_suffix}.png"
    fig.savefig(output_file, dpi=150)
    print(f"\nCombined plot saved to {output_file}")

    # Also save separate plots
    # Temporal speed
    fig_t_speed, ax = plt.subplots(figsize=(10, 6))
    ax.plot(frame_counts, t_dense_times, label="Dense Conv2d", marker="o")
    ax.plot(frame_counts, t_sparse_times, label="Sparse Conv2d", marker="x")
    ax.set_title(
        f"Speed vs Temporal Resolution (keep_rate={keep_rate}, dtype={dtype_str})"
    )
    ax.set_xlabel("Number of Frames")
    ax.set_ylabel("Time (ms)")
    ax.grid(True)
    ax.legend()
    ax_twin = ax.twiny()
    ax_twin.set_xlim(ax.get_xlim())
    ax_twin.set_xticks(frame_counts)
    ax_twin.set_xticklabels(
        [f"{int(t/1000)}k" for t in t_tokens], rotation=45, fontsize=8
    )
    ax_twin.set_xlabel("Kept Tokens")
    fig_t_speed.tight_layout()
    fig_t_speed.savefig(f"speed_vs_temporal_{dtype_str}{chunk_suffix}.png")
    print(
        f"Temporal speed plot saved to speed_vs_temporal_{dtype_str}{chunk_suffix}.png"
    )

    # Spatial speed
    fig_s_speed, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_pos, s_dense_times, label="Dense Conv2d", marker="o")
    ax.plot(x_pos, s_sparse_times, label="Sparse Conv2d", marker="x")
    ax.set_title(
        f"Speed vs Spatial Resolution (keep_rate={keep_rate}, dtype={dtype_str})"
    )
    ax.set_xlabel("Input Resolution")
    ax.set_ylabel("Time (ms)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(spatial_labels, rotation=45, ha="right")
    ax.grid(True)
    ax.legend()
    ax_twin = ax.twiny()
    ax_twin.set_xlim(ax.get_xlim())
    ax_twin.set_xticks(x_pos)
    ax_twin.set_xticklabels(
        [f"{int(t/1000)}k" for t in s_tokens], rotation=45, fontsize=8
    )
    ax_twin.set_xlabel("Kept Tokens")
    fig_s_speed.tight_layout()
    fig_s_speed.savefig(f"speed_vs_spatial_{dtype_str}{chunk_suffix}.png")
    print(f"Spatial speed plot saved to speed_vs_spatial_{dtype_str}{chunk_suffix}.png")

    # Temporal memory
    fig_t_mem, ax = plt.subplots(figsize=(10, 6))
    ax.plot(frame_counts, t_dense_mem, label="Dense Conv2d", marker="o")
    ax.plot(frame_counts, t_sparse_mem, label="Sparse Conv2d", marker="x")
    ax.set_title(
        f"Memory vs Temporal Resolution (keep_rate={keep_rate}, dtype={dtype_str})"
    )
    ax.set_xlabel("Number of Frames")
    ax.set_ylabel("Peak Memory (MB)")
    ax.grid(True)
    ax.legend()
    ax_twin = ax.twiny()
    ax_twin.set_xlim(ax.get_xlim())
    ax_twin.set_xticks(frame_counts)
    ax_twin.set_xticklabels(
        [f"{int(t/1000)}k" for t in t_tokens], rotation=45, fontsize=8
    )
    ax_twin.set_xlabel("Kept Tokens")
    fig_t_mem.tight_layout()
    fig_t_mem.savefig(f"memory_vs_temporal_{dtype_str}{chunk_suffix}.png")
    print(
        f"Temporal memory plot saved to memory_vs_temporal_{dtype_str}{chunk_suffix}.png"
    )

    # Spatial memory
    fig_s_mem, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_pos, s_dense_mem, label="Dense Conv2d", marker="o")
    ax.plot(x_pos, s_sparse_mem, label="Sparse Conv2d", marker="x")
    ax.set_title(
        f"Memory vs Spatial Resolution (keep_rate={keep_rate}, dtype={dtype_str})"
    )
    ax.set_xlabel("Input Resolution")
    ax.set_ylabel("Peak Memory (MB)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(spatial_labels, rotation=45, ha="right")
    ax.grid(True)
    ax.legend()
    ax_twin = ax.twiny()
    ax_twin.set_xlim(ax.get_xlim())
    ax_twin.set_xticks(x_pos)
    ax_twin.set_xticklabels(
        [f"{int(t/1000)}k" for t in s_tokens], rotation=45, fontsize=8
    )
    ax_twin.set_xlabel("Kept Tokens")
    fig_s_mem.tight_layout()
    fig_s_mem.savefig(f"memory_vs_spatial_{dtype_str}{chunk_suffix}.png")
    print(
        f"Spatial memory plot saved to memory_vs_spatial_{dtype_str}{chunk_suffix}.png"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark sparse vs dense convolution across resolutions"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type to use for benchmark (default: bfloat16)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=32768,
        help="Chunk size for memory-efficient sparse conv (0 = no chunking, default: 0)",
    )
    parser.add_argument(
        "--keep-rate",
        type=float,
        default=0.3,
        help="Keep rate for sparse convolution (default: 0.3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (number of videos) (default: 1)",
    )
    args = parser.parse_args()
    run_experiment(
        dtype_str=args.dtype,
        chunk_size=args.chunk_size,
        keep_rate=args.keep_rate,
        batch_size=args.batch_size,
    )
