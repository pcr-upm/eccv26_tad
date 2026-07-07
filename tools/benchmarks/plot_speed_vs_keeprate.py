import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 18})

# Add workspace root to path
sys.path.append(os.getcwd())
# %%
import math

embed_dims = 1280  # Example for ViT-Huge, 1For vit-l use 1024
mlp_ratio = 0.25
hidden_dims = int(embed_dims * mlp_ratio)
# Round to nearest power of 2
hidden_dims = 2 ** round(math.log2(hidden_dims))
hidden_dims
# %%
try:
    from opentad.models.bricks.sparse_conv_layer import SparseConv2d
except ImportError:
    print(
        "Error: Could not import SparseConv2d. Make sure you are running this script from the workspace root."
    )
    sys.exit(1)


def prepare_data_vit(B, T, C_in, C_out, H, W, keep_rate, dtype, device, num_blobs=4):
    N_full = T * H * W
    N_kept = int(N_full * keep_rate)
    if N_kept == 0:
        N_kept = 1

    effective_B = B

    # Generate per-sample masks/indices (Clustered/Block-wise for realism)
    kept_indices_list = []
    for _ in range(effective_B):
        if num_blobs == 0:
            # Pure random sampling
            indices = torch.randperm(N_full, device=device)[:N_kept]
        else:
            # Create a random mask that is spatially coherent (blobs) per frame
            full_mask_list = []

            for t in range(T):
                # Generate random centers for blobs
                centers_y = torch.randint(0, H, (num_blobs,), device=device)
                centers_x = torch.randint(0, W, (num_blobs,), device=device)

                # Create Gaussian-like blobs or simple distance threshold
                y_grid, x_grid = torch.meshgrid(
                    torch.arange(H, device=device),
                    torch.arange(W, device=device),
                    indexing="ij",
                )

                dist_mask = torch.zeros(H, W, device=device)
                for i in range(num_blobs):
                    dist = (y_grid - centers_y[i]) ** 2 + (x_grid - centers_x[i]) ** 2
                    dist_mask = torch.maximum(
                        dist_mask, (dist < 5).float()
                    )  # Radius squared ~ 5 (R~2.2)
                full_mask_list.append(dist_mask)

            # Flatten and select
            flat_mask = torch.stack(full_mask_list).flatten()
            # Ensure we have exactly N_kept (fill random if needed or cut)
            indices = torch.nonzero(flat_mask).squeeze()

            if indices.numel() < N_kept:
                # Add random to fill
                remaining = N_kept - indices.numel()
                extra = torch.randperm(N_full, device=device)[:remaining]
                indices = torch.cat([indices, extra])
            elif indices.numel() > N_kept:
                indices = indices[:N_kept]

        kept_indices_list.append(
            indices.sort().values
        )  # Sort for better memory locality simulation

    kept_indices = torch.stack(kept_indices_list)  # (effective_B, N_kept)

    # Create lookup table: (effective_B, N_full) -> Compressed Index (or -1)
    lookup = torch.full((effective_B, N_full), -1, dtype=torch.long, device=device)
    lookup.scatter_(
        1,
        kept_indices,
        torch.arange(N_kept, device=device).unsqueeze(0).expand(effective_B, N_kept),
    )

    # Grid coordinates
    t_grid, y_grid, x_grid = torch.meshgrid(
        torch.arange(T, device=device),
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )
    coords = torch.stack(
        (t_grid.flatten(), y_grid.flatten(), x_grid.flatten()), dim=1
    )  # (N_full, 3)

    # Neighbor offsets (spatial only, same frame)
    offsets_2d = torch.tensor(
        [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]],
        device=device,
    )
    offsets = torch.zeros((9, 3), device=device, dtype=torch.long)
    offsets[:, 1:] = offsets_2d

    # Get neighbors for kept tokens
    kept_coords = coords[kept_indices.view(-1)].view(
        effective_B, N_kept, 3
    )  # (effective_B, N_kept, 3)
    neighbor_coords = kept_coords.unsqueeze(2) + offsets.unsqueeze(0).unsqueeze(
        0
    )  # (effective_B, N_kept, 9, 3)

    nt = neighbor_coords[..., 0]
    ny = neighbor_coords[..., 1]
    nx = neighbor_coords[..., 2]
    valid = (nt >= 0) & (nt < T) & (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)

    # Convert neighbor coords to full indices
    neighbor_full_indices = nt * (H * W) + ny * W + nx
    neighbor_full_indices[~valid] = 0  # Dummy safe index

    # Map Full Indices -> Compressed Indices using lookup
    neighbor_compressed = torch.gather(
        lookup, 1, neighbor_full_indices.view(effective_B, -1)
    )
    neighbor_compressed = neighbor_compressed.view(effective_B, N_kept, 9)
    neighbor_compressed[~valid] = -1  # Mask out invalid spatial neighbors

    indices_local = neighbor_compressed.int()

    # Input Data
    x_batched = torch.randn(effective_B, N_kept, C_in, device=device, dtype=dtype)

    return x_batched, indices_local, kept_indices, N_full


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
            del out  # Ensure output is freed each iteration
        end.record()
    torch.cuda.synchronize()

    avg_time = start.elapsed_time(end) / iters
    # Peak memory minus baseline = memory used by forward pass only
    peak_memory_mb = (torch.cuda.max_memory_allocated() - baseline_memory) / (
        1024 * 1024
    )
    return avg_time, peak_memory_mb


def run_experiment(dtype_str="bfloat16", chunk_size=0):
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

    # Configuration
    B = 192
    T = 8
    C_in = 256
    C_out = 256
    H = 14
    W = 14
    # torch.Size([192, 1569, 384])
    keep_rates = np.linspace(0.05, 0.8, 10)

    dense_fwd_times = []
    sparse_fwd_times = []
    dense_memory = []
    sparse_memory = []
    token_counts = []

    chunk_str = f", chunk_size={chunk_size}" if chunk_size > 0 else ""
    print(f"Running benchmark on {device} with dtype {dtype}{chunk_str}")
    print(f"Config: B={B}, T={T}, C_in={C_in}, C_out={C_out}, H={H}, W={W}")
    print(
        f"{'Keep Rate':<10} | {'Tokens':<10} | {'Dense Fwd (ms)':<15} | {'Dense Mem (MB)':<15} | {'Sparse Fwd (ms)':<15} | {'Sparse Mem (MB)':<15}"
    )
    print("-" * 95)

    for keep in keep_rates:
        x_batched, indices_local, kept_indices, N_full = prepare_data_vit(
            B, T, C_in, C_out, H, W, keep, dtype, device
        )
        print(x_batched.shape, indices_local.shape, kept_indices.shape)
        n_tokens = x_batched.shape[1] * x_batched.shape[0]
        token_counts.append(n_tokens)

        # --- Dense Path ---
        conv2d_bench = (
            nn.Conv2d(C_in, C_out, kernel_size=3, padding=1, bias=True)
            .to(device)
            .to(dtype)
        )
        scatter_indices = kept_indices.unsqueeze(2).expand(B, x_batched.shape[1], C_in)

        def run_d_fwd():
            # 1. Create Dense Tensor (effective_B, N_full, C)
            x_dense_flat = torch.zeros(B, N_full, C_in, device=device, dtype=dtype)
            # 2. Scatter
            x_dense_flat.scatter_(1, scatter_indices, x_batched)
            # 3. View as Image
            x_img = (
                x_dense_flat.view(B * T, H, W, C_in).permute(0, 3, 1, 2).contiguous()
            )
            # 4. Conv2d
            out_img = conv2d_bench(x_img)
            # 5. View as Flat
            out_dense_flat = out_img.permute(0, 2, 3, 1).reshape(B, N_full, C_out)
            # 6. Gather
            out = torch.gather(out_dense_flat, 1, scatter_indices)
            return out

        df_t, df_mem = measure(run_d_fwd)
        dense_fwd_times.append(df_t)
        dense_memory.append(df_mem)

        # --- Sparse Path ---
        layer = SparseConv2d(C_in, C_out, chunk_size=chunk_size).to(device).to(dtype)

        def run_s_fwd():
            return layer(x_batched, indices_local)

        sf_t, sf_mem = measure(run_s_fwd)
        sparse_fwd_times.append(sf_t)
        sparse_memory.append(sf_mem)

        print(
            f"{keep:<10.2f} | {n_tokens:<10} | {df_t:<15.3f} | {df_mem:<15.1f} | {sf_t:<15.3f} | {sf_mem:<15.1f}"
        )

    # Plotting
    chunk_suffix = f"_chunk{chunk_size}" if chunk_size > 0 else ""

    # Time plot
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(keep_rates, dense_fwd_times, label="Dense Conv2d (Baseline)", marker="o")
    ax1.plot(keep_rates, sparse_fwd_times, label="Sparse Conv2d", marker="x")

    # ax1.set_title(
    #     f"Inference Speed vs Keep Rate: Input size T=768, C=3, H=224, W=224, Dtype={dtype_str}",
    #     fontsize=14,
    # )
    ax1.set_xlabel("Keep Rate")
    ax1.set_ylabel("Time (ms)")
    ax1.grid(True)
    ax1.legend()

    # Add secondary x-axis with token counts
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(keep_rates)
    ax2.set_xticklabels(
        [f"{int(t/1000)}k" for t in token_counts], rotation=45, fontsize=8
    )
    ax2.set_xlabel("Number of Kept Tokens")

    fig1.tight_layout()
    speed_file = f"speed_vs_keeprate_{dtype_str}{chunk_suffix}.png"
    fig1.savefig(speed_file)
    print(f"\nSpeed graph saved to {speed_file}")

    # Memory plot
    fig2, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(keep_rates, dense_memory, label="Dense Conv2d (Baseline)", marker="o")
    ax3.plot(keep_rates, sparse_memory, label="Sparse Conv2d", marker="x")
    ax3.set_title(
        f"Memory Usage vs Keep Rate: Input size T=768, C=3, H=224, W=224, Dtype={dtype_str}",
        fontsize=14,
    )
    ax3.set_xlabel("Keep Rate")
    ax3.set_ylabel("Peak Memory (MB)")
    ax3.grid(True)
    ax3.legend()

    # Add secondary x-axis with token counts
    ax4 = ax3.twiny()
    ax4.set_xlim(ax3.get_xlim())
    ax4.set_xticks(keep_rates)
    ax4.set_xticklabels(
        [f"{int(t/1000)}k" for t in token_counts], rotation=45, fontsize=8
    )
    ax4.set_xlabel("Number of Kept Tokens")

    fig2.tight_layout()
    memory_file = f"memory_vs_keeprate_{dtype_str}{chunk_suffix}.png"
    fig2.savefig(memory_file)
    print(f"Memory graph saved to {memory_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark sparse vs dense convolution"
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
        default=0,
        help="Chunk size for memory-efficient sparse conv (0 = no chunking, default: 0)",
    )
    args = parser.parse_args()
    run_experiment(dtype_str=args.dtype, chunk_size=args.chunk_size)
