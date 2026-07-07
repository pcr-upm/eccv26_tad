import torch
import torch.nn as nn
import argparse
import sys
import os

sys.path.append(os.getcwd())

from opentad.models.bricks.sparse_conv_layer import SparseConv2d


# ncu --set full -k sparse_fwd_tiled_v2  -o v3_sparse_2d_blob $(which python) tools/benchmarks/benchmark_ncu.py
def benchmark_ncu():
    parser = argparse.ArgumentParser(
        description="Benchmark SparseConv2d for NCU Profiling"
    )
    parser.add_argument(
        "--measure-dense", action="store_true", help="Also run the dense Conv2d path"
    )
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # ViT-Sparse Configuration
    # Input Video: (1, 3, 16, 224, 224)
    # PatchEmbed: Tubelet=2, Patch=16
    # T_tokens = 16 / 2 = 8
    # H_tokens = 224 / 16 = 14
    # W_tokens = 224 / 16 = 14

    B = 48
    T = 8
    H, W = 14, 14
    N_full = T * H * W
    keep_rate = 0.5
    N_kept = int(N_full * keep_rate)
    C = 256
    K = 9

    effective_B = B

    print(f"Preparing NCU Benchmark for SparseConv2d (Native Batching)")
    print(
        f"Config: B={B}, T={T} (Effective Batch={effective_B}), N_full={N_full}, N_kept={N_kept}, C={C}"
    )
    print(f"Precision: {dtype}")

    # Create Layer
    layer = SparseConv2d(C, C).to(device).to(dtype)

    # --- Data Preparation ---
    # Generate per-sample masks/indices (Clustered/Block-wise for realism)
    kept_indices_list = []
    for _ in range(effective_B):
        # Create a random mask that is spatially coherent (blobs) per frame
        full_mask_list = []

        for t in range(T):
            # Generate random centers for blobs
            num_blobs = 3
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

    if False:  # Visualization debug
        print("Visualizing mask for verification...")
        try:
            import matplotlib.pyplot as plt

            vis_mask = torch.zeros(N_full, device=device)
            vis_mask[kept_indices[0]] = 1
            vis_mask = vis_mask.view(T, H, W).cpu().float()

            plt.figure(figsize=(6, 6))
            plt.imshow(vis_mask[0], cmap="gray")
            plt.title("Generated Blob Mask (Sample 0, Frame 0)")
            plt.savefig("blob_mask_debug.png")
            print("Saved blob_mask_debug.png")
        except ImportError:
            print("matplotlib not installed, skipping visualization.")

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

    if False:  # Visualization debug for neighbors
        print("Visualizing neighbors for verification...")
        try:
            import matplotlib.pyplot as plt

            # Pick a sample and a token
            s_idx = 0
            # Find a token that has valid neighbors (not all -1)
            valid_mask = (indices_local[s_idx] != -1).sum(dim=1)

            # Pick one with 4-8 neighbors (edge/corner case or sparse region)
            candidates = torch.nonzero((valid_mask >= 4) & (valid_mask <= 8)).squeeze()

            if candidates.numel() > 0:
                if candidates.ndim == 0:
                    t_idx = candidates.item()
                else:
                    t_idx = candidates[0].item()
            else:
                # Fallback to max if no candidate found
                t_idx = torch.argmax(valid_mask).item()

            center_full_idx = kept_indices[s_idx, t_idx].item()
            neighbor_local_idxs = indices_local[s_idx, t_idx].long()

            # Filter -1
            valid_neighbors = neighbor_local_idxs[neighbor_local_idxs != -1]
            neighbor_full_idxs = kept_indices[s_idx, valid_neighbors]

            vis_grid = torch.zeros(N_full, device=device)
            vis_grid[center_full_idx] = 2  # Center
            vis_grid[neighbor_full_idxs] = 1  # Neighbors

            vis_grid = vis_grid.view(T, H, W).cpu()

            # Find which frame the center is in
            center_t = center_full_idx // (H * W)

            plt.figure(figsize=(6, 6))
            plt.imshow(vis_grid[center_t], cmap="viridis")
            plt.title(f"Neighbors for Token {t_idx} (Frame {center_t})")
            plt.colorbar()
            plt.savefig("neighbor_debug.png")
            print(
                f"Saved neighbor_debug.png (Center: {center_full_idx}, Neighbors: {neighbor_full_idxs.tolist()})"
            )

        except ImportError:
            print("matplotlib not installed, skipping neighbor visualization.")
        except Exception as e:
            print(f"Error visualizing neighbors: {e}")

    # Input Data
    x_batched = torch.randn(effective_B, N_kept, C, device=device, dtype=dtype)

    # Execution for NCU
    print("Running SparseConv2d for NCU capture...")
    torch.cuda.synchronize()

    # Run once
    torch.cuda.nvtx.range_push("SparseConv2d")
    layer(x_batched, indices_local)
    torch.cuda.nvtx.range_pop()

    torch.cuda.synchronize()

    if args.measure_dense:
        print("Running Dense Conv2d path for NCU capture...")
        conv2d_bench = (
            nn.Conv2d(C, C, kernel_size=3, padding=1, bias=True).to(device).to(dtype)
        )
        scatter_indices = kept_indices.unsqueeze(2).expand(effective_B, N_kept, C)

        torch.cuda.synchronize()

        torch.cuda.nvtx.range_push("DenseConv2d")
        # 1. Create Dense Tensor (effective_B, N_full, C)
        x_dense_flat = torch.zeros(effective_B, N_full, C, device=device, dtype=dtype)
        # 2. Scatter
        x_dense_flat.scatter_(1, scatter_indices, x_batched)
        # 3. View as Image
        x_img = (
            x_dense_flat.view(effective_B * T, H, W, C).permute(0, 3, 1, 2).contiguous()
        )
        # 4. Conv2d
        out_img = conv2d_bench(x_img)
        # 5. View as Flat
        out_dense_flat = out_img.permute(0, 2, 3, 1).reshape(effective_B, N_full, C)
        # 6. Gather
        out = torch.gather(out_dense_flat, 1, scatter_indices)
        torch.cuda.nvtx.range_pop()

        torch.cuda.synchronize()

    print("Done.")


if __name__ == "__main__":
    benchmark_ncu()
# ncu --set full     --nvtx     --nvtx-include "regex:(SparseConv2d|DenseConv2d)/"     -o v3sparse_vs_conv_batch     -f     $(which python) tools/benchmarks/benchmark_ncu.py --measure-dense
