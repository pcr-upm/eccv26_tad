"""
Memory analysis: why VitSparse has higher peak memory than AdaTAD at lower GFLOPs.

Theory: implicit_gemm_fwd_v2 allocates TWO large intermediate tensors per call:
  - x_gathered:  [9, N, Cin]  in fp16  (gather step)
  - out_features: [9, N, Cout] in fp32  (GEMM accumulation)

With chunk_size=32768, Cin=256 (after power-of-2 rounding of 768*0.25=192):
  x_gathered  = 9 * 32768 * 256 * 2B = 150 MB
  out_features = 9 * 32768 * 256 * 4B = 301 MB
  Total: ~451 MB per chunk — vs AdaTAD dwconv's ~58 MB

Run:
    python tools/benchmarks/test_memory_theory.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gc
import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def reset(device="cuda:0"):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def peak_mb(device="cuda:0"):
    return torch.cuda.max_memory_allocated(device) / 1e6


def current_mb(device="cuda:0"):
    return torch.cuda.memory_allocated(device) / 1e6


def header(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def bench_sparse_conv(N_flat, C_in, chunk_size, device, label):
    """Measure peak memory overhead of SparseConv2d on N_flat tokens."""
    from opentad.models.bricks.sparse_conv_layer import SparseConv2d

    conv = SparseConv2d(C_in, C_in, chunk_size=chunk_size).to(device).half()
    x = torch.randn(N_flat, C_in, device=device, dtype=torch.float16)
    # Random valid neighbour indices (no -1 padding for simplicity)
    neighbor_indices = torch.randint(0, N_flat, (N_flat, 9), device=device, dtype=torch.int32)

    baseline = current_mb(device)
    reset(device)
    baseline = current_mb(device)

    with torch.no_grad():
        out = conv(x, neighbor_indices)

    pk = peak_mb(device)
    overhead = pk - baseline

    # Theoretical: x_gathered + out_features per chunk
    actual_chunk = min(chunk_size, N_flat) if chunk_size > 0 else N_flat
    gather_mb  = 9 * actual_chunk * C_in  * 2 / 1e6   # fp16
    outfeat_mb = 9 * actual_chunk * C_in  * 4 / 1e6   # fp32
    theory_mb  = gather_mb + outfeat_mb

    print(
        f"  {label:<50}"
        f"  overhead={overhead:>7.1f} MB"
        f"  | theory={theory_mb:>7.1f} MB"
        f"  (gather={gather_mb:.0f}+outfp32={outfeat_mb:.0f})"
    )

    del x, neighbor_indices, conv, out
    gc.collect()
    torch.cuda.empty_cache()

    return overhead, theory_mb


def bench_dense_dwconv(B, T, H, W, C_in, device, label):
    """Measure peak memory for temporal_dwconv (AdaTAD adapter, no pruning)."""
    N_total = T * H * W
    x_proj = torch.randn(B, N_total, C_in, device=device, dtype=torch.float16)

    dwconv = nn.Conv1d(C_in, C_in, kernel_size=3, padding=1, groups=C_in).to(device).half()
    pwconv = nn.Conv1d(C_in, C_in, 1).to(device).half()

    baseline = current_mb(device)
    reset(device)
    baseline = current_mb(device)

    with torch.no_grad():
        # Reshape [B, T, H, W, C] → [B*H*W, C, T] for Conv1d
        conv_input = x_proj.reshape(B, T, H, W, C_in).permute(0, 2, 3, 4, 1).flatten(0, 2)
        out = pwconv(dwconv(conv_input))

    pk = peak_mb(device)
    overhead = pk - baseline

    # Theoretical: [B*H*W, C, T] in fp16 × 2 (input + output)
    bhw_ct_mb = B * H * W * C_in * T * 2 / 1e6
    theory_mb = bhw_ct_mb * 2  # input tensor + output tensor

    print(
        f"  {label:<50}"
        f"  overhead={overhead:>7.1f} MB"
        f"  | theory={theory_mb:>7.1f} MB"
        f"  ([{B*H*W},{C_in},{T}] fp16 × 2)"
    )

    del x_proj, dwconv, pwconv, out, conv_input
    gc.collect()
    torch.cuda.empty_cache()

    return overhead, theory_mb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda:0"
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    prop = torch.cuda.get_device_properties(device)
    print(f"GPU : {prop.name}")
    print(f"VRAM: {prop.total_memory / 1e9:.1f} GB")

    # ------------------------------------------------------------------
    # Actual config parameters
    # B=48 chunks (768 frames / 16 per clip = 48),  each clip: T=8 temporal
    # steps (16 frames / tubelet_size=2), H=14, W=14 (224 / patch_size=16)
    # N_per_chunk = T*H*W = 8*14*14 = 1568
    # keep_rate=0.6 → N_kept ≈ ceil(0.6 * 1568) = 941
    # Flat layout seen by SparseConv2d:
    #   before pruning : B * 1568 = 48 * 1568 = 75 264
    #   after  pruning : B * 941  = 48 *  941 = 45 168
    # hidden_dims: 768 * 0.25 = 192 → 2**round(log2(192))=2**8 = 256 (vitsparse)
    # ------------------------------------------------------------------
    B, T, H, W = 48, 8, 14, 14
    N_per_chunk = T * H * W          # 1568
    N_per_chunk_kept = math.ceil(0.6 * N_per_chunk)  # 941

    N_full   = B * N_per_chunk        # 75 264  (before pruning)
    N_pruned = B * N_per_chunk_kept   # 45 168  (after pruning)

    C_vitsparse = 256   # 768*0.25=192 → rounded to 2^8=256
    C_raw       = 192   # without rounding (also AdaTAD's value)

    print(f"\nKey parameters:")
    print(f"  B={B} chunks, T={T}, H={H}, W={W}")
    print(f"  N_full={N_full} (before pruning), N_pruned={N_pruned} (after keep_rate=0.6)")
    print(f"  VitSparse hidden_dims={C_vitsparse} (192 rounded to 256)")
    print(f"  AdaTAD   hidden_dims={C_raw}")

    # ------------------------------------------------------------------
    header("EXP 1 — chunk_size effect on peak memory overhead")
    print(f"  Config: N_flat={N_pruned} (after pruning), C_in={C_vitsparse}")
    print(f"  {'Label':<50}  {'overhead':>12}  {'theory':>12}")
    for cs in [0, 4096, 8192, 16384, 32768, 65536]:
        lbl = f"chunk_size={cs if cs > 0 else 'none (full gather)'}"
        bench_sparse_conv(N_pruned, C_vitsparse, chunk_size=cs, device=device, label=lbl)

    # ------------------------------------------------------------------
    header("EXP 2 — power-of-2 rounding: C_in=256 vs C_in=192")
    print(f"  Config: N_flat={N_pruned}, chunk_size=32768")
    bench_sparse_conv(N_pruned, C_vitsparse, chunk_size=32768, device=device,
                      label="C_in=256 (VitSparse, rounded)")
    bench_sparse_conv(N_pruned, C_raw,       chunk_size=32768, device=device,
                      label="C_in=192 (no rounding)")

    # ------------------------------------------------------------------
    header("EXP 3 — full N (before pruning) vs pruned N")
    print(f"  Config: C_in={C_vitsparse}, chunk_size=32768")
    bench_sparse_conv(N_full,   C_vitsparse, chunk_size=32768, device=device,
                      label=f"N_flat={N_full}  (before pruning, blocks 0-2)")
    bench_sparse_conv(N_pruned, C_vitsparse, chunk_size=32768, device=device,
                      label=f"N_flat={N_pruned} (after  pruning, blocks 3-11)")

    # ------------------------------------------------------------------
    header("EXP 4 — sparse_conv vs temporal_dwconv (AdaTAD)")
    print(f"  Full token count, no pruning (worst case for both)")
    bench_sparse_conv(N_full, C_vitsparse, chunk_size=32768, device=device,
                      label=f"sparse_conv  C_in=256 chunk=32768 (VitSparse)")
    bench_sparse_conv(N_full, C_raw,       chunk_size=32768, device=device,
                      label=f"sparse_conv  C_in=192 chunk=32768 (no rounding)")
    bench_dense_dwconv(B, T, H, W, C_raw, device=device,
                       label=f"temporal_dwconv C_in=192 (AdaTAD)")

    # ------------------------------------------------------------------
    header("EXP 5 — cumulative overhead across all adapter layers")
    # VitSparse: 4 × 2d_conv (full N before pruning or pruned) + 8 × sparse_conv (pruned)
    # AdaTAD:    12 × temporal_dwconv (full N, no pruning)
    print("  Summing overhead per adapter layer across the full 12-layer backbone")
    print()

    # VitSparse: blocks 0-2 (before first pruning): N_full tokens, 2d_conv adapter
    # block 3 prunes → blocks 3-5: N_pruned, adapter[3]=2d_conv (layer index 3)
    # Simplification: adapter_conv_types = ["2d_conv"]*4 + ["sparse_conv"]*8
    # Pruning at blocks 3, 6, 9. After block 3: N→0.6N, after block 6: N→0.6²N, ...
    keep = 0.6
    N_stages = [
        N_full,                              # blocks 0-2  (2d_conv)
        B * math.ceil(keep   * N_per_chunk), # blocks 3-5  (2d_conv at 3, sparse at 4,5)
        B * math.ceil(keep**2 * N_per_chunk),# blocks 6-8  (sparse)
        B * math.ceil(keep**3 * N_per_chunk),# blocks 9-11 (sparse)
    ]
    print(f"  Token counts per stage: {N_stages}")

    # 2d_conv overhead: scatter [B,N_total,C] → same as dense but without full-N scatter
    # For 2d_conv in VitSparse (no pruning case or pruned): scatter to full_N first
    # Actually the 2d_conv case in VitSparse's EfficientAdapter also uses full_N scatter
    # full_grid = [B_chunk, N_total_per_chunk, C] (before flattening with batch)
    # For B=48 chunks and N_per_chunk total:
    scatter_2d = B * N_per_chunk * C_vitsparse * 2 / 1e6  # [B, N_total_per_chunk, C] fp16

    total_sparse_overhead = 0.0
    total_adatad_overhead = 0.0

    conv_types = ["2d"] * 4 + ["sparse"] * 8
    # Pruning happens at blocks 3, 6, 9 (keep_rate=[1,1,1,0.6,1,1,0.6,1,1,0.6,1,1])
    stage_map = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3]

    print(f"\n  {'Block':<8} {'Type':<10} {'N_flat':<10} {'VitSparse MB':>14} {'AdaTAD MB':>12}")
    print(f"  {'-'*60}")

    for blk in range(12):
        stage = stage_map[blk]
        n_flat = N_stages[stage]
        ctype = conv_types[blk]

        if ctype == "sparse":
            # Peak per chunk for chunked implicit GEMM (chunk_size=32768)
            cs = 32768
            actual_chunk = min(cs, n_flat)
            gather_mb  = 9 * actual_chunk * C_vitsparse * 2 / 1e6
            outfp32_mb = 9 * actual_chunk * C_vitsparse * 4 / 1e6
            vs_overhead = gather_mb + outfp32_mb
        else:
            # 2d_conv: scatter to full grid [B, N_per_chunk, C] (one zero-fill + one scatter)
            vs_overhead = scatter_2d  # full grid allocation

        # AdaTAD dwconv: [B*H*W, C, T] × 2 (input + output)
        adatad_overhead = B * H * W * C_raw * T * 2 / 1e6 * 2

        total_sparse_overhead += vs_overhead
        total_adatad_overhead += adatad_overhead

        print(
            f"  {blk:<8} {ctype:<10} {n_flat:<10}"
            f" {vs_overhead:>14.1f}"
            f" {adatad_overhead:>12.1f}"
        )

    print(f"  {'-'*60}")
    print(f"  {'TOTAL':<18} {'':10} {total_sparse_overhead:>14.1f} {total_adatad_overhead:>12.1f}")
    print()
    print(f"  Peak difference (VitSparse − AdaTAD): {total_sparse_overhead - total_adatad_overhead:+.1f} MB")
    print(f"  (Note: peak ≠ sum; tensors are freed between layers)")
    print(f"  The per-layer peak is what sets the max_memory_allocated.")

    peak_vs  = max(
        9 * min(32768, N_stages[s]) * C_vitsparse * 2 / 1e6
        + 9 * min(32768, N_stages[s]) * C_vitsparse * 4 / 1e6
        if conv_types[i] == "sparse" else scatter_2d
        for i, s in enumerate(stage_map)
    )
    peak_ada = B * H * W * C_raw * T * 2 / 1e6 * 2
    print(f"\n  Worst-case single-layer peak:")
    print(f"    VitSparse : {peak_vs:.1f} MB  (blocks 0-2, sparse with full N)")
    print(f"    AdaTAD    : {peak_ada:.1f} MB")
    print(f"    Difference: {peak_vs - peak_ada:+.1f} MB  (theory only, not full model delta)")


if __name__ == "__main__":
    main()
