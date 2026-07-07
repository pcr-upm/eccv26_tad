"""
Granular per-operation memory profiling inside a single VitSparse Block.

Instead of a full model forward, we directly run the block with synthetic
inputs and record memory allocated BEFORE each sub-operation (after resetting
the peak counter).  This isolates exactly where memory spikes.

Run:
    python tools/benchmarks/profile_block.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gc, math, torch, torch.nn as nn, torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN

# ─── helpers ──────────────────────────────────────────────────────────────────

DEVICE = "cuda:0"

def alloc_mb(): return torch.cuda.memory_allocated(DEVICE) / 1e6
def peak_mb():  return torch.cuda.max_memory_allocated(DEVICE) / 1e6
def reset():
    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(DEVICE)

log = []

def tag(label):
    log.append((label, alloc_mb(), peak_mb()))

def print_log(title=""):
    if title: print(f"\n{'='*72}\n{title}\n{'='*72}")
    print(f"  {'Op':<55}  {'alloc':>10}  {'peak':>10}  {'delta':>8}")
    print(f"  {'-'*88}")
    prev = log[0][1] if log else 0
    for lbl, alloc, pk in log:
        d = alloc - prev
        print(f"  {lbl:<55}  {alloc:>8.1f} MB  {pk:>8.1f} MB  {d:>+8.1f}")
        prev = alloc
    print(f"\n  PEAK: {max(p for _,_,p in log):.1f} MB")

# ─── config mirrors the actual VitSparse-Base Thumos benchmark ────────────────
# B=48 (chunk_num), N=1569 (CLS + 1568 patches), embed=768, heads=12, ffn=4×
B, N, C, H = 48, 1569, 768, 12        # batch, tokens, embed, heads
N_patch = N - 1                        # 1568 patch tokens
h_patch, w_patch, T_patch = 14, 14, 8 # spatial and temporal patch grid

# adapter hidden: 768*0.25=192 → 2**round(log2(192))=256
hidden = 2 ** round(math.log2(int(C * 0.25)))   # = 256
attn_dim = hidden // 2                            # = 128

print(f"B={B}  N={N}  C={C}  hidden={hidden}  attn_dim={attn_dim}")

# ─── build minimal sub-modules ────────────────────────────────────────────────

norm_cfg = dict(type="LN", eps=1e-6)
norm1 = build_norm_layer(norm_cfg, C)[1].to(DEVICE)
norm2 = build_norm_layer(norm_cfg, C)[1].to(DEVICE)

qkv_w   = nn.Parameter(torch.randn(C * 3, C, device=DEVICE))
q_bias  = nn.Parameter(torch.zeros(C, device=DEVICE))
v_bias  = nn.Parameter(torch.zeros(C, device=DEVICE))
proj    = nn.Linear(C, C).to(DEVICE)

ffn = FFN(embed_dims=C, feedforward_channels=C*4,
          act_cfg=dict(type="GELU"), ffn_drop=0.0, add_identity=False).to(DEVICE)

# adapter sub-modules (2d_conv + use_attn=3)
down_proj  = nn.Linear(C, hidden).to(DEVICE)
up_proj    = nn.Linear(hidden, C).to(DEVICE)
conv2d     = nn.Conv2d(hidden, hidden, 3, padding=1).to(DEVICE)
attn_norm  = build_norm_layer(norm_cfg, hidden)[1].to(DEVICE)
down_attn  = nn.Linear(hidden, attn_dim).to(DEVICE)
qkv_attn   = nn.Linear(attn_dim, attn_dim * 3).to(DEVICE)
attn_proj  = nn.Linear(attn_dim, hidden).to(DEVICE)
gamma      = nn.Parameter(torch.ones(1, device=DEVICE))

# ─── synthetic input ─────────────────────────────────────────────────────────
# simulate state at entry to block 0 (no pruning yet)

reset(); log.clear()

x = torch.randn(B, N, C, dtype=torch.float16, device=DEVICE)
idx = torch.arange(N_patch, device=DEVICE).unsqueeze(0).expand(B, -1)
neighbor_indices = torch.randint(0, N_patch, (B, N_patch, 9),
                                 dtype=torch.int32, device=DEVICE)

tag("after inputs (x, idx, neigh_idx)")

# ─── ATTENTION ────────────────────────────────────────────────────────────────

with torch.cuda.amp.autocast(dtype=torch.float16, enabled=True):
  with torch.no_grad():

    tag("--- ATTENTION ---")
    x_norm = norm1(x)
    tag("norm1(x)")

    qkv_bias = torch.cat([q_bias, torch.zeros_like(v_bias), v_bias])
    qkv = F.linear(x_norm, qkv_w, qkv_bias)
    tag("F.linear → qkv")

    qkv_r = qkv.reshape(B, N, 3, H, C//H).permute(2,0,3,1,4)
    q, k, v = qkv_r[0], qkv_r[1], qkv_r[2]
    tag("reshape/permute qkv → q,k,v (views)")

    # Flash attention (keep_rate=1, no partial attn needed)
    attn_out = F.scaled_dot_product_attention(q, k, v)
    tag("scaled_dot_product_attn (flash)")

    attn_out = attn_out.transpose(1,2).reshape(B, N, C)
    tag("transpose+reshape attn_out")

    del qkv, qkv_r, q, k, v
    tag("del qkv tensors")

    attn_out = proj(attn_out)
    tag("proj(attn_out)")

    x = x + attn_out
    tag("x = x + attn_out (residual)")
    del attn_out, x_norm

    # ─── pruning (simulated: keep 60%) ───────────────────────────────────────
    tag("--- PRUNING (keep 60%) ---")
    N_keep = math.ceil(0.6 * N_patch)
    keep_idx = torch.randint(0, N_patch, (B, N_keep), device=DEVICE)
    x_key = x[:, :1]  # CLS
    x_nonkey = x[:, 1:]
    idx_exp = keep_idx.unsqueeze(-1).expand(-1, -1, C)
    x_kept = torch.gather(x_nonkey, 1, idx_exp)
    x = torch.cat([x_key, x_kept], dim=1)
    tag(f"pruned x → [{B},{N_keep+1},{C}]")
    del x_key, x_nonkey, x_kept, keep_idx, idx_exp

    # ─── FFN ─────────────────────────────────────────────────────────────────
    tag("--- FFN ---")
    x_n2 = norm2(x)
    tag("norm2(x) — pruned")

    # manual: linear1 → GELU → linear2  (ffn.layers[0][0] and ffn.layers[1])
    ffn_h = F.linear(x_n2, ffn.layers[0][0].weight, ffn.layers[0][0].bias)
    tag(f"FFN linear1 → [{B},{N_keep+1},{C*4}] fp16")

    ffn_h = F.gelu(ffn_h)
    tag("FFN GELU (in-place approx)")

    ffn_out = F.linear(ffn_h, ffn.layers[1].weight, ffn.layers[1].bias)
    tag("FFN linear2 → x-shape")

    del ffn_h, x_n2
    x = x + ffn_out
    tag("x = x + ffn_out (residual)")
    del ffn_out

    # ─── ADAPTER (2d_conv path, block 3 = first pruning block) ─────────────
    tag("--- ADAPTER (2d_conv, post-pruning) ---")
    inputs = x   # save residual

    x_proj = down_proj(x)                 # [B, N_keep+1, hidden]
    tag("down_proj → x_proj")
    x_proj = F.gelu(x_proj)
    tag("GELU x_proj")

    patch_proj = x_proj[:, 1:]            # [B, N_keep, hidden]
    tag("split special/patch tokens (view)")

    # 2d_conv with full_grid scatter (N_keep < N_patch)
    full_grid = torch.zeros(B, N_patch, hidden, dtype=x.dtype, device=DEVICE)
    tag(f"full_grid = zeros([{B},{N_patch},{hidden}])")

    # simulated scatter (no real idx — use random positions)
    rnd_idx = torch.randint(0, N_patch, (B, N_keep, 1), device=DEVICE).expand(-1,-1, hidden)
    full_grid = full_grid.scatter_(1, rnd_idx, patch_proj)
    tag("scatter_ patch_proj → full_grid")
    del rnd_idx

    conv_in = full_grid.reshape(B * T_patch, h_patch, w_patch, hidden)
    tag("reshape full_grid → [B*T, H, W, hid]")

    conv_in = conv_in.permute(0, 3, 1, 2).contiguous()
    tag("permute+contiguous → [B*T, hid, H, W]")
    del full_grid

    conv_out = conv2d(conv_in)
    tag("conv2d")
    del conv_in

    # scatter result back → gather at kept positions (skip for brevity)
    tag("(gather omitted for simplicity)")
    del conv_out

    # ─── ADAPTER ATTENTION (use_attn=3) ──────────────────────────────────────
    tag("--- ADAPTER ATTENTION (use_attn=3) ---")
    x_comb = x_proj                       # [B, N_keep+1, hidden]
    tag("x_combined (x_proj)")

    x_an = attn_norm(x_comb)
    tag("attn_norm(x_comb)")

    x_da = down_attn(x_an)               # [B, N_keep+1, attn_dim]
    tag("down_attn → attn_dim")
    del x_an

    qkv_a = qkv_attn(x_da)              # [B, N_keep+1, 3*attn_dim]
    tag("qkv_attn")
    del x_da

    qkv_a_r = qkv_a.reshape(B, N_keep+1, 3, 4, attn_dim//4).permute(2,0,3,1,4)
    qa, ka, va = qkv_a_r[0], qkv_a_r[1], qkv_a_r[2]
    tag("reshape/permute qkv_a → qa,ka,va (views)")

    # Option 3: CLS attends to visual
    q_key = qa[:, :, :1]
    k_vis = ka[:, :, 1:]
    v_vis = va[:, :, 1:]
    xa_key = F.scaled_dot_product_attention(q_key, k_vis, v_vis)
    tag("attn_key→vis (flash)")

    xa_vis = torch.zeros_like(qa[:, :, 1:])
    tag("zeros_like for visual attn output")

    xa = torch.cat([xa_key, xa_vis], dim=2)
    tag("cat attn results")
    del qkv_a, qkv_a_r, qa, ka, va, xa_key, xa_vis, q_key, k_vis, v_vis

    xa = xa.transpose(1,2).reshape(B, N_keep+1, attn_dim)
    tag("transpose+reshape xa")

    xa = attn_proj(xa)                    # [B, N_keep+1, hidden]
    tag("attn_proj")
    del xa

    # ─── UP-PROJECTION + RESIDUAL ─────────────────────────────────────────────
    tag("--- UP-PROJ + RESIDUAL ---")
    out = up_proj(x_comb)                 # [B, N_keep+1, C]
    tag("up_proj → embed_dims")
    del x_comb, x_proj

    x = out * gamma + inputs
    tag("x = out*gamma + inputs (final residual)")
    del out, inputs

    tag("=== BLOCK DONE ===")

print_log("Per-operation memory inside Block (B=48, N=1569, C=768, hidden=256)")

print("\n\nFor comparison — AdaTAD temporal_dwconv adapter (no pruning, full N):")
log2 = []

def tag2(label): log2.append((label, alloc_mb(), peak_mb()))

reset()
x2 = torch.randn(B, N, C, dtype=torch.float16, device=DEVICE)
tag2("after x (full N, fp16)")
N_full = N - 1   # 1568 patches

with torch.cuda.amp.autocast(dtype=torch.float16, enabled=True):
  with torch.no_grad():
    # dwconv adapter (AdaTAD): scatter → temporal_dwconv → scatter back
    tag2("--- AdaTAD temporal_dwconv adapter ---")
    hid_ada = int(C * 0.25)   # 192, no power-of-2 rounding
    dwconv = nn.Conv1d(hid_ada, hid_ada, 3, padding=1, groups=hid_ada).to(DEVICE).half()
    pwconv = nn.Conv1d(hid_ada, hid_ada, 1).to(DEVICE).half()
    dp2 = nn.Linear(C, hid_ada).to(DEVICE)
    up2 = nn.Linear(hid_ada, C).to(DEVICE)

    xp2 = dp2(x2)                  # [B, N, 192]
    tag2(f"down_proj → [B,N,{hid_ada}]")
    xp2 = F.gelu(xp2)
    tag2("GELU")

    # no pruning → no full_grid scatter needed
    patch2 = xp2[:, 1:]             # [B, N_full, 192] = [48, 1568, 192]
    tag2("patch split (view)")

    cv2 = patch2.reshape(B, T_patch, h_patch, w_patch, hid_ada).permute(0,2,3,4,1).flatten(0,2)
    tag2(f"reshape→[B*H*W,hid,T]=[{B*h_patch*w_patch},{hid_ada},{T_patch}]")

    cv2 = pwconv(dwconv(cv2))
    tag2("dwconv+pwconv")
    del cv2, patch2, xp2
    tag2("del temporaries")

print(f"\n  {'Op':<55}  {'alloc':>10}  {'peak':>10}  {'delta':>8}")
print(f"  {'-'*88}")
prev = log2[0][1] if log2 else 0
for lbl, alloc, pk in log2:
    d = alloc - prev
    print(f"  {lbl:<55}  {alloc:>8.1f} MB  {pk:>8.1f} MB  {d:>+8.1f}")
    prev = alloc
print(f"\n  PEAK: {max(p for _,_,p in log2):.1f} MB")
