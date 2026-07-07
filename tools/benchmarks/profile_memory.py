"""
Profile peak GPU memory at each stage of VitSparse and AdaTAD forward passes.

Run:
    python tools/benchmarks/profile_memory.py <config.py>
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gc
import argparse
import torch
from mmengine.config import Config, DictAction
from opentad.models import build_detector
from opentad.datasets import build_dataset
from opentad.utils import set_seed


def mb(device="cuda:0"):
    return torch.cuda.memory_allocated(device) / 1e6


def peak_mb(device="cuda:0"):
    return torch.cuda.max_memory_allocated(device) / 1e6


def reset(device="cuda:0"):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def checkpoint(label, device="cuda:0", prev=None):
    cur = mb(device)
    pk = peak_mb(device)
    delta = f"({cur - prev:+.1f})" if prev is not None else ""
    print(f"  {label:<55}  alloc={cur:>8.1f} MB  peak={pk:>8.1f} MB  {delta}")
    return cur


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cfg-options", nargs="+", action=DictAction)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(42)
    device = args.device

    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    print(f"GPU : {torch.cuda.get_device_name(device)}")
    print(f"VRAM: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    print(f"Config: {os.path.basename(args.config)}")

    # Build dataset and model
    test_dataset = build_dataset(cfg.dataset.test, default_args=dict(logger=None))
    sample = test_dataset[0]
    batch = {
        k: (v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else
            [v] if isinstance(v, dict) else v)
        for k, v in sample.items()
    }

    reset(device)
    print("\n--- Memory before model build ---")
    checkpoint("empty", device)

    model = build_detector(cfg.model).to(device)
    model.eval()

    print("\n--- After model build ---")
    cur = checkpoint("model on GPU", device)
    model_mem = mb(device)
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Model memory: {model_mem:.1f} MB")

    # -----------------------------------------------------------------------
    # Instrument the backbone forward with memory checkpoints
    # -----------------------------------------------------------------------
    from opentad.models.backbones import backbone_wrapper as bw_mod
    orig_backbone_forward = model.backbone.model.backbone.forward

    block_mems = []

    def instrumented_backbone_forward(x):
        reset(device)
        b_cur = checkpoint("[backbone] entry x", device)

        # patch embed
        b, _, _, h, w = x.shape
        h_patch = h // model.backbone.model.backbone.patch_size
        w_patch = w // model.backbone.model.backbone.patch_size
        x_emb, _ = model.backbone.model.backbone.patch_embed(x)
        b_cur = checkpoint("[backbone] after patch_embed", device, b_cur)

        # Add CLS, pos embed etc — just call the real forward instead
        # We'll measure by splitting at block boundaries using forward hooks
        return orig_backbone_forward(x)

    # Use per-block hooks
    bb = model.backbone.model.backbone
    block_peaks = []

    def make_pre_hook(idx):
        def hook(module, input):
            reset(device)
            return None
        return hook

    def make_post_hook(idx):
        def hook(module, input, output):
            pk = peak_mb(device)
            block_peaks.append((idx, pk))
        return hook

    pre_handles  = []
    post_handles = []
    if hasattr(bb, "blocks"):
        for i, blk in enumerate(bb.blocks):
            pre_handles.append(blk.register_forward_pre_hook(make_pre_hook(i)))
            post_handles.append(blk.register_forward_hook(make_post_hook(i)))

    # -----------------------------------------------------------------------
    # Run one full forward pass and capture peak at the wrapper level
    # -----------------------------------------------------------------------
    from opentad.models.utils.post_processing import build_classifier
    if (
        "external_cls" in cfg.post_processing
        and cfg.post_processing.external_cls is not None
    ):
        ext_cls = build_classifier(cfg.post_processing.external_cls)
    else:
        ext_cls = test_dataset.class_map

    cfg.post_processing.sliding_window = False

    gc.collect()
    torch.cuda.empty_cache()

    # --- Stage 1: just the input in memory ---
    reset(device)
    inputs = batch.get("inputs")
    print("\n--- Memory stages during one forward pass ---")
    print(f"  {'Stage':<55}  {'alloc':>10}  {'peak':>10}")
    print(f"  {'-'*80}")
    cur = checkpoint("baseline (model only)", device)

    # Feed inputs to GPU (they're already there from get_single_sample)
    checkpoint(f"inputs on GPU  shape={list(inputs.shape)}", device, cur)

    # -----------------------------------------------------------------------
    # Instrument BackboneWrapper.forward to get stage-level memory
    # -----------------------------------------------------------------------
    orig_bw_forward = model.backbone.forward

    stage_log = []

    def instrumented_bw_forward(frames, masks=None):
        def tag(label):
            stage_log.append((label, mb(device), peak_mb(device)))

        tag("bw.forward entry")

        # Step-by-step replication of BackboneWrapper.forward
        model.backbone.set_norm_layer()
        tag("set_norm_layer done")

        frames_list = model.backbone.tensor_to_list(frames)
        tag(f"tensor_to_list  ({len(frames_list)} items)")

        frames_proc, _ = model.backbone.model.data_preprocessor.preprocess(
            frames_list, data_samples=None, training=False
        )
        tag(f"data_preprocessor  shape={list(frames_proc.shape)}")

        if model.backbone.pre_processing_pipeline is not None:
            frames_proc = model.backbone.pre_processing_pipeline(
                dict(frames=frames_proc)
            )["frames"]
        tag(f"pre_processing_pipeline  shape={list(frames_proc.shape)}")

        batches, num_segs = frames_proc.shape[0:2]
        frames_flat = frames_proc.flatten(0, 1).contiguous()
        tag(f"flatten(0,1)+contiguous  shape={list(frames_flat.shape)}")

        # Delete intermediate to free memory before backbone
        del frames_proc
        tag("frames_proc deleted")

        backbone_output = model.backbone.model.backbone(frames_flat)
        tag(f"backbone(frames_flat) done")

        del frames_flat
        tag("frames_flat deleted")

        if isinstance(backbone_output, tuple):
            features, heatmap = backbone_output
            model.backbone.heatmap_output = heatmap
        else:
            features = backbone_output
            model.backbone.heatmap_output = None
        tag(f"backbone output  shape={list(features.shape)}")

        features = model.backbone.unflatten_and_pool_features(features, batches, num_segs)
        tag(f"unflatten_and_pool  shape={list(features.shape)}")

        if masks is not None and features.dim() == 3:
            features = features * masks.unsqueeze(1).detach().float()

        features = features.to(torch.float32)
        tag(f"to fp32  shape={list(features.shape)}")

        return features

    model.backbone.forward = instrumented_bw_forward

    reset(device)
    with torch.cuda.amp.autocast(dtype=torch.float16, enabled=True):
        with torch.no_grad():
            _ = model(
                **batch,
                return_loss=False,
                infer_cfg=cfg.inference,
                post_cfg=cfg.post_processing,
                ext_cls=ext_cls,
            )

    # Restore
    model.backbone.forward = orig_bw_forward
    for h in pre_handles + post_handles:
        h.remove()

    # -----------------------------------------------------------------------
    # Print results
    # -----------------------------------------------------------------------
    print(f"\n  {'Stage':<55}  {'alloc':>10}  {'peak':>10}")
    print(f"  {'-'*80}")
    prev_alloc = stage_log[0][1] if stage_log else 0
    for label, alloc, pk in stage_log:
        delta = alloc - prev_alloc
        print(f"  {label:<55}  {alloc:>8.1f} MB  {pk:>8.1f} MB  ({delta:+.1f})")
        prev_alloc = alloc

    if block_peaks:
        print(f"\n  {'Block':>6}  {'peak during block (MB)':>25}")
        print(f"  {'-'*35}")
        for blk_idx, pk in block_peaks:
            print(f"  {blk_idx:>6}  {pk:>25.1f}")
        overall_peak = max(pk for _, pk in block_peaks)
        peak_block = max(block_peaks, key=lambda x: x[1])[0]
        print(f"\n  Max block peak: {overall_peak:.1f} MB at block {peak_block}")

    print(f"\nOverall peak (full forward): {peak_mb(device):.1f} MB")


if __name__ == "__main__":
    main()
