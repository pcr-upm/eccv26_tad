"""Validate one (config, checkpoint) pair BEFORE spending GPU time on tools/test.py.

Usage:
  python tools/paper_test_preflight.py \
      --config configs/vitsparse/thumos/e2e_thumos_videomae_b_768x1_160_sparse_adapter.py \
      --checkpoint /media/ricardo/data/sv-tad-weights/vitb_thumos_best.pth \
      --cfg-options model.backbone.backbone.n_landmarks=0

Exit code 0 = safe to run, 1 = would produce a wrong number (or crash).

Machine-readable: the last line is always
  PREFLIGHT_RESULT <ok|fail> [EXTRA_CFG_OPTS=<space separated k=v>]
`EXTRA_CFG_OPTS` carries overrides the caller SHOULD append, currently only
`model.backbone.custom.pretrain=None` when the ImageNet/K400 pretrain file is
absent. Nulling it is safe at test time: the checkpoint overwrites every
backbone weight the pretrain would have provided.
"""

import argparse
import json
import os
import re
import sys

import torch
from mmengine.config import Config, DictAction


def parse_args():
    parser = argparse.ArgumentParser(description="Preflight a test config against its checkpoint")
    parser.add_argument("--config", required=True, help="path to the test config")
    parser.add_argument("--checkpoint", required=True, help="path to the .pth being tested")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="same as tools/test.py")
    parser.add_argument("--ann-file", type=str, default=None, help="same as tools/test.py")
    parser.add_argument("--class-map", type=str, default=None, help="same as tools/test.py")
    parser.add_argument("--data-root", type=str, default=None, help="same as tools/test.py")
    parser.add_argument("--block-list", type=str, default=None, help="same as tools/test.py")
    parser.add_argument("--external-cls-path", type=str, default=None, help="same as tools/test.py")
    parser.add_argument(
        "--allow-partial-data",
        action="store_true",
        help="downgrade incomplete-dataset errors to warnings (smoke tests on a debug subset)",
    )
    return parser.parse_args()


def check_data_coverage(subset_name, data_path, ann_file, block_list):
    """How much of the test split is actually on this machine?

    """
    try:
        with open(ann_file) as f:
            db = json.load(f)
    except Exception:
        return None
    db = db.get("database", db)
    if not isinstance(db, dict):
        return None

    wanted = subset_name if isinstance(subset_name, (list, tuple)) else [subset_name]
    ids = [k for k, v in db.items() if isinstance(v, dict) and v.get("subset") in wanted]
    if not ids:
        return None

    if block_list and os.path.isfile(block_list):
        with open(block_list) as f:
            blocked = {line.strip() for line in f if line.strip()}
        ids = [i for i in ids if i not in blocked and f"v_{i}" not in blocked]

    n_gt = sum(len(db[i].get("annotations", [])) for i in ids)

    # An id maps either to a file (<data_path>/<id>.mp4, THUMOS/ANet) or to a
    # directory holding the video (<data_path>/<id>/, ATTACH). ANet ids in the
    # annotation lack the "v_" prefix the files carry.
    try:
        entries = set(os.listdir(data_path))
    except OSError:
        return (len(ids), 0, n_gt)
    stems = {os.path.splitext(e)[0] for e in entries}
    present = sum(1 for i in ids if i in stems or f"v_{i}" in stems)

    return (len(ids), present, n_gt)


def probe_checkpoint(path, prefer_ema):
    """Recover the architecture the checkpoint was trained with, from key names only."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if prefer_ema and "state_dict_ema" in ckpt:
        sd = ckpt["state_dict_ema"]
    else:
        sd = ckpt.get("state_dict", ckpt)

    # checkpoints may or may not carry the DDP "module." prefix
    sd = {k[len("module.") :] if k.startswith("module.") else k: v for k, v in sd.items()}

    depths = [int(m.group(1)) for k in sd if (m := re.search(r"blocks\.(\d+)\.", k))]
    hm = next((v for k, v in sd.items() if k.endswith("heatmap_head.final_layer.weight")), None)
    cls = next((v for k, v in sd.items() if k.endswith("rpn_head.cls_head.weight")), None)
    dim = next((v.shape[0] for k, v in sd.items() if k.endswith("blocks.0.norm1.weight")), None)

    return dict(
        epoch=ckpt.get("epoch"),
        has_ema="state_dict_ema" in ckpt,
        used_ema=prefer_ema and "state_dict_ema" in ckpt,
        depth=max(depths) + 1 if depths else None,
        embed_dims=dim,
        num_classes=cls.shape[0] if cls is not None else None,
        n_landmarks=hm.shape[0] if hm is not None else 0,
        n_dense_conv=sum(1 for k in sd if k.endswith("adapter.conv.weight")),
        n_sparse_conv=sum(1 for k in sd if k.endswith("adapter.sparse_conv.weight")),
    )


def probe_config(cfg):
    bb = cfg.model.backbone.backbone
    depth = bb.get("depth")
    # VideoMAE backbones use embed_dims/adapter_conv_types, InternVideoNextBackbone
    # uses the singular embed_dim/adapter_conv_type. Accept either spelling.
    conv_types = bb.get("adapter_conv_types") or bb.get("adapter_conv_type") or []
    if isinstance(conv_types, str):  # a single type is broadcast over every block
        conv_types = [conv_types] * (depth or 0)
    # The list is indexed by BLOCK, not by adapter: block i only gets an adapter
    # when i is in adapter_index, and its conv type is conv_types[i]. Counting the
    # whole list overcounts whenever adapters are sparse (e.g. range(0, 24, 2)).
    adapter_index = bb.get("adapter_index")
    if adapter_index is None:
        adapter_index = range(len(conv_types))
    built = [conv_types[i] for i in adapter_index if 0 <= i < len(conv_types)]
    return dict(
        depth=depth,
        embed_dims=bb.get("embed_dims") or bb.get("embed_dim"),
        n_landmarks=bb.get("n_landmarks", 0) or 0,
        n_dense_conv=sum(1 for t in built if t == "2d_conv"),
        n_sparse_conv=sum(1 for t in built if t == "sparse_conv"),
        num_classes=cfg.model.rpn_head.get("num_classes"),
        keep_rate=bb.get("keep_rate"),
        adapter_use_attn=bb.get("adapter_use_attn"),
    )


# (config key, checkpoint key, human label, how to fix it)
ARCH_CHECKS = [
    ("depth", "depth", "backbone depth", "model.backbone.backbone.depth"),
    ("embed_dims", "embed_dims", "backbone embed_dims", "model.backbone.backbone.embed_dims"),
    ("num_classes", "num_classes", "rpn_head num_classes", "model.rpn_head.num_classes"),
    ("n_landmarks", "n_landmarks", "n_landmarks (POGUISE heatmap tokens)", "model.backbone.backbone.n_landmarks"),
    ("n_dense_conv", "n_dense_conv", "adapter 2d_conv layers", "model.backbone.backbone.adapter_conv_type[s]"),
    ("n_sparse_conv", "n_sparse_conv", "adapter sparse_conv layers", "model.backbone.backbone.adapter_conv_type[s]"),
]


def main():
    args = parse_args()
    errors, warnings, extra_opts = [], [], []

    if not os.path.isfile(args.config):
        print(f"FATAL: config not found: {args.config}")
        print("PREFLIGHT_RESULT fail")
        return 1
    if not os.path.isfile(args.checkpoint):
        print(f"FATAL: checkpoint not found: {args.checkpoint}")
        print("PREFLIGHT_RESULT fail")
        return 1

    try:
        cfg = Config.fromfile(args.config)
    except Exception as exc:  # broken _base_ chains show up here
        print(f"FATAL: config failed to load: {type(exc).__name__}: {exc}")
        print("PREFLIGHT_RESULT fail")
        return 1
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    use_ema = bool(getattr(cfg.solver, "ema", False))
    ck = probe_checkpoint(args.checkpoint, prefer_ema=use_ema)
    cf = probe_config(cfg)

    print(f"config     : {args.config}")

    # 1. architecture: config-built model vs checkpoint contents
    for cf_key, ck_key, label, fix in ARCH_CHECKS:
        want, got = cf[cf_key], ck[ck_key]
        if want is None or got is None:
            continue
        if want != got:
            errors.append(
                f"{label}: config builds {want}, checkpoint has {got}. "
                f"Fix with --cfg-options {fix}=<matching value>"
            )

    # 2. pretrain: loaded at model-build time even for testing (BackboneWrapper),
    #    but immediately overwritten by the checkpoint -> absence is not fatal.
    pretrain = cfg.model.backbone.custom.get("pretrain")
    if pretrain is not None and not os.path.isfile(pretrain):
        warnings.append(f"pretrain not on this machine: {pretrain} -> nulling it (checkpoint supplies the weights)")
        extra_opts.append("model.backbone.custom.pretrain=None")

    # 3. dataset paths the test split will actually open
    test_split = cfg.dataset.test
    paths = {
        "dataset.test.data_path": args.data_root or test_split.get("data_path"),
        "dataset.test.ann_file": args.ann_file or test_split.get("ann_file"),
        "dataset.test.class_map": args.class_map or test_split.get("class_map"),
        "dataset.test.block_list": args.block_list or test_split.get("block_list"),
        "dataset.test.skeleton_data_path_2d": test_split.get("skeleton_data_path_2d"),
    }
    ext = cfg.get("post_processing", {}).get("external_cls")
    if isinstance(ext, dict) and ext.get("path"):
        paths["post_processing.external_cls.path"] = args.external_cls_path or ext["path"]

    for key, path in paths.items():
        if not path or not isinstance(path, str):
            continue
        if not os.path.exists(path):
            errors.append(f"{key} does not exist: {path}")

    print(f"test split : subset={test_split.get('subset_name')} batch_size={cfg.solver.test.get('batch_size')}")

    # 4. is the whole test split actually on this machine?
    cov = check_data_coverage(
        test_split.get("subset_name"),
        paths["dataset.test.data_path"],
        paths["dataset.test.ann_file"],
        paths["dataset.test.block_list"],
    )
    if cov is None:
        warnings.append("could not read the annotation file -- dataset coverage unchecked")
    else:
        n_ann, n_present, n_gt = cov
        print(f"             {n_present}/{n_ann} videos present on disk, {n_gt} GT instances in this annotation")
        if n_present < n_ann:
            msg = (
                f"INCOMPLETE DATASET: only {n_present} of {n_ann} test videos are on this machine. "
                f"OpenTAD drops missing videos silently, so the run will finish and print an mAP "
                f"computed on a subset -- not comparable to the reported number. "
                f"Fix --data-root, or pass --allow-partial-data for a deliberate smoke test."
            )
            (warnings if args.allow_partial_data else errors).append(msg)

    for w in warnings:
        print(f"WARN  {w}")
    for e in errors:
        print(f"ERROR {e}")

    if errors:
        print(f"PREFLIGHT_RESULT fail")
        return 1
    print("PREFLIGHT_RESULT ok" + (f" EXTRA_CFG_OPTS={' '.join(extra_opts)}" if extra_opts else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
