import argparse
import torch
from mmaction.registry import MODELS
from mmengine.runner import save_checkpoint
from mmaction.utils import register_all_modules


register_all_modules()


def process_checkpoint(in_path, out_path, arch, num_classes):
    already_prefixed = False
    if in_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        video_state_dict = load_file(in_path)
        print("Loaded safetensors checkpoint with keys:", video_state_dict.keys())
    else:
        videomae_checkpoint = torch.load(in_path, map_location="cpu")
        if "module" in videomae_checkpoint:
            # raw VideoMAEv2 release format, e.g. vit_g_hybrid_pt_1200e.pth
            video_state_dict = videomae_checkpoint["module"]
        elif "state_dict" in videomae_checkpoint:
            # mmaction-style training checkpoint (meta/state_dict/optimizer),
            # e.g. some checkpoints mirrored on the OpenGVLab/VideoMAE2 HF repo.
            # Keys are already "backbone."-prefixed and may include a
            # finetuned cls_head for an unrelated task/dataset.
            video_state_dict = videomae_checkpoint["state_dict"]
            already_prefixed = True
            print(
                "Loaded mmaction-style checkpoint (state_dict/meta/optimizer); "
                "treating keys as already backbone-prefixed."
            )
        else:
            video_state_dict = videomae_checkpoint

    if arch == "small":
        model_cfg = dict(
            type="Recognizer3D",
            backbone=dict(
                type="VisionTransformer",
                img_size=224,
                patch_size=16,
                embed_dims=384,
                depth=12,
                num_heads=6,
                mlp_ratio=4,
                qkv_bias=True,
                num_frames=16,
                norm_cfg=dict(type="LN", eps=1e-6),
            ),
            cls_head=dict(
                type="TimeSformerHead",
                num_classes=num_classes,
                in_channels=384,
                average_clips="prob",
            ),
            data_preprocessor=dict(
                type="ActionDataPreprocessor",
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                format_shape="NCTHW",
            ),
        )
    elif arch == "base":
        model_cfg = dict(
            type="Recognizer3D",
            backbone=dict(
                type="VisionTransformer",
                img_size=224,
                patch_size=16,
                embed_dims=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4,
                qkv_bias=True,
                num_frames=16,
                norm_cfg=dict(type="LN", eps=1e-6),
            ),
            cls_head=dict(
                type="TimeSformerHead",
                num_classes=num_classes,
                in_channels=768,
                average_clips="prob",
            ),
            data_preprocessor=dict(
                type="ActionDataPreprocessor",
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                format_shape="NCTHW",
            ),
        )
    elif arch == "large":
        model_cfg = dict(
            type="Recognizer3D",
            backbone=dict(
                type="VisionTransformer",
                img_size=224,
                patch_size=16,
                embed_dims=1024,
                depth=24,
                num_heads=16,
                mlp_ratio=4,
                qkv_bias=True,
                num_frames=16,
                norm_cfg=dict(type="LN", eps=1e-6),
            ),
            cls_head=dict(
                type="TimeSformerHead",
                num_classes=num_classes,
                in_channels=1024,
                average_clips="prob",
            ),
            data_preprocessor=dict(
                type="ActionDataPreprocessor",
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                format_shape="NCTHW",
            ),
        )
    elif arch == "huge":
        model_cfg = dict(
            type="Recognizer3D",
            backbone=dict(
                type="VisionTransformer",
                img_size=224,
                patch_size=16,
                embed_dims=1280,
                depth=32,
                num_heads=16,
                mlp_ratio=4,
                qkv_bias=True,
                num_frames=16,
                norm_cfg=dict(type="LN", eps=1e-6),
            ),
            cls_head=dict(
                type="TimeSformerHead",
                num_classes=num_classes,
                in_channels=1280,
                average_clips="prob",
            ),
            data_preprocessor=dict(
                type="ActionDataPreprocessor",
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                format_shape="NCTHW",
            ),
        )
    elif arch == "giant":
        model_cfg = dict(
            type="Recognizer3D",
            backbone=dict(
                type="VisionTransformer",
                img_size=224,
                patch_size=14,
                embed_dims=1408,
                depth=40,
                num_heads=16,
                mlp_ratio=48 / 11,
                qkv_bias=True,
                num_frames=16,
                norm_cfg=dict(type="LN", eps=1e-6),
            ),
            cls_head=dict(
                type="TimeSformerHead",
                num_classes=num_classes,
                in_channels=1408,
                average_clips="prob",
            ),
            data_preprocessor=dict(
                type="ActionDataPreprocessor",
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                format_shape="NCTHW",
            ),
        )
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    model = MODELS.build(model_cfg)
    # video_state_dict is already loaded

    new_state_dict = {}
    model_state_dict = model.state_dict()
    for key, value in video_state_dict.items():
        # Strip 'model.' prefix if present (common in safetensors from HF)
        if key.startswith("model."):
            key = key[6:]

        # mmaction-style checkpoints already have the "backbone." prefix
        # (and a "cls_head." prefix for the head, which is left untouched)
        if already_prefixed and key.startswith("backbone."):
            key = key[len("backbone."):]

        # convert keys
        if "fc1" in key:
            key = key.replace("fc1", "layers.0.0")
        elif "fc2" in key:
            key = key.replace("fc2", "layers.1")
        elif "patch_embed.proj" in key:
            key = key.replace("patch_embed.proj", "patch_embed.projection")
        elif "head" in key and not key.startswith("cls_head"):
            key = key.replace("head", "cls_head.fc_cls")

        if "backbone." + key in model_state_dict:  # blocks.0.xxx
            candidate = "backbone." + key
        elif key.startswith("cls_head") and key in model_state_dict:
            candidate = key
        else:
            continue

        if model_state_dict[candidate].shape != value.shape:
            # e.g. a cls_head finetuned for a different number of classes
            print(f"Skipping {candidate}: shape mismatch {model_state_dict[candidate].shape} vs {value.shape}")
            continue
        new_state_dict[candidate] = value

    print("The following keys exist in model_cfg but not in the new checkpoint:")
    for key, value in model.state_dict().items():
        if key not in new_state_dict.keys():
            print(key)

    model.load_state_dict(new_state_dict, strict=False)
    save_checkpoint(model.state_dict(), out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert VideoMAEv2 checkpoint")
    parser.add_argument("in_file", help="input checkpoint path")
    parser.add_argument("out_file", help="output checkpoint path")
    parser.add_argument(
        "--arch",
        default="giant",
        choices=["small", "base", "large", "huge", "giant"],
        help="model architecture",
    )
    parser.add_argument(
        "--num_classes", default=710, type=int, help="number of classes"
    )
    args = parser.parse_args()

    process_checkpoint(args.in_file, args.out_file, args.arch, args.num_classes)

"""example
python tools/model_converters/convert_videomaev2.py \
   /home/ricardo/Documents/OpenTAD/pretrained/vit_b_k710_dl_from_giant.pth pretrained/vit_b_k710_dl_from_giant_k710_ft_my.pth --arch base
"""
