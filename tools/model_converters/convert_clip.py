import argparse
import torch


def process_checkpoint(in_path, out_path):
    """Extract the visual encoder weights from a JIT-traced OpenAI CLIP checkpoint
    and save them as a plain state_dict, loadable via load_checkpoint_with_prefix.

    The key names (conv1, class_embedding, positional_embedding, ln_pre,
    transformer.resblocks.*, ln_post) match VisionTransformerCLIPFreqAdapter
    directly, so no key remapping is needed. Extra keys present only in the
    original CLIP checkpoint (e.g. `proj`) are simply ignored (strict=False).
    """
    clip_model = torch.jit.load(in_path, map_location="cpu")
    state_dict = {k: v.float() for k, v in clip_model.visual.state_dict().items()}

    torch.save(state_dict, out_path)
    print(f"Saved {len(state_dict)} tensors from the CLIP visual encoder to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a JIT-traced OpenAI CLIP checkpoint's visual encoder "
        "weights to a plain state_dict for VisionTransformerCLIPFreqAdapter"
    )
    parser.add_argument("in_file", help="path to the CLIP .pt checkpoint (JIT-traced)")
    parser.add_argument("out_file", help="output .pth path")
    args = parser.parse_args()

    process_checkpoint(args.in_file, args.out_file)

"""example
python tools/model_converters/convert_clip.py \
    pretrained/ViT-B-16.pt pretrained/clip_vit_b16_visual.pth
"""
