"""
Convert InternVideoNext safetensors checkpoint to PyTorch format.

This script converts HuggingFace safetensors checkpoints to be compatible
with the native (non-flash_attn) InternVideoNext implementation.

Example usage:
    python tools/model_converters/convert_internvideo_next.py \
        internvideo_next_large_p14_res224_f16/model.safetensors \
        pretrained/internvideo_next_large_native.pth \
        --arch large

    # Or directly use the HuggingFace model path:
    python tools/model_converters/convert_internvideo_next.py \
        internvideo_next_large_p14_res224_f16 \
        pretrained/internvideo_next_large_native.pth \
        --arch large
"""

import argparse
import os
import torch


def load_checkpoint(in_path):
    """Load checkpoint from safetensors or pth file."""
    if os.path.isdir(in_path):
        # HuggingFace model directory
        safetensors_path = os.path.join(in_path, "model.safetensors")
        if os.path.exists(safetensors_path):
            in_path = safetensors_path
        else:
            pth_path = os.path.join(in_path, "pytorch_model.bin")
            if os.path.exists(pth_path):
                in_path = pth_path
            else:
                raise FileNotFoundError(f"No model weights found in {in_path}")
    
    if in_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(in_path)
        print(f"Loaded safetensors checkpoint from {in_path}")
    else:
        checkpoint = torch.load(in_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "module" in checkpoint:
            state_dict = checkpoint["module"]
        else:
            state_dict = checkpoint
        print(f"Loaded PyTorch checkpoint from {in_path}")
    
    return state_dict


def get_model_config(arch):
    """Get model configuration based on architecture."""
    configs = {
        "base": {
            "img_size": 224,
            "patch_size": 14,
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "mlp_ratio": 4,
            "attn_pool_num_heads": 16,
            "clip_embed_dim": 768,
            "num_frames": 16,
            "tubelet_size": 1,
        },
        "large": {
            "img_size": 224,
            "patch_size": 14,
            "embed_dim": 1024,
            "depth": 24,
            "num_heads": 16,
            "mlp_ratio": 4,
            "attn_pool_num_heads": 16,
            "clip_embed_dim": 768,
            "num_frames": 16,
            "tubelet_size": 1,
        },
        "giant": {
            "img_size": 224,
            "patch_size": 14,
            "embed_dim": 1408,
            "depth": 40,
            "num_heads": 16,
            "mlp_ratio": 4.3637,
            "attn_pool_num_heads": 16,
            "clip_embed_dim": 768,
            "num_frames": 16,
            "tubelet_size": 1,
        },
    }
    
    if arch not in configs:
        raise ValueError(f"Unsupported architecture: {arch}. Supported: {list(configs.keys())}")
    
    return configs[arch]


def convert_key(key):
    """
    Convert state dict key from original to native format.
    
    The native implementation doesn't use fused operations, so some keys
    might need remapping. Currently the architecture is identical so
    most keys remain the same.
    """
    # Strip 'model.' prefix if present (common in HuggingFace safetensors)
    if key.startswith("model."):
        key = key[6:]
    
    # The native model uses the same key names, so no further conversion needed
    return key


def process_checkpoint(in_path, out_path, arch, verify=True):
    """
    Process and convert InternVideoNext checkpoint.
    
    Args:
        in_path: Path to input checkpoint (safetensors or pth)
        out_path: Path to output PyTorch checkpoint
        arch: Model architecture ('base', 'large', 'giant')
        verify: Whether to verify the conversion by loading the model
    """
    # Load source checkpoint
    src_state_dict = load_checkpoint(in_path)
    
    # Convert keys
    new_state_dict = {}
    for key, value in src_state_dict.items():
        new_key = convert_key(key)
        new_state_dict[new_key] = value
        if key != new_key:
            print(f"  Renamed: {key} -> {new_key}")
    
    print(f"\nConverted {len(new_state_dict)} parameters")
    
    # Optionally verify by loading the model
    if verify:
        print("\nVerifying conversion by loading model...")
        try:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            from internvideo_next_large_p14_res224_f16.modeling_internvideo_next_native import InternVideoNextBackbone
            
            model_cfg = get_model_config(arch)
            model = InternVideoNextBackbone(**model_cfg)
            
            # Check for missing/unexpected keys
            model_keys = set(model.state_dict().keys())
            ckpt_keys = set(new_state_dict.keys())
            
            missing_keys = model_keys - ckpt_keys
            unexpected_keys = ckpt_keys - model_keys
            
            if missing_keys:
                print(f"\nMissing keys in checkpoint ({len(missing_keys)}):")
                for k in sorted(missing_keys)[:20]:
                    print(f"  {k}")
                if len(missing_keys) > 20:
                    print(f"  ... and {len(missing_keys) - 20} more")
            
            if unexpected_keys:
                print(f"\nUnexpected keys in checkpoint ({len(unexpected_keys)}):")
                for k in sorted(unexpected_keys)[:20]:
                    print(f"  {k}")
                if len(unexpected_keys) > 20:
                    print(f"  ... and {len(unexpected_keys) - 20} more")
            
            # Load with strict=False to see what loads
            load_result = model.load_state_dict(new_state_dict, strict=False)
            print(f"\nLoaded checkpoint with {len(model_keys) - len(load_result.missing_keys)} matching keys")
            
            # Quick forward pass test
            print("\nTesting forward pass...")
            model.eval()
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 16, 224, 224)
                output = model(dummy_input, projected=True)
                print(f"  Input shape: {dummy_input.shape}")
                print(f"  Output shape: {output.shape}")
                print("  Forward pass successful!")
            
        except Exception as e:
            print(f"\nWarning: Verification failed: {e}")
            print("The checkpoint will still be saved, but please verify manually.")
    
    # Save converted checkpoint
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    torch.save(new_state_dict, out_path)
    print(f"\nSaved converted checkpoint to: {out_path}")
    
    return new_state_dict


def main():
    parser = argparse.ArgumentParser(
        description="Convert InternVideoNext checkpoint to native format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "in_file",
        help="Input checkpoint path (safetensors, pth, or HuggingFace model directory)"
    )
    parser.add_argument(
        "out_file",
        help="Output checkpoint path (.pth)"
    )
    parser.add_argument(
        "--arch",
        default="large",
        choices=["base", "large", "giant"],
        help="Model architecture (default: large)"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification step"
    )
    
    args = parser.parse_args()
    
    process_checkpoint(
        args.in_file,
        args.out_file,
        args.arch,
        verify=not args.no_verify
    )


if __name__ == "__main__":
    main()
