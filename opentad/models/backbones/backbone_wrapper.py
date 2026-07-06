import copy
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
import torch.utils.checkpoint as cp

from mmengine.dataset import Compose
from mmengine.registry import MODELS as MM_BACKBONES
from mmengine.runner import load_checkpoint

BACKBONES = MM_BACKBONES


def load_checkpoint_with_prefix(model, checkpoint_path, prefix="backbone."):
    """Load checkpoint with key remapping to add prefix if needed.

    This handles cases where the checkpoint was saved from a standalone backbone
    but needs to be loaded into a wrapped model (e.g., Recognizer3D).
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Get state dict from checkpoint
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Get model state dict keys
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())

    # Check if keys need prefix
    # If checkpoint has "blocks.0.norm1.weight" but model expects "backbone.blocks.0.norm1.weight"
    sample_ckpt_key = next(iter(checkpoint_keys), None)
    needs_prefix = sample_ckpt_key and (prefix + sample_ckpt_key) in model_keys

    if needs_prefix:
        print(f"Adding '{prefix}' prefix to checkpoint keys")
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = prefix + k
            if new_key in model_keys:
                new_state_dict[new_key] = v
            else:
                # Key doesn't exist even with prefix, skip (e.g., adapter weights)
                pass
        state_dict = new_state_dict

    # Load with strict=False to allow missing adapter weights
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        # Filter out adapter-related missing keys for cleaner output
        adapter_missing = [k for k in missing if "adapter" in k]
        other_missing = [k for k in missing if "adapter" not in k]

        if other_missing:
            print(f"Missing keys (non-adapter): {len(other_missing)} keys")
            if len(other_missing) <= 10:
                for k in other_missing:
                    print(f"  - {k}")
        if adapter_missing:
            print(f"Missing keys (adapter, expected): {len(adapter_missing)} keys")

    if unexpected:
        print(f"Unexpected keys: {len(unexpected)} keys")
        if len(unexpected) <= 10:
            for k in unexpected:
                print(f"  - {k}")

    print(f"Successfully loaded checkpoint from {checkpoint_path}")


class BackboneWrapper(nn.Module):
    def __init__(self, cfg):
        super(BackboneWrapper, self).__init__()
        custom_cfg = cfg.custom
        model_cfg = copy.deepcopy(cfg)
        model_cfg.pop("custom")

        # build the backbone
        self.model = BACKBONES.build(model_cfg)
        self.heatmap_output = None
        # custom settings: pretrained checkpoint, post_processing_pipeline, norm_eval, freeze_backbone
        # 1. load the pretrained model
        if hasattr(custom_cfg, "pretrain") and custom_cfg.pretrain is not None:
            load_checkpoint_with_prefix(
                self.model, custom_cfg.pretrain, prefix="backbone."
            )
        else:
            print(
                "Warning: no pretrain path is provided, the backbone will be randomly initialized, "
                "unless you have initialized the weights in the model.py."
            )

        # 2. pre_processing_pipeline
        if hasattr(custom_cfg, "pre_processing_pipeline"):
            self.pre_processing_pipeline = Compose(custom_cfg.pre_processing_pipeline)
        else:
            self.pre_processing_pipeline = None

        # 3. post_processing_pipeline for pooling and other operations
        if hasattr(custom_cfg, "post_processing_pipeline"):
            self.post_processing_pipeline = Compose(custom_cfg.post_processing_pipeline)
        else:
            self.post_processing_pipeline = None

        # 4. norm_eval: set all norm layers to eval mode
        self.norm_eval = getattr(custom_cfg, "norm_eval", True)

        # 5. freeze_backbone: whether to freeze the backbone, default is False
        self.freeze_backbone = getattr(custom_cfg, "freeze_backbone", False)

        print(
            "freeze_backbone: {}, norm_eval: {}".format(
                self.freeze_backbone, self.norm_eval
            )
        )

        # 6. whether to use temporal activation checkpointing
        self.use_temporal_checkpointing = getattr(
            custom_cfg, "temporal_checkpointing", False
        )
        if self.use_temporal_checkpointing:
            assert hasattr(
                custom_cfg, "temporal_checkpointing_chunk_num"
            ), "temporal_checkpointing_chunk_num should be provided when using temporal checkpointing"
            assert hasattr(
                custom_cfg, "temporal_checkpointing_chunk_dim"
            ), "temporal_checkpointing_chunk_dim should be provided when using temporal checkpointing"
            self.temporal_checkpointing_chunk_num = (
                custom_cfg.temporal_checkpointing_chunk_num
            )
            self.temporal_checkpointing_chunk_dim = (
                custom_cfg.temporal_checkpointing_chunk_dim
            )

    def forward(self, frames, masks=None):
        # two types: snippet or frame

        # snippet: 3D backbone, [bs, T, 3, clip_len, H, W]
        # frame: 3D backbone, [bs, 1, 3, T, H, W]

        # set all normalization layers
        self.set_norm_layer()
        # data preprocessing: normalize mean and std
        frames, _ = self.model.data_preprocessor.preprocess(
            self.tensor_to_list(frames),  # need list input
            data_samples=None,
            training=False,  # for blending, which is not used in openTAD
        )

        # pre_processing_pipeline:
        if self.pre_processing_pipeline is not None:
            frames = self.pre_processing_pipeline(dict(frames=frames))["frames"]

        # flatten the batch dimension and num_segs dimension
        batches, num_segs = frames.shape[0:2]
        frames = frames.flatten(0, 1).contiguous()  # [bs*num_seg, ...]

        # go through the video backbone
        if self.freeze_backbone:  # freeze everything even in training
            with torch.no_grad():
                if self.use_temporal_checkpointing:
                    backbone_output = self.temporal_checkpointing(
                        frames,
                        self.temporal_checkpointing_chunk_num,
                        self.temporal_checkpointing_chunk_dim,
                    )
                else:
                    backbone_output = self.model.backbone(frames)
        else:
            if self.use_temporal_checkpointing:
                backbone_output = self.temporal_checkpointing(
                    frames,
                    self.temporal_checkpointing_chunk_num,
                    self.temporal_checkpointing_chunk_dim,
                )
            else:
                backbone_output = self.model.backbone(frames)

        if isinstance(backbone_output, tuple):
            # If the output is a tuple, we assume the first element is the main feature `x`
            # and the second is the auxiliary heatmap `x_heatmap`.
            features, heatmap = backbone_output

            # Unflatten the heatmap's batch dimension and store it
            # if heatmap is not None:
            #     heatmap = heatmap.unflatten(dim=0, sizes=(batches, num_segs))
            self.heatmap_output = heatmap
        else:
            # If the output is not a tuple, it's the old single-output behavior
            features = backbone_output
            self.heatmap_output = None
        if isinstance(features, (tuple, list)):
            features = torch.cat(
                [
                    self.unflatten_and_pool_features(f, batches, num_segs)
                    for f in features
                ],
                dim=1,
            )
        else:
            features = self.unflatten_and_pool_features(features, batches, num_segs)

        # apply mask
        if masks is not None and features.dim() == 3:
            features = features * masks.unsqueeze(1).detach().float()

        # make sure detector has the float32 input
        features = features.to(torch.float32)
        return features

    def get_heatmap(self):
        """
        Returns the heatmap generated during the last forward pass.
        This allows the main training loop to access it for loss calculation.
        """
        return self.heatmap_output

    def tensor_to_list(self, tensor):
        return [t for t in tensor]

    def unflatten_and_pool_features(self, features, batches, num_segs):
        # unflatten the batch dimension and num_segs dimension
        features = features.unflatten(
            dim=0, sizes=(batches, num_segs)
        )  # [bs, num_seg, ...]

        # convert the feature to [B,C,T]: pooling and other operations
        if self.post_processing_pipeline is not None:
            features = self.post_processing_pipeline(dict(feats=features))["feats"]
        return features

    def set_norm_layer(self):
        if self.norm_eval:
            for m in self.modules():
                if isinstance(m, (nn.LayerNorm, nn.GroupNorm, _BatchNorm)):
                    m.eval()

                    for param in m.parameters():
                        param.requires_grad = False

    def temporal_checkpointing(self, frames, chunk_num, chunk_dim):
        """Temporal Checkpointing for Video Backbone.

        Temporal checkpointing will 1) split the video frames along the temporal dimension and sequentially forward each chunk with
        no gradients. 2) The backward pass will recompute the intermediate activations and compute each chunk's gradient. 3) Backbone's
        gradients will be accumulated along different chunks.

        Args:
            frames (Tensor): input frames, [B*N,3,T,H,W]
            chunk_num (int): number of chunks to split the temporal dimension
            chunk_dim (int): input shape is [B*N,3,T,H,W], so either dim=0 or 2 is fine
        """

        def _inner_forward(frames):
            return self.model.backbone(frames)

        # This part of the logic remains the same
        video_chunks_output = []
        for mini_frames in torch.chunk(frames, chunk_num, dim=chunk_dim):
            mini_output = cp.checkpoint(
                _inner_forward, mini_frames, use_reentrant=False
            )
            video_chunks_output.append(mini_output)

        # --- NEW LOGIC to correctly re-assemble the tuple output ---
        first_output = video_chunks_output[0]
        if isinstance(first_output, tuple):
            # If the output is a tuple (features, heatmap), we need to re-assemble each part separately
            num_outputs = len(first_output)
            final_output = []
            for i in range(num_outputs):
                # For each element of the tuple, gather all the chunks and concatenate them
                if first_output[i] is not None:
                    concatenated_element = torch.cat(
                        [chunk[i] for chunk in video_chunks_output], dim=chunk_dim
                    )
                    final_output.append(concatenated_element)
                else:
                    final_output.append(None)
            return tuple(final_output)
        else:
            # If the output is a single tensor, handle it as before
            return torch.cat(video_chunks_output, dim=chunk_dim)
