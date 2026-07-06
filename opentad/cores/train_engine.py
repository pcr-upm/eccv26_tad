import copy
import torch
import tqdm
from opentad.utils.misc import AverageMeter, reduce_loss
import wandb
import os

import matplotlib
import numpy as np
from mmpose.codecs import UDPHeatmap

from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # Set the backend before importing pyplot


def unnormalize_image(
    tensor, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
):
    """Reverses the ImageNet normalization on a tensor image."""
    tensor = tensor.clone()
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor


def visualize_and_log_batch(
    data_dict,
    gt_motion_heatmap,
    epoch_num=1,
    iter_num=1,
    mean=None,
    std=None,
    save_path="./",
    log_to_wandb=False,
):
    """
    Visualizes a sample frame, its motion heatmap, and the trajectory of
    keypoints over the first 16 frames.
    """
    # --- 1. Prepare Image ---
    imgs_batch = data_dict["inputs"][0].cpu()
    single_img_clip = imgs_batch[0]  # Shape: (C, T, H, W)
    # Use the first frame as the static background
    img_frame = single_img_clip[:, 7, :, :]
    img_unnorm = unnormalize_image(img_frame)
    img_np = img_unnorm.permute(1, 2, 0).numpy().clip(0, 255).astype(np.uint8)

    # --- 2. Prepare Motion Heatmap ---
    # Select the first sample from the batch. Assuming the temporal dimension
    # of the heatmap corresponds to chunks, we take the first chunk's heatmap.
    single_motion_heatmap_chunk = gt_motion_heatmap[0, :, :, :].cpu().detach().float()
    aggregated_heatmap = torch.sum(single_motion_heatmap_chunk, dim=0)
    heatmap_resized = (
        F.interpolate(
            aggregated_heatmap.unsqueeze(0).unsqueeze(0),
            size=img_np.shape[:2],
            mode="bilinear",
            align_corners=False,
        )
        .squeeze()
        .numpy()
    )

    # --- 3. Prepare Keypoint Coordinates for Plotting ---
    # Get the augmented 2D keypoint coordinates for the first sample
    keypoints_gt_clip = (
        data_dict["keypoint"][0].cpu().detach().float()
    )  # Shape: (T, V, 2)
    # Select only the first 16 frames to visualize the trajectory
    keypoints_for_trajectory = keypoints_gt_clip[:, :16, :, :].squeeze(
        -1
    )  # Shape: (3, 16, V)
    print(keypoints_for_trajectory.shape, keypoints_gt_clip.shape)
    # --- 4. Create and save the comprehensive plot ---
    fig, ax = plt.subplots(1, figsize=(12, 12), dpi=120)

    # Plot the base image
    ax.imshow(img_np)

    # Plot the motion heatmap overlay with transparency
    ax.imshow(heatmap_resized, cmap="hot", alpha=0.5)

    # --- NEW: Plot Keypoint Trajectories ---
    # Define skeleton connections
    skeleton_connections = [
        (2, 3),
        (3, 26),
        (26, 27),
        (2, 11),
        (11, 12),
        (12, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (14, 17),
        (2, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (7, 10),
        (0, 1),
        (1, 2),
        (0, 22),
        (22, 23),
        (23, 24),
        (24, 25),
        (0, 18),
        (18, 19),
        (19, 20),
        (20, 21),
    ]

    # Use a colormap to show the passage of time
    colors = plt.cm.viridis(np.linspace(0, 1, 16))

    # Plot each of the first 16 frames' skeletons
    for frame_idx in range(keypoints_for_trajectory.shape[1]):
        keypoints_for_frame = keypoints_for_trajectory[:, frame_idx, :]  # Shape: (3, V)
        x_coords = keypoints_for_frame[0, :]
        y_coords = keypoints_for_frame[1, :]

        # Scatter plot for the joints at this frame
        ax.scatter(
            x_coords,
            y_coords,
            s=15,
            color=colors[frame_idx],
            marker="o",
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
        )

        # Plot the connections for this frame
        for start_idx, end_idx in skeleton_connections:
            if start_idx < len(x_coords) and end_idx < len(x_coords):
                ax.plot(
                    [x_coords[start_idx], x_coords[end_idx]],
                    [y_coords[start_idx], y_coords[end_idx]],
                    color=colors[frame_idx],
                    linewidth=1.5,
                    zorder=2,
                    alpha=0.7,
                )

    ax.set_title(f"GT Motion Heatmap + Keypoint Trajectory (Epoch {epoch_num})")
    ax.axis("off")
    fig.tight_layout()

    # ... (Saving and logging logic is the same) ...
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        filename = os.path.join(
            save_path, f"epoch_{epoch_num}_batch_{iter_num}_full_debug.png"
        )
        fig.savefig(filename)
        print(f"DEBUG: Successfully saved full debug visualization to {filename}")
    if log_to_wandb and wandb.run:
        wandb.log({f"train/full_debug_epoch_{epoch_num}": wandb.Image(fig)})
        print("DEBUG: Successfully logged full debug sample to W&B.")

    plt.close(fig)


# def visualize_and_log_batch(
#     data_dict,
#     epoch_num=1,
#     iter_num=1,
#     mean=None,
#     std=None,
#     save_path="./",
#     log_to_wandb=False,
# ):
#     """
#     Selects a sample from the batch, overlays keypoints on the image,
#     and optionally logs/saves it.
#     """
#     # --- 1. Select a sample from the batch ---
#     # `data_dict['inputs']` is a list: [imgs_batch, keypoints_batch]
#     # `imgs_batch` has a shape of (B, C, T, H, W)
#     imgs_batch = data_dict["inputs"][0].cpu()

#     # `keypoints_batch` has a shape of (B, T, V, C) after your transforms
#     keypoints_batch = data_dict["keypoint"][0].cpu()

#     # Select the first sample from the batch for visualization
#     single_img_clip = imgs_batch[0]  # Shape: (C, T, H, W)

#     # --- 2. Select the middle frame to visualize ---
#     middle_frame_idx = 0
#     img_frame = single_img_clip[:, middle_frame_idx, :, :]  # Shape: (C, H, W)
#     keypoints_for_frame = keypoints_batch[:, middle_frame_idx, :, :].squeeze(
#         -1
#     )  # Shape: (C, V)
#     # --- 3. Prepare image and keypoints for plotting ---
#     # Un-normalize the image using the correct mean and std
#     img_unnorm = unnormalize_image(img_frame)
#     # Convert to NumPy, clip to be safe, and set the correct INTEGER data type
#     img_np = img_unnorm.permute(1, 2, 0).numpy().clip(0, 255).astype(np.uint8)

#     # Extract the now-transformed X and Y coordinates for plotting
#     x_coords = keypoints_for_frame[0, :]  # X coordinates for all 32 joints
#     y_coords = keypoints_for_frame[1, :]  # Y coordinates for all 32 joints

#     # --- 4. Create and save the plot ---
#     fig, ax = plt.subplots(1, figsize=(10, 10), dpi=100)
#     ax.imshow(img_np)
#     ax.scatter(
#         x_coords, y_coords, s=25, c="cyan", marker="o", edgecolors="black", zorder=2
#     )

#     # Define connections for the Azure Kinect skeleton (32 joints) to draw lines
#     skeleton_connections = [
#         (2, 3),
#         (3, 26),
#         (26, 27),
#         (2, 11),
#         (11, 12),
#         (12, 13),
#         (13, 14),
#         (14, 15),
#         (15, 16),
#         (14, 17),
#         (2, 4),
#         (4, 5),
#         (5, 6),
#         (6, 7),
#         (7, 8),
#         (8, 9),
#         (7, 10),
#         (0, 1),
#         (1, 2),
#         (0, 22),
#         (22, 23),
#         (23, 24),
#         (24, 25),
#         (0, 18),
#         (18, 19),
#         (19, 20),
#         (20, 21),
#     ]

#     # Draw the skeleton lines
#     for start_idx, end_idx in skeleton_connections:
#         if start_idx < len(x_coords) and end_idx < len(x_coords):
#             ax.plot(
#                 [x_coords[start_idx], x_coords[end_idx]],
#                 [y_coords[start_idx], y_coords[end_idx]],
#                 "c-",
#                 linewidth=2,
#                 zorder=1,
#             )

#     ax.axis("off")
#     fig.tight_layout()

#     # Save the plot locally
#     if save_path:
#         os.makedirs(save_path, exist_ok=True)
#         filename = os.path.join(
#             save_path, f"epoch_{epoch_num}_batch_{iter_num}_sample.png"
#         )
#         fig.savefig(filename)
#         print(f"DEBUG: Successfully saved data visualization to {filename}")

#     # Log to W&B if enabled
#     if log_to_wandb and wandb.run:
#         wandb.log({f"train/data_sample_epoch_{epoch_num}": wandb.Image(fig)})
#         print("DEBUG: Successfully logged data visualization sample to W&B.")

#     # Close the figure to free up memory
#     plt.close(fig)


# https://mmpretrain.readthedocs.io/en/latest/get_started.html
class KeypointMSELoss(nn.Module):
    """MSE loss for heatmaps.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
            Defaults to ``False``
        skip_empty_channel (bool): If ``True``, heatmap channels with no
            non-zero value (which means no visible ground-truth keypoint
            in the image) will not be used to calculate the loss. Defaults to
            ``False``
        loss_weight (float): Weight of the loss. Defaults to 1.0
    """

    def __init__(
        self,
        use_target_weight: bool = False,
        skip_empty_channel: bool = False,
        loss_weight: float = 1.0,
    ):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.skip_empty_channel = skip_empty_channel
        self.loss_weight = loss_weight

    def forward(
        self,
        output: Tensor,
        target: Tensor,
        target_weights: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward function of loss.

        Note:
            - batch_size: B
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (Tensor): The output heatmaps with shape [B, K, H, W]
            target (Tensor): The target heatmaps with shape [B, K, H, W]
            target_weights (Tensor, optional): The target weights of differet
                keypoints, with shape [B, K] (keypoint-wise) or
                [B, K, H, W] (pixel-wise).
            mask (Tensor, optional): The masks of valid heatmap pixels in
                shape [B, K, H, W] or [B, 1, H, W]. If ``None``, no mask will
                be applied. Defaults to ``None``

        Returns:
            Tensor: The calculated loss.
        """

        _mask = self._get_mask(target, target_weights, mask)
        if _mask is None:
            loss = F.mse_loss(output, target)
        else:
            _loss = F.mse_loss(output, target, reduction="none")
            loss = (_loss * _mask).mean()

        return loss * self.loss_weight

    def _get_mask(
        self, target: Tensor, target_weights: Optional[Tensor], mask: Optional[Tensor]
    ) -> Optional[Tensor]:
        """Generate the heatmap mask w.r.t. the given mask, target weight and
        `skip_empty_channel` setting.

        Returns:
            Tensor: The mask in shape (B, K, *) or ``None`` if no mask is
            needed.
        """
        # Given spatial mask
        if mask is not None:
            # check mask has matching type with target
            assert mask.ndim == target.ndim and all(
                d_m == d_t or d_m == 1 for d_m, d_t in zip(mask.shape, target.shape)
            ), (
                f"mask and target have mismatched shapes {mask.shape} v.s."
                f"{target.shape}"
            )

        # Mask by target weights (keypoint-wise mask)
        if target_weights is not None:
            # check target weight has matching shape with target
            assert (
                target_weights.ndim in (2, 4)
                and target_weights.shape == target.shape[: target_weights.ndim]
            ), (
                "target_weights and target have mismatched shapes "
                f"{target_weights.shape} v.s. {target.shape}"
            )

            ndim_pad = target.ndim - target_weights.ndim
            _mask = target_weights.view(target_weights.shape + (1,) * ndim_pad)

            if mask is None:
                mask = _mask
            else:
                mask = mask * _mask

        # Mask by ``skip_empty_channel``
        if self.skip_empty_channel:
            _mask = (target != 0).flatten(2).any(dim=2)
            ndim_pad = target.ndim - _mask.ndim
            _mask = _mask.view(_mask.shape + (1,) * ndim_pad)

            if mask is None:
                mask = _mask
            else:
                mask = mask * _mask

        return mask


def generate_gt_heatmaps(
    heatmap_generator, keypoints_2d, device="cuda", chunk_size=16, target_chunks=None
):
    """
    Generates ground-truth heatmaps from keypoint coordinates and scores.

    Args:
        heatmap_generator (UDPHeatmap): The mmpose heatmap codec.
        keypoints_2d (torch.Tensor): Augmented keypoint coordinates. Shape (B, T, M, V, 3).
        target_chunks (int, optional): The target number of chunks to align with the model output.

    Returns:
        torch.Tensor: The generated ground-truth heatmaps. Shape (B, T, V, H, W).
    """

    target_t = 0
    if target_chunks is not None:
        target_t = target_chunks * chunk_size

    if isinstance(keypoints_2d, list):
        # We need to pad/truncate tensors to target_t (if specified) or max_t
        # and pad M dimension to max_m
        if target_t > 0:
            # Use exact target_t when target_chunks is specified
            max_t = target_t
        else:
            # Find max T across all samples
            max_t = 0
            for x in keypoints_2d:
                max_t = max(max_t, x.shape[0])

        max_m = 0
        for x in keypoints_2d:
            # x shape: (T, M, V, 3)
            max_m = max(max_m, x.shape[1])

        padded_list = []
        for x in keypoints_2d:
            t, m, v, c = x.shape

            # Truncate T dimension if it exceeds max_t
            if t > max_t:
                temp = x[:max_t]
                t = max_t
            else:
                temp = x

            pad_t = max_t - t
            pad_m = max_m - m

            if pad_m > 0:
                # Pad M dimension (dim 1)
                padding_m = torch.zeros(
                    (t, pad_m, v, c), dtype=x.dtype, device=x.device
                )
                temp = torch.cat([temp, padding_m], dim=1)

            if pad_t > 0:
                # Pad T dimension (dim 0).
                # Note: temp might have updated M dimension now (t, max_m, v, c)
                # We need padding of shape (pad_t, max_m, v, c)
                padding_t = torch.zeros(
                    (pad_t, max_m, v, c), dtype=x.dtype, device=x.device
                )
                temp = torch.cat([temp, padding_t], dim=0)

            padded_list.append(temp)

        keypoints_2d = torch.stack(padded_list, dim=0)

    elif isinstance(keypoints_2d, torch.Tensor) and target_t > 0:
        # keypoints_2d is (B, T, M, V, 3)
        curr_t = keypoints_2d.shape[1]
        if curr_t > target_t:
            # Truncate T dimension to target_t
            keypoints_2d = keypoints_2d[:, :target_t]
        elif curr_t < target_t:
            pad_t = target_t - curr_t
            padding = torch.zeros(
                (keypoints_2d.shape[0], pad_t, *keypoints_2d.shape[2:]),
                dtype=keypoints_2d.dtype,
                device=keypoints_2d.device,
            )
            keypoints_2d = torch.cat([keypoints_2d, padding], dim=1)

    # Input shape: (B, T, M, V, 3) where last dim is (x, y, score)
    # Take first 2 channels (x, y) -> (B, T, M, V, 2)
    # And permute to (B, M, T, V, 2) which is what logic below expects
    keypoints_2d = keypoints_2d[..., :2].permute(0, 2, 1, 3, 4)

    # From (B, T, V, M) -> (B, M, T, V)
    batch_size, num_persons, num_frames, num_joints, _ = keypoints_2d.shape
    # Move tensors to CPU and convert to NumPy for the codec
    keypoints_np = keypoints_2d.cpu().numpy()

    # The codec expects input shape (num_instances, num_joints, 2/3)
    # We will process frame by frame.
    # Move to CPU and convert to NumPy for the codec
    keypoints_np = keypoints_2d.cpu().numpy()
    if num_frames % chunk_size != 0:
        # Trim the last few frames if necessary to make it divisible
        print(
            f"Warning: Trimming {num_frames % chunk_size} frames for motion heatmap generation."
        )
        num_frames = (num_frames // chunk_size) * chunk_size
        keypoints_2d = keypoints_2d[:, :, :num_frames, :, :]

    heatmaps_batch = []
    for b in range(batch_size):
        # We iterate all people and sum their heatmaps?
        # The original code only took person 0: `person_coords = keypoints_np[b, 0]`
        # We should sum over all people (M dimension)

        # Dimensions: (M, T, V, 2)
        people_coords = keypoints_np[b]

        heatmaps_frames = []
        for t in range(num_frames):
            frame_coords = people_coords[:, t, :, :]  # Shape: (M, V, 2)

            accumulated_heatmap = None

            for m in range(frame_coords.shape[0]):
                coords = frame_coords[m : m + 1]  # Shape (1, V, 2)

                # Check if coordinates are valid (not all zeros) to save compute
                # Also to avoid adding gaussian noise at (0,0) for padding
                if np.max(coords) <= 1e-3:
                    continue

                # Encode single person
                hm = heatmap_generator.encode(coords)["heatmaps"]  # Shape (K, H, W)

                if accumulated_heatmap is None:
                    accumulated_heatmap = hm
                else:
                    # Merge using element-wise maximum
                    accumulated_heatmap = np.maximum(accumulated_heatmap, hm)

            if accumulated_heatmap is None:
                # If no valid people, return zero heatmap with correct shape
                W, H = heatmap_generator.heatmap_size
                K = frame_coords.shape[1]
                accumulated_heatmap = np.zeros((K, H, W), dtype=np.float32)

            heatmaps_frames.append(accumulated_heatmap)

        heatmaps_batch.append(np.stack(heatmaps_frames, axis=0))

    per_frame_heatmaps = torch.from_numpy(np.stack(heatmaps_batch, axis=0)).to(device)
    # Shape of per_frame_heatmaps is now (B, T, V, H, W)

    # 3. Reshape and sum to create motion heatmaps
    B, T, V, H, W = per_frame_heatmaps.shape
    num_chunks = T // chunk_size

    # Reshape from (B, T, V, H, W) -> (B, num_chunks, chunk_size, V, H, W)
    chunked_heatmaps = per_frame_heatmaps.view(B, num_chunks, chunk_size, V, H, W)

    # Sum along the chunk_size dimension to aggregate motion
    motion_heatmaps = torch.sum(chunked_heatmaps, dim=2)

    # The final shape is (B, num_chunks, V, H, W), e.g., (1, 48, 32, 40, 40)
    return motion_heatmaps


def train_one_epoch(
    train_loader,
    model,
    optimizer,
    scheduler,
    curr_epoch,
    logger,
    rank,
    model_ema=None,
    clip_grad_l2norm=-1,
    logging_interval=200,
    scaler=None,
    use_amp=False,
):
    """Training the model for one epoch"""

    logger.info("[Train]: Epoch {:d} started".format(curr_epoch))
    losses_tracker = {}
    num_iters = len(train_loader)
    keypoint_loss_fn = KeypointMSELoss(loss_weight=2.0)
    heatmap_generator = UDPHeatmap(
        input_size=(160, 160), heatmap_size=(56, 56), sigma=1.5
    )
    model.train()
    for iter_idx, data_dict in enumerate(train_loader):
        optimizer.zero_grad()
        # print(data_dict['metas'])
        # current learning rate
        curr_backbone_lr = None
        if hasattr(model.module, "backbone"):  # if backbone exists
            if model.module.backbone.freeze_backbone == False:  # not frozen
                curr_backbone_lr = scheduler.get_last_lr()[0]
        curr_det_lr = scheduler.get_last_lr()[-1]

        # forward pass
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_amp):
            losses = model(**data_dict, return_loss=True)
            heatmap_pred = model.module.backbone.get_heatmap()

        if heatmap_pred is not None and "keypoint" in data_dict:
            kp_data = data_dict["keypoint"]
            B = len(kp_data) if isinstance(kp_data, list) else kp_data.shape[0]
            total_chunks = heatmap_pred.shape[0]
            target_chunks = total_chunks // B
            gt_heatmaps = generate_gt_heatmaps(
                heatmap_generator,
                data_dict["keypoint"],
                device=heatmap_pred.device,
                target_chunks=target_chunks,
            )
            gt_heatmaps = gt_heatmaps.to(heatmap_pred.dtype)
            # merge first two dimensions
            gt_heatmaps = gt_heatmaps.view(
                gt_heatmaps.shape[0] * gt_heatmaps.shape[1], *gt_heatmaps.shape[2:]
            )

            loss_heatmap = keypoint_loss_fn(heatmap_pred, gt_heatmaps)
            losses["loss_heatmap"] = loss_heatmap
            losses["cost"] += loss_heatmap
        # visualize_and_log_batch(data_dict, gt_heatmaps)
        # compute the gradients
        if scaler is not None:
            scaler.scale(losses["cost"]).backward()
        else:
            losses["cost"].backward()

        # gradient clipping (to stabilize training if necessary)
        if clip_grad_l2norm > 0.0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_l2norm)

        # update parameters
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        # update scheduler
        scheduler.step()
        # update ema
        if model_ema is not None:
            model_ema.update(model.module)

        # track all losses
        losses = reduce_loss(losses)  # only for log
        for key, value in losses.items():
            if key not in losses_tracker:
                losses_tracker[key] = AverageMeter()
            losses_tracker[key].update(value.item())
            # if loss is nan throw error
            if torch.isnan(value).all():
                print(f"NaN detected in loss component: {key}")
                if key == "loss_heatmap":
                    if torch.isnan(heatmap_pred).any():
                        print("heatmap_pred contains NaNs")
                    if torch.isnan(gt_heatmaps).any():
                        print("gt_heatmaps contains NaNs")
                raise ValueError(f"Loss {key} is NaN")

        # printing each logging_interval
        if ((iter_idx != 0) and (iter_idx % logging_interval) == 0) or (
            (iter_idx + 1) == num_iters
        ):
            # print to terminal
            block1 = "[Train]: [{:03d}][{:05d}/{:05d}]".format(
                curr_epoch, iter_idx, num_iters - 1
            )
            block2 = "Loss={:.4f}".format(losses_tracker["cost"].avg)
            block3 = [
                "{:s}={:.4f}".format(key, value.avg)
                for key, value in losses_tracker.items()
                if key != "cost"
            ]
            block4 = "lr_det={:.1e}".format(curr_det_lr)
            if curr_backbone_lr is not None:
                block4 = "lr_backbone={:.1e}".format(curr_backbone_lr) + "  " + block4
            block5 = "mem={:.0f}MB".format(
                torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
            )
            logger.info("  ".join([block1, block2, "  ".join(block3), block4, block5]))

            # wandb logging
            if rank == 0 and wandb.run:
                log_dict = {
                    f"train/{key}": value.avg for key, value in losses_tracker.items()
                }
                log_dict["epoch"] = curr_epoch
                log_dict["step"] = curr_epoch * num_iters + iter_idx
                log_dict["lr/det"] = curr_det_lr
                if curr_backbone_lr is not None:
                    log_dict["lr/backbone"] = curr_backbone_lr
                wandb.log(log_dict)


def val_one_epoch(
    val_loader,
    model,
    logger,
    rank,
    curr_epoch,
    model_ema=None,
    use_amp=False,
):
    """Validating the model for one epoch: compute the loss"""

    # load the ema dict for evaluation
    if model_ema != None:
        current_dict = copy.deepcopy(model.module.state_dict())
        model.module.load_state_dict(model_ema.module.state_dict())

    logger.info("[Val]: Epoch {:d} Loss".format(curr_epoch))
    losses_tracker = {}

    model.eval()
    for data_dict in tqdm.tqdm(val_loader, disable=(rank != 0)):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_amp):
            with torch.no_grad():
                losses = model(**data_dict, return_loss=True)

        # track all losses
        losses = reduce_loss(losses)  # only for log
        for key, value in losses.items():
            if key not in losses_tracker:
                losses_tracker[key] = AverageMeter()
            losses_tracker[key].update(value.item())

    # print to terminal
    block1 = "[Val]: [{:03d}]".format(curr_epoch)
    block2 = "Loss={:.4f}".format(losses_tracker["cost"].avg)
    block3 = [
        "{:s}={:.4f}".format(key, value.avg)
        for key, value in losses_tracker.items()
        if key != "cost"
    ]
    logger.info("  ".join([block1, block2, "  ".join(block3)]))

    # wandb logging
    if rank == 0 and wandb.run:
        log_dict = {f"val/{key}": value.avg for key, value in losses_tracker.items()}
        wandb.log(log_dict)

    # load back the normal model dict
    if model_ema != None:
        model.module.load_state_dict(current_dict)
    return losses_tracker["cost"].avg
