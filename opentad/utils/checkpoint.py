import os
import torch
import glob
import re


def save_checkpoint(
    model, model_ema, optimizer, scheduler, epoch, work_dir=None, max_keep=15
):
    save_dir = os.path.join(work_dir, "checkpoint")

    save_states = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }

    if model_ema != None:
        save_states.update({"state_dict_ema": model_ema.module.state_dict()})

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    checkpoint_path = os.path.join(save_dir, f"epoch_{epoch}.pth")
    torch.save(save_states, checkpoint_path)

    if max_keep > 0:
        checkpoints = glob.glob(os.path.join(save_dir, "epoch_*.pth"))

        def get_epoch_from_path(path):
            match = re.search(r"epoch_(\d+)\.pth", os.path.basename(path))
            return int(match.group(1)) if match else -1

        checkpoints.sort(key=get_epoch_from_path)

        if len(checkpoints) > max_keep:
            for ckpt in checkpoints[:-max_keep]:
                os.remove(ckpt)


def remap_legacy_sparse_conv_weights(state_dict, model_state_dict):
    """Older checkpoints stored ``SparseConv2d.weight`` as ``(C_in, 9, C_out)``.
    The current module stores the pre-permuted layout ``(9, C_in, C_out)``
    (see opentad/models/bricks/sparse_conv_layer.py). Permute any mismatched
    sparse_conv weights so such checkpoints can still be loaded.
    """
    for key, param in state_dict.items():
        if key.endswith("sparse_conv.weight") and key in model_state_dict:
            target_shape = model_state_dict[key].shape
            if param.shape != target_shape and param.dim() == 3:
                permuted = param.permute(1, 0, 2).contiguous()
                if permuted.shape == target_shape:
                    state_dict[key] = permuted
    return state_dict


def save_best_checkpoint(model, model_ema, epoch, work_dir=None):
    save_dir = os.path.join(work_dir, "checkpoint")

    save_states = {"epoch": epoch, "state_dict": model.state_dict()}

    if model_ema != None:
        save_states.update({"state_dict_ema": model_ema.module.state_dict()})

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    checkpoint_path = os.path.join(save_dir, f"best.pth")
    torch.save(save_states, checkpoint_path)
