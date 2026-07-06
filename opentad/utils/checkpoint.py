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


def save_best_checkpoint(model, model_ema, epoch, work_dir=None):
    save_dir = os.path.join(work_dir, "checkpoint")

    save_states = {"epoch": epoch, "state_dict": model.state_dict()}

    if model_ema != None:
        save_states.update({"state_dict_ema": model_ema.module.state_dict()})

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    checkpoint_path = os.path.join(save_dir, f"best.pth")
    torch.save(save_states, checkpoint_path)
