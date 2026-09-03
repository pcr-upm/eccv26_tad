#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import os
import sys
sys.path.append(os.getcwd())
import cv2
import torch
import wandb
import random
import string
from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks
from opentad.datasets import build_dataset, build_dataloader
from opentad.cores import train_one_epoch, val_one_epoch, eval_one_epoch, build_optimizer, build_scheduler
from opentad.utils import update_workdir, override_dataset_paths,create_folder, save_config, setup_logger, ModelEma, save_checkpoint, save_best_checkpoint
from images_framework.src.constants import Modes
from images_framework.src.composite import Composite
from src.eccv26_tad import ECCV26TAD


def parse_options():
    """
    Parse options from command line.
    """
    import argparse
    from mmengine.config import DictAction
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, 
                        help="resume from a checkpoint.")
    parser.add_argument("--wandb", action="store_true", 
                        help="whether to use wandb for logging.")
    parser.add_argument("--project", type=str, default="thumos", 
                        help="wandb project name.")
    parser.add_argument("--ann-file", type=str, default=None,
                        help="override the annotation file path for all dataset splits and evaluation.")
    parser.add_argument("--class-map", type=str, default=None,
                        help="override the class map / category index file path for all dataset splits.")
    parser.add_argument("--data-root", type=str, default=None,
                        help="override the raw video / feature data root path for all dataset splits.")
    parser.add_argument("--block-list", type=str, default=None,
                        help="override the block list file path for all dataset splits.")
    parser.add_argument("--external-cls-path", type=str, default=None,
                        help="override the external classifier (post_processing.external_cls) path.")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, default=None,
                        help="override settings in the config (e.g., --cfg-options model.backbone.backbone.n_landmarks=32)")
    args, unknown = parser.parse_known_args()
    print(parser.format_usage())
    resume = args.resume
    use_wandb = args.wandb
    project = args.project
    ann_file = args.ann_file
    class_map = args.class_map
    data_root = args.data_root
    block_list = args.block_list
    external_cls_path = args.external_cls_path
    cfg_options = args.cfg_options or {}
    return unknown, resume, use_wandb, project, ann_file, class_map, data_root, block_list, external_cls_path, cfg_options


def main():
    """
    SV-TAD: Native Sparse Convolutions for Efficient Temporal Action Detection train database script.
    """
    print('OpenCV ' + cv2.__version__)
    unknown, resume, use_wandb, project, ann_file, class_map, data_root, block_list, external_cls_path, cfg_options = parse_options()

    # Load vision components
    composite = Composite()
    sr = ECCV26TAD('')
    composite.add(sr)
    composite.parse_options(unknown)
    composite.load(Modes.TRAIN)
    if ann_file or class_map or data_root or block_list or external_cls_path:
        sr.cfg = override_dataset_paths(sr.cfg, ann_file=ann_file, class_map=class_map, data_path=data_root, block_list=block_list, external_cls_path=external_cls_path)
    if cfg_options:
        sr.cfg.merge_from_dict(cfg_options)
    # Generate random three letter run id
    run_id = "".join([random.choice(string.ascii_lowercase) for _ in range(3)])
    sr.cfg.work_dir = os.path.join(sr.cfg.work_dir, run_id)
    id = run_id

    sr.cfg = update_workdir(sr.cfg, id, sr.world_size)
    if sr.rank == 0:
        create_folder(sr.cfg.work_dir)
        save_config('configs/vitsparse/thumos/e2e_thumos_videomae_b_768x1_160_sparse_adapter.py', sr.cfg.work_dir)

    # setup logger
    logger = setup_logger("Train", save_dir=sr.cfg.work_dir, distributed_rank=sr.rank)
    logger.info(f"Using torch version: {torch.__version__}, CUDA version: {torch.version.cuda}")
    logger.info(f"Config: \n{sr.cfg.pretty_text}")

    # setup wandb
    if use_wandb and sr.rank == 0:
        run_name = os.path.basename('configs/vitsparse/thumos/e2e_thumos_videomae_b_768x1_160_sparse_adapter.py').split(".")[0] + f"_id{id}"
        wandb.init(project=project, name=run_name, config=sr.cfg.to_dict())

    # build dataset
    train_dataset = build_dataset(sr.cfg.dataset.train, default_args=dict(logger=logger))
    train_loader = build_dataloader(train_dataset, rank=sr.rank, world_size=sr.world_size, shuffle=True, drop_last=True, **sr.cfg.solver.train)

    val_dataset = build_dataset(sr.cfg.dataset.val, default_args=dict(logger=logger))
    val_loader = build_dataloader(val_dataset, rank=sr.rank, world_size=sr.world_size, shuffle=False, drop_last=False, **sr.cfg.solver.val)

    test_dataset = build_dataset(sr.cfg.dataset.test, default_args=dict(logger=logger))
    test_loader = build_dataloader(test_dataset, rank=sr.rank, world_size=sr.world_size, shuffle=False, drop_last=False, **sr.cfg.solver.test)

    # FP16 compression
    use_fp16_compress = getattr(sr.cfg.solver, "fp16_compress", False)
    if use_fp16_compress:
        logger.info("Using FP16 compression ...")
        sr.model.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)

    # Model EMA
    use_ema = getattr(sr.cfg.solver, "ema", False)
    if use_ema:
        logger.info("Using Model EMA...")
        model_ema = ModelEma(sr.model.module)
    else:
        model_ema = None

    # AMP: automatic mixed precision
    use_amp = getattr(sr.cfg.solver, "amp", False)
    if use_amp:
        logger.info("Using Automatic Mixed Precision...")
        # GradScaler is only needed for float16 AMP, not bfloat16.
        # bfloat16 has the same dynamic range as float32, so loss scaling
        # is unnecessary and can introduce NaN from rounding errors.
        scaler = None
    else:
        scaler = None

    # build optimizer and scheduler
    optimizer = build_optimizer(sr.cfg.optimizer, sr.model, logger)
    scheduler, max_epoch = build_scheduler(sr.cfg.scheduler, optimizer, len(train_loader))

    # override the max_epoch
    max_epoch = sr.cfg.workflow.get("end_epoch", max_epoch)

    # --- Early Stopping Parameters Initialization ---
    val_loss_best = 1e6  # Initialize best validation loss
    epochs_no_improve = 0  # Counter for epochs without improvement
    # Get early stopping patience and min_delta from config, with defaults
    early_stopping_patience = sr.cfg.workflow.get("early_stopping_patience", -1)  # -1 means disabled
    early_stopping_min_delta = sr.cfg.workflow.get("early_stopping_min_delta", 0.0)  # 0.0 means any improvement counts
    # measure gflops

    # resume: reset epoch, load checkpoint / best rmse
    if resume is not None:
        logger.info("Resume training from: {}".format(resume))
        device = f"cuda:{sr.local_rank}"
        checkpoint = torch.load(resume, map_location=device)
        resume_epoch = checkpoint["epoch"]
        logger.info("Resume epoch is {}".format(resume_epoch))
        sr.model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if model_ema is not None:
            model_ema.module.load_state_dict(checkpoint["state_dict_ema"])

        # If resuming, ideally load previous best_val_loss and epochs_no_improve
        # For simplicity here, we re-initialize them, meaning early stopping
        # will start fresh from the resume point. The loaded model state
        # will still reflect the best from prior training.
        # val_loss_best = checkpoint.get("val_loss_best", val_loss_best)
        # epochs_no_improve = checkpoint.get("epochs_no_improve", epochs_no_improve)

        del checkpoint  #  save memory if the model is very large such as ViT-g
        torch.cuda.empty_cache()
    else:
        resume_epoch = -1

    # train the detector
    logger.info("Training Starts...\n")
    val_start_epoch = sr.cfg.workflow.get("val_start_epoch", 0)  # Already defined, just keeping it here for clarity
    for epoch in range(resume_epoch + 1, max_epoch):
        train_loader.sampler.set_epoch(epoch)
        # train for one epoch
        train_one_epoch(train_loader, sr.model, optimizer, scheduler, epoch, logger, rank=sr.rank, model_ema=model_ema, clip_grad_l2norm=sr.cfg.solver.clip_grad_norm, logging_interval=sr.cfg.workflow.logging_interval, scaler=scaler, use_amp=use_amp)

        # save checkpoint
        if (epoch == max_epoch - 1) or ((epoch + 1) % sr.cfg.workflow.checkpoint_interval == 0):
            if sr.rank == 0:
                save_checkpoint(sr.model, model_ema, optimizer, scheduler, epoch, work_dir=sr.cfg.work_dir)

        # val for one epoch and early stopping check
        # None = no val loss computed this epoch, so eval falls back to its interval
        val_loss_improved = None
        if epoch >= val_start_epoch:
            if (sr.cfg.workflow.val_loss_interval > 0) and ((epoch + 1) % sr.cfg.workflow.val_loss_interval == 0):
                val_loss = val_one_epoch(val_loader, sr.model, logger, sr.rank, epoch, model_ema=model_ema, use_amp=use_amp)

                # --- Early Stopping and Best Checkpoint Saving Logic ---
                # Only activate early stopping if patience is set (i.e., not -1)
                if early_stopping_patience > 0:
                    # Check for significant improvement
                    if val_loss < val_loss_best - early_stopping_min_delta:
                        logger.info(f"Validation loss improved from {val_loss_best:.4f} to {val_loss:.4f}. Resetting early stopping counter.")
                        val_loss_best = val_loss  # Update the best loss
                        epochs_no_improve = 0  # Reset counter
                        val_loss_improved = True
                        if sr.rank == 0:
                            # Save the best model only when significant improvement is observed
                            save_best_checkpoint(sr.model, model_ema, epoch, work_dir=sr.cfg.work_dir)
                    else:
                        # No significant improvement, increment counter
                        epochs_no_improve += 1
                        val_loss_improved = False
                        logger.info(f"Validation loss did not improve significantly. Early stopping counter: {epochs_no_improve}/{early_stopping_patience}.")

                    # Check if early stopping condition is met
                    if epochs_no_improve >= early_stopping_patience:
                        logger.info(f"Early stopping triggered after {epochs_no_improve} epochs without significant improvement. Training will stop.")
                        break  # Exit the training loop
                else:
                    # If early stopping is not enabled, use the original logic for saving best checkpoint
                    if val_loss < val_loss_best:
                        logger.info(f"New best epoch {epoch} based on val_loss: {val_loss:.4f}.")
                        val_loss_best = val_loss
                        val_loss_improved = True
                        if sr.rank == 0:
                            save_best_checkpoint(sr.model, model_ema, epoch, work_dir=sr.cfg.work_dir)
                    else:
                        val_loss_improved = False

        # eval for one epoch (evaluation metrics, not for early stopping loss)
        # skipped when the val loss did not improve this epoch
        if epoch >= val_start_epoch and val_loss_improved is not False:
            if (sr.cfg.workflow.val_eval_interval > 0) and ((epoch + 1) % sr.cfg.workflow.val_eval_interval == 0):
                eval_one_epoch(test_loader, sr.model, sr.cfg, logger, sr.rank, model_ema=model_ema, use_amp=use_amp, world_size=sr.world_size, not_eval=True, training=True)
        elif val_loss_improved is False and (sr.cfg.workflow.val_eval_interval > 0) and ((epoch + 1) % sr.cfg.workflow.val_eval_interval == 0):
            logger.info(f"Skipping evaluation at epoch {epoch}: val_loss did not improve.")
    logger.info("Training Over...\n")

    # Load best model if exists
    best_checkpoint_path = os.path.join(sr.cfg.work_dir, "checkpoint", "best.pth")
    if os.path.exists(best_checkpoint_path):
        logger.info(f"Loading best checkpoint from {best_checkpoint_path} for final evaluation...")
        checkpoint = torch.load(best_checkpoint_path, map_location=f"cuda:{sr.local_rank}")
        sr.model.load_state_dict(checkpoint["state_dict"])
        if model_ema is not None and "state_dict_ema" in checkpoint:
            model_ema.module.load_state_dict(checkpoint["state_dict_ema"])
    else:
        logger.info("Best checkpoint not found. Using the last model for final evaluation.")
    eval_one_epoch(test_loader, sr.model, sr.cfg, logger, sr.rank, model_ema=model_ema, use_amp=use_amp, world_size=sr.world_size, not_eval=True, training=True)
    if use_wandb and sr.rank == 0:
        wandb.finish()


if __name__ == "__main__":
    main()
