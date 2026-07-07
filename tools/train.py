import os
import sys

sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)

import argparse
import torch
import torch.distributed as dist
import wandb
from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks
from torch.nn.parallel import DistributedDataParallel
from mmengine.config import Config, DictAction
from opentad.models import build_detector
from opentad.datasets import build_dataset, build_dataloader
from opentad.cores import (
    train_one_epoch,
    val_one_epoch,
    eval_one_epoch,
    build_optimizer,
    build_scheduler,
)
from opentad.utils import (
    set_seed,
    update_workdir,
    override_dataset_paths,
    create_folder,
    save_config,
    setup_logger,
    ModelEma,
    save_checkpoint,
    save_best_checkpoint,
)
import random
import string


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Temporal Action Detector")
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--id", type=int, default=0, help="repeat experiment id")
    parser.add_argument(
        "--resume", type=str, default=None, help="resume from a checkpoint"
    )
    parser.add_argument(
        "--not_eval", action="store_true", help="whether not to eval, only do inference"
    )
    parser.add_argument(
        "--disable_deterministic",
        action="store_true",
        help="disable deterministic for faster speed",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="whether to use wandb for logging"
    )
    parser.add_argument(
        "--project", type=str, default="thumos", help="wandb project name"
    )
    parser.add_argument(
        "--cfg-options", nargs="+", action=DictAction, help="override settings"
    )
    parser.add_argument(
        "--ann-file", type=str, default=None,
        help="override the annotation file path for all dataset splits and evaluation",
    )
    parser.add_argument(
        "--class-map", type=str, default=None,
        help="override the class map / category index file path for all dataset splits",
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="override the raw video / feature data root path for all dataset splits",
    )
    parser.add_argument(
        "--block-list", type=str, default=None,
        help="override the block list file path for all dataset splits",
    )
    parser.add_argument(
        "--external-cls-path", type=str, default=None,
        help="override the external classifier (post_processing.external_cls) path",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    # generate random three letter run id
    run_id = "".join([random.choice(string.ascii_lowercase) for _ in range(3)])
    cfg.work_dir = os.path.join(cfg.work_dir, run_id)
    args.id = run_id
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg = override_dataset_paths(
        cfg,
        ann_file=args.ann_file,
        class_map=args.class_map,
        data_path=args.data_root,
        block_list=args.block_list,
        external_cls_path=args.external_cls_path,
    )

    # DDP init
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.rank = int(os.environ["RANK"])
    print(
        f"Distributed init (rank {args.rank}/{args.world_size}, local rank {args.local_rank})"
    )
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group("nccl", rank=args.rank, world_size=args.world_size)

    # set random seed, create work_dir, and save config
    set_seed(args.seed, True)
    cfg = update_workdir(cfg, args.id, args.world_size)
    if args.rank == 0:
        create_folder(cfg.work_dir)
        save_config(args.config, cfg.work_dir)

    # setup logger
    logger = setup_logger("Train", save_dir=cfg.work_dir, distributed_rank=args.rank)
    logger.info(
        f"Using torch version: {torch.__version__}, CUDA version: {torch.version.cuda}"
    )
    logger.info(f"Config: \n{cfg.pretty_text}")

    # setup wandb
    if args.wandb and args.rank == 0:
        run_name = os.path.basename(args.config).split(".")[0] + f"_id{args.id}"
        wandb.init(project=args.project, name=run_name, config=cfg.to_dict())

    # build dataset
    train_dataset = build_dataset(cfg.dataset.train, default_args=dict(logger=logger))
    train_loader = build_dataloader(
        train_dataset,
        rank=args.rank,
        world_size=args.world_size,
        shuffle=True,
        drop_last=True,
        **cfg.solver.train,
    )

    val_dataset = build_dataset(cfg.dataset.val, default_args=dict(logger=logger))
    val_loader = build_dataloader(
        val_dataset,
        rank=args.rank,
        world_size=args.world_size,
        shuffle=False,
        drop_last=False,
        **cfg.solver.val,
    )

    test_dataset = build_dataset(cfg.dataset.test, default_args=dict(logger=logger))
    test_loader = build_dataloader(
        test_dataset,
        rank=args.rank,
        world_size=args.world_size,
        shuffle=False,
        drop_last=False,
        **cfg.solver.test,
    )

    # build model
    model = build_detector(cfg.model)

    # DDP
    use_static_graph = getattr(cfg.solver, "static_graph", False)
    model = model.to(args.local_rank)
    model = DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=False if use_static_graph else True,
        static_graph=use_static_graph,  # default is False, should be true when use activation checkpointing in E2E
    )
    logger.info(f"Using DDP with total {args.world_size} GPUS...")

    # FP16 compression
    use_fp16_compress = getattr(cfg.solver, "fp16_compress", False)
    if use_fp16_compress:
        logger.info("Using FP16 compression ...")
        model.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)

    # Model EMA
    use_ema = getattr(cfg.solver, "ema", False)
    if use_ema:
        logger.info("Using Model EMA...")
        model_ema = ModelEma(model.module)
    else:
        model_ema = None

    # AMP: automatic mixed precision
    use_amp = getattr(cfg.solver, "amp", False)
    if use_amp:
        logger.info("Using Automatic Mixed Precision...")
        # GradScaler is only needed for float16 AMP, not bfloat16.
        # bfloat16 has the same dynamic range as float32, so loss scaling
        # is unnecessary and can introduce NaN from rounding errors.
        scaler = None
    else:
        scaler = None

    # build optimizer and scheduler
    optimizer = build_optimizer(cfg.optimizer, model, logger)
    scheduler, max_epoch = build_scheduler(cfg.scheduler, optimizer, len(train_loader))

    # override the max_epoch
    max_epoch = cfg.workflow.get("end_epoch", max_epoch)

    # --- Early Stopping Parameters Initialization ---
    val_loss_best = 1e6  # Initialize best validation loss
    epochs_no_improve = 0  # Counter for epochs without improvement
    # Get early stopping patience and min_delta from config, with defaults
    early_stopping_patience = cfg.workflow.get(
        "early_stopping_patience", -1
    )  # -1 means disabled
    early_stopping_min_delta = cfg.workflow.get(
        "early_stopping_min_delta", 0.0
    )  # 0.0 means any improvement counts
    # measure gflops

    # resume: reset epoch, load checkpoint / best rmse
    if args.resume is not None:
        logger.info("Resume training from: {}".format(args.resume))
        device = f"cuda:{args.local_rank}"
        checkpoint = torch.load(args.resume, map_location=device)
        resume_epoch = checkpoint["epoch"]
        logger.info("Resume epoch is {}".format(resume_epoch))
        model.load_state_dict(checkpoint["state_dict"])
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
    val_start_epoch = cfg.workflow.get(
        "val_start_epoch", 0
    )  # Already defined, just keeping it here for clarity
    for epoch in range(resume_epoch + 1, max_epoch):
        train_loader.sampler.set_epoch(epoch)
        # train for one epoch
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            logger,
            rank=args.rank,
            model_ema=model_ema,
            clip_grad_l2norm=cfg.solver.clip_grad_norm,
            logging_interval=cfg.workflow.logging_interval,
            scaler=scaler,
            use_amp=use_amp,
        )

        # save checkpoint
        if (epoch == max_epoch - 1) or (
            (epoch + 1) % cfg.workflow.checkpoint_interval == 0
        ):
            if args.rank == 0:
                save_checkpoint(
                    model, model_ema, optimizer, scheduler, epoch, work_dir=cfg.work_dir
                )

        # val for one epoch and early stopping check
        if epoch >= val_start_epoch:
            if (cfg.workflow.val_loss_interval > 0) and (
                (epoch + 1) % cfg.workflow.val_loss_interval == 0
            ):
                val_loss = val_one_epoch(
                    val_loader,
                    model,
                    logger,
                    args.rank,
                    epoch,
                    model_ema=model_ema,
                    use_amp=use_amp,
                )

                # --- Early Stopping and Best Checkpoint Saving Logic ---
                # Only activate early stopping if patience is set (i.e., not -1)
                if early_stopping_patience > 0:
                    # Check for significant improvement
                    if val_loss < val_loss_best - early_stopping_min_delta:
                        logger.info(
                            f"Validation loss improved from {val_loss_best:.4f} to {val_loss:.4f}. Resetting early stopping counter."
                        )
                        val_loss_best = val_loss  # Update the best loss
                        epochs_no_improve = 0  # Reset counter
                        if args.rank == 0:
                            # Save the best model only when significant improvement is observed
                            save_best_checkpoint(
                                model, model_ema, epoch, work_dir=cfg.work_dir
                            )
                    else:
                        # No significant improvement, increment counter
                        epochs_no_improve += 1
                        logger.info(
                            f"Validation loss did not improve significantly. Early stopping counter: {epochs_no_improve}/{early_stopping_patience}."
                        )

                    # Check if early stopping condition is met
                    if epochs_no_improve >= early_stopping_patience:
                        logger.info(
                            f"Early stopping triggered after {epochs_no_improve} epochs without significant improvement. Training will stop."
                        )
                        break  # Exit the training loop
                else:
                    # If early stopping is not enabled, use the original logic for saving best checkpoint
                    if val_loss < val_loss_best:
                        logger.info(
                            f"New best epoch {epoch} based on val_loss: {val_loss:.4f}."
                        )
                        val_loss_best = val_loss
                        if args.rank == 0:
                            save_best_checkpoint(
                                model, model_ema, epoch, work_dir=cfg.work_dir
                            )

        # eval for one epoch (evaluation metrics, not for early stopping loss)
        if epoch >= val_start_epoch:
            if (cfg.workflow.val_eval_interval > 0) and (
                (epoch + 1) % cfg.workflow.val_eval_interval == 0
            ):
                eval_one_epoch(
                    test_loader,
                    model,
                    cfg,
                    logger,
                    args.rank,
                    model_ema=model_ema,
                    use_amp=use_amp,
                    world_size=args.world_size,
                    not_eval=args.not_eval,
                    training=True,
                )
    logger.info("Training Over...\n")

    # Load best model if exists
    best_checkpoint_path = os.path.join(cfg.work_dir, "checkpoint", "best.pth")
    if os.path.exists(best_checkpoint_path):
        logger.info(
            f"Loading best checkpoint from {best_checkpoint_path} for final evaluation..."
        )
        checkpoint = torch.load(
            best_checkpoint_path, map_location=f"cuda:{args.local_rank}"
        )
        model.load_state_dict(checkpoint["state_dict"])
        if model_ema is not None and "state_dict_ema" in checkpoint:
            model_ema.module.load_state_dict(checkpoint["state_dict_ema"])
    else:
        logger.info(
            "Best checkpoint not found. Using the last model for final evaluation."
        )
    eval_one_epoch(
        test_loader,
        model,
        cfg,
        logger,
        args.rank,
        model_ema=model_ema,
        use_amp=use_amp,
        world_size=args.world_size,
        not_eval=args.not_eval,
        training=True,
    )
    if args.wandb and args.rank == 0:
        wandb.finish()


if __name__ == "__main__":
    main()
