import os
import sys

sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)

import json
import argparse
import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from mmengine.config import Config, DictAction

from opentad.models import build_detector
from opentad.datasets import build_dataset
from opentad.datasets.builder import collate
from opentad.models.utils.post_processing import batched_nms
from opentad.utils import (
    set_seed,
    remap_legacy_sparse_conv_weights,
    override_dataset_paths,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference on a single video with a Temporal Action Detector"
    )
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
    parser.add_argument(
        "--video", type=str, required=True,
        help="video name (id) to run inference on, e.g. video_test_0000004",
    )
    parser.add_argument("--checkpoint", type=str, default="none", help="the checkpoint path")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="override settings")
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
        "--device", type=str, default="cuda:0",
        help="device to run inference on, e.g. cuda:0 or cpu",
    )
    parser.add_argument(
        "--score-thresh", type=float, default=0.0,
        help="only show predictions with score above this threshold",
    )
    parser.add_argument(
        "--topk", type=int, default=20,
        help="show at most this many predictions (sorted by score, -1 for all)",
    )
    return parser.parse_args()


def load_ground_truth(ann_file, video_name):
    """Read the annotations of a single video directly from the annotation json."""
    with open(ann_file, "r") as f:
        database = json.load(f)["database"]
    if video_name not in database:
        return None, []
    video_info = database[video_name]
    gt = []
    for anno in video_info.get("annotations", []):
        if anno["label"] == "Ambiguous":
            continue
        gt.append(dict(segment=anno["segment"], label=anno["label"]))
    gt.sort(key=lambda x: x["segment"][0])
    return video_info, gt


def nms_single_video(predictions, nms_cfg):
    """Apply the same NMS used for sliding-window evaluation, but for one video."""
    segments = torch.Tensor([data["segment"] for data in predictions])
    scores = torch.Tensor([data["score"] for data in predictions])
    class_idx = []
    labels = []
    for data in predictions:
        if data["label"] not in class_idx:
            class_idx.append(data["label"])
        labels.append(class_idx.index(data["label"]))
    labels = torch.Tensor(labels)

    segments, scores, labels = batched_nms(segments, scores, labels, **nms_cfg)

    results = []
    for segment, label, score in zip(segments, labels, scores):
        results.append(
            dict(
                segment=[round(seg.item(), 2) for seg in segment],
                label=class_idx[int(label.item())],
                score=round(score.item(), 4),
            )
        )
    return results


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg = override_dataset_paths(
        cfg,
        ann_file=args.ann_file,
        class_map=args.class_map,
        data_path=args.data_root,
    )

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")
    print(f"Using device: {device}")

    # build test dataset and keep only the windows belonging to the requested video
    test_dataset = build_dataset(cfg.dataset.test)
    print(test_dataset.data_list)
    test_dataset.data_list = [d for d in test_dataset.data_list if d[0] == args.video]
    if len(test_dataset.data_list) == 0:
        raise ValueError(
            f"Video '{args.video}' not found in the '{cfg.dataset.test.subset_name}' subset. "
            f"Check the video name and that it belongs to this subset."
        )
    print(f"Video '{args.video}' split into {len(test_dataset.data_list)} sliding windows.")

    # build model
    model = build_detector(cfg.model)
    model = model.to(device)

    # load checkpoint (args -> config -> best)
    if args.checkpoint != "none":
        checkpoint_path = args.checkpoint
    elif "test_epoch" in cfg.inference.keys():
        checkpoint_path = os.path.join(cfg.work_dir, f"checkpoint/epoch_{cfg.inference.test_epoch}.pth")
    else:
        checkpoint_path = os.path.join(cfg.work_dir, "checkpoint/best.pth")
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Checkpoint is epoch {checkpoint.get('epoch', 'unknown')}.")

    # Model EMA
    use_ema = getattr(cfg.solver, "ema", False)
    state_dict = checkpoint["state_dict_ema"] if use_ema else checkpoint["state_dict"]
    consume_prefix_in_state_dict_if_present(state_dict, prefix="module.")
    state_dict = remap_legacy_sparse_conv_weights(state_dict, model.state_dict())
    model.load_state_dict(state_dict)
    if use_ema:
        print("Using Model EMA...")

    use_amp = getattr(cfg.solver, "amp", False)

    # this is a sliding window dataset, so NMS is applied after merging windows
    cfg.post_processing.sliding_window = True
    external_cls = test_dataset.class_map  # list: class index -> class name

    # inference, window by window
    model.eval()
    result_dict = {}
    print("Running inference...")
    for index in range(len(test_dataset)):
        data_dict = collate([test_dataset[index]])
        data_dict["inputs"] = data_dict["inputs"].to(device)
        data_dict["masks"] = data_dict["masks"].to(device)

        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
            with torch.no_grad():
                results = model(
                    **data_dict,
                    return_loss=False,
                    infer_cfg=cfg.inference,
                    post_cfg=cfg.post_processing,
                    ext_cls=external_cls,
                )

        for k, v in results.items():
            if k in result_dict:
                result_dict[k].extend(v)
            else:
                result_dict[k] = v

    # merge windows with NMS (same as sliding-window evaluation)
    predictions = result_dict.get(args.video, [])
    if len(predictions) > 0 and cfg.post_processing.nms is not None:
        predictions = nms_single_video(predictions, dict(cfg.post_processing.nms))
    predictions.sort(key=lambda x: x["score"], reverse=True)

    # ground truth
    ann_file = args.ann_file if args.ann_file is not None else cfg.dataset.test.ann_file
    video_info, ground_truth = load_ground_truth(ann_file, args.video)

    # ---- report ----
    print("\n" + "=" * 70)
    print(f"VIDEO: {args.video}")
    if video_info is not None:
        print(f"  duration: {video_info.get('duration', '?')} s | frames: {video_info.get('frame', '?')}")
    print("=" * 70)

    print(f"\nGROUND TRUTH ({len(ground_truth)} segments):")
    if len(ground_truth) == 0:
        print("  (no ground truth annotations found for this video)")
    else:
        print(f"  {'start':>8}  {'end':>8}  label")
        print(f"  {'-'*8}  {'-'*8}  {'-'*20}")
        for gt in ground_truth:
            s, e = gt["segment"]
            print(f"  {s:>8.2f}  {e:>8.2f}  {gt['label']}")

    shown = [p for p in predictions if p["score"] >= args.score_thresh]
    if args.topk >= 0:
        shown = shown[: args.topk]
    print(
        f"\nPREDICTIONS (showing {len(shown)} of {len(predictions)}"
        f"{f', score >= {args.score_thresh}' if args.score_thresh > 0 else ''}):"
    )
    if len(shown) == 0:
        print("  (no predictions)")
    else:
        print(f"  {'start':>8}  {'end':>8}  {'score':>7}  label")
        print(f"  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*20}")
        for p in shown:
            s, e = p["segment"]
            print(f"  {s:>8.2f}  {e:>8.2f}  {p['score']:>7.4f}  {p['label']}")
    print()


if __name__ == "__main__":
    main()
