import pickle
import numpy as np
import os
import multiprocessing as mp
from .builder import EVALUATORS
from copy import deepcopy


def compute_tube_iou(pred_tube, gt_tube):
    """
    Compute Spatio-Temporal IoU between two tubes.
    pred_tube: (start_frame, boxes_array(T, 4))
    gt_tube: (start_frame, boxes_array(T, 4))
    boxes are [x1, y1, x2, y2]
    """
    t_pred_start = pred_tube[0]
    boxes_pred = pred_tube[1]
    T_pred = boxes_pred.shape[0]
    t_pred_end = t_pred_start + T_pred  # exclusive

    t_gt_start = gt_tube[0]
    boxes_gt = gt_tube[1]
    T_gt = boxes_gt.shape[0]
    t_gt_end = t_gt_start + T_gt

    # Calculate temporal intersection
    t_start = max(t_pred_start, t_gt_start)
    t_end = min(t_pred_end, t_gt_end)

    if t_end <= t_start:
        # print(f"DEBUG IoU: No temporal overlap. Pred: {t_pred_start}-{t_pred_end}, GT: {t_gt_start}-{t_gt_end}")
        return 0.0

    # Intersection over union of frames
    # Union Duration: (t_pred_end - t_pred_start) + (t_gt_end - t_gt_start) - (t_end - t_start)
    # But we want 3D IoU which is Sum(Area_Inter) / Sum(Area_Union)
    # Area_Union = Area_Pred + Area_GT - Area_Inter

    intersection_vol = 0.0
    union_vol = 0.0

    # We iterate through the intersection frames
    # pred indices: t_start - t_pred_start to t_end - t_pred_start
    # gt indices: t_start - t_gt_start to t_end - t_gt_start

    p_s = int(t_start - t_pred_start)
    g_s = int(t_start - t_gt_start)
    length = int(t_end - t_start)

    # Extract overlapping boxes
    bp = boxes_pred[p_s : p_s + length]
    bg = boxes_gt[g_s : g_s + length]

    # Vectorized IoU for overlap
    # Intersection
    xx1 = np.maximum(bp[:, 0], bg[:, 0])
    yy1 = np.maximum(bp[:, 1], bg[:, 1])
    xx2 = np.minimum(bp[:, 2], bg[:, 2])
    yy2 = np.minimum(bp[:, 3], bg[:, 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter_area = w * h
    intersection_vol = np.sum(inter_area)

    # Area of each box in the overlap part
    area_p = (bp[:, 2] - bp[:, 0]) * (bp[:, 3] - bp[:, 1])
    area_g = (bg[:, 2] - bg[:, 0]) * (bg[:, 3] - bg[:, 1])

    # Union volume part 1: overlap frames
    union_area_overlap = area_p + area_g - inter_area

    # Parts that are not overlapping in time
    # Pred non-overlap: [0, p_s] and [p_s+length, T_pred]
    # GT non-overlap: [0, g_s] and [g_s+length, T_gt]

    # Just sum all areas and subtract intersection
    full_area_p = np.sum(
        (boxes_pred[:, 2] - boxes_pred[:, 0]) * (boxes_pred[:, 3] - boxes_pred[:, 1])
    )
    full_area_g = np.sum(
        (boxes_gt[:, 2] - boxes_gt[:, 0]) * (boxes_gt[:, 3] - boxes_gt[:, 1])
    )

    union_vol = full_area_p + full_area_g - intersection_vol

    if union_vol <= 0:
        return 0.0

    iou = intersection_vol / union_vol

    return iou


def iou_2d(box1, box2):
    """
    Compute 2D IoU between two boxes [x1, y1, x2, y2].
    """
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])

    w = max(0, inter_x2 - inter_x1)
    h = max(0, inter_y2 - inter_y1)
    inter_area = w * h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def link_predictions(results, iou_thresh=0.5):
    """
    Link tubelets into full video tubes.
    results: {video_id: [ {label, score, boxes, start_frame}, ... ]}
    """
    linked_results = {}

    for vid, preds in results.items():
        # Sort by start frame, then by score desc
        # This ensures high-confidence detections get first pick at linking (greedy-best approximation)
        preds.sort(key=lambda x: (x["start_frame"], -x["score"]))

        # Group by label
        by_label = {}
        for p in preds:
            l = p["label"]
            if l not in by_label:
                by_label[l] = []
            by_label[l].append(p)

        vid_linked = []

        for label, tubes in by_label.items():
            # Greedy linking
            # paths: list of {'score_sum': float, 'count': int, 'boxes': list, 'start': float, 'end': float}
            paths = []

            for t in tubes:
                # t has 'boxes' (list/array), 'start_frame', 'score', 'segment'
                t_start = t["start_frame"]
                # Assume t['boxes'] is list or np array
                t_boxes = t["boxes"]

                # Determine end frame based on box count (assuming contiguous)
                # If segment is available, use it, else infer
                if "segment" in t:
                    t_end = t["segment"][1]  # float
                else:
                    # Fallback if segment not present
                    t_end = t_start + len(t_boxes)

                best_match = -1
                max_iou = -1

                # Iterate backwards to find recent match
                for i in range(len(paths) - 1, -1, -1):
                    path = paths[i]

                    # Check continuity (temporal gap)
                    # For chunks [0,16], [16,32], gap is 0.
                    gap = t_start - path["end"]

                    # Allow small tolerance (e.g. +/- 2 frames)
                    if -2 <= gap <= 2:
                        # Check spatial IoU between transition frames
                        last_box = path["boxes"][-1]
                        first_box = t_boxes[0]

                        iou = iou_2d(last_box, first_box)

                        if iou > iou_thresh and iou > max_iou:
                            max_iou = iou
                            best_match = i
                            # optimization: break if greedy match found?
                            # Closest in time is usually preferred.
                            break
                    elif gap > 20:
                        # Optimization: if gap is too large, stop checking further back
                        # Since paths are not necessarily strictly sorted by end time, be careful.
                        # But we are iterating prediction stream which is sorted.
                        pass

                if best_match != -1:
                    # Extend existing path
                    path = paths[best_match]
                    path["boxes"].extend(t_boxes)
                    path["score_sum"] += t["score"]
                    path["count"] += 1
                    path["end"] = t_end
                else:
                    # Start new path
                    paths.append(
                        {
                            "label": label,
                            "score_sum": t["score"],
                            "count": 1,
                            "boxes": list(t_boxes),
                            "start_frame": t_start,
                            "end": t_end,
                            "video_id": vid,
                        }
                    )

            # Convert paths back to result format
            for path in paths:
                avg_score = path["score_sum"] / path["count"]
                vid_linked.append(
                    {
                        "label": path["label"],
                        "score": avg_score,
                        "boxes": path["boxes"],
                        "start_frame": path["start_frame"],
                        "video_id": vid,
                    }
                )

        linked_results[vid] = vid_linked

    return linked_results


def evaluate_video_ap(gt_db, pred_db, iou_threshold=0.5):
    """
    gt_db: {video_id: {label: [ (start_frame, boxes), ... ] } }
    pred_db: {video_id: [ {label, score, boxes, start_frame}, ... ] }
    """

    # Collect all classes
    classes = set()
    for v in gt_db.values():
        classes.update(v.keys())

    aps = []

    for cls in classes:
        # Collect all predictions for this class across all videos
        # List of (score, tp/fp(initially None), video_id, tube_idx)
        all_preds = []
        n_pos = 0

        for vid, preds in pred_db.items():
            # GT tubes for this class in this video
            gt_tubes = gt_db.get(vid, {}).get(cls, [])
            n_pos += len(gt_tubes)

            # Pred tubes for this class in this video
            cls_preds = [p for p in preds if p["label"] == cls]
            for p in cls_preds:
                all_preds.append(
                    {
                        "score": p["score"],
                        "boxes": np.array(p["boxes"]),
                        "start_frame": p["start_frame"],
                        "video_id": vid,
                    }
                )

        # Sort predictions by score desc
        all_preds.sort(key=lambda x: x["score"], reverse=True)

        tp = np.zeros(len(all_preds))
        fp = np.zeros(len(all_preds))

        # Track which GTs are detected
        gt_detected = {
            vid: [False] * len(gt_db.get(vid, {}).get(cls, [])) for vid in gt_db.keys()
        }

        for i, p in enumerate(all_preds):
            vid = p["video_id"]
            pred_tube = (p["start_frame"], p["boxes"])

            gt_tubes = gt_db.get(vid, {}).get(cls, [])

            max_iou = -1.0
            max_idx = -1

            if len(gt_tubes) > 0:
                for idx, gt_tube in enumerate(gt_tubes):
                    # gt_tube is (start_frame, boxes)
                    # Note: gt_tubes in our db structure might need adjustment
                    # if we store them as (start, boxes)
                    iou = compute_tube_iou(pred_tube, gt_tube)
                    if iou > max_iou:
                        max_iou = iou
                        max_idx = idx

            if max_iou >= iou_threshold:
                if not gt_detected[vid][max_idx]:
                    tp[i] = 1.0
                    gt_detected[vid][max_idx] = True
                else:
                    fp[i] = 1.0  # Duplicate detection
            else:
                fp[i] = 1.0

            # --- DEBUG EVALUATION MISMATCH ---
            if i < 3 and cls == list(classes)[0]:
                print(
                    f"[DEBUG EVAL] Vid: {vid}, Class: {cls}, PredScore: {p['score']:.4f}"
                )
                print(f"  Pred Box Sample: {p['boxes'][0]}")
                print(f"  Pred Range: {p['boxes'].min():.2f} - {p['boxes'].max():.2f}")
                if len(gt_tubes) > 0:
                    print(f"  GT Box Sample: {gt_tubes[0][1][0]}")
                    print(
                        f"  GT Range: {gt_tubes[0][1].min():.4f} - {gt_tubes[0][1].max():.4f}"
                    )
                    print(f"  Max IoU Found: {max_iou:.6f}")
            # ---------------------------------

        # Compute AP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        rec = tp_cumsum / max(n_pos, 1)
        prec = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, np.finfo(np.float64).eps)

        # 11-point AP or VOC07? Let's use continuous area
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.0

        aps.append(ap)

    return np.mean(aps) if len(aps) > 0 else 0.0


@EVALUATORS.register_module()
class VideoMAP:
    def __init__(
        self,
        ground_truth_filename,
        subset="val",
        iou_thresholds=[0.2, 0.5, 0.75],
        **kwargs,
    ):
        self.ground_truth_filename = ground_truth_filename
        self.subset = subset
        self.iou_thresholds = iou_thresholds
        self.class_agnostic = kwargs.get("class_agnostic", False)
        self.skip_linking = kwargs.get("skip_linking", False)
        self.link_iou_thresh = kwargs.get("link_iou_thresh", 0.5)

        # OpenTAD test_engine passes results in 'prediction_filename'
        self.results = None
        if "prediction_filename" in kwargs:
            res = kwargs["prediction_filename"]
            if isinstance(res, dict) and "results" in res:
                self.results = res["results"]
            else:
                self.results = res

        self.gt_db = self._load_gt()
        self.subset = subset
        self.iou_thresholds = iou_thresholds
        self.gt_db = self._load_gt()

    def _load_gt(self):
        print(f"Loading GT from {self.ground_truth_filename} for VideoMAP...")
        with open(self.ground_truth_filename, "rb") as f:
            data = pickle.load(f)

        # Structure: data['gttubes'][video_name][label_idx] = list of arrays(N, 5)
        # We need to filter by subset
        # data['train_videos'], data['test_videos'] (and maybe val?)

        target_videos = set()

        vid_list = []
        if self.subset == "train":
            vid_list = data.get("train_videos", [])
        elif self.subset == "test":
            vid_list = data.get("test_videos", [])
        elif self.subset == "val":
            vid_list = data.get("val_videos", data.get("test_videos", []))

        # Handle case where video list is wrapped in another list
        if len(vid_list) > 0 and isinstance(vid_list[0], list):
            vid_list = vid_list[0]

        target_videos = set(vid_list)

        # If target_videos empty, might be due to incomplete pkl or custom
        # For simplicity, we process all videos present in gttubes that are passed to evaluate

        db = {}
        # Convert to convenient structure:
        # db[vid][label] = list of (start_frame, boxes)

        if "gttubes" not in data:
            print("Warning: gttubes not found in annotation file.")
            return db

        for vid, tubes_dict in data["gttubes"].items():
            if target_videos and vid not in target_videos:
                continue

            db[vid] = {}
            for label, tubes in tubes_dict.items():
                label_key = 0 if self.class_agnostic else label
                if label_key not in db[vid]:
                    db[vid][label_key] = []
                for tube in tubes:
                    if tube.shape[0] == 0:
                        continue

                    res = data["resolution"].get(vid, None)
                    if res is None:
                        # Fallback or error
                        h, w = 720, 1280
                    else:
                        h, w = res

                    # Normalize
                    t_frames = tube[:, 0]
                    t_boxes = tube[:, 1:5].astype(np.float32)

                    if w > 0 and h > 0:
                        t_boxes[:, [0, 2]] /= w
                        t_boxes[:, [1, 3]] /= h

                    # DEBUG: Check normalization
                    if vid == list(data["gttubes"].keys())[0] and len(t_boxes) > 0:
                        print(
                            f"DEBUG Evaluator: GT Normalized sample for {vid}. Box[0]: {t_boxes[0]}"
                        )

                    start_frame = t_frames[0]
                    db[vid][label_key].append((start_frame, t_boxes))

        return db

    def evaluate(self, results=None, logger=None):
        """
        results: {video_id: [ {label, score, boxes, start_frame}, ... ]}
        """
        if results is None:
            results = self.results

        if results is None:
            print("No results to evaluate!")
            return {}

        if self.class_agnostic:
            # Collapse all prediction labels to 0 for class-agnostic eval
            results = {
                vid: [dict(p, label=0) for p in preds] for vid, preds in results.items()
            }

        # Basic Alignmnent Check
        pred_vids = set(results.keys())
        gt_vids = set(self.gt_db.keys())
        common_vids = pred_vids.intersection(gt_vids)

        if len(common_vids) == 0:
            print(
                f"WARNING: No common videos between predictions ({len(pred_vids)}) and GT ({len(gt_vids)}). mAP will be 0."
            )

        print(f"Evaluating Video mAP on {len(common_vids)} common videos...")

        # Link predictions (Tubelet stitching)
        if self.skip_linking:
            print("Skipping tubelet linking (debug mode)...")
        else:
            print("Linking tubelets into action tubes...")
            results = link_predictions(results, iou_thresh=self.link_iou_thresh)

        # Debug: per-video best IoU (class-agnostic) and label match stats
        for vid in list(common_vids)[:3]:
            gt_labels = set(self.gt_db.get(vid, {}).keys())
            pred_labels = set([p["label"] for p in results.get(vid, [])])
            label_overlap = len(gt_labels.intersection(pred_labels))

            pred_starts = [p["start_frame"] for p in results.get(vid, [])]
            pred_lengths = [len(p["boxes"]) for p in results.get(vid, [])]
            gt_starts = []
            gt_lengths = []
            gt_invalid = 0
            gt_total = 0
            for tubes in self.gt_db.get(vid, {}).values():
                for t in tubes:
                    gt_starts.append(t[0])
                    gt_lengths.append(t[1].shape[0])
                    if t[1].size > 0:
                        boxes = t[1]
                        invalid = (boxes[:, 2] <= boxes[:, 0]) | (
                            boxes[:, 3] <= boxes[:, 1]
                        )
                        gt_invalid += int(invalid.sum())
                        gt_total += int(boxes.shape[0])

            # Temporal overlap diagnostic (any overlap between any pred and gt)
            temporal_overlap = False
            for p in results.get(vid, []):
                p_s = p["start_frame"]
                p_e = p_s + len(p["boxes"])
                for tubes in self.gt_db.get(vid, {}).values():
                    for t in tubes:
                        g_s = t[0]
                        g_e = g_s + t[1].shape[0]
                        if min(p_e, g_e) > max(p_s, g_s):
                            temporal_overlap = True
                            break
                    if temporal_overlap:
                        break
                if temporal_overlap:
                    break

            best_iou = 0.0
            best_pair = None
            best_pred = None
            best_gt = None
            best_iou_swap = 0.0
            for p in results.get(vid, []):
                p_label = p["label"]
                p_tube = (p["start_frame"], np.array(p["boxes"], dtype=np.float32))
                for gt_label, tubes in self.gt_db.get(vid, {}).items():
                    for gt_tube in tubes:
                        iou = compute_tube_iou(p_tube, gt_tube)
                        if iou > best_iou:
                            best_iou = iou
                            best_pair = (p_label, gt_label)
                            best_pred = p_tube
                            best_gt = gt_tube

                        # Check swapped GT coords (y1,x1,y2,x2) hypothesis
                        gt_boxes = gt_tube[1]
                        if gt_boxes.size > 0:
                            gt_swapped = gt_boxes[:, [1, 0, 3, 2]]
                            iou_swap = compute_tube_iou(
                                p_tube, (gt_tube[0], gt_swapped)
                            )
                            if iou_swap > best_iou_swap:
                                best_iou_swap = iou_swap

            # Extra spatial stats for best match
            if best_pred is not None and best_gt is not None:
                p_boxes = best_pred[1]
                g_boxes = best_gt[1]
                # Align temporal overlap for stats
                t_pred_start = best_pred[0]
                t_gt_start = best_gt[0]
                t_start = max(t_pred_start, t_gt_start)
                t_end = min(
                    t_pred_start + p_boxes.shape[0], t_gt_start + g_boxes.shape[0]
                )
                if t_end > t_start:
                    p_s = int(t_start - t_pred_start)
                    g_s = int(t_start - t_gt_start)
                    length = int(t_end - t_start)
                    bp = p_boxes[p_s : p_s + length]
                    bg = g_boxes[g_s : g_s + length]

                    p_cx = (bp[:, 0] + bp[:, 2]) * 0.5
                    p_cy = (bp[:, 1] + bp[:, 3]) * 0.5
                    g_cx = (bg[:, 0] + bg[:, 2]) * 0.5
                    g_cy = (bg[:, 1] + bg[:, 3]) * 0.5
                    center_dist = np.sqrt((p_cx - g_cx) ** 2 + (p_cy - g_cy) ** 2)
                    mean_center_dist = float(np.mean(center_dist))

                    p_area = np.clip(
                        (bp[:, 2] - bp[:, 0]) * (bp[:, 3] - bp[:, 1]), 1e-6, None
                    )
                    g_area = np.clip(
                        (bg[:, 2] - bg[:, 0]) * (bg[:, 3] - bg[:, 1]), 1e-6, None
                    )
                    area_ratio = float(np.mean(p_area / g_area))
                else:
                    mean_center_dist = float("nan")
                    area_ratio = float("nan")
            else:
                mean_center_dist = float("nan")
                area_ratio = float("nan")

            print(
                f"[DEBUG EVAL VIDEO] {vid} pred_labels={len(pred_labels)} "
                f"gt_labels={len(gt_labels)} overlap={label_overlap} "
                f"best_iou={best_iou:.4f} best_pair={best_pair} "
                f"center_dist={mean_center_dist:.4f} area_ratio={area_ratio:.3f} "
                f"temporal_overlap={temporal_overlap} "
                f"pred_start_range=({min(pred_starts) if pred_starts else 'na'},"
                f"{max(pred_starts) if pred_starts else 'na'}) "
                f"gt_start_range=({min(gt_starts) if gt_starts else 'na'},"
                f"{max(gt_starts) if gt_starts else 'na'}) "
                f"gt_invalid={gt_invalid}/{gt_total} best_iou_swap={best_iou_swap:.4f}"
            )

        metrics = {}
        for th in self.iou_thresholds:
            map_score = evaluate_video_ap(self.gt_db, results, iou_threshold=th)
            metrics[f"v-mAP@{th}"] = map_score
            print(f"v-mAP@{th}: {map_score:.4f}")

        metrics["mAP"] = np.mean(list(metrics.values()))
        self.metrics = metrics

        # Diagnostics for 0% mAP
        if metrics["mAP"] == 0.0:
            print("\n!!! Warning: mAP is 0.0% !!!")
            print("Troubleshooting checks:")
            print(f"1. Common Videos: {len(common_vids)}")
            if len(common_vids) > 0:
                sample_vid = list(common_vids)[0]
                print(f"2. Sample Video {sample_vid}:")
                p_labels = set(p["label"] for p in results[sample_vid])
                g_labels = set(self.gt_db[sample_vid].keys())
                print(
                    f"   Pred Labels: {len(p_labels)} present (e.g. {list(p_labels)[:5]}...)"
                )
                print(
                    f"   GT Labels: {len(g_labels)} present (e.g. {list(g_labels)[:5]}...)"
                )

                scores = [p["score"] for p in results[sample_vid]]
                if scores:
                    print(f"   Score Range: {min(scores):.4f} - {max(scores):.4f}")

                # Check first tube duration
                if len(results[sample_vid]) > 0:
                    p0 = results[sample_vid][0]
                    print(
                        f"   Sample Pred Tube: Start={p0['start_frame']:.1f}, Duration={len(p0['boxes'])}"
                    )

                    if len(self.gt_db[sample_vid]) > 0:
                        g0_key = list(self.gt_db[sample_vid].keys())[0]
                        if len(self.gt_db[sample_vid][g0_key]) > 0:
                            g0 = self.gt_db[sample_vid][g0_key][0]
                            print(
                                f"   Sample GT Tube: Start={g0[0]:.1f}, Duration={len(g0[1])}"
                            )

                            # Test IoU
                            t_iou = compute_tube_iou(
                                (p0["start_frame"], np.array(p0["boxes"])), g0
                            )
                            print(f"   Sample IoU (Label Match): {t_iou:.4f}")

                            # Debug: Class Agnostic IoU
                            max_iou_ca = 0.0
                            for l_key in self.gt_db[sample_vid]:
                                for gt_t in self.gt_db[sample_vid][l_key]:
                                    # gt_t is (start, boxes)
                                    iou_ca = compute_tube_iou(
                                        (p0["start_frame"], np.array(p0["boxes"])), gt_t
                                    )
                                    if iou_ca > max_iou_ca:
                                        max_iou_ca = iou_ca
                            print(
                                f"   Sample Max IoU (Class Agnostic): {max_iou_ca:.4f}"
                            )

        return metrics

    def logging(self, logger=None):
        if logger is None:
            pprint = print
        else:
            pprint = logger.info

        if hasattr(self, "metrics"):
            pprint(f"Video mAP (v-mAP): {self.metrics['mAP']*100:.2f}%")
            for k, v in self.metrics.items():
                if k != "mAP":
                    pprint(f"{k}: {v*100:.2f}%")
