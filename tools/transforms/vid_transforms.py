import numpy as np
import mmcv
from mmaction.registry import TRANSFORMS
from mmcv.transforms import BaseTransform
import random
import mmengine


@TRANSFORMS.register_module()
class VideoNormalize(BaseTransform):
    """Normalize a list of images.

    Required Keys:
    - imgs

    Modified Keys:
    - imgs

    Added Keys:
    - img_norm_cfg
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def transform(self, results: dict) -> dict:
        """
        Apply normalization to each image in the 'imgs' list.
        """
        # Apply normalization to each frame
        results["imgs"] = [
            mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
            for img in results["imgs"]
        ]

        results["img_norm_cfg"] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})"
        return repr_str


@TRANSFORMS.register_module()
class PadSkelSequence(BaseTransform):
    """
    Pads or truncates a skeleton sequence to a maximum length.
    """

    def __init__(
        self, max_len: int, key: str = "raw_keypoints", out_key: str = "keypoints"
    ):
        """
        Args:
            max_len (int): The target length of the sequence.
            key (str): The key in the results dictionary holding the input skeleton sequence.
            out_key (str): The key to save the padded sequence to.
        """
        self.max_len = max_len
        self.key = key
        self.out_key = out_key

    def transform(self, results: dict) -> dict:
        """
        Performs the padding/truncating logic.
        """
        skeleton_seq = results[self.key]

        if skeleton_seq.shape[0] > self.max_len:
            # Truncate the sequence if it's too long
            padded_seq = skeleton_seq[: self.max_len, :]
        elif skeleton_seq.shape[0] < self.max_len:
            # Pad with zeros if it's too short
            padding_shape = (
                self.max_len - skeleton_seq.shape[0],
                skeleton_seq.shape[1],
            )
            padding = np.zeros(padding_shape, dtype=skeleton_seq.dtype)
            padded_seq = np.concatenate([skeleton_seq, padding], axis=0)
        else:
            # Length is already correct
            padded_seq = skeleton_seq

        results[self.out_key] = padded_seq
        return results

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"max_len={self.max_len}, key={self.key}, out_key={self.out_key})"
        )


@TRANSFORMS.register_module()
class FormatSkelShape(BaseTransform):
    """
    Reshapes a skeleton sequence into the format expected by many models.
    Example: (T, V, C) -> (C, T, V, M)
    """

    def __init__(
        self,
        # --- MODIFIED: input_format is now T_V_C by default ---
        input_format: str = "T_V_C",
        target_format: str = "C_T_V_M",
        key: str = "keypoint",  # <-- The key should be 'keypoint'
    ):
        self.input_format = input_format
        self.target_format = target_format
        self.key = key

    def transform(self, results: dict) -> dict:
        skeleton_seq = results[self.key]
        # The input is now already in (T, V, C) format, so the
        # complicated reshaping logic is no longer needed.

        if self.input_format == "T_V_C":
            # Input shape is already (T, V, C), e.g., (989, 32, 4)
            # We can directly use it.
            seq_reshaped = skeleton_seq
        else:
            # You could add back the old logic here if you need to support
            # the flat (T, V*C) format for other datasets.
            raise NotImplementedError(
                f"Input format {self.input_format} not supported yet."
            )
        # Now transpose to the target format, e.g., C, T, V
        # From (T, V, C) -> (C, T, V)
        seq_transposed = seq_reshaped.transpose(2, 0, 1)

        if self.target_format == "C_T_V_M":
            # Add a 'member' dimension for single-person tracking
            # (C, T, V) -> (C, T, V, M=1)
            final_seq = np.expand_dims(seq_transposed, axis=-1)
        else:
            raise NotImplementedError(
                f"Target format {self.target_format} not supported yet."
            )

        results[self.key] = final_seq
        return results

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"input_format={self.input_format}, target_format={self.target_format})"
        )


def _init_lazy_if_proper(results, lazy):
    """Initialize lazy operation properly.

    Make sure that a lazy operation is properly initialized,
    and avoid a non-lazy operation accidentally getting mixed in.

    Required keys in results are "imgs" if "img_shape" not in results,
    otherwise, Required keys in results are "img_shape", add or modified keys
    are "img_shape", "lazy".
    Add or modified keys in "lazy" are "original_shape", "crop_bbox", "flip",
    "flip_direction", "interpolation".

    Args:
        results (dict): A dict stores data pipeline result.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    if "img_shape" not in results:
        results["img_shape"] = results["imgs"][0].shape[:2]
    if lazy:
        if "lazy" not in results:
            img_h, img_w = results["img_shape"]
            lazyop = dict()
            lazyop["original_shape"] = results["img_shape"]
            lazyop["crop_bbox"] = np.array([0, 0, img_w, img_h], dtype=np.float32)
            lazyop["flip"] = False
            lazyop["flip_direction"] = None
            lazyop["interpolation"] = None
            results["lazy"] = lazyop
    else:
        assert "lazy" not in results, "Use Fuse after lazy operations"


@TRANSFORMS.register_module()
class RandomCropKP(BaseTransform):
    """Vanilla square random crop that specifics the output size.

    Required keys in results are "img_shape", "keypoint" (optional), "imgs"
    (optional), added or modified keys are "keypoint", "imgs", "lazy"; Required
    keys in "lazy" are "flip", "crop_bbox", added or modified key is
    "crop_bbox".

    Args:
        size (int): The output size of the images.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self, size, lazy=False):
        if not isinstance(size, int):
            raise TypeError(f"Size must be an int, but got {type(size)}")
        self.size = size
        self.lazy = lazy

    @staticmethod
    def _crop_kps(kps, crop_bbox):
        """Static method for cropping keypoint."""
        if kps.shape[-1] > 2:
            # Handle keypoints with more than 2 dims (e.g., x, y, z, conf)
            # Only subtract the offset from the x and y coordinates.
            kps_aug = kps.copy()
            kps_aug[..., :2] = kps_aug[..., :2] - crop_bbox[:2]
            return kps_aug
        else:
            # Original behavior for 2D keypoints
            return kps - crop_bbox[:2]

    @staticmethod
    def _crop_imgs(imgs, crop_bbox):
        """Static method for cropping images."""
        x1, y1, x2, y2 = crop_bbox
        return [img[y1:y2, x1:x2] for img in imgs]

    @staticmethod
    def _box_crop(box, crop_bbox):
        """Crop the bounding boxes according to the crop_bbox.

        Args:
            box (np.ndarray): The bounding boxes.
            crop_bbox(np.ndarray): The bbox used to crop the original image.
        """

        x1, y1, x2, y2 = crop_bbox
        img_w, img_h = x2 - x1, y2 - y1

        box_ = box.copy()
        box_[..., 0::2] = np.clip(box[..., 0::2] - x1, 0, img_w - 1)
        box_[..., 1::2] = np.clip(box[..., 1::2] - y1, 0, img_h - 1)
        return box_

    def _all_box_crop(self, results, crop_bbox):
        """Crop the gt_bboxes and proposals in results according to crop_bbox.

        Args:
            results (dict): All information about the sample, which contain
                'gt_bboxes' and 'proposals' (optional).
            crop_bbox(np.ndarray): The bbox used to crop the original image.
        """
        results["gt_bboxes"] = self._box_crop(results["gt_bboxes"], crop_bbox)
        if "proposals" in results and results["proposals"] is not None:
            assert results["proposals"].shape[1] == 4
            results["proposals"] = self._box_crop(results["proposals"], crop_bbox)
        return results

    def transform(self, results):
        """Performs the RandomCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)
        if "keypoint" in results:
            assert not self.lazy, (
                "Keypoint Augmentations are not compatible " "with lazy == True"
            )

        img_h, img_w = results["img_shape"]
        assert self.size <= img_h and self.size <= img_w

        y_offset = 0
        x_offset = 0
        if img_h > self.size:
            y_offset = int(np.random.randint(0, img_h - self.size))
        if img_w > self.size:
            x_offset = int(np.random.randint(0, img_w - self.size))

        if "crop_quadruple" not in results:
            results["crop_quadruple"] = np.array(
                [0, 0, 1, 1], dtype=np.float32  # x, y, w, h
            )

        x_ratio, y_ratio = x_offset / img_w, y_offset / img_h
        w_ratio, h_ratio = self.size / img_w, self.size / img_h

        old_crop_quadruple = results["crop_quadruple"]
        old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio,
            old_y_ratio + y_ratio * old_h_ratio,
            w_ratio * old_w_ratio,
            h_ratio * old_h_ratio,
        ]
        results["crop_quadruple"] = np.array(new_crop_quadruple, dtype=np.float32)

        new_h, new_w = self.size, self.size

        crop_bbox = np.array([x_offset, y_offset, x_offset + new_w, y_offset + new_h])
        results["crop_bbox"] = crop_bbox

        results["img_shape"] = (new_h, new_w)

        if not self.lazy:
            if "keypoint" in results:
                results["keypoint"] = self._crop_kps(results["keypoint"], crop_bbox)
            if "imgs" in results:
                results["imgs"] = self._crop_imgs(results["imgs"], crop_bbox)
        else:
            lazyop = results["lazy"]
            if lazyop["flip"]:
                raise NotImplementedError("Put Flip at last for now")

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop["crop_bbox"]
            left = x_offset * (lazy_right - lazy_left) / img_w
            right = (x_offset + new_w) * (lazy_right - lazy_left) / img_w
            top = y_offset * (lazy_bottom - lazy_top) / img_h
            bottom = (y_offset + new_h) * (lazy_bottom - lazy_top) / img_h
            lazyop["crop_bbox"] = np.array(
                [
                    (lazy_left + left),
                    (lazy_top + top),
                    (lazy_left + right),
                    (lazy_top + bottom),
                ],
                dtype=np.float32,
            )

        # Process entity boxes
        if "gt_bboxes" in results:
            assert not self.lazy
            results = self._all_box_crop(results, results["crop_bbox"])

        return results

    def __repr__(self):
        repr_str = f"{self.__class__.__name__}(size={self.size}, " f"lazy={self.lazy})"
        return repr_str


@TRANSFORMS.register_module()
class RandomResizedCropKP(RandomCropKP):
    """Random crop that specifics the area and height-weight ratio range.

    Required keys in results are "img_shape", "crop_bbox", "imgs" (optional),
    "keypoint" (optional), added or modified keys are "imgs", "keypoint",
    "crop_bbox" and "lazy"; Required keys in "lazy" are "flip", "crop_bbox",
    added or modified key is "crop_bbox".

    Args:
        area_range (Tuple[float]): The candidate area scales range of
            output cropped images. Default: (0.08, 1.0).
        aspect_ratio_range (Tuple[float]): The candidate aspect ratio range of
            output cropped images. Default: (3 / 4, 4 / 3).
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(
        self, area_range=(0.08, 1.0), aspect_ratio_range=(3 / 4, 4 / 3), lazy=False
    ):
        self.area_range = area_range
        self.aspect_ratio_range = aspect_ratio_range
        self.lazy = lazy
        if not mmengine.is_tuple_of(self.area_range, float):
            raise TypeError(
                f"Area_range must be a tuple of float, " f"but got {type(area_range)}"
            )
        if not mmengine.is_tuple_of(self.aspect_ratio_range, float):
            raise TypeError(
                f"Aspect_ratio_range must be a tuple of float, "
                f"but got {type(aspect_ratio_range)}"
            )

    @staticmethod
    def get_crop_bbox(img_shape, area_range, aspect_ratio_range, max_attempts=10):
        """Get a crop bbox given the area range and aspect ratio range.

        Args:
            img_shape (Tuple[int]): Image shape
            area_range (Tuple[float]): The candidate area scales range of
                output cropped images. Default: (0.08, 1.0).
            aspect_ratio_range (Tuple[float]): The candidate aspect
                ratio range of output cropped images. Default: (3 / 4, 4 / 3).
                max_attempts (int): The maximum of attempts. Default: 10.
            max_attempts (int): Max attempts times to generate random candidate
                bounding box. If it doesn't qualified one, the center bounding
                box will be used.
        Returns:
            (list[int]) A random crop bbox within the area range and aspect
            ratio range.
        """
        assert 0 < area_range[0] <= area_range[1] <= 1
        assert 0 < aspect_ratio_range[0] <= aspect_ratio_range[1]

        img_h, img_w = img_shape
        area = img_h * img_w

        min_ar, max_ar = aspect_ratio_range
        aspect_ratios = np.exp(
            np.random.uniform(np.log(min_ar), np.log(max_ar), size=max_attempts)
        )
        target_areas = np.random.uniform(*area_range, size=max_attempts) * area
        candidate_crop_w = np.round(np.sqrt(target_areas * aspect_ratios)).astype(
            np.int32
        )
        candidate_crop_h = np.round(np.sqrt(target_areas / aspect_ratios)).astype(
            np.int32
        )

        for i in range(max_attempts):
            crop_w = candidate_crop_w[i]
            crop_h = candidate_crop_h[i]
            if crop_h <= img_h and crop_w <= img_w:
                x_offset = random.randint(0, img_w - crop_w)
                y_offset = random.randint(0, img_h - crop_h)
                return x_offset, y_offset, x_offset + crop_w, y_offset + crop_h

        # Fallback
        crop_size = min(img_h, img_w)
        x_offset = (img_w - crop_size) // 2
        y_offset = (img_h - crop_size) // 2
        return x_offset, y_offset, x_offset + crop_size, y_offset + crop_size

    def transform(self, results):
        """Performs the RandomResizeCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)
        if "keypoint" in results:
            assert not self.lazy, (
                "Keypoint Augmentations are not compatible " "with lazy == True"
            )

        img_h, img_w = results["img_shape"]

        left, top, right, bottom = self.get_crop_bbox(
            (img_h, img_w), self.area_range, self.aspect_ratio_range
        )
        new_h, new_w = bottom - top, right - left

        if "crop_quadruple" not in results:
            results["crop_quadruple"] = np.array(
                [0, 0, 1, 1], dtype=np.float32  # x, y, w, h
            )

        x_ratio, y_ratio = left / img_w, top / img_h
        w_ratio, h_ratio = new_w / img_w, new_h / img_h

        old_crop_quadruple = results["crop_quadruple"]
        old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio,
            old_y_ratio + y_ratio * old_h_ratio,
            w_ratio * old_w_ratio,
            h_ratio * old_h_ratio,
        ]
        results["crop_quadruple"] = np.array(new_crop_quadruple, dtype=np.float32)

        crop_bbox = np.array([left, top, right, bottom])
        results["crop_bbox"] = crop_bbox
        results["img_shape"] = (new_h, new_w)

        if not self.lazy:
            if "keypoint" in results:
                results["keypoint"] = self._crop_kps(results["keypoint"], crop_bbox)
            if "imgs" in results:
                results["imgs"] = self._crop_imgs(results["imgs"], crop_bbox)
        else:
            lazyop = results["lazy"]
            if lazyop["flip"]:
                raise NotImplementedError("Put Flip at last for now")

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop["crop_bbox"]
            left = left * (lazy_right - lazy_left) / img_w
            right = right * (lazy_right - lazy_left) / img_w
            top = top * (lazy_bottom - lazy_top) / img_h
            bottom = bottom * (lazy_bottom - lazy_top) / img_h
            lazyop["crop_bbox"] = np.array(
                [
                    (lazy_left + left),
                    (lazy_top + top),
                    (lazy_left + right),
                    (lazy_top + bottom),
                ],
                dtype=np.float32,
            )

        if "gt_bboxes" in results:
            assert not self.lazy
            results = self._all_box_crop(results, results["crop_bbox"])

        return results

    def __repr__(self):
        repr_str = (
            f"{self.__class__.__name__}("
            f"area_range={self.area_range}, "
            f"aspect_ratio_range={self.aspect_ratio_range}, "
            f"lazy={self.lazy})"
        )
        return repr_str


@TRANSFORMS.register_module()
class ResizeKP(BaseTransform):
    """Resize images to a specific size.

    Required keys are "img_shape", "modality", "imgs" (optional), "keypoint"
    (optional), added or modified keys are "imgs", "img_shape", "keep_ratio",
    "scale_factor", "lazy", "resize_size". Required keys in "lazy" is None,
    added or modified key is "interpolation".

    Args:
        scale (float | Tuple[int]): If keep_ratio is True, it serves as scaling
            factor or maximum size:
            If it is a float number, the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, the image will
            be rescaled as large as possible within the scale.
            Otherwise, it serves as (w, h) of output size.
        keep_ratio (bool): If set to True, Images will be resized without
            changing the aspect ratio. Otherwise, it will resize images to a
            given size. Default: True.
        interpolation (str): Algorithm used for interpolation:
            "nearest" | "bilinear". Default: "bilinear".
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self, scale, keep_ratio=True, interpolation="bilinear", lazy=False):
        if isinstance(scale, float):
            if scale <= 0:
                raise ValueError(f"Invalid scale {scale}, must be positive.")
        elif isinstance(scale, tuple):
            max_long_edge = max(scale)
            max_short_edge = min(scale)
            if max_short_edge == -1:
                # assign np.inf to long edge for rescaling short edge later.
                scale = (np.inf, max_long_edge)
        else:
            raise TypeError(
                f"Scale must be float or tuple of int, but got {type(scale)}"
            )
        self.scale = scale
        self.keep_ratio = keep_ratio
        self.interpolation = interpolation
        self.lazy = lazy

    def _resize_imgs(self, imgs, new_w, new_h):
        """Static method for resizing keypoint."""
        return [
            mmcv.imresize(img, (new_w, new_h), interpolation=self.interpolation)
            for img in imgs
        ]

    @staticmethod
    def _resize_kps(kps, scale_factor):
        """Static method for resizing keypoint."""
        if kps.shape[-1] > 2:
            # Handle keypoints with more than 2 dims (e.g., x, y, z, conf)
            # Only apply scaling to the x and y coordinates.
            kps_aug = kps.copy()
            kps_aug[..., :2] = kps_aug[..., :2] * scale_factor
            return kps_aug
        else:
            # Original behavior for 2D keypoints
            return kps * scale_factor

    @staticmethod
    def _box_resize(box, scale_factor):
        """Rescale the bounding boxes according to the scale_factor.

        Args:
            box (np.ndarray): The bounding boxes.
            scale_factor (np.ndarray): The scale factor used for rescaling.
        """
        assert len(scale_factor) == 2
        scale_factor = np.concatenate([scale_factor, scale_factor])
        return box * scale_factor

    def transform(self, results):
        """Performs the Resize augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """

        _init_lazy_if_proper(results, self.lazy)
        if "keypoint" in results:
            assert not self.lazy, (
                "Keypoint Augmentations are not compatible " "with lazy == True"
            )

        if "scale_factor" not in results:
            results["scale_factor"] = np.array([1, 1], dtype=np.float32)
        img_h, img_w = results["img_shape"]

        if self.keep_ratio:
            new_w, new_h = mmcv.rescale_size((img_w, img_h), self.scale)
        else:
            new_w, new_h = self.scale

        self.scale_factor = np.array([new_w / img_w, new_h / img_h], dtype=np.float32)

        results["img_shape"] = (new_h, new_w)
        results["keep_ratio"] = self.keep_ratio
        results["scale_factor"] = results["scale_factor"] * self.scale_factor

        if not self.lazy:
            if "imgs" in results:
                results["imgs"] = self._resize_imgs(results["imgs"], new_w, new_h)
            if "keypoint" in results:
                results["keypoint"] = self._resize_kps(
                    results["keypoint"], self.scale_factor
                )
        else:
            lazyop = results["lazy"]
            if lazyop["flip"]:
                raise NotImplementedError("Put Flip at last for now")
            lazyop["interpolation"] = self.interpolation

        if "gt_bboxes" in results:
            assert not self.lazy
            results["gt_bboxes"] = self._box_resize(
                results["gt_bboxes"], self.scale_factor
            )
            if "proposals" in results and results["proposals"] is not None:
                assert results["proposals"].shape[1] == 4
                results["proposals"] = self._box_resize(
                    results["proposals"], self.scale_factor
                )

        return results

    def __repr__(self):
        repr_str = (
            f"{self.__class__.__name__}("
            f"scale={self.scale}, keep_ratio={self.keep_ratio}, "
            f"interpolation={self.interpolation}, "
            f"lazy={self.lazy})"
        )
        return repr_str


@TRANSFORMS.register_module()
class RandomRescaleKP(BaseTransform):
    """Randomly resize images so that the short_edge is resized to a specific
    size in a given range. The scale ratio is unchanged after resizing.

    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs", "img_shape", "keep_ratio", "scale_factor", "resize_size",
    "short_edge".

    Args:
        scale_range (tuple[int]): The range of short edge length. A closed
            interval.
        interpolation (str): Algorithm used for interpolation:
            "nearest" | "bilinear". Default: "bilinear".
    """

    def __init__(self, scale_range, interpolation="bilinear"):
        self.scale_range = scale_range
        # make sure scale_range is legal, first make sure the type is OK
        assert mmengine.is_tuple_of(scale_range, int)
        assert len(scale_range) == 2
        assert scale_range[0] < scale_range[1]
        assert np.all([x > 0 for x in scale_range])

        self.keep_ratio = True
        self.interpolation = interpolation

    def transform(self, results):
        """Performs the Resize augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        short_edge = np.random.randint(self.scale_range[0], self.scale_range[1] + 1)
        resize = ResizeKP(
            (-1, short_edge),
            keep_ratio=True,
            interpolation=self.interpolation,
            lazy=False,
        )
        results = resize(results)

        results["short_edge"] = short_edge
        return results

    def __repr__(self):
        scale_range = self.scale_range
        repr_str = (
            f"{self.__class__.__name__}("
            f"scale_range=({scale_range[0]}, {scale_range[1]}), "
            f"interpolation={self.interpolation})"
        )
        return repr_str


from mmaction.datasets.transforms import Flip
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend to prevent windows from popping up
import matplotlib.pyplot as plt


@TRANSFORMS.register_module()
class Augment_Keypoints_and_Imgs(BaseTransform):
    """
    A unified transform to correctly augment both images and their corresponding
    HYBRID (u, v, Z) keypoints step-by-step. Includes a debug mode to
    visualize the output of each stage.
    """

    def __init__(self, resize_cfg, crop_cfg, last_resize_cfg, flip_cfg, debug=False):
        self.resize = ResizeKP(**resize_cfg)
        self.crop = RandomResizedCropKP(**crop_cfg)
        self.final_resize = ResizeKP(**last_resize_cfg)
        self.flip = Flip(**flip_cfg)
        self.debug = False
        # Define skeleton connections once for plotting
        self.skeleton_connections = [
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

    def _debug_save_plot(self, img, kps, save_path, title):
        """Helper function to save a debug visualization."""
        if img is None or kps is None:
            return

        # Ensure image is a NumPy array
        if not isinstance(img, np.ndarray):
            img = np.array(img)

        # Ensure image is in uint8 format for plotting
        if img.dtype != np.uint8:
            img = img.clip(0, 255).astype(np.uint8)

        fig, ax = plt.subplots(1, figsize=(16, 9), dpi=100)
        ax.imshow(img)

        if kps.size > 0:
            # Handle keypoints shape:
            # (K, C) -> Single person
            # (M, K, C) -> Multiple people

            kps_list = []
            if kps.ndim == 2:
                kps_list.append(kps)
            elif kps.ndim == 3:
                for i in range(kps.shape[0]):
                    kps_list.append(kps[i])

            colors = ["cyan", "lime", "magenta", "yellow", "orange"]

            for i, person_kps in enumerate(kps_list):
                col = colors[i % len(colors)]
                x_coords, y_coords = person_kps[:, 0], person_kps[:, 1]

                # Check if valid (not all zeros)
                if np.all(x_coords == 0) and np.all(y_coords == 0):
                    continue

                ax.scatter(
                    x_coords,
                    y_coords,
                    s=20,
                    c=col,
                    marker="o",
                    edgecolors="black",
                    zorder=2,
                )
                for start, end in self.skeleton_connections:
                    if start < len(x_coords) and end < len(x_coords):
                        # Don't plot lines to/from (0,0) points usually
                        if (x_coords[start] > 1e-1 or y_coords[start] > 1e-1) and (
                            x_coords[end] > 1e-1 or y_coords[end] > 1e-1
                        ):
                            ax.plot(
                                [x_coords[start], x_coords[end]],
                                [y_coords[start], y_coords[end]],
                                color=col,
                                linestyle="-",
                                lw=2.0,
                                zorder=1,
                            )

        ax.set_title(title, fontsize=16)
        ax.axis("off")
        fig.tight_layout(pad=0)
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        print(f"--- [DEBUG] Saved visualization: {save_path} ---")

    def transform(self, results: dict) -> dict:
        imgs = results["imgs"]
        kps = results.get("keypoint", np.array([]))  # Shape (T, M, K, C)

        results["img_shape"] = imgs[0].shape[:2]
        idx_vis = 130  # Frame index to visualize
        # --- Temporal Slicing: Sync Keypoints with Selected Frames ---
        if "frame_inds" in results and kps.size > 0:
            frame_inds = np.array(results["frame_inds"])
            if kps.shape[0] >= (frame_inds.max() if len(frame_inds) > 0 else 0):
                if frame_inds.max() < kps.shape[0]:
                    kps = kps[frame_inds]
                    results["keypoint"] = kps
                else:
                    # This can happen if keypoints length < total frames declared in video info
                    print(
                        f"[Warning] Temporal Slicing: frame_inds.max() {frame_inds.max()} >= kps length {kps.shape[0]}. "
                        f"Clipping indices."
                    )
                    valid_inds = frame_inds[frame_inds < kps.shape[0]]
                    kps = kps[valid_inds]
                    results["keypoint"] = kps
        # --- Pre-check: Resize keypoints if they don't match image dimensions ---
        # This part handles normalization to pixel coordinates
        if kps.size > 0:
            # print(results.keys())
            # kp_h_ref, kp_w_ref = results["original_shape_kp"]
            # print(f"Keypoint original shape: {kp_h_ref}, {kp_w_ref}")
            if kps.max() <= 1.0:
                # Assume normalized, scale to image size
                h, w = results["img_shape"]
                kps[..., 0] *= w
                kps[..., 1] *= h

            # Ensure we update results['keypoint'] with the denormalized kps
            results["keypoint"] = kps

        # --- Initial State (for debugging) ---
        if self.debug:
            # print(kps.shape, results["video_name"])
            self._debug_save_plot(
                imgs[idx_vis],
                kps[idx_vis] if kps.size > 0 else None,
                "debug_00_initial_state.png",
                "Initial State (Before Augmentation)",
            )

        # --- Step 1: Initial Resize ---
        results = self.resize(results)
        # RELOAD KPS: Standard transforms modify results['keypoint'] in-place or return new
        kps = results.get("keypoint", np.array([]))

        if self.debug:
            self._debug_save_plot(
                results["imgs"][idx_vis],
                kps[idx_vis] if kps.size > 0 else None,
                "debug_01_after_resize.png",
                f"Step 1: After Resize to {results['img_shape']}",
            )

        # --- Step 2: Random Crop ---
        results = self.crop(results)
        kps = results.get("keypoint", np.array([]))

        if self.debug:
            self._debug_save_plot(
                results["imgs"][idx_vis],
                kps[idx_vis] if kps.size > 0 else None,
                "debug_02_after_crop.png",
                f"Step 2: After Random Crop to {results['img_shape']}",
            )

        # --- Step 3: Final Resize ---
        results = self.final_resize(results)
        kps = results.get("keypoint", np.array([]))

        if self.debug:
            self._debug_save_plot(
                results["imgs"][idx_vis],
                kps[idx_vis] if kps.size > 0 else None,
                "debug_03_after_final_resize.png",
                f"Step 3: After Final Resize to {results['img_shape']}",
            )

        # --- Step 4: Flip ---
        results = self.flip(results)
        kps = results.get("keypoint", np.array([]))

        if self.debug:
            flip_status = "FLIPPED" if results.get("flip", False) else "NOT FLIPPED"
            self._debug_save_plot(
                results["imgs"][idx_vis],
                kps[idx_vis] if kps.size > 0 else None,
                "debug_04_after_flip.png",
                f"Step 4: After Flip ({flip_status})",
            )
            raise RuntimeError("Debug mode active - stopping after visualizations.")

        results["keypoint"] = kps
        return results
