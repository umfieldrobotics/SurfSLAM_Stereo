"""Stereo rectification from a Kalibr calibration file.

Disparity is only meaningful on a rectified pair: the network assumes a
corresponding point sits on the same image row in both views, at ``x - disp``.
The SUDS release ships *raw* frames (``raw_left``/``raw_right``), so every
consumer -- the dataset loaders, the demo, anything else that reads those
images -- has to undistort and rectify first, and has to do it the same way, or
the disparities it produces are not comparable.

This module owns that one piece of geometry. Give it a Kalibr yaml
(``cam0``/``cam1`` with ``intrinsics``, ``distortion_coeffs`` and
``T_cn_cnm1``) and it hands back the remap tables, the rectified projection
matrices, and the baseline:

    rect = load_kalibr_calibration("demo/data/stereo_calib.yaml")
    left, right = rect.rectify(left, right)
    depth = rect.focal_px * rect.baseline_m / disparity

Only the radtan (plumb bob) model is supported, which is what our ZED
calibrations use.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence, Tuple, Union

import cv2
import numpy as np
import yaml


def _as_K(intrinsics: Sequence[float]) -> np.ndarray:
    """Kalibr's ``[fx, fy, cx, cy]`` as a 3x3 camera matrix."""
    fx, fy, cx, cy = intrinsics
    return np.array([[fx, 0.0, cx],
                     [0.0, fy, cy],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _as_D(distortion_coeffs: Sequence[float]) -> np.ndarray:
    """Kalibr's radtan ``[k1, k2, p1, p2]`` as OpenCV's 5-vector (k3 = 0)."""
    k1, k2, p1, p2 = distortion_coeffs
    return np.array([k1, k2, p1, p2, 0.0], dtype=np.float64)


@dataclass(frozen=True, eq=False)
class StereoRectification:
    """Everything needed to rectify one stereo rig and interpret its disparities.

    Attributes:
        size: ``(width, height)`` the calibration was made at. Images of any
            other size cannot be rectified with these maps.
        map1_l, map2_l, map1_r, map2_r: ``cv2.remap`` tables for each camera.
        P1, P2: 3x4 projection matrices of the rectified cameras.
        baseline_m: distance between the rectified camera centres, in metres.
    """

    size: Tuple[int, int]
    map1_l: np.ndarray
    map2_l: np.ndarray
    map1_r: np.ndarray
    map2_r: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    baseline_m: float

    @property
    def focal_px(self) -> float:
        """Focal length of the rectified left camera, in pixels."""
        return float(self.P1[0, 0])

    @property
    def left_intrinsics(self) -> np.ndarray:
        """3x3 intrinsics of the rectified left camera."""
        return self.P1[:3, :3]

    @property
    def right_intrinsics(self) -> np.ndarray:
        """3x3 intrinsics of the rectified right camera."""
        return self.P2[:3, :3]

    def rectify(self, left: np.ndarray, right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Undistort and rectify a pair of images taken with this rig.

        Raises:
            ValueError: if either image is not the size the rig was calibrated at.
        """
        expected = (self.size[1], self.size[0])
        for name, img in (("left", left), ("right", right)):
            if img.shape[:2] != expected:
                raise ValueError(
                    f"{name} image is {img.shape[1]}x{img.shape[0]}, but this calibration "
                    f"is for {self.size[0]}x{self.size[1]}. Rectify with the calibration "
                    f"that matches your images, or pass images at the calibrated size.")

        rect_l = cv2.remap(left, self.map1_l, self.map2_l, interpolation=cv2.INTER_LINEAR)
        rect_r = cv2.remap(right, self.map1_r, self.map2_r, interpolation=cv2.INTER_LINEAR)
        return rect_l, rect_r


@lru_cache(maxsize=4)
def load_kalibr_calibration(path: Union[str, Path]) -> StereoRectification:
    """Build the rectification for the rig described by a Kalibr yaml.

    Cached: the remap tables are two float32 arrays per camera at full image
    resolution (~32 MB for 1920x1080), and rebuilding them per call is wasteful
    when the same rig is used over and over.
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    try:
        cam0, cam1 = data["cam0"], data["cam1"]
    except (TypeError, KeyError) as err:
        raise ValueError(
            f"{path} does not look like a Kalibr stereo calibration: "
            f"expected top-level 'cam0' and 'cam1' entries.") from err

    K0, D0 = _as_K(cam0["intrinsics"]), _as_D(cam0["distortion_coeffs"])
    K1, D1 = _as_K(cam1["intrinsics"]), _as_D(cam1["distortion_coeffs"])
    size = tuple(int(v) for v in cam0["resolution"])

    # T_cn_cnm1 is the transform from the previous camera (cam0) into cam1.
    # The translation must be a column vector: OpenCV 4 tolerated a flat (3,)
    # array here, OpenCV 5 fails inside gemm.
    T10 = np.array(cam1["T_cn_cnm1"], dtype=np.float64)
    R, t = T10[:3, :3], T10[:3, 3].reshape(3, 1)

    # alpha=0 crops to the largest all-valid rectangle, so the rectified images
    # have no invalid border for the network to hallucinate disparity on.
    R1, R2, P1, P2, _Q, _roi1, _roi2 = cv2.stereoRectify(
        K0, D0, K1, D1, size, R, t,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.0,
        newImageSize=size,
    )

    map1_l, map2_l = cv2.initUndistortRectifyMap(K0, D0, R1, P1, size, cv2.CV_32FC1)
    map1_r, map2_r = cv2.initUndistortRectifyMap(K1, D1, R2, P2, size, cv2.CV_32FC1)

    # After rectification the right camera sits at -baseline along x, encoded in
    # P2 as a translation of -f * baseline.
    baseline = -float(P2[0, 3]) / float(P2[0, 0]) if P2[0, 0] != 0 else float(t[0])

    return StereoRectification(
        size=size,
        map1_l=map1_l, map2_l=map2_l,
        map1_r=map1_r, map2_r=map2_r,
        P1=P1, P2=P2,
        baseline_m=baseline,
    )
