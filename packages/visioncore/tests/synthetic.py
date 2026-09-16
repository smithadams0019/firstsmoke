"""Synthetic scenes whose right answer is known by construction.

Kept out of conftest.py so the module name is unique across the workspace:
two conftest modules with the same basename collide in sys.modules when pytest
runs both packages in one session.
"""

from __future__ import annotations

import cv2
import numpy as np


def render_marker_scene(
    marker_id: int = 7,
    marker_px: int = 200,
    canvas: tuple[int, int] = (800, 600),
    top_left: tuple[int, int] = (200, 150),
    quiet_zone: int = 30,
    dictionary: str = "DICT_4X4_50",
) -> tuple[np.ndarray, float]:
    """A marker of an exactly known pixel edge on a white canvas.

    Returns (image, marker_edge_px). Ground truth is `marker_px` by construction:
    generateImageMarker writes a marker_px square, so px_per_mm must come out at
    marker_px / marker_length_mm.
    """
    from visioncore.calibration import ARUCO_DICTS

    adict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dictionary])
    marker = cv2.aruco.generateImageMarker(adict, marker_id, marker_px)
    w, h = canvas
    image = np.full((h, w), 255, dtype=np.uint8)
    x, y = top_left
    assert x - quiet_zone >= 0 and y - quiet_zone >= 0
    assert x + marker_px + quiet_zone <= w and y + marker_px + quiet_zone <= h
    image[y : y + marker_px, x : x + marker_px] = marker
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), float(marker_px)


def warp_like_tilt(image: np.ndarray, tilt_deg: float) -> np.ndarray:
    """Foreshorten an image horizontally by cos(tilt), which is what a plane tilt does.

    Keeps the canvas size, so the detector still sees a complete marker; only the
    aspect ratio changes, exactly the signal `quad_obliquity_deg` reads.
    """
    h, w = image.shape[:2]
    scale = float(np.cos(np.deg2rad(tilt_deg)))
    new_w = max(8, round(w * scale))
    squeezed = cv2.resize(image, (new_w, h), interpolation=cv2.INTER_AREA)
    canvas = np.full_like(image, 255)
    canvas[:, :new_w] = squeezed
    return canvas


def band_mask(
    width_px: int, *, length: int = 300, canvas: tuple[int, int] = (200, 400)
) -> np.ndarray:
    """A horizontal band of exactly `width_px` rows set, with a clear margin.

    Built by array slicing, not by cv2.line, so the ground truth is exact.
    """
    h, w = canvas
    mask = np.zeros((h, w), np.uint8)
    top = (h - width_px) // 2
    left = (w - length) // 2
    mask[top : top + width_px, left : left + length] = 255
    return mask


def project_marker(
    tilt_deg: float,
    *,
    marker_mm: float = 50.0,
    distance_mm: float = 500.0,
    focal_px: float = 1400.0,
    canvas: tuple[int, int] = (1200, 900),
    marker_id: int = 7,
    dictionary: str = "DICT_4X4_50",
    axis: str = "y",
) -> tuple[np.ndarray, dict[str, float]]:
    """Render a marker through a real pinhole camera with the plane tilted by `tilt_deg`.

    Unlike `warp_like_tilt`, this is a genuine perspective projection: the plane is
    rotated in 3D about the camera's y (or x) axis and projected with
    u = f*X/Z + cx. That means the quad is a true perspective quadrilateral, not
    an affine squeeze, so the calibration code is tested against the geometry it
    will actually meet.

    Returns (image, truth) where truth carries the ground-truth scales:

      fronto_px_per_mm   f / distance - the scale of an untilted plane
      centre_px_per_mm   the isotropic (geometric-mean) scale at the marker
                         centre, which is fronto * sqrt(cos(tilt))
    """
    from visioncore.calibration import ARUCO_DICTS

    width, height = canvas
    cx, cy = width / 2.0, height / 2.0
    half = marker_mm / 2.0
    theta = np.deg2rad(tilt_deg)
    cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))

    if axis == "y":
        rotation = np.array([[cos_t, 0.0, sin_t], [0.0, 1.0, 0.0], [-sin_t, 0.0, cos_t]])
    else:
        rotation = np.array([[1.0, 0.0, 0.0], [0.0, cos_t, -sin_t], [0.0, sin_t, cos_t]])

    # ArUco corner order: top-left, top-right, bottom-right, bottom-left
    object_mm = np.array(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]]
    )
    camera = object_mm @ rotation.T + np.array([0.0, 0.0, distance_mm])
    projected = np.stack(
        [focal_px * camera[:, 0] / camera[:, 2] + cx,
         -focal_px * camera[:, 1] / camera[:, 2] + cy],
        axis=1,
    ).astype(np.float32)

    adict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dictionary])
    side = 400
    marker = cv2.aruco.generateImageMarker(adict, marker_id, side)
    source = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], np.float32)

    homography = cv2.getPerspectiveTransform(source, projected)
    canvas_img = cv2.warpPerspective(
        cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR),
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    fronto = focal_px / distance_mm
    return canvas_img, {
        "fronto_px_per_mm": fronto,
        "centre_px_per_mm": fronto * float(np.sqrt(max(cos_t, 1e-9))),
        "tilt_deg": tilt_deg,
        "focal_px": focal_px,
        "distance_mm": distance_mm,
        "marker_mm": marker_mm,
        "cx": cx,
        "cy": cy,
    }


def camera_matrix(truth: dict[str, float]) -> np.ndarray:
    f = truth["focal_px"]
    return np.array([[f, 0.0, truth["cx"]], [0.0, f, truth["cy"]], [0.0, 0.0, 1.0]])
