"""Turning a change map into described regions.

`connectedComponentsWithStats` gives blobs. A blob is not evidence. What makes a
blob smoke-like is a set of measurable properties, and the honest thing is to
measure all of them, attach them to the region, and let the scoring stage weigh
them in the open rather than hiding a decision inside a filter.

The properties that matter, and why:

* **Texture drop.** Smoke is a translucent scattering medium. It does not replace
  the hillside behind it, it veils it, so the local Laplacian variance inside the
  region falls relative to what the background reference had there. A shadow
  darkens texture but keeps it; an object replaces it with its own. The *ratio*
  separates all three.
* **Desaturation.** Wildfire smoke is grey to grey-brown. It pulls the colour of
  whatever is behind it toward neutral. Cloud does this too, which is why this
  test alone is useless and why the column analysis exists.
* **Soft edges.** A smoke boundary has a gradient measured over tens of pixels.
  A ridge line misregistered by camera shake has a gradient over two.
* **Attachment.** A plume from an ignition starts at the ground and is connected
  to it. A cloud floats. Measuring how far the region's base sits below the sky
  boundary is the single most useful geometric test in this pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .background import ChangeMap
from .scene import Horizon

MIN_AREA_PX = 40
MAX_AREA_FRACTION = 0.30
"""A region covering a third of the frame is weather, a re-exposure or a lens
event. Real early smoke is small; by the time a plume fills the frame, the
people who needed telling have been told."""


@dataclass
class Region:
    """One candidate patch of change, with everything we measured about it."""

    x: int
    y: int
    w: int
    h: int
    area: int
    cx: float
    cy: float
    mask: np.ndarray
    """uint8 0/255, full-frame size, so overlays and crops stay simple."""

    # measured properties
    area_fraction: float = 0.0
    elongation: float = 1.0
    fill: float = 1.0
    mean_signed: float = 0.0
    texture_ratio: float = 1.0
    saturation_ratio: float = 1.0
    greyness: float = 0.0
    edge_softness: float = 0.0
    base_below_horizon: float = 0.0
    top_above_horizon: float = 0.0
    sky_fraction: float = 0.0
    frame_index: int = 0
    base_cx: float = 0.0
    """Centroid x of the lowest slice of the mask: where the thing meets the ground.

    Not the bounding box centre. A plume that shears downwind widens at the top,
    which drags the box centre sideways even though the fire has not moved, and
    that alone was enough to make the cloud rejector fire on real smoke."""
    base_cy: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def base_y(self) -> int:
        return self.y + self.h

    @property
    def top_y(self) -> int:
        return self.y

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    def crop(self, image: np.ndarray, pad: int = 8) -> np.ndarray:
        h, w = image.shape[:2]
        x0 = max(0, self.x - pad)
        y0 = max(0, self.y - pad)
        x1 = min(w, self.x + self.w + pad)
        y1 = min(h, self.y + self.h + pad)
        return image[y0:y1, x0:x1]

    def iou(self, other: Region) -> float:
        ax0, ay0, ax1, ay1 = self.x, self.y, self.x + self.w, self.y + self.h
        bx0, by0, bx1, by1 = other.x, other.y, other.x + other.w, other.y + other.h
        ix = max(0, min(ax1, bx1) - max(ax0, bx0))
        iy = max(0, min(ay1, by1) - max(ay0, by0))
        inter = ix * iy
        union = self.w * self.h + other.w * other.h - inter
        return inter / union if union else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": [self.x, self.y, self.w, self.h],
            "centroid": [round(self.cx, 1), round(self.cy, 1)],
            "area": self.area,
            "area_fraction": round(self.area_fraction, 5),
            "elongation": round(self.elongation, 3),
            "fill": round(self.fill, 3),
            "mean_signed": round(self.mean_signed, 2),
            "texture_ratio": round(self.texture_ratio, 3),
            "saturation_ratio": round(self.saturation_ratio, 3),
            "greyness": round(self.greyness, 3),
            "edge_softness": round(self.edge_softness, 3),
            "base_below_horizon": round(self.base_below_horizon, 1),
            "top_above_horizon": round(self.top_above_horizon, 1),
            "sky_fraction": round(self.sky_fraction, 3),
            "base_point": [round(self.base_cx, 1), round(self.base_cy, 1)],
            "frame_index": self.frame_index,
            **self.metrics,
        }


def extract_regions(
    frame: np.ndarray,
    change: ChangeMap,
    horizon: Horizon,
    *,
    reference: np.ndarray | None = None,
    has_colour: bool = True,
    frame_index: int = 0,
    min_area: int = MIN_AREA_PX,
    limit: int = 12,
    diff_threshold: float = 6.0,
) -> list[Region]:
    """Find candidate regions and measure them.

    The mask is the union of two things: MOG2's foreground, and anywhere the
    photometric difference is large. MOG2 alone misses a plume that grows slowly
    enough to be learned; the photometric difference alone lights up on every
    exposure change. Together they are stricter than either, because the scoring
    stage sees which of the two fired.
    """
    h, w = frame.shape[:2]
    magnitude = np.abs(change.signed)
    combined = cv2.bitwise_or(change.foreground, _hysteresis(magnitude, diff_threshold))

    # Nothing below the ridge by more than a plume's worth of height is
    # interesting, and nothing in the top few rows is: a column that starts at the
    # very top of frame came in from outside it, which makes it weather.
    keep = horizon.band_mask((h, w), above_px=h, below_px=int(h * 0.30))
    keep[: int(h * 0.02), :] = 0
    combined = cv2.bitwise_and(combined, keep)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(combined, connectivity=8)
    frame_area = float(h * w)

    # Choose which components are worth measuring *before* measuring any of
    # them. A real HPWREN frame produces around seventy components that pass the
    # area filter; the first version measured all seventy and then kept the
    # twelve largest, so 83% of the most expensive work in the pipeline was
    # thrown away. That one line was 1.1 of the 1.3 seconds a frame.
    candidates = [
        (int(stats[label][4]), label)
        for label in range(1, count)
        if min_area <= int(stats[label][4]) <= MAX_AREA_FRACTION * frame_area
    ]
    candidates.sort(reverse=True)
    candidates = candidates[:limit]
    if not candidates:
        return []

    shared = _FrameContext(frame, change, gray_of(frame), reference, has_colour)

    regions: list[Region] = []
    for area, label in candidates:
        x, y, bw, bh = (int(v) for v in stats[label][:4])
        mask = (labels == label).astype(np.uint8) * 255
        region = Region(
            x=x, y=y, w=bw, h=bh, area=area,
            cx=float(centroids[label][0]), cy=float(centroids[label][1]),
            mask=mask, frame_index=frame_index,
        )
        _measure(region, shared, horizon)
        regions.append(region)
    return regions


def gray_of(frame: np.ndarray) -> np.ndarray:
    return frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


class _FrameContext:
    """Everything the per-region measurements share, computed once a frame.

    Each of these used to be recomputed inside every region: a full-frame
    Gaussian blur, two Sobels, an HSV conversion and a channel-spread image, all
    for a region a few hundred pixels across. Hoisting them out is most of the
    remaining speed-up, and it is also simply the right shape — these are
    properties of the frame, not of the region.
    """

    __slots__ = (
        "change", "frame", "gray", "has_colour", "hsv", "laplacian", "luma",
        "ref_laplacian", "reference", "signed", "slope", "spread",
    )

    def __init__(self, frame, change, gray, reference, has_colour) -> None:
        self.frame = frame
        self.change = change
        self.gray = gray
        self.reference = reference
        self.has_colour = has_colour
        self.laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        self.ref_laplacian = (
            cv2.Laplacian(reference.astype(np.uint8), cv2.CV_32F, ksize=3)
            if reference is not None
            else None
        )
        self.signed = np.abs(change.signed)
        smoothed = cv2.GaussianBlur(self.signed, (0, 0), 2.0)
        self.slope = cv2.magnitude(
            cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3),
        )
        if has_colour and frame.ndim == 3:
            self.hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            b, g, r = (frame[:, :, i].astype(np.float32) for i in range(3))
            self.spread = np.maximum(np.maximum(np.abs(b - g), np.abs(g - r)), np.abs(b - r))
            self.luma = frame.max(axis=2)
        else:
            self.hsv = None
            self.spread = None
            self.luma = gray


def _hysteresis(
    magnitude: np.ndarray, floor: float, *, high_k: float = 4.0, low_k: float = 1.6
) -> np.ndarray:
    """Keep faint pixels only where they join a confident core.

    The first version of this used one fixed threshold and cost us the entire
    top of every plume: a column is dense near its base and fades to two or
    three counts of difference by the time it clears the ridge. A threshold high
    enough to ignore sensor noise cut the plume off at the shoulders, the tracked
    box stopped growing upward, and the rise term — the single most diagnostic
    measurement in the product — read zero on real smoke.

    So this is Canny's trick applied to a difference image. Seeds come from a
    high threshold; a connected component is kept whole if it contains one. The
    thresholds are set from the frame's own robust noise level (a median
    absolute deviation, which a plume covering a few percent of the frame cannot
    shift) rather than fixed numbers, so a noisy camera at dusk and a clean one
    at noon get the same treatment in units that mean something.
    """
    sigma = float(np.median(np.abs(magnitude - np.median(magnitude)))) * 1.4826
    sigma = max(sigma, 0.6)
    high = max(floor, high_k * sigma)
    low = max(floor * 0.45, low_k * sigma)

    weak = (magnitude > low).astype(np.uint8)
    if not weak.any():
        return np.zeros(magnitude.shape, np.uint8)
    count, labels = cv2.connectedComponents(weak, connectivity=8)
    if count <= 1:
        return np.zeros(magnitude.shape, np.uint8)
    seeded = np.zeros(count, dtype=bool)
    seeds = labels[magnitude > high]
    if seeds.size:
        seeded[np.unique(seeds)] = True
    seeded[0] = False  # label 0 is the background
    return (seeded[labels]).astype(np.uint8) * 255


def _measure(region: Region, ctx: _FrameContext, horizon: Horizon) -> None:
    """Measure one region, working inside its own bounding box.

    Everything here used to index full-frame boolean masks, so measuring a
    300-pixel region read 786,000 pixels a dozen times over. Cropping to the
    bounding box first gives identical numbers — the mask is zero outside the
    box by construction — for a fraction of the memory traffic.
    """
    h, w = ctx.gray.shape[:2]
    pad = 24
    x0 = max(0, region.x - pad)
    y0 = max(0, region.y - pad)
    x1 = min(w, region.x + region.w + pad)
    y1 = min(h, region.y + region.h + pad)
    box = (slice(y0, y1), slice(x0, x1))

    mask = region.mask[box] > 0
    n = int(mask.sum())
    if n == 0:
        return
    outside = ~mask  # the padded ring around the region, in the same crop

    region.area_fraction = n / float(h * w)
    region.elongation = region.h / max(region.w, 1)
    region.fill = n / float(max(region.w * region.h, 1))
    region.mean_signed = float(ctx.change.signed[box][mask].mean())

    # Texture, measured two ways, because each fails differently.
    #
    # Against the background reference: exactly the right question — has the
    # detail that was here gone? — but the reference is a running average, so it
    # is softer than any single frame and the ratio drifts above 1.
    #
    # Against the ring of pixels just outside the region in this same frame: no
    # reference needed and no temporal drift, but it is fooled when the region
    # happens to sit on a naturally smooth patch.
    #
    # Taking the lower of the two means a region has to look veiled on both
    # counts to score as veiled, which is the conservative direction.
    laplacian = ctx.laplacian[box]
    now = float(laplacian[mask].var())
    region.metrics["texture_now"] = round(now, 2)
    ratios = []
    if ctx.ref_laplacian is not None:
        was = float(ctx.ref_laplacian[box][mask].var())
        if was > 0.5:
            ratios.append(now / was)
    if outside.any():
        around = float(laplacian[outside].var())
        if around > 0.5:
            ratios.append(now / around)
            region.metrics["texture_surround"] = round(around, 2)
    region.texture_ratio = float(np.clip(min(ratios), 0.0, 3.0)) if ratios else 1.0

    # Colour. Smoke pulls what is behind it toward neutral grey.
    if ctx.hsv is not None:
        saturation = ctx.hsv[:, :, 1][box]
        sat_now = float(saturation[mask].mean())
        region.metrics["saturation_now"] = round(sat_now, 2)
        sat_around = float(saturation[outside].mean()) if outside.any() else sat_now
        region.saturation_ratio = sat_now / sat_around if sat_around > 1e-3 else 1.0
        region.greyness = float(
            1.0 - np.clip(ctx.spread[box][mask].mean() / 60.0, 0.0, 1.0)
        )
    else:
        region.saturation_ratio = 1.0
        region.greyness = 0.5  # unknown, not "grey"

    region.metrics["mean_luma"] = round(float(ctx.luma[box][mask].mean()), 1)
    region.metrics["blown_fraction"] = round(float(np.mean(ctx.luma[box][mask] > 250)), 4)

    # Edge softness, measured on the difference image rather than on the frame.
    #
    # The question is how many pixels the change takes to go from nothing to
    # full. A droplet on the glass or a misregistered ridge gets there in one or
    # two; a plume takes twenty. Comparing the gradient of the difference on the
    # region's boundary against the difference's own peak inside it gives that
    # directly, and in units that do not care how textured the scene is — which
    # the earlier version, normalised against the frame's own gradients, very
    # much did.
    boundary = cv2.morphologyEx(
        region.mask[box], cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ) > 0
    peak = float(np.percentile(ctx.signed[box][mask], 95))
    if boundary.any() and peak > 1.0:
        edge_slope = float(ctx.slope[box][boundary].mean())
        region.edge_softness = float(np.clip(1.0 - edge_slope / (0.6 * peak), 0.0, 1.0))
        region.metrics["edge_slope"] = round(edge_slope, 2)
        region.metrics["diff_peak"] = round(peak, 2)
    else:
        region.edge_softness = 0.0

    # Geometry against the ridge.
    columns = np.clip(np.arange(region.x, region.x + region.w), 0, len(horizon.boundary) - 1)
    local_horizon = float(np.median(horizon.boundary[columns])) if columns.size else h / 2
    region.base_below_horizon = float(region.base_y - local_horizon)
    region.top_above_horizon = float(local_horizon - region.top_y)
    rows, cols = np.where(mask)
    rows = rows + y0
    cols = cols + x0
    region.sky_fraction = float(np.mean(rows < local_horizon)) if rows.size else 0.0

    # The base point: the centroid of the lowest tenth of the mask.
    if rows.size:
        cutoff = np.percentile(rows, 90)
        low = rows >= cutoff
        region.base_cx = float(cols[low].mean())
        region.base_cy = float(rows[low].mean())
    else:
        region.base_cx, region.base_cy = region.cx, float(region.base_y)

    region.metrics["local_horizon_y"] = round(local_horizon, 1)
    if ctx.reference is not None:
        region.metrics["reference_luma"] = round(float(ctx.reference[box][mask].mean()), 2)


def _surround_mask(region: Region, shape: tuple[int, int], pad: int = 24) -> np.ndarray:
    """A ring just outside the region, for comparing colour against its neighbours."""
    h, w = shape
    ring = np.zeros((h, w), dtype=np.uint8)
    x0, y0 = max(0, region.x - pad), max(0, region.y - pad)
    x1, y1 = min(w, region.x + region.w + pad), min(h, region.y + region.h + pad)
    ring[y0:y1, x0:x1] = 255
    ring[region.mask > 0] = 0
    return ring > 0


def draw_regions(
    frame: np.ndarray,
    regions: list[Region],
    *,
    colour: tuple[int, int, int] = (26, 48, 180),
    labels: list[str] | None = None,
    thickness: int = 2,
) -> np.ndarray:
    """Overlay for the evidence panel. BGR in, BGR out."""
    canvas = frame.copy()
    overlay = canvas.copy()
    for region in regions:
        overlay[region.mask > 0] = colour
    cv2.addWeighted(overlay, 0.28, canvas, 0.72, 0, canvas)
    # OpenCV 5's TrueType text path. Note the argument order: the FontFace
    # overload is putText(img, text, org, colour, face, size, weight), which
    # swaps colour and font relative to the legacy Hershey overload. Passing the
    # old order raises "Argument 'fontFace' is required to be an integer", which
    # is a confusing way to be told the arguments are the wrong way round.
    face = cv2.FontFace("sans")
    for i, region in enumerate(regions):
        cv2.rectangle(canvas, (region.x, region.y), (region.x + region.w, region.y + region.h),
                      colour, thickness)
        if labels and i < len(labels):
            cv2.putText(canvas, labels[i], (region.x, max(16, region.y - 6)),
                        (255, 255, 255), face, 16, 600)
    return canvas


def draw_horizon(
    frame: np.ndarray, horizon: Horizon, colour: tuple[int, int, int] = (200, 170, 60)
) -> np.ndarray:
    canvas = frame.copy()
    points = np.stack(
        [np.arange(len(horizon.boundary)), np.clip(horizon.boundary, 0, frame.shape[0] - 1)], axis=1
    ).astype(np.int32)
    cv2.polylines(canvas, [points], False, colour, 2, cv2.LINE_AA)
    return canvas
