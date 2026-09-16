"""What the camera can see, and where the sky stops.

Two jobs live here, and they run before any detection does.

**Where the sky stops.** A smoke column that matters starts at the ground and
rises into the sky, so the horizon is where we look hardest. It is also where
every false positive lives: a ridge line is the highest-contrast edge in the
frame, so a one-pixel camera nudge lights the whole ridge up in a difference
image. We therefore recover a per-column sky boundary rather than a single line,
because a mountain camera looks at a ridge and a ridge is not a line.

**Whether the camera can see at all.** A lookout that reports "no smoke" from a
fogged lens is worse than one that says nothing, because the silence is read as
evidence. Every camera carries a usability verdict on every frame, and an
unusable camera is excluded from the network's coverage rather than counted as a
clean look.

Method note: the sky boundary uses the threshold-search idea from the classical
sky-region literature — sweep a gradient threshold, take the first strong
gradient down each column as the boundary, and keep the threshold that best
separates the two regions. It needs no training data and no model file, which
matters for a camera we have never seen before.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# sky and horizon
# --------------------------------------------------------------------------- #
HORIZON_SEARCH_WIDTH = 256
"""The boundary is found at this width and scaled up. A ridge is a low-frequency
shape; finding it at full resolution costs 12x more and moves the answer by
about a pixel."""


@dataclass
class Horizon:
    """The per-column sky boundary, in working-image pixels."""

    boundary: np.ndarray
    """One row index per image column. Height means 'no sky in this column'."""
    confidence: float
    """0 to 1. How crisply the boundary separated the two regions."""
    sky_fraction: float
    threshold: int
    """The gradient threshold that won the sweep, for the record."""
    flat: bool = False
    """True when no boundary was found and we fell back to a fixed split."""

    def sky_mask(self, shape: tuple[int, int], margin_px: int = 0) -> np.ndarray:
        """Binary mask of everything above the boundary."""
        h, _w = shape
        rows = np.arange(h, dtype=np.int32)[:, None]
        limit = (self.boundary + margin_px)[None, :]
        return ((rows < limit).astype(np.uint8)) * 255

    def band_mask(self, shape: tuple[int, int], above_px: int, below_px: int) -> np.ndarray:
        """The strip around the boundary, where a new column first appears."""
        h, _w = shape
        rows = np.arange(h, dtype=np.int32)[:, None]
        b = self.boundary[None, :]
        return (((rows >= b - above_px) & (rows <= b + below_px)).astype(np.uint8)) * 255

    def height_above(self, x: float, y: float) -> float:
        """How far above the boundary a point sits, in pixels. Negative below."""
        index = int(np.clip(round(x), 0, len(self.boundary) - 1))
        return float(self.boundary[index] - y)

    def to_dict(self) -> dict[str, Any]:
        return {
            "confidence": round(self.confidence, 3),
            "sky_fraction": round(self.sky_fraction, 3),
            "threshold": self.threshold,
            "flat": self.flat,
            "median_row": int(np.median(self.boundary)),
            "relief_px": int(np.percentile(self.boundary, 95) - np.percentile(self.boundary, 5)),
        }


def find_horizon(
    image: np.ndarray, *, thresholds: tuple[int, ...] = (6, 10, 16, 24, 36, 52)
) -> Horizon:
    """Recover the sky boundary by sweeping a gradient threshold.

    For each candidate threshold we walk down every column and take the first row
    whose gradient magnitude exceeds it. That gives a boundary; we score it by how
    well the two regions separate in intensity, divided by how ragged the boundary
    is. Smoke, cloud and haze all soften the ridge, so the winning threshold also
    tells us how confident to be.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    scale = HORIZON_SEARCH_WIDTH / float(w)
    small = cv2.resize(
        gray, (HORIZON_SEARCH_WIDTH, max(8, int(h * scale))), interpolation=cv2.INTER_AREA
    )
    small = cv2.GaussianBlur(small, (5, 5), 0)
    sh, sw = small.shape

    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    best: tuple[float, np.ndarray, int] | None = None
    for threshold in thresholds:
        strong = grad > threshold
        # argmax on a boolean gives the first True; columns with none give 0, so
        # mark those as "all sky" explicitly rather than "boundary at the top".
        first = np.argmax(strong, axis=0)
        none = ~strong.any(axis=0)
        first = first.astype(np.int32)
        first[none] = sh - 1
        smoothed = _smooth_boundary(first, sh)
        score = _boundary_score(small, smoothed, sh)
        if best is None or score > best[0]:
            best = (score, smoothed, threshold)

    assert best is not None
    score, boundary_small, threshold = best

    # A boundary pinned to the bottom in most columns means we never found a
    # ridge: a fogged frame, a wall, or a camera pointed at the ground.
    pinned = float(np.mean(boundary_small >= sh - 2))
    flat = pinned > 0.6 or score < 0.08
    if flat:
        boundary_small = np.full(sw, int(sh * 0.55), dtype=np.int32)

    boundary = np.interp(
        np.linspace(0, sw - 1, w), np.arange(sw), boundary_small.astype(np.float64)
    )
    boundary = np.clip(np.round(boundary / max(scale, 1e-6)), 0, h - 1).astype(np.int32)
    if not flat:
        boundary = _refine(gray, boundary)
    sky_fraction = float(np.mean(boundary) / h)
    return Horizon(
        boundary=boundary,
        confidence=0.0 if flat else float(np.clip(score * 2.5, 0.0, 1.0)),
        sky_fraction=sky_fraction,
        threshold=int(threshold),
        flat=flat,
    )


REFINE_WINDOW_PX = 14


def _refine(gray: np.ndarray, coarse: np.ndarray) -> np.ndarray:
    """Pull the coarse boundary back onto the real ridge, at full resolution.

    The search runs at 256 px wide and the image is blurred before the Sobel, so
    the first-strong-gradient rule fires about two pixels early in the small
    image. Scaled back up that is eight pixels, and it is a *bias*, not noise:
    every column comes out the same distance high. Eight pixels matters here,
    because the attachment term asks how far a candidate's base sits from the
    skyline and a plume's base is often within twenty pixels of it.

    So each column's boundary is moved to the strongest vertical gradient within
    a small window of the coarse estimate, at native resolution. Columns where
    that window contains nothing convincing keep the coarse answer.
    """
    h, w = gray.shape[:2]
    gy = np.abs(cv2.Sobel(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_32F, 0, 1, ksize=3))
    refined = coarse.copy()
    lo = np.clip(coarse - REFINE_WINDOW_PX, 0, h - 1)
    hi = np.clip(coarse + REFINE_WINDOW_PX + 1, 1, h)
    for x in range(w):
        window = gy[lo[x] : hi[x], x]
        if window.size == 0:
            continue
        peak = int(np.argmax(window))
        if window[peak] > 1.5 * float(window.mean()):
            refined[x] = lo[x] + peak
    return _smooth_boundary(refined, h, window=7)


def _smooth_boundary(first: np.ndarray, height: int, window: int = 9) -> np.ndarray:
    """Median-filter the raw per-column boundary. A ridge is rough; noise is rougher."""
    padded = np.pad(first.astype(np.float32), window // 2, mode="edge")
    strides = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.clip(np.median(strides, axis=1), 0, height - 1).astype(np.int32)


def _boundary_score(gray: np.ndarray, boundary: np.ndarray, height: int) -> float:
    """Separation between the two regions, penalised by a ragged boundary."""
    rows = np.arange(height, dtype=np.int32)[:, None]
    sky = rows < boundary[None, :]
    ground = ~sky
    n_sky, n_ground = int(sky.sum()), int(ground.sum())
    if n_sky < 64 or n_ground < 64:
        return 0.0
    values = gray.astype(np.float32)
    mean_sky = float(values[sky].mean())
    mean_ground = float(values[ground].mean())
    spread = float(values.std()) + 1e-6
    separation = abs(mean_sky - mean_ground) / spread
    roughness = float(np.mean(np.abs(np.diff(boundary)))) / max(height * 0.05, 1.0)
    return float(separation / (1.0 + roughness))


# --------------------------------------------------------------------------- #
# usability
# --------------------------------------------------------------------------- #
class Usability(StrEnum):
    """Can this camera be believed right now?"""

    USABLE = "usable"
    DEGRADED = "degraded"
    """Seeing something, but not well. Detections are reported with the reason attached."""
    NIGHT = "night"
    FOG = "fog"
    LENS_OBSCURED = "lens_obscured"
    """Rain, spray, dust or dirt on the glass."""
    GLARE = "glare"
    FROZEN = "frozen"
    """The feed is repeating the same picture. The camera is up; the world is not."""

    @property
    def blind(self) -> bool:
        return self not in (Usability.USABLE, Usability.DEGRADED)


USABILITY_TEXT = {
    Usability.USABLE: "clear",
    Usability.DEGRADED: "degraded, still watching",
    Usability.NIGHT: "dark, no usable illumination",
    Usability.FOG: "fog or heavy haze, the ridge is not visible",
    Usability.LENS_OBSCURED: "something on the lens",
    Usability.GLARE: "sun in the frame",
    Usability.FROZEN: "feed frozen, the same picture is repeating",
}


@dataclass
class SceneState:
    """What one frame looks like before anyone asks about smoke."""

    usability: Usability
    reason: str
    mean_luma: float
    contrast: float
    """Laplacian variance, the standard blur/contrast proxy."""
    dark_channel: float
    """Mean of the per-pixel minimum across colour channels. High means milky."""
    saturated_fraction: float
    horizon: Horizon
    repeat_count: int = 0
    """Consecutive frames identical to this one."""
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def blind(self) -> bool:
        return self.usability.blind

    def to_dict(self) -> dict[str, Any]:
        return {
            "usability": self.usability.value,
            "reason": self.reason,
            "mean_luma": round(self.mean_luma, 2),
            "contrast": round(self.contrast, 2),
            "dark_channel": round(self.dark_channel, 2),
            "saturated_fraction": round(self.saturated_fraction, 4),
            "repeat_count": self.repeat_count,
            "horizon": self.horizon.to_dict(),
            **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.metrics.items()},
        }


# Thresholds.
#
# These were set from measurements, not taste, and the measurements are worth
# stating because they are the reason the first set was wrong.
#
# Laplacian variance ("contrast") on real HPWREN frames runs from about 1,080 on
# a hazy south-facing view to 9,590 on a sharp one. It is a property of the
# camera, the lens and the hillside, and it varies by nearly a factor of ten
# between two cameras on the same mast. Any absolute threshold is therefore a
# blunt instrument, and the real test is against the camera's own recent normal.
# The absolute floors below exist only for a camera with no history yet.
#
# Clipped-pixel fraction on the same real frames reaches 0.09 on a clear
# afternoon with the sun high, so a glare threshold anywhere near a few percent
# would declare a working camera blind every day at four o'clock.
NIGHT_MEAN_LUMA = 42.0
NIGHT_P98_LUMA = 110.0

FOG_ABSOLUTE_CONTRAST = 300.0
FOG_RELATIVE_CONTRAST = 0.25
FOG_DARK_CHANNEL = 140.0
"""Mean of the per-pixel channel minimum. Real clear frames measure 94 to 135;
fog and heavy haze push it past 145 because scattered light fills the darkest
channel everywhere."""

DEGRADED_ABSOLUTE_CONTRAST = 450.0
DEGRADED_RELATIVE_CONTRAST = 0.55

GLARE_FRACTION = 0.20
GLARE_MEAN_LUMA = 150.0

FROZEN_MAD = 0.35
"""Mean absolute difference below this between consecutive frames means the feed
is repeating. A live sensor always carries read noise, so a genuinely static
scene still measures around 1.0 to 2.5 here; an exactly repeated JPEG measures 0."""
FROZEN_REPEATS = 2


def assess_scene(
    image: np.ndarray,
    *,
    previous: np.ndarray | None = None,
    repeat_count: int = 0,
    horizon: Horizon | None = None,
    baseline_contrast: float | None = None,
) -> SceneState:
    """Decide whether this frame can be believed, and why.

    ``baseline_contrast`` is this camera's own median contrast over its recent
    frames. When it is available the fog and haze tests run against it; when it
    is not, they fall back to the absolute floors, which are deliberately
    generous so that a cold-started camera is not declared blind on frame one.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_luma = float(gray.mean())
    p98 = float(np.percentile(gray, 98))
    contrast = float(cv2.Laplacian(gray, cv2.CV_32F, ksize=3).var())
    dark = float(image.min(axis=2).mean())
    saturated = float(np.mean(gray > 250))
    horizon = horizon if horizon is not None else find_horizon(image)

    mad = None
    if previous is not None and previous.shape == image.shape:
        mad = float(cv2.absdiff(gray, cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)).mean())
        if mad < FROZEN_MAD:
            repeat_count += 1
        else:
            repeat_count = 0

    metrics: dict[str, Any] = {"p98_luma": p98}
    if mad is not None:
        metrics["frame_mad"] = mad
    if baseline_contrast:
        metrics["contrast_ratio"] = contrast / baseline_contrast

    def state(usability: Usability, reason: str) -> SceneState:
        return SceneState(
            usability, reason, mean_luma, contrast, dark, saturated, horizon, repeat_count, metrics
        )

    fog_contrast = (
        contrast < FOG_RELATIVE_CONTRAST * baseline_contrast
        if baseline_contrast
        else contrast < FOG_ABSOLUTE_CONTRAST
    )
    haze_contrast = (
        contrast < DEGRADED_RELATIVE_CONTRAST * baseline_contrast
        if baseline_contrast
        else contrast < DEGRADED_ABSOLUTE_CONTRAST
    )

    # Order matters. A frozen feed at night is frozen, because "it is dark" is a
    # condition that fixes itself at dawn and a dead feed is not.
    if repeat_count >= FROZEN_REPEATS:
        return state(
            Usability.FROZEN,
            f"the last {repeat_count + 1} frames are the same picture "
            f"(mean difference {mad:.2f} of 255)",
        )
    if mean_luma < NIGHT_MEAN_LUMA and p98 < NIGHT_P98_LUMA:
        return state(
            Usability.NIGHT,
            f"mean brightness {mean_luma:.0f} of 255 with no bright detail; this camera has no "
            "night capability",
        )
    if saturated > GLARE_FRACTION and mean_luma > GLARE_MEAN_LUMA:
        return state(
            Usability.GLARE,
            f"{saturated * 100:.0f}% of the frame is blown out; the sun is in shot",
        )
    if fog_contrast and dark > FOG_DARK_CHANNEL:
        return state(
            Usability.FOG,
            f"detail contrast is down to {contrast:.0f} with a milky dark channel of {dark:.0f}; "
            "the ridge is not visible",
        )
    if horizon.flat and haze_contrast:
        return state(
            Usability.LENS_OBSCURED,
            f"no ridge line anywhere in frame and detail contrast is only {contrast:.0f}; the "
            "glass is probably wet or dirty",
        )
    if haze_contrast or horizon.confidence < 0.25:
        return state(
            Usability.DEGRADED,
            f"haze is softening the scene (contrast {contrast:.0f}, horizon confidence "
            f"{horizon.confidence:.2f}); detections from this camera are reported with that noted",
        )
    return state(Usability.USABLE, USABILITY_TEXT[Usability.USABLE])


# --------------------------------------------------------------------------- #
# camera shake
# --------------------------------------------------------------------------- #
@dataclass
class Alignment:
    """How far the camera moved between two frames, and the corrected image."""

    dx: float
    dy: float
    response: float
    """Phase-correlation peak height. Low means the two frames do not match."""
    image: np.ndarray
    corrected: bool

    @property
    def shift_px(self) -> float:
        return float(np.hypot(self.dx, self.dy))

    def to_dict(self) -> dict[str, Any]:
        return {
            "dx": round(self.dx, 2),
            "dy": round(self.dy, 2),
            "shift_px": round(self.shift_px, 2),
            "response": round(self.response, 4),
            "corrected": self.corrected,
        }


MAX_CORRECTABLE_SHIFT_PX = 24.0
MIN_ALIGNMENT_RESPONSE = 0.10


def align_to(image: np.ndarray, reference: np.ndarray) -> Alignment:
    """Undo camera shake with phase correlation.

    A mast-mounted camera in wind moves by a few pixels between frames. Left
    uncorrected, every ridge edge in the scene appears in the difference image and
    the detector spends its whole confidence budget rejecting them. Phase
    correlation recovers a global translation in about 3 ms at 1024 px, which is
    the cheapest useful thing in this pipeline.

    Rotation is not corrected. A pole that twists is a camera that needs a
    service visit, and the alignment response drops far enough to say so.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
    window = cv2.createHanningWindow((gray.shape[1], gray.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(ref, gray, window)

    shift = float(np.hypot(dx, dy))
    if response < MIN_ALIGNMENT_RESPONSE or shift > MAX_CORRECTABLE_SHIFT_PX or shift < 0.25:
        return Alignment(dx, dy, float(response), image, corrected=False)

    matrix = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], dtype=np.float32)
    warped = cv2.warpAffine(
        image, matrix, (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    return Alignment(dx, dy, float(response), warped, corrected=True)
