"""Per-camera background modelling, built for one frame a minute.

The textbook answer is `cv2.createBackgroundSubtractorMOG2` and it is the right
starting point, but a lookout camera breaks two of its assumptions.

**The sun moves between frames.** At one frame a minute the illumination changes
measurably from frame to frame and dramatically over an hour. MOG2's per-pixel
Gaussians absorb that eventually, but "eventually" is exactly the window in
which a fire starts. So before differencing we fit a **global gain and offset**
that maps the current frame onto the background's exposure, estimated by robust
regression on the pixels that did not change much. A sun-angle shift then
cancels out; a smoke column does not, because it is a small part of the frame
and the robust fit ignores it.

**Time of day repeats.** The shadow on a west-facing slope at 16:00 today looks
like the shadow at 16:00 yesterday and nothing like the one at 10:00. We keep a
small ring of references keyed by time-of-day bucket, so a camera coming back
from a gap compares against a like-for-like reference instead of learning from
scratch.

What comes out is a signed difference: positive where the frame got brighter
than its background. Smoke against terrain is almost always brighter, smoke
against bright sky is darker, and the sign is evidence rather than noise, so we
keep both and let the candidate stage decide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .scene import Alignment, align_to

TOD_BUCKETS = 12
"""Two-hour buckets. Finer buckets need more history than a demo sequence has;
coarser ones put dawn and noon in the same box."""

HISTORY = 24
MOG2_VAR_THRESHOLD = 20.0


def tod_bucket(hour_utc: float) -> int:
    return int(hour_utc / (24.0 / TOD_BUCKETS)) % TOD_BUCKETS


@dataclass
class ChangeMap:
    """What the background model thinks changed."""

    signed: np.ndarray
    """float32, current minus photometrically aligned background."""
    foreground: np.ndarray
    """uint8 0/255 from MOG2, after shadow removal and morphological cleanup."""
    gain: float
    offset: float
    residual: float
    """Robust residual of the photometric fit. High means the exposure model failed."""
    alignment: Alignment
    warmed: bool
    """False until the model has seen enough frames to be worth believing."""
    frames_seen: int
    bucket: int
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def brighter(self) -> np.ndarray:
        return np.clip(self.signed, 0, None)

    @property
    def darker(self) -> np.ndarray:
        return np.clip(-self.signed, 0, None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gain": round(self.gain, 4),
            "offset": round(self.offset, 2),
            "residual": round(self.residual, 3),
            "warmed": self.warmed,
            "frames_seen": self.frames_seen,
            "tod_bucket": self.bucket,
            "alignment": self.alignment.to_dict(),
            "foreground_fraction": round(float(np.mean(self.foreground > 0)), 5),
            **self.metrics,
        }


def photometric_fit(
    current: np.ndarray, background: np.ndarray, *, iterations: int = 3
) -> tuple[float, float, float]:
    """Least-squares gain and offset from background to current, ignoring outliers.

    Solves ``current ~ gain * background + offset`` on a subsample, then re-fits
    twice while dropping the worst 20% of residuals. The dropped pixels are the
    ones that actually changed — which is the point: we want the exposure model
    fitted to the scene that stayed the same.
    """
    bg = background[::4, ::4].astype(np.float32).ravel()
    cur = current[::4, ::4].astype(np.float32).ravel()
    if bg.size < 64:
        return 1.0, 0.0, 0.0

    keep = np.ones(bg.size, dtype=bool)
    gain, offset = 1.0, 0.0
    for _ in range(iterations):
        b, c = bg[keep], cur[keep]
        if b.size < 32:
            break
        var = float(b.var())
        if var < 1e-3:
            gain, offset = 1.0, float(c.mean() - b.mean())
            break
        gain = float(((b - b.mean()) * (c - c.mean())).mean() / var)
        gain = float(np.clip(gain, 0.4, 2.5))
        offset = float(c.mean() - gain * b.mean())
        residuals = np.abs(cur - (gain * bg + offset))
        cutoff = float(np.percentile(residuals, 80))
        keep = residuals <= max(cutoff, 1.0)

    final = np.abs(cur[keep] - (gain * bg[keep] + offset))
    return gain, offset, float(final.mean()) if final.size else 0.0


SHADING_SIGMA_PX = 96.0
SHADING_DOWNSCALE = 16
SHADING_CLAMP = 10.0


def _low_frequency(residual: np.ndarray) -> np.ndarray:
    """The part of the difference that is a shading change, not an object.

    A single global gain and offset cannot describe what the sun does to a
    hillside. As the sun swings west, one aspect brightens and the opposite one
    darkens, so after the global fit there is a smooth residual field across the
    frame, tens of pixels wide and a few counts deep. It was producing detections
    that scored around 0.5 on the evaluation set: real changes in the image, and
    nothing to do with fire.

    A plume is compact. Blurring the residual with a 96 px kernel leaves a
    shading gradient almost untouched and spreads a 3,000 px plume at 25 counts
    into well under one count, so subtracting the blurred field removes the
    shading and leaves the plume. The correction is clamped so that a genuinely
    large event cannot subtract itself away.

    The blur is done on a sixteenth-scale copy. A sigma of 96 px on a full
    1024 px frame is a 577-tap kernel and cost 1.4 seconds a frame, which was
    97% of the entire pipeline's time; the same field computed at 64 px wide and
    bilinearly resampled is visually identical, because a field this smooth has
    no detail for the downscale to lose, and it costs a fraction of a
    millisecond.
    """
    h, w = residual.shape[:2]
    small_w = max(16, w // SHADING_DOWNSCALE)
    small_h = max(16, h // SHADING_DOWNSCALE)
    small = cv2.resize(residual, (small_w, small_h), interpolation=cv2.INTER_AREA)
    smooth = cv2.GaussianBlur(small, (0, 0), SHADING_SIGMA_PX / SHADING_DOWNSCALE)
    field = cv2.resize(smooth, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(field, -SHADING_CLAMP, SHADING_CLAMP)


class BackgroundModel:
    """One per camera. Feed it frames in time order.

    Holds a MOG2 subtractor for the foreground mask and, separately, a set of
    time-of-day reference images for the photometric difference. They answer
    different questions — MOG2 asks "is this pixel unusual for this pixel", the
    reference asks "how much brighter is this patch than it should be" — and the
    candidate stage wants both.
    """

    def __init__(
        self,
        camera_id: str,
        *,
        history: int = HISTORY,
        var_threshold: float = MOG2_VAR_THRESHOLD,
        warmup: int = 5,
    ) -> None:
        self.camera_id = camera_id
        self.warmup = warmup
        self.frames_seen = 0
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=True
        )
        self._references: dict[int, np.ndarray] = {}
        self._reference_counts: dict[int, int] = {}
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._last_frame: np.ndarray | None = None

    # -- state ---------------------------------------------------------------
    @property
    def warmed(self) -> bool:
        return self.frames_seen >= self.warmup

    def reference_for(self, hour_utc: float) -> np.ndarray | None:
        """The stored reference for this time of day, or the nearest one we have."""
        bucket = tod_bucket(hour_utc)
        if bucket in self._references:
            return self._references[bucket]
        if not self._references:
            return None
        nearest = min(
            self._references,
            key=lambda b: min(abs(b - bucket), TOD_BUCKETS - abs(b - bucket)),
        )
        return self._references[nearest]

    def snapshot(self) -> np.ndarray | None:
        """The current background image, for showing a judge what 'normal' looks like."""
        bg = self._mog2.getBackgroundImage()
        return bg if bg is not None and bg.size else None

    # -- the work ------------------------------------------------------------
    def update(
        self,
        image: np.ndarray,
        hour_utc: float,
        *,
        learning_rate: float = -1.0,
        stabilise: bool = True,
        learn: bool = True,
    ) -> ChangeMap:
        """Take one frame, return what changed.

        ``learn=False`` runs the comparison without teaching the model, which is
        what you want once a detection is live: a smoke column left to soak into
        the background disappears in about ten frames, and then the camera
        reports all clear while the hillside burns.
        """
        bucket = tod_bucket(hour_utc)
        alignment = (
            align_to(image, self._last_frame)
            if stabilise and self._last_frame is not None
            else Alignment(0.0, 0.0, 1.0, image, corrected=False)
        )
        frame = alignment.image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        reference = self.reference_for(hour_utc)
        if reference is None:
            gain, offset, residual = 1.0, 0.0, 0.0
            signed = np.zeros(gray.shape, dtype=np.float32)
        else:
            gain, offset, residual = photometric_fit(gray, reference)
            predicted = cv2.convertScaleAbs(reference, alpha=gain, beta=offset)
            signed = gray.astype(np.float32) - predicted.astype(np.float32)
            signed = signed - _low_frequency(signed)

        mask = self._mog2.apply(frame, learningRate=learning_rate if learn else 0.0)
        # MOG2 marks shadows 127 and foreground 255. A shadow is a sun going behind
        # a cloud, which is exactly the thing we do not want to alarm on.
        foreground = np.where(mask == 255, 255, 0).astype(np.uint8)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, self._kernel, iterations=1)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, self._kernel, iterations=2)

        if learn:
            self._absorb(bucket, gray)
            self._last_frame = frame
            self.frames_seen += 1

        return ChangeMap(
            signed=signed,
            foreground=foreground,
            gain=gain,
            offset=offset,
            residual=residual,
            alignment=alignment,
            warmed=self.warmed,
            frames_seen=self.frames_seen,
            bucket=bucket,
            metrics={"shadow_fraction": round(float(np.mean(mask == 127)), 5)},
        )

    def _absorb(self, bucket: int, gray: np.ndarray) -> None:
        """Roll the time-of-day reference toward the new frame.

        The weight falls off as 1/n up to a floor of 0.2, so the first few frames
        establish the reference quickly and later frames nudge it. The floor
        matters: a reference that stops moving cannot follow a season.
        """
        existing = self._references.get(bucket)
        if existing is None or existing.shape != gray.shape:
            self._references[bucket] = gray.astype(np.float32).copy()
            self._reference_counts[bucket] = 1
            return
        count = self._reference_counts[bucket] + 1
        self._reference_counts[bucket] = count
        weight = max(1.0 / count, 0.2)
        cv2.accumulateWeighted(gray.astype(np.float32), self._references[bucket], weight)

    def prime(self, images: list[np.ndarray], hours: list[float]) -> None:
        """Feed a run of known-clean frames before the sequence under test."""
        for image, hour in zip(images, hours, strict=True):
            self.update(image, hour)


class NetworkBackground:
    """One :class:`BackgroundModel` per camera, created on first sight."""

    def __init__(self, **kwargs: Any) -> None:
        self._models: dict[str, BackgroundModel] = {}
        self._kwargs = kwargs

    def __contains__(self, camera_id: object) -> bool:
        return camera_id in self._models

    def for_camera(self, camera_id: str) -> BackgroundModel:
        model = self._models.get(camera_id)
        if model is None:
            model = BackgroundModel(camera_id, **self._kwargs)
            self._models[camera_id] = model
        return model

    def reset(self, camera_id: str | None = None) -> None:
        if camera_id is None:
            self._models.clear()
        else:
            self._models.pop(camera_id, None)
