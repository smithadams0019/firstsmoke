"""One camera, watched over time, with a score that shows its working.

Everything upstream measures. This module is where measurements become a number
between zero and one, and it is deliberately the most boring file in the
project: a weighted sum of named terms, each between zero and one, each with a
sentence explaining what it saw. No hidden nonlinearity, no learned fusion, no
threshold buried three calls deep.

That choice costs a little accuracy. It buys the thing this product is actually
for: when the system wakes someone at two in the morning, the alert says *why*,
in terms a lookout can check against the picture, and when it stands down it
says why it stood down. A gradient-boosted fusion of forty features would score
a point or two better on the evaluation set and be unarguable in exactly the
situation where someone needs to argue with it.

The confidence is
    support x (1 - r1)(1 - r2)...
where ``support`` is the weighted evidence and each ``r`` is a rejection's
confidence, discounted slightly so that no single rule can zero a detection
outright.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

import cv2
import numpy as np

from .background import BackgroundModel
from .cameras import Camera
from .candidates import Region, extract_regions
from .confirm import Confirmation, SmokeConfirmer
from .frames import CameraFrame
from .geometry import bearing_from_pixel
from .rejectors import Rejection, apply_all, blind_reason
from .scene import Alignment, SceneState, assess_scene
from .tracks import Growth, Track, Tracker

SUSPECT_AT = 0.35
"""Below this a camera stays on watch and says nothing."""
CONFIRM_AT = 0.68
"""At or above this one camera is willing to assert a column on its own. Between
the two is the band the escalation loop exists to resolve."""

MINIMUM_FRAMES = 3


class Verdict(StrEnum):
    CLEAR = "clear"
    SUSPECT = "suspect"
    COLUMN = "column"
    BLIND = "blind"
    """The camera cannot see. This is not 'clear' and must never be read as it."""


@dataclass(frozen=True)
class Reason:
    """One named piece of positive evidence."""

    name: str
    value: float
    weight: float
    text: str

    @property
    def contribution(self) -> float:
        return self.value * self.weight

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": round(self.value, 3),
            "weight": round(self.weight, 3),
            "contribution": round(self.contribution, 4),
            "text": self.text,
        }


# The weights. They sum to 1.0 for a colour camera. On a monochrome imager the
# desaturation term is dropped and the rest are renormalised, rather than being
# scored as zero — a camera that cannot measure colour has not measured "no
# colour change", and scoring it as though it had would punish the mono cameras
# for their own hardware.
WEIGHTS = {
    "growth": 0.20,
    "rise": 0.16,
    "anchored": 0.16,
    "veiling": 0.14,
    "attachment": 0.12,
    "soft_edges": 0.08,
    "desaturation": 0.08,
    "persistence": 0.06,
}
LEARNED_WEIGHT = 0.12
"""Added on top and renormalised when the ONNX confirmer is available."""


@dataclass
class Detection:
    """One camera's opinion about one candidate at one moment."""

    camera_id: str
    frame_index: int
    timestamp: datetime
    track_id: int
    confidence: float
    support: float
    verdict: Verdict
    bearing_deg: float
    bearing_sigma_deg: float
    region: Region
    growth: Growth
    reasons: list[Reason] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)
    confirmation: Confirmation | None = None
    evidence: dict[str, str] = field(default_factory=dict)

    @property
    def stood_down(self) -> bool:
        return bool(self.rejections) and self.confidence < SUSPECT_AT

    def summary(self) -> str:
        """One sentence a person can read at a glance."""
        if self.verdict is Verdict.COLUMN:
            lead = "A column"
        elif self.verdict is Verdict.SUSPECT:
            lead = "Something"
        else:
            lead = "Nothing worth calling"
        top = sorted(self.reasons, key=lambda r: r.contribution, reverse=True)[:2]
        because = "; ".join(r.text for r in top) if top else "no positive evidence"
        if self.rejections:
            against = self.rejections[0].message
            return (
                f"{lead} at bearing {self.bearing_deg:.1f} degrees: {because}. "
                f"Against it: {against}."
            )
        return f"{lead} at bearing {self.bearing_deg:.1f} degrees: {because}."

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "frame_index": self.frame_index,
            "timestamp": self.timestamp.isoformat(),
            "track_id": self.track_id,
            "confidence": round(self.confidence, 4),
            "support": round(self.support, 4),
            "verdict": self.verdict.value,
            "bearing_deg": round(self.bearing_deg, 2),
            "bearing_sigma_deg": round(self.bearing_sigma_deg, 2),
            "summary": self.summary(),
            "region": self.region.to_dict(),
            "growth": self.growth.to_dict(),
            "reasons": [r.to_dict() for r in self.reasons],
            "rejections": [r.to_dict() for r in self.rejections],
            "confirmation": self.confirmation.to_dict() if self.confirmation else None,
            "evidence": dict(self.evidence),
        }


@dataclass
class CameraReading:
    """Everything one camera produced from one frame."""

    camera_id: str
    frame_index: int
    timestamp: datetime
    verdict: Verdict
    scene: SceneState
    alignment: Alignment
    detections: list[Detection] = field(default_factory=list)
    ms: float = 0.0
    warmed: bool = True

    @property
    def best(self) -> Detection | None:
        return max(self.detections, key=lambda d: d.confidence, default=None)

    @property
    def confidence(self) -> float:
        best = self.best
        return best.confidence if best else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "frame_index": self.frame_index,
            "timestamp": self.timestamp.isoformat(),
            "verdict": self.verdict.value,
            "confidence": round(self.confidence, 4),
            "usability": self.scene.usability.value,
            "usability_reason": self.scene.reason,
            "warmed": self.warmed,
            "ms": round(self.ms, 2),
            "scene": self.scene.to_dict(),
            "alignment": self.alignment.to_dict(),
            "detections": [d.to_dict() for d in self.detections],
        }


class CameraWatch:
    """The per-camera state machine's lower half: look at a frame, form an opinion.

    One instance per camera, fed frames in time order. It owns the background
    model, the tracker and the camera's own contrast baseline, because all three
    are only meaningful relative to that one camera's own history.
    """

    def __init__(
        self,
        camera: Camera,
        *,
        confirmer: SmokeConfirmer | None = None,
        minimum_frames: int = MINIMUM_FRAMES,
        suspect_at: float = SUSPECT_AT,
        confirm_at: float = CONFIRM_AT,
    ) -> None:
        self.camera = camera
        self.background = BackgroundModel(camera.camera_id)
        self.tracker = Tracker(camera.camera_id)
        self.confirmer = confirmer
        self.minimum_frames = minimum_frames
        self.suspect_at = suspect_at
        self.confirm_at = confirm_at
        self._previous: np.ndarray | None = None
        self._repeats = 0
        self._contrasts: list[float] = []
        self._readings: list[CameraReading] = []

    @property
    def readings(self) -> list[CameraReading]:
        return list(self._readings)

    @property
    def baseline_contrast(self) -> float | None:
        """This camera's own recent normal, so 'the scene went milky' is relative."""
        usable = self._contrasts[-12:]
        return float(np.median(usable)) if len(usable) >= 4 else None

    def observe(self, frame: CameraFrame, *, learn: bool = True) -> CameraReading:
        start = cv2.getTickCount()
        scene = assess_scene(
            frame.image,
            previous=self._previous,
            repeat_count=self._repeats,
            baseline_contrast=self.baseline_contrast,
        )
        self._repeats = scene.repeat_count
        self._previous = frame.image

        if scene.blind:
            reading = CameraReading(
                camera_id=self.camera.camera_id,
                frame_index=frame.index,
                timestamp=frame.timestamp,
                verdict=Verdict.BLIND,
                scene=scene,
                alignment=Alignment(0.0, 0.0, 1.0, frame.image, corrected=False),
                warmed=self.background.warmed,
                ms=_ms_since(start),
            )
            self._readings.append(reading)
            return reading

        self._contrasts.append(scene.contrast)
        change = self.background.update(frame.image, frame.hour_utc, learn=learn)

        if not change.warmed:
            # Do not extract, and above all do not start tracks. A background
            # model with two samples produces large, confident nonsense, and a
            # track seeded from it carries that nonsense forward for the rest of
            # the sequence with a respectable frame count behind it.
            reading = CameraReading(
                camera_id=self.camera.camera_id,
                frame_index=frame.index,
                timestamp=frame.timestamp,
                verdict=Verdict.CLEAR,
                scene=scene,
                alignment=change.alignment,
                warmed=False,
                ms=_ms_since(start),
            )
            self._readings.append(reading)
            return reading

        reference = self.background.reference_for(frame.hour_utc)
        regions = extract_regions(
            change.alignment.image,
            change,
            scene.horizon,
            reference=reference,
            has_colour=self.camera.has_colour,
            frame_index=frame.index,
        )
        touched = self.tracker.update(frame.index, frame.timestamp, regions)

        detections: list[Detection] = []
        for track in touched:
            if track.latest.frame_index != frame.index:
                continue
            detections.append(
                self._score(track, change.alignment.image, scene, change.alignment, frame)
            )
        detections.sort(key=lambda d: d.confidence, reverse=True)

        top = detections[0].confidence if detections else 0.0
        verdict = (
            Verdict.COLUMN if top >= self.confirm_at
            else Verdict.SUSPECT if top >= self.suspect_at
            else Verdict.CLEAR
        )

        reading = CameraReading(
            camera_id=self.camera.camera_id,
            frame_index=frame.index,
            timestamp=frame.timestamp,
            verdict=verdict,
            scene=scene,
            alignment=change.alignment,
            detections=detections,
            warmed=True,
            ms=_ms_since(start),
        )
        self._readings.append(reading)
        return reading

    # -- scoring -------------------------------------------------------------
    def _score(
        self,
        track: Track,
        frame_image: np.ndarray,
        scene: SceneState,
        alignment: Alignment,
        frame: CameraFrame,
    ) -> Detection:
        region = track.latest.region
        growth = track.growth()
        height, width = frame_image.shape[:2]

        reasons = self._reasons(growth, region, height)
        weights = dict(WEIGHTS)
        if not self.camera.has_colour:
            weights.pop("desaturation", None)

        confirmation: Confirmation | None = None
        if self.confirmer is not None:
            confirmation = self.confirmer.confirm(region.crop(frame_image))
            weights["learned"] = LEARNED_WEIGHT
            reasons.append(
                Reason(
                    "learned",
                    confirmation.probability,
                    LEARNED_WEIGHT,
                    f"the appearance classifier puts this crop at "
                    f"{confirmation.probability * 100:.0f}% smoke",
                )
            )

        total_weight = sum(weights.values())
        support = sum(
            r.value * weights[r.name] for r in reasons if r.name in weights
        ) / max(total_weight, 1e-6)

        rejections = apply_all(
            track, growth, region, frame_image, scene, self.camera, alignment,
            minimum_frames=self.minimum_frames,
            baseline_contrast=self.baseline_contrast,
        )
        confidence = support
        for rejection in rejections:
            confidence *= 1.0 - 0.9 * rejection.confidence

        bearing = bearing_from_pixel(
            region.cx, width, self.camera.azimuth_deg, self.camera.hfov_deg
        )
        sigma = self._bearing_sigma(region, width)

        verdict = (
            Verdict.COLUMN if confidence >= self.confirm_at
            else Verdict.SUSPECT if confidence >= self.suspect_at
            else Verdict.CLEAR
        )
        return Detection(
            camera_id=self.camera.camera_id,
            frame_index=frame.index,
            timestamp=frame.timestamp,
            track_id=track.track_id,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            support=float(np.clip(support, 0.0, 1.0)),
            verdict=verdict,
            bearing_deg=bearing,
            bearing_sigma_deg=sigma,
            region=region,
            growth=growth,
            reasons=[r for r in reasons if r.name in weights],
            rejections=rejections,
            confirmation=confirmation,
        )

    def _reasons(self, growth: Growth, region: Region, height: int) -> list[Reason]:
        """Each term scaled to 0..1, with the sentence that explains it."""
        growth_value = float(np.clip((growth.area_ratio - 1.0) / 2.0, 0.0, 1.0))
        growth_value = growth_value * (0.4 + 0.6 * growth.monotonic_fraction)

        # A plume rises at roughly 1 to 4 percent of frame height per minute at
        # these ranges. Saturating at 3% keeps a fast riser from swamping the sum.
        rise_value = float(np.clip(growth.top_rise_px_per_min / (0.03 * height), 0.0, 1.0))
        veiling = float(np.clip(1.0 - region.texture_ratio, 0.0, 1.0))
        desaturation = float(np.clip((1.0 - region.saturation_ratio) * 1.6, 0.0, 1.0))
        attachment = float(
            np.clip(1.0 - abs(region.base_below_horizon) / (0.12 * height), 0.0, 1.0)
        )
        persistence = float(np.clip((growth.frames - 1) / 5.0, 0.0, 1.0))

        return [
            Reason("growth", growth_value, WEIGHTS["growth"],
                   f"it grew from {growth.area_first} to {growth.area_last} px over "
                   f"{growth.span_s / 60:.0f} minutes, without shrinking in "
                   f"{growth.monotonic_fraction * 100:.0f}% of steps"),
            Reason("rise", rise_value, WEIGHTS["rise"],
                   f"its top edge climbed {growth.top_rise_px_per_min:.1f} px a minute"),
            Reason("anchored", growth.anchor_score, WEIGHTS["anchored"],
                   f"its base stayed put while its top moved (anchor {growth.anchor_score:.2f}, "
                   f"base drift {growth.base_drift_px_per_min:.1f} px a minute)"),
            Reason("veiling", veiling, WEIGHTS["veiling"],
                   f"the detail behind it dropped to {region.texture_ratio * 100:.0f}% of what the "
                   "background reference has there, which is what a translucent veil does"),
            Reason("attachment", attachment, WEIGHTS["attachment"],
                   f"its base sits {region.base_below_horizon:+.0f} px from the skyline, so it is "
                   "joined to the ground"),
            Reason("soft_edges", region.edge_softness, WEIGHTS["soft_edges"],
                   f"its boundary is soft (softness {region.edge_softness:.2f}), not a hard edge"),
            Reason("desaturation", desaturation, WEIGHTS["desaturation"],
                   f"it is {(1 - region.saturation_ratio) * 100:.0f}% less saturated than its "
                   "surroundings"),
            Reason("persistence", persistence, WEIGHTS["persistence"],
                   f"it has been there for {growth.frames} frames"),
        ]

    def _bearing_sigma(self, region: Region, width: int) -> float:
        """How well we know the bearing, in degrees.

        Two sources of error. The plume is wide, so its centroid column is only
        defined to about a quarter of its width; and the camera's own azimuth
        calibration is good to perhaps half a degree. A wide, diffuse plume
        therefore yields a soft bearing, and the map shows it as a fan rather
        than a line.
        """
        degrees_per_px = self.camera.hfov_deg / float(width)
        centroid_sigma = 0.25 * region.w * degrees_per_px
        mounting_sigma = 0.5
        return float(np.hypot(centroid_sigma, mounting_sigma))


def blind_detection_reason(scene: SceneState) -> Rejection | None:
    return blind_reason(scene)


def _ms_since(start: int) -> float:
    return (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000.0
