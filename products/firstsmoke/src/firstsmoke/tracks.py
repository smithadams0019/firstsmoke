"""Following a candidate across frames, and asking whether it behaves like a column.

This is the module that decides the product's central argument: **a single frame
cannot tell you whether a grey patch is smoke.** Cloud, fog bank, dust plume and
smoke all look alike in one still. What separates them is what they do over the
next ten minutes.

A smoke column from a new ignition:

* has a **base that does not move**, because the fire is on the ground and the
  ground is not going anywhere;
* has a **top that rises**, because hot gas is buoyant;
* **grows in area** roughly monotonically, because more fuel keeps burning;
* **widens with height**, because the plume entrains air as it climbs.

A cloud translates bodily: base, centroid and top all move together at the wind
speed, and the area stays about the same. A dust plume from a vehicle moves
along the ground with its base, and does not rise far. A fog bank arrives across
the whole frame at once and has no base at all.

Those are four different measurements, and this module makes all four, with no
thresholds applied. The thresholds live in :mod:`firstsmoke.detector`, in one
place, where they can be argued with.

OpenCV 5 note: the tracking here is our own. OpenCV 5 removed `TrackerCSRT`,
`TrackerKCF` and the whole `legacy` namespace from the main wheel, so there is
no built-in tracker to call. That turned out to be a good thing — appearance
trackers lock onto texture, and a smoke plume's defining property is that it has
almost none. Greedy overlap association on the segmentation masks is both
simpler and better suited.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from .candidates import Region

MATCH_IOU = 0.10
"""Deliberately loose. A growing plume overlaps its previous self by less than
you would expect: doubling in area with a fixed base gives an IoU near 0.5, and
a fast riser can drop to 0.15. Association also checks centroid distance, so a
low bar here does not mean promiscuous matching."""

MATCH_DISTANCE_FRACTION = 0.06
"""Centroid may move by this fraction of the frame width between frames."""

MAX_MISSES = 2
"""A plume can thin below the detection threshold for a frame without ceasing to
exist. Three consecutive misses ends the track."""


@dataclass
class Observation:
    """One region, at one moment, on one track."""

    frame_index: int
    timestamp: datetime
    region: Region

    @property
    def area(self) -> int:
        return self.region.area


@dataclass
class Growth:
    """How a track behaved over time. Measurements only; no verdict."""

    frames: int
    span_s: float
    area_first: int
    area_last: int
    area_slope_px_per_min: float
    area_ratio: float
    monotonic_fraction: float
    """Fraction of consecutive steps where the area did not shrink."""
    top_rise_px_per_min: float
    """Positive means the top edge climbed."""
    base_drift_px_per_min: float
    """Speed of the base of the region. Near zero for an anchored column."""
    centroid_drift_px_per_min: float
    horizontal_drift_px_per_min: float
    width_growth_px_per_min: float
    aspect_first: float
    aspect_last: float
    base_jitter_px: float
    """Median frame-to-frame wobble of the base. Zero means it is on the lens."""
    anchor_score: float
    """0 to 1. High when the top moves and the base does not — the signature of a
    column. Computed as ``rise / (rise + base drift)``, so a bodily translation
    where both move equally scores 0.5, and a cloud drifting with a still top
    scores near 0."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "span_s": round(self.span_s, 1),
            "area_first": self.area_first,
            "area_last": self.area_last,
            "area_slope_px_per_min": round(self.area_slope_px_per_min, 1),
            "area_ratio": round(self.area_ratio, 3),
            "monotonic_fraction": round(self.monotonic_fraction, 3),
            "top_rise_px_per_min": round(self.top_rise_px_per_min, 2),
            "base_drift_px_per_min": round(self.base_drift_px_per_min, 2),
            "centroid_drift_px_per_min": round(self.centroid_drift_px_per_min, 2),
            "horizontal_drift_px_per_min": round(self.horizontal_drift_px_per_min, 2),
            "width_growth_px_per_min": round(self.width_growth_px_per_min, 2),
            "aspect_first": round(self.aspect_first, 3),
            "aspect_last": round(self.aspect_last, 3),
            "base_jitter_px": round(self.base_jitter_px, 2),
            "anchor_score": round(self.anchor_score, 3),
        }


@dataclass
class Track:
    """One candidate followed across frames."""

    track_id: int
    camera_id: str
    observations: list[Observation] = field(default_factory=list)
    misses: int = 0
    closed: bool = False

    def __len__(self) -> int:
        return len(self.observations)

    @property
    def latest(self) -> Observation:
        return self.observations[-1]

    @property
    def first(self) -> Observation:
        return self.observations[0]

    @property
    def alive(self) -> bool:
        return not self.closed and self.misses <= MAX_MISSES

    def add(self, frame_index: int, timestamp: datetime, region: Region) -> None:
        """Record an observation, keeping the list in time order.

        Order matters because the agent deliberately replays *earlier* frames
        when it re-examines a suspect camera. Appending those blindly put the
        oldest observation last, and every rate in :meth:`growth` is measured
        against ``observations[0]``, so the whole track reported a negative time
        span and growth rates with the sign inverted. An insert keeps the
        re-examination doing what it was meant to do, which is give the growth
        analysis more history rather than corrupt it.
        """
        observation = Observation(frame_index, timestamp, region)
        if self.observations and timestamp < self.observations[-1].timestamp:
            if any(o.frame_index == frame_index for o in self.observations):
                return  # already seen this frame; a replay must not double-count
            stamps = [o.timestamp for o in self.observations]
            self.observations.insert(bisect_left(stamps, timestamp), observation)
        else:
            self.observations.append(observation)
        self.misses = 0

    def miss(self) -> None:
        self.misses += 1
        if self.misses > MAX_MISSES:
            self.closed = True

    def growth(self) -> Growth:
        """Measure the track. Safe on a single observation; everything reads zero."""
        obs = self.observations
        if len(obs) < 2:
            r = obs[0].region if obs else None
            return Growth(
                frames=len(obs), span_s=0.0,
                area_first=r.area if r else 0, area_last=r.area if r else 0,
                area_slope_px_per_min=0.0, area_ratio=1.0, monotonic_fraction=1.0,
                top_rise_px_per_min=0.0, base_drift_px_per_min=0.0,
                centroid_drift_px_per_min=0.0, horizontal_drift_px_per_min=0.0,
                width_growth_px_per_min=0.0,
                aspect_first=r.elongation if r else 1.0, aspect_last=r.elongation if r else 1.0,
                base_jitter_px=0.0, anchor_score=0.0,
            )

        minutes = np.array(
            [(o.timestamp - obs[0].timestamp).total_seconds() / 60.0 for o in obs], dtype=np.float64
        )
        span = float(minutes[-1])
        areas = np.array([float(o.region.area) for o in obs])
        tops = np.array([float(o.region.top_y) for o in obs])
        # The base point, not the bounding box. See Region.base_cx.
        bases_y = np.array([o.region.base_cy for o in obs])
        bases_x = np.array([o.region.base_cx for o in obs])
        cxs = np.array([o.region.cx for o in obs])
        cys = np.array([o.region.cy for o in obs])
        widths = np.array([float(o.region.w) for o in obs])

        area_slope = _slope(minutes, areas)
        top_rise = -_slope(minutes, tops)  # image y grows downward
        base_drift = _net_speed(minutes, bases_x, bases_y)
        centroid_drift = _net_speed(minutes, cxs, cys)
        base_jitter = _jitter(bases_x, bases_y)
        horizontal_drift = abs(_slope(minutes, cxs))
        width_growth = _slope(minutes, widths)

        steps = np.diff(areas)
        monotonic = float(np.mean(steps >= -0.02 * areas[:-1])) if steps.size else 1.0

        rise = max(top_rise, 0.0)
        anchor = rise / (rise + base_drift) if (rise + base_drift) > 1e-6 else 0.0

        return Growth(
            frames=len(obs),
            span_s=span * 60.0,
            area_first=int(areas[0]),
            area_last=int(areas[-1]),
            area_slope_px_per_min=area_slope,
            area_ratio=float(areas[-1] / max(areas[0], 1.0)),
            monotonic_fraction=monotonic,
            top_rise_px_per_min=top_rise,
            base_drift_px_per_min=base_drift,
            centroid_drift_px_per_min=centroid_drift,
            horizontal_drift_px_per_min=horizontal_drift,
            width_growth_px_per_min=width_growth,
            aspect_first=float(obs[0].region.elongation),
            aspect_last=float(obs[-1].region.elongation),
            base_jitter_px=base_jitter,
            anchor_score=float(anchor),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "camera_id": self.camera_id,
            "frames": [o.frame_index for o in self.observations],
            "first_frame": self.first.frame_index,
            "last_frame": self.latest.frame_index,
            "first_seen": self.first.timestamp.isoformat(),
            "last_seen": self.latest.timestamp.isoformat(),
            "growth": self.growth().to_dict(),
            "latest_region": self.latest.region.to_dict(),
        }


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope of y against x. Zero when x does not vary."""
    var = float(((x - x.mean()) ** 2).sum())
    if var < 1e-9:
        return 0.0
    return float(((x - x.mean()) * (y - y.mean())).sum() / var)


def _net_speed(minutes: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> float:
    """How far the point actually travelled, per minute, ignoring jitter.

    The first version of this summed the frame-to-frame path length, on the
    argument that a blob jittering back and forth has not stayed put. That was
    wrong, and it cost us real detections: a growing plume's mask breathes by
    five or ten pixels a frame at its lower edge as the hysteresis threshold
    catches and loses the thin skirt, so path length accumulated 20 px a minute
    on a base that had not moved at all, and the erratic-motion rule then threw
    away the smoke.

    So this compares the median position over the first third of the track with
    the median over the last third. Medians are unmoved by the breathing; a real
    translation still shows up in full.
    """
    n = len(minutes)
    if n < 2:
        return 0.0
    third = max(1, n // 3)
    x0, y0 = float(np.median(xs[:third])), float(np.median(ys[:third]))
    x1, y1 = float(np.median(xs[-third:])), float(np.median(ys[-third:]))
    t0 = float(np.median(minutes[:third]))
    t1 = float(np.median(minutes[-third:]))
    elapsed = t1 - t0
    if elapsed <= 1e-6:
        elapsed = float(minutes[-1] - minutes[0])
    return float(np.hypot(x1 - x0, y1 - y0) / elapsed) if elapsed > 1e-6 else 0.0


def _jitter(xs: np.ndarray, ys: np.ndarray) -> float:
    """Median frame-to-frame wobble in pixels, after removing any steady drift.

    This is the quantity path length was accidentally measuring. Kept as its own
    number because it is genuinely diagnostic: a mask that breathes is a soft
    object, and a mask that does not move at all is something on the glass."""
    if len(xs) < 3:
        return 0.0
    steps = np.hypot(np.diff(xs), np.diff(ys))
    return float(np.median(steps))


class Tracker:
    """Greedy overlap association, one instance per camera."""

    def __init__(self, camera_id: str, frame_width: int = 1024) -> None:
        self.camera_id = camera_id
        self.frame_width = frame_width
        self.tracks: list[Track] = []
        self._next_id = 1

    @property
    def live(self) -> list[Track]:
        return [t for t in self.tracks if t.alive]

    def update(self, frame_index: int, timestamp: datetime, regions: list[Region]) -> list[Track]:
        """Associate this frame's regions with the live tracks. Returns the updated tracks."""
        live = self.live
        max_distance = self.frame_width * MATCH_DISTANCE_FRACTION

        pairs: list[tuple[float, Track, Region]] = []
        for track in live:
            previous = track.latest.region
            for region in regions:
                iou = previous.iou(region)
                distance = math.hypot(previous.cx - region.cx, previous.cy - region.cy)
                if iou >= MATCH_IOU or distance <= max_distance:
                    # Score prefers overlap but lets proximity rescue a plume that
                    # grew so fast its boxes barely intersect.
                    score = iou + max(0.0, 1.0 - distance / max(max_distance, 1e-6))
                    pairs.append((score, track, region))

        pairs.sort(key=lambda p: p[0], reverse=True)
        used_tracks: set[int] = set()
        used_regions: set[int] = set()
        touched: list[Track] = []
        for _score, track, region in pairs:
            if track.track_id in used_tracks or id(region) in used_regions:
                continue
            track.add(frame_index, timestamp, region)
            used_tracks.add(track.track_id)
            used_regions.add(id(region))
            touched.append(track)

        for track in live:
            if track.track_id not in used_tracks:
                track.miss()

        for region in regions:
            if id(region) not in used_regions:
                track = Track(self._next_id, self.camera_id)
                self._next_id += 1
                track.add(frame_index, timestamp, region)
                self.tracks.append(track)
                touched.append(track)

        return touched

    def best(self, key) -> Track | None:
        candidates = [t for t in self.tracks if t.observations]
        return max(candidates, key=key) if candidates else None
