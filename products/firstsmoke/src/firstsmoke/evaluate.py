"""Measuring the detector against recorded fires, with the ground truth we have.

The evaluation set is HPWREN's FIgLib. Each sequence is a run of stills from one
real camera around one real ignition, and the filenames carry the label:

    <unix_epoch>_<offset_seconds>.jpg

where the offset is signed seconds from the moment a human first saw the plume.
Frames before zero are labelled clear; frames from zero onward are labelled
smoke. That gives four numbers worth reporting and one that is not available.

**Detection rate.** The fraction of fires the detector reaches its threshold on
at any point after the plume became visible. This is a per-fire number, not a
per-frame one, because a fire you find four minutes late is found and a fire you
find on one frame in the middle and never again is not.

**Time to alert.** The offset of the first frame that crosses the threshold.
Negative means we crossed it before the human's mark, which happens and which we
report rather than clipping to zero — the human mark is one annotator's judgement
of when a plume became visible, not a physical ignition time, and treating it as
ground truth to the second would be overclaiming.

**False positives per camera-day.** Counted only on frames labelled clear, from
the same cameras, the same weather and the same hours as the positives. This is
the number that decides whether anyone keeps the system switched on, and it is
the number most easily flattered by choosing negatives from somewhere else.

**Confusion cases.** Which rejector fired, and on what.

What is *not* available: the coordinates of these fires. FIgLib does not publish
them, so the localisation accuracy cannot be measured against recorded data at
all. It is measured instead on synthetic incidents where we placed the fire
ourselves, and separately as a self-consistency check on the dates where several
cameras recorded the same fire — if two independent pairs of bearings cross in
the same place, the geometry is working, even though neither pair can be checked
against a surveyed point. Both are reported, and neither is called accuracy.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .calibration import (
    CalibrationSet,
    CameraCalibration,
    ClearFrameEvidence,
    combine,
    downscale_mask,
    empty_coverage,
)
from .cameras import Camera, Network, parse_sites_js
from .detector import SUSPECT_AT, CameraWatch, Verdict
from .frames import CameraFrame, Sequence_, prepare
from .geometry import CrossingRefused, Ray, cross_rays, haversine_m

FIGLIB_CACHE = Path.home() / ".cache" / "firstsmoke" / "figlib"
SWEEP_THRESHOLDS = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65)
FRAME_RE = re.compile(r"^(\d{10})_([+-]?\d+)\.jpg$")
SEQUENCE_RE = re.compile(r"^(\d{8})_FIRE_(.+)$")


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
@dataclass
class LabelledFrame:
    frame: CameraFrame
    offset_s: int

    @property
    def is_smoke(self) -> bool:
        return self.offset_s >= 0


def default_network() -> Network:
    """The real HPWREN geometry, from the metadata shipped in ``data/network``."""
    from .paths import network_file

    path = network_file()
    return Network.from_hpwren_sites(
        parse_sites_js(path.read_text(encoding="utf-8")),
        name="HPWREN",
        attribution="Imagery and camera metadata: HPWREN, UC San Diego. http://hpwren.ucsd.edu",
    )


def load_figlib_sequence(
    directory: Path, network: Network, *, stride: int = 1
) -> tuple[Sequence_, list[LabelledFrame]] | None:
    """Read one cached FIgLib sequence and join it to its camera's geometry."""
    match = SEQUENCE_RE.match(directory.name)
    if not match:
        return None
    camera_id = match.group(2)
    camera = (
        network.cameras.get(camera_id)
        or Camera(camera_id, camera_id.split("-")[0], camera_id, 0.0, 0.0, 0.0, 0.0, 90.0)
    )

    entries: list[tuple[int, int, Path]] = []
    for path in directory.glob("*.jpg"):
        file_match = FRAME_RE.match(path.name)
        if file_match:
            entries.append((int(file_match.group(1)), int(file_match.group(2)), path))
    if len(entries) < 8:
        return None
    entries.sort(key=lambda item: item[1])
    entries = entries[::stride]

    sequence = Sequence_(camera=camera)
    labelled: list[LabelledFrame] = []
    for index, (epoch, offset, path) in enumerate(entries):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        prepared, scale, original = prepare(image)
        frame = CameraFrame(
            camera_id=camera.camera_id,
            index=index,
            image=prepared,
            timestamp=datetime.fromtimestamp(epoch, UTC),
            source=path.name,
            scale=scale,
            original_shape=original,
        )
        sequence.frames.append(frame)
        labelled.append(LabelledFrame(frame, offset))
    return (sequence, labelled) if sequence.frames else None


# --------------------------------------------------------------------------- #
# per-sequence run
# --------------------------------------------------------------------------- #
@dataclass
class SequenceResult:
    sequence: str
    camera_id: str
    site_name: str
    frames: int
    negative_frames: int
    positive_frames: int
    detected: bool
    first_alert_offset_s: int | None
    peak_confidence: float
    peak_negative_confidence: float
    false_positive_frames: int
    negative_minutes: float
    unusable_frames: int
    unusable_reasons: Counter = field(default_factory=Counter)
    rejection_counts: Counter = field(default_factory=Counter)
    ms_per_frame: float = 0.0
    bearing_deg: float | None = None
    bearing_sigma_deg: float | None = None
    series: list[tuple[int, float, float]] = field(default_factory=list)
    """``(offset_seconds, confidence, bearing_degrees)`` for every frame judged.

    Keeping the whole series is what makes an operating curve honest and cheap:
    every threshold is a different reading of the same numbers, so the sweep
    costs one pass rather than one pass per threshold, and no threshold gets a
    slightly different run of the pipeline to be compared against.
    """
    first_bearings: dict[str, float] = field(default_factory=dict)
    """The bearing at the first crossing of each swept threshold."""
    evidence: ClearFrameEvidence | None = None
    """What this sequence's clear frames contribute to its camera's calibration.

    Never used to calibrate this sequence's own run. It is pooled with the
    camera's *other* sequences to score a different date."""
    calibrated: bool = False
    calibration_sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "camera_id": self.camera_id,
            "site_name": self.site_name,
            "frames": self.frames,
            "negative_frames": self.negative_frames,
            "positive_frames": self.positive_frames,
            "detected": self.detected,
            "first_alert_offset_s": self.first_alert_offset_s,
            "peak_confidence": round(self.peak_confidence, 4),
            "peak_negative_confidence": round(self.peak_negative_confidence, 4),
            "false_positive_frames": self.false_positive_frames,
            "negative_minutes": round(self.negative_minutes, 1),
            "unusable_frames": self.unusable_frames,
            "unusable_reasons": dict(self.unusable_reasons),
            "rejection_counts": dict(self.rejection_counts),
            "ms_per_frame": round(self.ms_per_frame, 1),
            "bearing_deg": None if self.bearing_deg is None else round(self.bearing_deg, 2),
            "series": [[o, c, b] for o, c, b in self.series],
            "calibrated": self.calibrated,
            "calibration_sources": list(self.calibration_sources),
        }


def run_sequence(
    sequence: Sequence_,
    labelled: list[LabelledFrame],
    *,
    threshold: float = SUSPECT_AT,
    confirmer=None,
    calibration: CameraCalibration | None = None,
) -> SequenceResult:
    """Watch one recorded sequence and score it against its labels.

    ``calibration`` must have been fitted from this camera's frames on *other*
    dates. The harness enforces that; nothing here checks it, so if you call
    this directly, check it yourself.
    """
    watch = CameraWatch(sequence.camera, confirmer=confirmer, calibration=calibration)
    first_alert: int | None = None
    peak = 0.0
    peak_negative = 0.0
    false_positives = 0
    unusable = 0
    unusable_reasons: Counter = Counter()
    rejections: Counter = Counter()
    durations: list[float] = []
    bearing = None
    bearing_sigma = None
    series: list[tuple[int, float, float]] = []
    first_bearings: dict[str, float] = {}
    coverage = empty_coverage()
    clear_confidences: list[float] = []
    clear_frames_seen = 0

    for item in labelled:
        reading = watch.observe(item.frame)
        durations.append(reading.ms)

        if reading.verdict is Verdict.BLIND:
            unusable += 1
            unusable_reasons[reading.scene.usability.value] += 1
            continue
        if not reading.warmed:
            continue

        best = reading.best
        if best is None:
            continue
        for rejection in best.rejections:
            rejections[rejection.code] += 1

        peak = max(peak, best.confidence)
        series.append((item.offset_s, round(best.confidence, 4), round(best.bearing_deg, 3)))

        if not item.is_smoke:
            # Everything a calibration is allowed to learn from, gathered here
            # and nowhere else: the clear frames only, every candidate region on
            # them, and the confidences they reached.
            clear_frames_seen += 1
            clear_confidences.append(round(best.confidence, 4))
            for detection in reading.detections:
                coverage += downscale_mask(detection.region.mask)
        if item.is_smoke:
            if best.confidence >= threshold and first_alert is None:
                first_alert = item.offset_s
                bearing = best.bearing_deg
                bearing_sigma = best.bearing_sigma_deg
            for level in SWEEP_THRESHOLDS:
                key = f"{level:.2f}"
                if key not in first_bearings and best.confidence >= level:
                    first_bearings[key] = best.bearing_deg
        else:
            peak_negative = max(peak_negative, best.confidence)
            if best.confidence >= threshold:
                false_positives += 1

    negatives = [item for item in labelled if not item.is_smoke]
    negative_minutes = 0.0
    if len(negatives) >= 2:
        negative_minutes = (negatives[-1].offset_s - negatives[0].offset_s) / 60.0

    return SequenceResult(
        sequence=sequence.camera.camera_id,
        camera_id=sequence.camera.camera_id,
        site_name=sequence.camera.site_name,
        frames=len(labelled),
        negative_frames=len(negatives),
        positive_frames=len(labelled) - len(negatives),
        detected=first_alert is not None,
        first_alert_offset_s=first_alert,
        peak_confidence=peak,
        peak_negative_confidence=peak_negative,
        false_positive_frames=false_positives,
        negative_minutes=abs(negative_minutes),
        unusable_frames=unusable,
        unusable_reasons=unusable_reasons,
        rejection_counts=rejections,
        ms_per_frame=float(statistics.mean(durations)) if durations else 0.0,
        bearing_deg=bearing,
        bearing_sigma_deg=bearing_sigma,
        series=series,
        first_bearings=first_bearings,
        evidence=ClearFrameEvidence(
            camera_id=sequence.camera.camera_id,
            sequence="",
            frames=clear_frames_seen,
            coverage=np.clip(coverage, 0.0, float(max(clear_frames_seen, 1))),
            confidences=clear_confidences,
        ),
        calibrated=calibration is not None and calibration.trustworthy,
        calibration_sources=list(calibration.sources) if calibration else [],
    )


# --------------------------------------------------------------------------- #
# the whole set
# --------------------------------------------------------------------------- #
@dataclass
class Evaluation:
    results: list[SequenceResult]
    threshold: float
    crossings: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def detection_rate(self) -> float:
        usable = [r for r in self.results if r.positive_frames > 0]
        return sum(r.detected for r in usable) / len(usable) if usable else 0.0

    @property
    def alert_offsets(self) -> list[int]:
        return [r.first_alert_offset_s for r in self.results if r.first_alert_offset_s is not None]

    @property
    def false_positives_per_camera_day(self) -> float:
        frames = sum(r.false_positive_frames for r in self.results)
        minutes = sum(r.negative_minutes for r in self.results)
        return frames / (minutes / 1440.0) if minutes else 0.0

    def summary(self) -> dict[str, Any]:
        offsets = self.alert_offsets
        rejections: Counter = Counter()
        unusable: Counter = Counter()
        for result in self.results:
            rejections.update(result.rejection_counts)
            unusable.update(result.unusable_reasons)
        timings = [r.ms_per_frame for r in self.results if r.ms_per_frame]
        return {
            "sequences": len(self.results),
            "frames": sum(r.frames for r in self.results),
            "threshold": self.threshold,
            "detection_rate": round(self.detection_rate, 4),
            "detected": sum(r.detected for r in self.results),
            "missed": sum(not r.detected for r in self.results),
            "time_to_alert_s": {
                "n": len(offsets),
                "median": int(statistics.median(offsets)) if offsets else None,
                "mean": round(statistics.mean(offsets), 1) if offsets else None,
                "p10": int(np.percentile(offsets, 10)) if offsets else None,
                "p90": int(np.percentile(offsets, 90)) if offsets else None,
                "min": min(offsets) if offsets else None,
                "max": max(offsets) if offsets else None,
                "before_the_human_mark": sum(1 for o in offsets if o < 0),
            },
            "false_positives": {
                "frames": sum(r.false_positive_frames for r in self.results),
                "clear_frames": sum(r.negative_frames for r in self.results),
                "clear_camera_hours": round(
                    sum(r.negative_minutes for r in self.results) / 60.0, 1
                ),
                "per_camera_day": round(self.false_positives_per_camera_day, 2),
            },
            "unusable_frames": dict(unusable),
            "rejections_fired": dict(rejections.most_common()),
            "ms_per_frame": {
                "median": round(statistics.median(timings), 1) if timings else None,
                "max": round(max(timings), 1) if timings else None,
            },
            "cross_camera": self.cross_camera_summary(),
            "notes": list(self.notes),
        }

    def cross_camera_summary(self) -> dict[str, Any]:
        if not self.crossings:
            return {"pairs": 0}
        spreads = [c["spread_m"] for c in self.crossings if c.get("spread_m") is not None]
        return {
            "incidents": len(self.crossings),
            "with_a_fix": sum(1 for c in self.crossings if c.get("fix")),
            "median_pair_spread_m": round(statistics.median(spreads), 1) if spreads else None,
            "detail": self.crossings,
        }

    def operating_curve(self) -> list[dict[str, Any]]:
        """Detection rate against false positives, at every swept threshold.

        Read off the stored per-frame confidences, so every row describes the
        same pipeline run. Nothing is re-analysed and nothing can drift between
        the rows a reader is invited to compare.
        """
        rows = []
        for threshold in SWEEP_THRESHOLDS:
            detected = 0
            fires = 0
            offsets: list[int] = []
            false_frames = 0
            clear_minutes = 0.0
            for result in self.results:
                positives = [(o, c) for o, c, _b in result.series if o >= 0]
                negatives = [(o, c) for o, c, _b in result.series if o < 0]
                if positives:
                    fires += 1
                    hits = [o for o, c in positives if c >= threshold]
                    if hits:
                        detected += 1
                        offsets.append(min(hits))
                false_frames += sum(1 for _o, c in negatives if c >= threshold)
                clear_minutes += result.negative_minutes
            rows.append(
                {
                    "threshold": threshold,
                    "detection_rate": round(detected / fires, 4) if fires else 0.0,
                    "detected": detected,
                    "fires": fires,
                    "median_time_to_alert_s": int(statistics.median(offsets)) if offsets else None,
                    "false_positive_frames": false_frames,
                    "false_positives_per_camera_day": round(
                        false_frames / (clear_minutes / 1440.0), 2
                    ) if clear_minutes else 0.0,
                }
            )
        return rows

    def corroboration_curve(
        self, network: Network, *, tolerance_s: int = 120
    ) -> list[dict[str, Any]]:
        """What the second camera is worth.

        The operating curve above measures one camera on its own, which is not
        what this product does. The product requires a second camera on a
        different summit to see the same thing on a bearing that crosses the
        first one in front of both of them. This measures exactly that rule,
        against the same stored numbers.

        It can only be measured on the dates where two or more cameras on
        different summits recorded the same fire, which in this evaluation set
        is a subset. The comparison is therefore run on that subset alone, so
        the single-camera and corroborated numbers describe the same fires.

        The bearing test is the real one from `firstsmoke.geometry.cross_rays`:
        the two rays must meet in front of both cameras and within range. What
        is *not* modelled is the agent's frame-count bar and its re-reads, both
        of which raise precision further, so this understates the rule's value.
        """
        by_date: dict[str, list[SequenceResult]] = {}
        for result in self.results:
            camera = network.cameras.get(result.camera_id)
            if camera is None:
                continue
            by_date.setdefault(result.sequence.split("_")[0], []).append(result)
        groups = [
            entries for entries in by_date.values()
            if len({network.cameras[r.camera_id].site_id for r in entries}) >= 2
        ]
        if not groups:
            return []

        def crossing_exists(a: SequenceResult, bearing_a: float,
                            b: SequenceResult, bearing_b: float) -> bool:
            ca, cb = network.cameras[a.camera_id], network.cameras[b.camera_id]
            if ca.site_id == cb.site_id:
                return False
            try:
                cross_rays([
                    Ray(ca.camera_id, ca.lat, ca.lon, bearing_a, 2.0, ca.range_m),
                    Ray(cb.camera_id, cb.lat, cb.lon, bearing_b, 2.0, cb.range_m),
                ])
            except CrossingRefused:
                return False
            return True

        rows = []
        for threshold in SWEEP_THRESHOLDS:
            alone_hits = alone_fp = both_hits = both_fp = 0
            fires = clear_frames = 0
            clear_minutes = 0.0
            for entries in groups:
                fires += sum(1 for r in entries if any(o >= 0 for o, _c, _b in r.series))
                clear_minutes += sum(r.negative_minutes for r in entries)
                clear_frames += sum(
                    sum(1 for o, _c, _b in r.series if o < 0) for r in entries
                )
                for result in entries:
                    others = [r for r in entries if r is not result]
                    fired_smoke = False
                    corroborated_smoke = False
                    for offset, confidence, bearing in result.series:
                        if confidence < threshold:
                            continue
                        agreed = any(
                            crossing_exists(result, bearing, other, other_bearing)
                            for other in others
                            for other_offset, other_conf, other_bearing in other.series
                            if other_conf >= threshold
                            and abs(other_offset - offset) <= tolerance_s
                        )
                        if offset >= 0:
                            fired_smoke = True
                            corroborated_smoke = corroborated_smoke or agreed
                        else:
                            alone_fp += 1
                            if agreed:
                                both_fp += 1
                    alone_hits += int(fired_smoke)
                    both_hits += int(corroborated_smoke)
            per_day = (clear_minutes / 1440.0) or 1.0
            rows.append(
                {
                    "threshold": threshold,
                    "fires": fires,
                    "one_camera": {
                        "detected": alone_hits,
                        "detection_rate": round(alone_hits / fires, 4) if fires else 0.0,
                        "false_positive_frames": alone_fp,
                        "false_positives_per_camera_day": round(alone_fp / per_day, 2),
                    },
                    "corroborated": {
                        "detected": both_hits,
                        "detection_rate": round(both_hits / fires, 4) if fires else 0.0,
                        "false_positive_frames": both_fp,
                        "false_positives_per_camera_day": round(both_fp / per_day, 2),
                    },
                    "clear_frames": clear_frames,
                }
            )
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "operating_curve": self.operating_curve(),
            "corroboration_curve": self.corroboration_curve(default_network()),
            "sequences": [r.to_dict() for r in self.results],
        }


def evaluate_calibrated(
    cache: Path = FIGLIB_CACHE,
    *,
    threshold: float = SUSPECT_AT,
    stride: int = 1,
    limit: int | None = None,
    confirmer=None,
    progress=None,
) -> tuple[Evaluation, Evaluation, CalibrationSet]:
    """Two passes: measure uncalibrated, then measure again with a holdout.

    Pass one runs every sequence with no calibration and, while doing so,
    records what each sequence's *clear* frames say about its camera.

    Pass two re-runs every sequence with a calibration pooled from that camera's
    **other sequences** — different dates, different weather, and never the day
    being scored. A camera that appears only once in the set gets no calibration
    at all rather than one fitted on itself, and is reported as uncalibrated.

    Returns ``(before, after, calibrations)``. Comparing the two on the same
    sequences is the only comparison worth making, and the harness never lets
    them diverge, because both come from the same loaded frames.
    """
    network = default_network()
    directories = sorted(d for d in cache.glob("*_FIRE_*") if d.is_dir())
    if limit:
        directories = directories[:limit]

    loaded: list[tuple[str, Sequence_, list[LabelledFrame]]] = []
    for directory in directories:
        item = load_figlib_sequence(directory, network, stride=stride)
        if item is not None:
            loaded.append((directory.name, item[0], item[1]))

    # ---- pass one: no calibration, and gather the evidence -----------------
    before: list[SequenceResult] = []
    evidence: list[ClearFrameEvidence] = []
    for index, (name, sequence, labelled) in enumerate(loaded):
        result = run_sequence(sequence, labelled, threshold=threshold, confirmer=confirmer)
        result.sequence = name
        if result.evidence is not None:
            result.evidence.sequence = name
            evidence.append(result.evidence)
        before.append(result)
        if progress:
            progress(index + 1, len(loaded), f"{name} (uncalibrated)", result)

    # ---- fit, holding out the sequence each calibration will be used on -----
    calibrations = CalibrationSet()
    per_sequence: dict[str, CameraCalibration | None] = {}
    for name, sequence, _labelled in loaded:
        camera_id = sequence.camera.camera_id
        others = [e for e in evidence if e.camera_id == camera_id and e.sequence != name]
        fitted = combine(others, camera_id)
        per_sequence[name] = fitted
        if fitted is not None:
            calibrations.add(fitted)

    # ---- pass two: the same frames, with the held-out calibration ----------
    after: list[SequenceResult] = []
    for index, (name, sequence, labelled) in enumerate(loaded):
        result = run_sequence(
            sequence, labelled, threshold=threshold, confirmer=confirmer,
            calibration=per_sequence[name],
        )
        result.sequence = name
        after.append(result)
        if progress:
            progress(index + 1, len(loaded), f"{name} (calibrated)", result)

    notes_before = [
        *_notes(),
        "This pass runs with no per-camera calibration. It is the baseline the "
        "calibrated pass is compared against.",
    ]
    notes_after = [
        *_notes(),
        "Each camera's calibration was fitted only from that camera's clear frames on "
        "OTHER dates, never from the sequence being scored. Cameras appearing once in "
        "the set receive no calibration and are counted as uncalibrated.",
    ]
    by_date_before: dict[str, list[tuple[str, SequenceResult]]] = {}
    by_date_after: dict[str, list[tuple[str, SequenceResult]]] = {}
    for result in before:
        by_date_before.setdefault(result.sequence.split("_")[0], []).append(
            (result.sequence, result)
        )
    for result in after:
        by_date_after.setdefault(result.sequence.split("_")[0], []).append(
            (result.sequence, result)
        )

    return (
        Evaluation(before, threshold, cross_camera_consistency(by_date_before, network),
                   notes_before),
        Evaluation(after, threshold, cross_camera_consistency(by_date_after, network),
                   notes_after),
        calibrations,
    )


def _notes() -> list[str]:
    return [
        "Imagery: HPWREN, University of California San Diego (http://hpwren.ucsd.edu), "
        "CC BY-NC-ND 4.0. Frames are cached locally and are not redistributed with this "
        "repository.",
        "Labels are the signed offsets in FIgLib's own filenames: the seconds between a "
        "frame and the moment a human first marked the plume as visible.",
        "No field deployment trial exists for this system. These numbers are a replay "
        "against recorded imagery, not evidence of operational outcome.",
    ]


def evaluate_cache(
    cache: Path = FIGLIB_CACHE,
    *,
    threshold: float = SUSPECT_AT,
    stride: int = 1,
    limit: int | None = None,
    confirmer=None,
    progress=None,
) -> Evaluation:
    network = default_network()
    directories = sorted(d for d in cache.glob("*_FIRE_*") if d.is_dir())
    if limit:
        directories = directories[:limit]

    results: list[SequenceResult] = []
    by_date: dict[str, list[tuple[str, SequenceResult]]] = {}
    for index, directory in enumerate(directories):
        loaded = load_figlib_sequence(directory, network, stride=stride)
        if loaded is None:
            continue
        sequence, labelled = loaded
        result = run_sequence(sequence, labelled, threshold=threshold, confirmer=confirmer)
        result.sequence = directory.name
        results.append(result)
        date = directory.name.split("_")[0]
        by_date.setdefault(date, []).append((directory.name, result))
        if progress:
            progress(index + 1, len(directories), directory.name, result)

    return Evaluation(
        results=results,
        threshold=threshold,
        crossings=cross_camera_consistency(by_date, network),
        notes=[
            "Imagery: HPWREN, University of California San Diego (http://hpwren.ucsd.edu), "
            "CC BY-NC-ND 4.0. Frames are cached locally and are not redistributed with this "
            "repository.",
            "Labels are the signed offsets in FIgLib's own filenames: the seconds between a "
            "frame and the moment a human first marked the plume as visible.",
            "No field deployment trial exists for this system. These numbers are a replay "
            "against recorded imagery, not evidence of operational outcome.",
        ],
    )


def cross_camera_consistency(
    by_date: dict[str, list[tuple[str, SequenceResult]]], network: Network
) -> list[dict[str, Any]]:
    """Where several cameras recorded the same fire, do their bearings agree?

    FIgLib publishes no coordinates, so there is nothing to be accurate
    *against*. What can be checked is whether independent pairs of bearings
    cross in the same place. They are independent measurements of the same
    unknown, so their spread is a real bound on the geometry even though it is
    not an accuracy figure, and it is reported as a spread and never as an error.
    """
    out: list[dict[str, Any]] = []
    for date, entries in sorted(by_date.items()):
        rays: list[Ray] = []
        for _name, result in entries:
            camera = network.cameras.get(result.camera_id)
            if camera is None or result.bearing_deg is None:
                continue
            if camera.lat == 0.0 and camera.lon == 0.0:
                continue
            rays.append(
                Ray(
                    camera_id=result.camera_id,
                    lat=camera.lat,
                    lon=camera.lon,
                    bearing_deg=result.bearing_deg,
                    sigma_deg=result.bearing_sigma_deg or 1.5,
                    max_range_m=camera.range_m,
                )
            )
        distinct_sites = {network.cameras[r.camera_id].site_id for r in rays}
        if len(rays) < 2 or len(distinct_sites) < 2:
            continue

        record: dict[str, Any] = {
            "date": date,
            "cameras": [r.camera_id for r in rays],
            "bearings": {r.camera_id: round(r.bearing_deg, 2) for r in rays},
        }
        try:
            fix = cross_rays(rays)
            record["fix"] = fix.to_dict()
        except CrossingRefused as exc:
            record["refused"] = {"code": exc.code, "message": exc.message}

        pair_fixes = []
        for i, a in enumerate(rays):
            for b in rays[i + 1 :]:
                if network.cameras[a.camera_id].site_id == network.cameras[b.camera_id].site_id:
                    continue
                try:
                    pair = cross_rays([a, b])
                except CrossingRefused:
                    continue
                pair_fixes.append((pair.lat, pair.lon))
        if len(pair_fixes) >= 2:
            centre_lat = sum(p[0] for p in pair_fixes) / len(pair_fixes)
            centre_lon = sum(p[1] for p in pair_fixes) / len(pair_fixes)
            spread = [haversine_m(lat, lon, centre_lat, centre_lon) for lat, lon in pair_fixes]
            record["pairs"] = len(pair_fixes)
            record["spread_m"] = round(float(statistics.mean(spread)), 1)
        out.append(record)
    return out


def write_report(evaluation: Evaluation, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evaluation.to_dict(), indent=2))
    return path
