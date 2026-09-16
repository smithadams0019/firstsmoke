"""Learning what each camera gets wrong, from that camera's own clear frames.

The problem this exists to solve
--------------------------------
The first evaluation found 487 false positives per camera-day, and the shape of
them was the useful part. They were not spread evenly: on eight of forty-four
cameras the *highest confidence reached anywhere in the sequence* fell on a
frame labelled clear, while seventeen sequences produced none at all. The
detector was locking onto something persistent in those particular views — the
likely candidates are a marine layer creeping up a valley and slope shadow
rotating across a ridge — and that something passes all seven named rejectors,
because it genuinely does grow, stay anchored, veil the hillside and soften at
its edges.

No threshold fixes that, and no extra impostor rule catches it, because the
thing really does look like smoke. What separates it from smoke is not its
appearance at all. It is that **it was already there**.

So the rule is the one the coordinator put best: a feature that is present in a
camera's clear frames is by definition not smoke.

What is learned
---------------
Two things per camera, both from frames labelled clear and never from a frame
containing a fire:

1. **A nuisance map.** A coarse grid over the image recording, for each cell,
   the fraction of clear frames in which the detector put a candidate region
   there. A cell at 0.60 means this camera raises something in that part of its
   view on three clear frames in five, which is a property of the view rather
   than of an event.

2. **A confidence ceiling.** The 95th percentile of the confidences this camera
   reaches on clear frames. A camera that routinely reaches 0.55 with nothing
   happening should not be believed at 0.55.

Why both. The map is spatial and catches a nuisance that sits still; the ceiling
is scalar and catches a camera that is simply noisy everywhere. A camera can
fail either way and they are cheap to carry together.

Honesty
-------
Everything here is fitted on held-out data or it is worthless. The evaluation
harness fits each camera's calibration from that camera's sequences on **other
dates** and never from the sequence being scored, so the calibration has never
seen the day it is tested on. Cameras with only one sequence in the set cannot
be calibrated this way at all and are reported separately rather than quietly
given a calibration fitted on themselves. See `docs/evaluation.md`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

GRID_W = 32
GRID_H = 24
"""The nuisance map's resolution. At a 1024 px working width each cell is 32 px
across, which is about the size of an early plume and far smaller than the
persistent features this is built to catch. Finer than this and a nuisance that
drifts by a few pixels a day stops overlapping itself between dates."""

MIN_FRAMES_TO_TRUST = 25
"""Below this many clear frames the rates are noise. A camera with twelve clear
frames that happened to show a bird twice would otherwise learn a 17% nuisance
rate at that cell and start suppressing real detections there."""

HABITUAL_AT = 0.35
"""A cell this camera raises something in, on this fraction of its clear frames,
is habitual. Chosen so that a nuisance appearing in roughly one clear frame in
three is caught, while a location that fired once or twice is not."""


@dataclass
class CameraCalibration:
    """What one camera habitually does when nothing is happening."""

    camera_id: str
    nuisance: np.ndarray
    """``GRID_H x GRID_W`` float32 in 0..1: the fraction of clear frames in which
    a candidate region covered this cell."""
    clear_frames: int
    clear_p95: float
    """95th percentile of the confidences reached on clear frames."""
    clear_p50: float
    sources: list[str] = field(default_factory=list)
    """Which sequences this was fitted from. Printed in the evaluation so the
    holdout can be checked rather than taken on trust."""

    @property
    def trustworthy(self) -> bool:
        return self.clear_frames >= MIN_FRAMES_TO_TRUST

    def habituation(self, mask: np.ndarray) -> float:
        """How habitual this region's location is for this camera, 0 to 1.

        Takes the mean over the cells the region covers, weighted by how much of
        each cell it covers, so a large region spanning one noisy cell and five
        quiet ones is not condemned by the one.
        """
        if not self.trustworthy:
            return 0.0
        grid = downscale_mask(mask)
        weight = grid.sum()
        if weight <= 0:
            return 0.0
        return float((grid * self.nuisance).sum() / weight)

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "clear_frames": self.clear_frames,
            "clear_p95": round(self.clear_p95, 4),
            "clear_p50": round(self.clear_p50, 4),
            "trustworthy": self.trustworthy,
            "sources": list(self.sources),
            "habitual_cells": int((self.nuisance >= HABITUAL_AT).sum()),
            "peak_nuisance": round(float(self.nuisance.max()), 4),
            "grid": [round(float(v), 4) for v in self.nuisance.ravel()],
            "grid_shape": [GRID_H, GRID_W],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CameraCalibration:
        shape = tuple(data.get("grid_shape", [GRID_H, GRID_W]))
        grid = np.array(data["grid"], np.float32).reshape(shape)
        return cls(
            camera_id=data["camera_id"],
            nuisance=grid,
            clear_frames=int(data["clear_frames"]),
            clear_p95=float(data["clear_p95"]),
            clear_p50=float(data["clear_p50"]),
            sources=list(data.get("sources", [])),
        )


def downscale_mask(mask: np.ndarray) -> np.ndarray:
    """A region mask as cell coverage in 0..1, on the nuisance grid.

    Uses area averaging rather than sampling, so a thin plume that crosses a
    cell contributes its real fraction of that cell instead of being rounded to
    present or absent.
    """
    import cv2

    binary = (mask > 0).astype(np.float32)
    return cv2.resize(binary, (GRID_W, GRID_H), interpolation=cv2.INTER_AREA)


@dataclass
class ClearFrameEvidence:
    """What one sequence's clear frames contribute to a calibration.

    Accumulated once per sequence during the first evaluation pass, so a
    leave-one-date-out calibration is a sum over the other sequences rather than
    a second run of the detector over everything.
    """

    camera_id: str
    sequence: str
    frames: int
    coverage: np.ndarray
    """``GRID_H x GRID_W`` float32: summed cell coverage over the clear frames."""
    confidences: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "sequence": self.sequence,
            "frames": self.frames,
            "coverage": [round(float(v), 4) for v in self.coverage.ravel()],
            "confidences": [round(c, 4) for c in self.confidences],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClearFrameEvidence:
        return cls(
            camera_id=data["camera_id"],
            sequence=data["sequence"],
            frames=int(data["frames"]),
            coverage=np.array(data["coverage"], np.float32).reshape(GRID_H, GRID_W),
            confidences=[float(c) for c in data.get("confidences", [])],
        )


def empty_coverage() -> np.ndarray:
    return np.zeros((GRID_H, GRID_W), np.float32)


def combine(evidence: list[ClearFrameEvidence], camera_id: str) -> CameraCalibration | None:
    """Build one camera's calibration by pooling the evidence handed in.

    The caller decides what goes in, which is the whole point: the evaluation
    hands in a camera's *other dates* and nothing from the sequence being
    scored. Returns ``None`` when there is not enough to be worth fitting, so a
    thin calibration is absent rather than weak.
    """
    mine = [e for e in evidence if e.camera_id == camera_id]
    frames = sum(e.frames for e in mine)
    if not mine or frames < MIN_FRAMES_TO_TRUST:
        return None
    coverage = np.zeros((GRID_H, GRID_W), np.float32)
    confidences: list[float] = []
    for item in mine:
        coverage += item.coverage
        confidences.extend(item.confidences)
    nuisance = np.clip(coverage / float(frames), 0.0, 1.0)
    values = np.array(confidences, np.float32) if confidences else np.zeros(1, np.float32)
    return CameraCalibration(
        camera_id=camera_id,
        nuisance=nuisance,
        clear_frames=frames,
        clear_p95=float(np.percentile(values, 95)),
        clear_p50=float(np.percentile(values, 50)),
        sources=sorted(e.sequence for e in mine),
    )


class CalibrationSet:
    """Calibrations for a whole network, and where they came from."""

    def __init__(self, calibrations: dict[str, CameraCalibration] | None = None) -> None:
        self._by_camera: dict[str, CameraCalibration] = dict(calibrations or {})

    def __len__(self) -> int:
        return len(self._by_camera)

    def __contains__(self, camera_id: object) -> bool:
        return camera_id in self._by_camera

    def get(self, camera_id: str) -> CameraCalibration | None:
        return self._by_camera.get(camera_id)

    def add(self, calibration: CameraCalibration) -> None:
        self._by_camera[calibration.camera_id] = calibration

    def summary(self) -> dict[str, Any]:
        cams = list(self._by_camera.values())
        return {
            "cameras": len(cams),
            "trustworthy": sum(c.trustworthy for c in cams),
            "median_clear_frames": (
                int(np.median([c.clear_frames for c in cams])) if cams else 0
            ),
            "median_clear_p95": (
                round(float(np.median([c.clear_p95 for c in cams])), 4) if cams else 0.0
            ),
            "cameras_with_habitual_cells": sum(
                1 for c in cams if (c.nuisance >= HABITUAL_AT).any()
            ),
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "grid_shape": [GRID_H, GRID_W],
                    "habitual_at": HABITUAL_AT,
                    "cameras": [c.to_dict() for c in self._by_camera.values()],
                },
                indent=2,
            )
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> CalibrationSet:
        data = json.loads(Path(path).read_text())
        return cls({c["camera_id"]: CameraCalibration.from_dict(c) for c in data["cameras"]})
