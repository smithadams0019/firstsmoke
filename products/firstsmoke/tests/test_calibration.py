"""Per-camera calibration: what is learnt, from what, and what it refuses to learn."""

from __future__ import annotations

import numpy as np

from firstsmoke.calibration import (
    GRID_H,
    GRID_W,
    MIN_FRAMES_TO_TRUST,
    CalibrationSet,
    ClearFrameEvidence,
    combine,
    downscale_mask,
)
from firstsmoke.rejectors import reject_habitual


def evidence(sequence, frames, cells=(), rate=1.0, confidences=None, camera="cam"):
    coverage = np.zeros((GRID_H, GRID_W), np.float32)
    for row, col in cells:
        coverage[row, col] = frames * rate
    return ClearFrameEvidence(camera, sequence, frames, coverage, confidences or [0.2] * frames)


def mask_on_cell(row, col, shape=(768, 1024)):
    mask = np.zeros(shape, np.uint8)
    ch, cw = shape[0] // GRID_H, shape[1] // GRID_W
    mask[row * ch : (row + 1) * ch, col * cw : (col + 1) * cw] = 255
    return mask


class FakeRegion:
    def __init__(self, mask):
        self.mask = mask


def test_a_cell_lit_on_every_clear_frame_is_fully_habitual():
    cal = combine([evidence("a", 40, cells=[(10, 12)])], "cam")
    assert cal is not None
    assert cal.nuisance[10, 12] == 1.0
    assert cal.habituation(mask_on_cell(10, 12)) > 0.95
    assert cal.habituation(mask_on_cell(3, 3)) == 0.0


def test_too_few_clear_frames_gives_no_calibration_rather_than_a_weak_one():
    assert combine([evidence("a", MIN_FRAMES_TO_TRUST - 1, cells=[(1, 1)])], "cam") is None


def test_only_the_named_camera_is_pooled():
    cal = combine(
        [evidence("a", 40, cells=[(5, 5)]), evidence("b", 40, cells=[(9, 9)], camera="other")],
        "cam",
    )
    assert cal.nuisance[5, 5] == 1.0
    assert cal.nuisance[9, 9] == 0.0
    assert cal.sources == ["a"]


def test_rates_pool_across_dates():
    cal = combine([evidence("a", 40, cells=[(5, 5)]), evidence("b", 40)], "cam")
    assert abs(cal.nuisance[5, 5] - 0.5) < 1e-6


def test_the_confidence_ceiling_is_the_clear_frame_95th_percentile():
    confs = list(np.linspace(0.0, 1.0, 101))
    cal = combine([evidence("a", 101, confidences=confs)], "cam")
    assert abs(cal.clear_p95 - 0.95) < 0.01


def test_a_new_thing_somewhere_quiet_is_not_rejected():
    cal = combine([evidence("a", 40, cells=[(10, 12)], confidences=[0.2] * 40)], "cam")
    assert reject_habitual(FakeRegion(mask_on_cell(2, 2)), cal, confidence=0.6) is None


def test_a_candidate_where_the_camera_always_fires_is_rejected():
    cal = combine([evidence("a", 40, cells=[(10, 12)])], "cam")
    rejection = reject_habitual(FakeRegion(mask_on_cell(10, 12)), cal, confidence=0.9)
    assert rejection is not None and rejection.code == "HABITUAL_REGION"
    assert "other days" in rejection.message


def test_a_score_this_camera_reaches_on_clear_days_is_not_unusual():
    cal = combine([evidence("a", 40, confidences=[0.5] * 40)], "cam")
    rejection = reject_habitual(FakeRegion(mask_on_cell(2, 2)), cal, confidence=0.45)
    assert rejection is not None and rejection.code == "BELOW_CAMERA_BASELINE"


def test_no_calibration_means_no_rejection():
    assert reject_habitual(FakeRegion(mask_on_cell(1, 1)), None, confidence=0.1) is None


def test_downscale_is_area_weighted():
    mask = np.zeros((768, 1024), np.uint8)
    mask[:16, :16] = 255  # a quarter of the first cell
    assert abs(downscale_mask(mask)[0, 0] - 0.25) < 0.02


def test_a_calibration_set_round_trips(tmp_path):
    cal = combine([evidence("a", 40, cells=[(4, 7)])], "cam")
    path = CalibrationSet({"cam": cal}).save(tmp_path / "c.json")
    back = CalibrationSet.load(path).get("cam")
    assert back.nuisance[4, 7] == 1.0 and back.sources == ["a"]
