"""Width measurement, with the 380x regression pinned down by name."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from synthetic import band_mask, render_marker_scene
from visioncore import calibration as cal
from visioncore import measure


def _rendered_stroke_px(mask: np.ndarray) -> float:
    """True perpendicular width of a rendered stroke, read off the pixels themselves."""
    columns = np.count_nonzero(mask, axis=0)
    return float(np.median(columns[columns > 0]))


class TestKnownWidthByConstruction:
    @pytest.mark.parametrize("width", [3, 5, 7, 9, 15, 21])
    def test_exact_band_width_recovered_within_one_pixel(self, width):
        """A band of exactly `width` rows. Ground truth is the array, not a drawing API."""
        profile = measure.stroke_width_profile(band_mask(width))
        assert profile.ok, profile.refusal
        assert profile.p50_px == pytest.approx(float(width), abs=measure.WIDTH_PRECISION_PX)
        assert profile.max_px == pytest.approx(float(width), abs=measure.WIDTH_PRECISION_PX)

    @pytest.mark.parametrize("thickness", [3, 5, 9])
    def test_cv2_drawn_line_width_recovered(self, thickness):
        """cv2.line renders `thickness` as roughly thickness+2 rows, so the ground
        truth is the rendered mask, not the argument. Counting rows keeps the
        assertion about our estimator rather than about OpenCV's rasteriser."""
        mask = np.zeros((300, 500), np.uint8)
        cv2.line(mask, (60, 150), (440, 150), 255, thickness)
        profile = measure.stroke_width_profile(mask)
        assert profile.ok
        assert profile.p50_px == pytest.approx(_rendered_stroke_px(mask), abs=1.0)

    def test_diagonal_line_width_recovered(self):
        mask = np.zeros((400, 400), np.uint8)
        cv2.line(mask, (50, 50), (350, 350), 255, 7)
        profile = measure.stroke_width_profile(mask)
        assert profile.ok
        # A 45-degree stroke of rendered width w has perpendicular width w/sqrt(2)
        # at worst; bound it rather than pretend to a single exact number.
        assert 5.0 <= profile.p50_px <= 11.0

    def test_width_converts_to_millimetres_through_a_calibration(self):
        image, _ = render_marker_scene(marker_px=200)  # 200 px marker
        calib = cal.calibrate_from_aruco(image, marker_length_mm=50.0)  # 4 px/mm
        assert calib.ok
        profile = measure.stroke_width_profile(band_mask(8), calibration=calib)
        assert profile.ok
        # 8 px at 4 px/mm is 2 mm.
        assert profile.p50_mm == pytest.approx(2.0, abs=0.3)
        assert profile.precision_mm == pytest.approx(0.25, rel=0.01)


class TestTheThreeHundredAndEightyTimesBug:
    """filling an open polyline reported 283 mm for a 0.75 mm crack.

    A crack detector hands you a *path*: an open polyline of points along the
    crack. Filling that path closes it into a polygon and the distance transform
    then measures the enclosed area. Nothing raises. The number looks plausible.
    """

    @staticmethod
    def _crack_path(n: int = 60, shape: tuple[int, int] = (2160, 3840)) -> np.ndarray:
        """A long, shallow crack across a 4K frame, as a polyline of points."""
        t = np.linspace(0.0, 1.0, n)
        x = (80 + t * (shape[1] - 160)).astype(np.int32)
        y = (shape[0] // 2 + 380 * np.sin(t * 2.2)).astype(np.int32)
        return np.stack([x, y], axis=1)

    @staticmethod
    def _crack_mask(path: np.ndarray, shape: tuple[int, int], thickness: int) -> np.ndarray:
        mask = np.zeros(shape, np.uint8)
        cv2.polylines(mask, [path.reshape(-1, 1, 2)], isClosed=False, color=255,
                      thickness=thickness)
        return mask

    def test_medial_axis_measures_the_stroke_not_the_enclosed_area(self):
        shape = (1080, 1920)
        path = self._crack_path(shape=shape)
        mask = self._crack_mask(path, shape, thickness=3)
        profile = measure.stroke_width_profile(mask)
        assert profile.ok, profile.refusal
        rendered = _rendered_stroke_px(mask)
        assert profile.p50_px == pytest.approx(rendered, abs=1.5)
        assert profile.p95_px < rendered + 4.0, "the tail must stay near the true stroke width"

    def test_the_filled_polyline_approach_is_wrong_by_an_order_of_magnitude(self):
        """This is the executable form of the lesson. If it ever passes trivially,
        someone has changed the wrong function."""
        shape = (1080, 1920)
        path = self._crack_path(shape=shape)
        right = measure.stroke_width_profile(self._crack_mask(path, shape, 3)).p50_px
        wrong = measure.filled_polyline_width_max(path, shape)
        assert wrong > 10 * right, (
            "filling an open polyline is supposed to be catastrophically wrong here; "
            f"got {wrong:.1f} px vs {right:.1f} px"
        )

    def test_a_calibrated_crack_would_have_been_reported_as_metres(self):
        """The same arithmetic that produced '283 mm for a 0.75 mm crack'."""
        shape = (1080, 1920)
        path = self._crack_path(shape=shape)
        px_per_mm = 4.0
        right_mm = measure.stroke_width_profile(self._crack_mask(path, shape, 3)).p50_px / px_per_mm
        wrong_mm = measure.filled_polyline_width_max(path, shape) / px_per_mm
        assert right_mm < 3.0
        assert wrong_mm > 20.0

    def test_reporting_a_distribution_survives_a_single_bad_pixel(self):
        """One fat blob attached to a thin crack must not become the headline number."""
        mask = band_mask(5, length=400, canvas=(300, 500))
        cv2.circle(mask, (420, 150), 20, 255, -1)  # a blob at one end
        profile = measure.stroke_width_profile(mask)
        assert profile.ok
        assert profile.p50_px == pytest.approx(5.0, abs=1.0), "median resists the blob"
        assert profile.max_px > 30.0, "max does not, which is exactly why we report both"


class TestRefusals:
    def test_empty_mask_refuses(self):
        profile = measure.stroke_width_profile(np.zeros((100, 100), np.uint8))
        assert not profile.ok
        assert profile.refusal.code == "EMPTY_MASK"
        assert profile.p50_px is None

    def test_a_handful_of_pixels_refuses_for_want_of_evidence(self):
        mask = np.zeros((100, 100), np.uint8)
        mask[50:53, 50:53] = 255
        profile = measure.stroke_width_profile(mask, min_samples=8)
        assert not profile.ok
        assert profile.refusal.code == "TOO_FEW_SAMPLES"

    def test_uncalibrated_profile_reports_pixels_only(self):
        profile = measure.stroke_width_profile(band_mask(5))
        assert profile.ok
        assert profile.p50_px is not None
        assert profile.p50_mm is None
        assert profile.to_dict()["mm"]["p50"] is None


class TestComponentFiltering:
    def test_keeps_elongated_components_and_drops_blobs(self):
        mask = np.zeros((400, 600), np.uint8)
        cv2.line(mask, (50, 100), (550, 110), 255, 4)  # elongated
        cv2.circle(mask, (300, 300), 40, 255, -1)  # round blob
        kept, stats = measure.elongated_components(mask, min_area_px=30, min_elongation=3.0)
        assert len(stats) == 1
        assert stats[0]["elongation"] > 3.0
        assert kept[300, 300] == 0
        assert np.count_nonzero(kept) > 0

    def test_small_specks_are_dropped_by_area(self):
        mask = np.zeros((200, 200), np.uint8)
        mask[10:12, 10:30] = 255  # 40 px, elongated but small
        kept, stats = measure.elongated_components(mask, min_area_px=100)
        assert stats == []
        assert np.count_nonzero(kept) == 0


class TestMedialAxis:
    def test_ridge_of_a_band_is_its_centre_line(self):
        mask = band_mask(9, length=200, canvas=(100, 300))
        ridge = measure.medial_axis(mask)
        rows = np.unique(np.nonzero(ridge)[0])
        assert len(rows) <= 2, f"a 9 px band should ridge on one row, got {rows}"
        assert abs(int(rows.mean()) - 50) <= 1
