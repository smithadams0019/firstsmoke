from __future__ import annotations

import cv2
import numpy as np
import pytest
from synthetic import camera_matrix, project_marker, render_marker_scene, warp_like_tilt
from visioncore import calibration as cal


class TestArucoScale:
    def test_recovers_px_per_mm_to_within_one_percent(self):
        """Ground truth by construction: a 200 px marker printed at 50 mm is 4 px/mm."""
        image, _ = render_marker_scene(marker_px=200)
        result = cal.calibrate_from_aruco(image, marker_length_mm=50.0)
        assert result.ok, result.refusal
        assert result.px_per_mm == pytest.approx(200.0 / 50.0, rel=0.01)
        assert result.mm_per_px == pytest.approx(0.25, rel=0.01)
        assert result.precision_mm == pytest.approx(0.25, rel=0.01)
        assert result.marker_ids == (7,)

    @pytest.mark.parametrize("marker_px,mm", [(120, 40.0), (200, 50.0), (320, 100.0)])
    def test_scale_tracks_marker_size_and_physical_size(self, marker_px, mm):
        image, _ = render_marker_scene(marker_px=marker_px, canvas=(900, 700), top_left=(250, 180))
        result = cal.calibrate_from_aruco(image, marker_length_mm=mm)
        assert result.ok, result.refusal
        assert result.px_per_mm == pytest.approx(marker_px / mm, rel=0.01)

    def test_segment_mm_matches_the_scalar_scale_in_the_marker_plane(self):
        image, _ = render_marker_scene(marker_px=200)
        result = cal.calibrate_from_aruco(image, marker_length_mm=50.0)
        # The marker spans x=200..400 at y=150, i.e. exactly 50 mm.
        length = result.segment_mm((200.0, 150.0), (400.0, 150.0))
        assert length == pytest.approx(50.0, abs=0.6)

    def test_obliquity_is_near_zero_for_a_fronto_parallel_marker(self):
        image, _ = render_marker_scene()
        result = cal.calibrate_from_aruco(image, marker_length_mm=50.0)
        assert result.obliquity_deg is not None
        assert result.obliquity_deg < 3.0

    def test_residual_is_tiny_for_a_clean_synthetic_marker(self):
        image, _ = render_marker_scene()
        result = cal.calibrate_from_aruco(image, marker_length_mm=50.0)
        assert result.residual_px is not None
        assert result.residual_px < 1.0


class TestRefusals:
    def test_blank_image_refuses_rather_than_guessing(self):
        blank = np.full((400, 400, 3), 255, np.uint8)
        result = cal.calibrate_from_aruco(blank, marker_length_mm=50.0)
        assert not result.ok
        assert result.refusal is not None
        assert result.refusal.code == "NO_FIDUCIAL"
        assert result.px_per_mm is None
        assert result.px_to_mm(100) is None
        assert result.segment_mm((0, 0), (10, 10)) is None

    def test_require_raises_with_the_refusal_attached(self):
        blank = np.full((400, 400, 3), 255, np.uint8)
        result = cal.calibrate_from_aruco(blank, marker_length_mm=50.0)
        with pytest.raises(cal.CalibrationRefused) as excinfo:
            result.require()
        assert excinfo.value.refusal.code == "NO_FIDUCIAL"

    def test_a_steeply_tilted_marker_is_refused_not_mis_measured(self):
        """The whole point: at 60 degrees the scalar scale would be ~2x wrong."""
        image, _ = render_marker_scene(marker_px=260, canvas=(900, 700), top_left=(250, 200))
        tilted = warp_like_tilt(image, 60.0)
        result = cal.calibrate_from_aruco(tilted, marker_length_mm=50.0)
        if result.ok:
            pytest.fail(
                f"accepted a 60 deg tilt and reported {result.px_per_mm:.3f} px/mm "
                f"(obliquity {result.obliquity_deg})"
            )
        assert result.refusal.code == "TOO_OBLIQUE"
        assert result.refusal.details["obliquity_deg"] > 35.0

    def test_a_small_marker_is_refused_on_precision_grounds(self):
        image, _ = render_marker_scene(marker_px=36, canvas=(300, 300), top_left=(120, 120))
        result = cal.calibrate_from_aruco(image, marker_length_mm=50.0, min_marker_px=40.0)
        assert not result.ok
        assert result.refusal.code in {"MARKER_TOO_SMALL", "NO_FIDUCIAL"}

    def test_markers_that_disagree_on_scale_refuse_a_single_scalar(self):
        """Two markers of different apparent size cannot share one px/mm."""
        canvas = np.full((700, 1100, 3), 255, np.uint8)
        near, _ = render_marker_scene(
            marker_id=7, marker_px=200, canvas=(520, 520), top_left=(60, 60)
        )
        far, _ = render_marker_scene(
            marker_id=11, marker_px=120, canvas=(520, 520), top_left=(60, 60)
        )
        canvas[60:580, 20:540] = near
        canvas[60:580, 560:1080] = far
        result = cal.calibrate_from_aruco(canvas, marker_length_mm=50.0)
        assert not result.ok
        assert result.refusal.code == "INCONSISTENT_SCALE"
        assert len(result.refusal.details["px_per_mm_per_marker"]) == 2

    def test_a_negative_marker_length_is_a_programming_error_not_a_refusal(self):
        image, _ = render_marker_scene()
        with pytest.raises(ValueError):
            cal.calibrate_from_aruco(image, marker_length_mm=0.0)


class TestObliquityEstimator:
    def test_square_quad_reads_zero_degrees(self):
        quad = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], np.float64)
        assert cal.quad_obliquity_deg(quad) == pytest.approx(0.0, abs=0.01)

    @pytest.mark.parametrize("tilt", [20.0, 40.0, 60.0])
    def test_foreshortened_quad_recovers_the_tilt(self, tilt):
        w = 100.0 * float(np.cos(np.deg2rad(tilt)))
        quad = np.array([[0, 0], [w, 0], [w, 100], [0, 100]], np.float64)
        assert cal.quad_obliquity_deg(quad) == pytest.approx(tilt, abs=0.5)

    def test_estimator_is_monotonic_in_tilt(self):
        angles = [cal.quad_obliquity_deg(
            np.array([[0, 0], [100 * np.cos(np.deg2rad(t)), 0],
                      [100 * np.cos(np.deg2rad(t)), 100], [0, 100]], np.float64)
        ) for t in (0, 10, 25, 45, 70)]
        assert angles == sorted(angles)


class TestLocalScale:
    def test_pure_scale_homography_returns_that_scale(self):
        h = np.array([[3.0, 0, 10.0], [0, 3.0, 20.0], [0, 0, 1.0]])
        assert cal.local_scale(h, 0, 0) == pytest.approx(3.0)
        assert cal.local_scale(h, 50, 50) == pytest.approx(3.0)

    def test_anisotropic_scale_returns_the_geometric_mean(self):
        h = np.array([[4.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
        assert cal.local_scale(h, 0, 0) == pytest.approx(2.0)

    def test_perspective_homography_scale_varies_across_the_plane(self):
        h = np.array([[2.0, 0, 0], [0, 2.0, 0], [0.002, 0, 1.0]])
        near, far = cal.local_scale(h, -100, 0), cal.local_scale(h, 100, 0)
        assert near > far


class TestCheckerboard:
    @staticmethod
    def _board(cols: int, rows: int, square_px: int, margin: int = 60) -> np.ndarray:
        """Inner corners (cols, rows) means (cols+1) x (rows+1) squares."""
        w = (cols + 1) * square_px + 2 * margin
        h = (rows + 1) * square_px + 2 * margin
        image = np.full((h, w), 255, np.uint8)
        for r in range(rows + 1):
            for c in range(cols + 1):
                if (r + c) % 2 == 0:
                    y, x = margin + r * square_px, margin + c * square_px
                    image[y : y + square_px, x : x + square_px] = 0
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    def test_recovers_scale_from_a_synthetic_board(self):
        square_px, square_mm = 60, 20.0
        image = self._board(7, 5, square_px)
        result = cal.calibrate_from_checkerboard(image, (7, 5), square_mm)
        assert result.ok, result.refusal
        assert result.px_per_mm == pytest.approx(square_px / square_mm, rel=0.02)

    def test_absent_board_refuses(self):
        blank = np.full((400, 400, 3), 255, np.uint8)
        result = cal.calibrate_from_checkerboard(blank, (7, 5), 20.0)
        assert not result.ok
        assert result.refusal.code == "NO_FIDUCIAL"


class TestFacade:
    def test_calibrate_prefers_aruco_and_falls_through(self):
        image, _ = render_marker_scene()
        result = cal.calibrate(image, marker_length_mm=50.0)
        assert result.ok
        assert result.method.startswith("aruco")

    def test_calibrate_with_no_fiducial_configured_refuses_clearly(self):
        image, _ = render_marker_scene()
        result = cal.calibrate(image)
        assert not result.ok
        assert "marker_length_mm" in result.refusal.message

    def test_to_dict_round_trips_through_json(self):
        import json

        image, _ = render_marker_scene()
        payload = json.loads(json.dumps(cal.calibrate(image, marker_length_mm=50.0).to_dict()))
        assert payload["ok"] is True
        assert payload["px_per_mm"] == pytest.approx(4.0, rel=0.01)
        assert payload["refusal"] is None


class TestAgainstATruePerspectiveCamera:
    """The tests that matter: a real pinhole projection, not an affine squeeze.

    `synthetic.project_marker` rotates the marker plane in 3D and projects it
    through a pinhole camera, so the detected quad is a genuine perspective
    quadrilateral. Ground truth for every scale below is computed from the camera
    parameters, not measured off the image.
    """

    ACCEPTED_TILTS = (0.0, 10.0, 20.0, 30.0)
    REFUSED_TILTS = (40.0, 50.0, 60.0)

    @pytest.mark.parametrize("tilt", ACCEPTED_TILTS)
    def test_an_accepted_result_is_accurate(self, tilt):
        """The contract: if we do not refuse, the number is right to within 1%."""
        image, truth = project_marker(tilt)
        result = cal.calibrate_from_aruco(image, marker_length_mm=truth["marker_mm"])
        assert result.ok, result.refusal
        assert result.px_per_mm == pytest.approx(truth["centre_px_per_mm"], rel=0.01)

    @pytest.mark.parametrize("tilt", REFUSED_TILTS)
    def test_a_steep_tilt_is_refused(self, tilt):
        image, truth = project_marker(tilt)
        result = cal.calibrate_from_aruco(image, marker_length_mm=truth["marker_mm"])
        assert not result.ok
        assert result.refusal.code == "TOO_OBLIQUE"

    @pytest.mark.parametrize("tilt", ACCEPTED_TILTS)
    def test_segment_mm_is_exact_where_the_scalar_scale_is_not(self, tilt):
        """`segment_mm` goes through the homography, so it undoes foreshortening.

        The marker's top edge is `marker_mm` long by construction; recovering it
        to 0.5% at 30 degrees of tilt is what makes this the API to measure with.
        """
        image, truth = project_marker(tilt)
        result = cal.calibrate_from_aruco(image, marker_length_mm=truth["marker_mm"])
        assert result.ok, result.refusal
        corners, _ = cal.detect_aruco(image)
        top_left, top_right = tuple(corners[0][0]), tuple(corners[0][1])
        assert result.segment_mm(top_left, top_right) == pytest.approx(
            truth["marker_mm"], rel=0.005
        )

    def test_the_scalar_scale_degrades_with_tilt_and_segment_mm_does_not(self):
        """Why px_per_mm carries a caveat in its docstring and segment_mm does not."""
        image, _truth = project_marker(30.0)
        result = cal.calibrate_from_aruco(image, marker_length_mm=50.0)
        corners, _ = cal.detect_aruco(image)
        scalar_estimate = 50.0 * result.px_per_mm  # px the marker edge "should" be
        actual_px = float(np.linalg.norm(corners[0][1] - corners[0][0]))
        assert abs(scalar_estimate - actual_px) / actual_px > 0.02, (
            "at 30 degrees the isotropic scalar should visibly disagree with a "
            "length measured along the tilt axis"
        )
        assert result.segment_mm(tuple(corners[0][0]), tuple(corners[0][1])) == pytest.approx(
            50.0, rel=0.005
        )

    @pytest.mark.parametrize("tilt", [0.0, 10.0, 20.0, 30.0])
    def test_with_intrinsics_the_tilt_estimate_is_metric(self, tilt):
        image, truth = project_marker(tilt)
        result = cal.calibrate_from_aruco(
            image, marker_length_mm=50.0, camera_matrix=camera_matrix(truth)
        )
        assert result.ok, result.refusal
        assert result.obliquity_deg == pytest.approx(tilt, abs=2.0)
        assert result.diagnostics["obliquity_source"] == "homography"

    @pytest.mark.parametrize("tilt", [10.0, 20.0, 30.0, 40.0, 50.0])
    def test_the_intrinsics_free_estimate_errs_towards_refusing(self, tilt):
        """It may over-read tilt. It must never under-read it, or the gate leaks."""
        image, _ = project_marker(tilt)
        corners, _ = cal.detect_aruco(image)
        assert corners, f"marker not detected at {tilt} degrees"
        estimate = cal.quad_obliquity_deg(corners[0])
        assert estimate >= tilt - 1.0, f"under-read {tilt} deg as {estimate:.1f} deg"
        assert estimate <= tilt + 4.0

    def test_solvepnp_ippe_square_is_why_we_decompose_the_homography(self):
        """A regression guard on a real bug, not a hypothetical one.

        solvePnP with SOLVEPNP_IPPE_SQUARE reported 2.0 deg for a true 20 deg
        tilt: it returns one of two ambiguous solutions and picked the mirror.
        A refusal gate built on it would wave through the exact geometry it
        exists to catch. If this test ever fails because OpenCV fixed the
        ambiguity, the homography path is still correct - just delete the test.
        """
        image, truth = project_marker(20.0)
        corners, _ = cal.detect_aruco(image)
        quad = corners[0].astype(np.float64)
        half = 25.0
        obj = np.array(
            [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]]
        )
        ok, rvec, tvec = cv2.solvePnP(
            obj, quad, camera_matrix(truth), None, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        assert ok
        rot, _ = cv2.Rodrigues(rvec)
        normal = rot @ np.array([0.0, 0.0, 1.0])
        view = np.asarray(tvec).reshape(3)
        pnp_deg = np.degrees(
            np.arccos(min(1.0, abs(float(np.dot(normal, view / np.linalg.norm(view))))))
        )

        result = cal.calibrate_from_aruco(
            image, marker_length_mm=50.0, camera_matrix=camera_matrix(truth)
        )
        assert abs(result.obliquity_deg - 20.0) < 2.0, "the homography path must be right"
        assert abs(pnp_deg - 20.0) > 5.0, (
            f"solvePnP was expected to be badly wrong here; it said {pnp_deg:.1f} deg"
        )

    def test_a_distant_marker_is_refused_on_size_before_anything_else(self):
        image, _truth = project_marker(0.0, distance_mm=4000.0)
        result = cal.calibrate_from_aruco(image, marker_length_mm=50.0)
        assert not result.ok
        assert result.refusal.code in {"MARKER_TOO_SMALL", "NO_FIDUCIAL"}
