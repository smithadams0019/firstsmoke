"""Bearings and crossings, checked against answers worked out by hand."""

from __future__ import annotations

import math

import pytest

from firstsmoke.geometry import (
    CrossingRefused,
    Ray,
    bearing_between,
    bearing_from_pixel,
    covers_bearing,
    cross_rays,
    haversine_m,
    pixel_from_bearing,
    project_bearing,
    signed_delta,
)


class TestBearingFromPixel:
    def test_centre_column_is_the_azimuth(self):
        assert bearing_from_pixel(512, 1024, 90.0, 90.0) == pytest.approx(90.0)

    def test_edges_are_half_the_field_of_view_away(self):
        assert bearing_from_pixel(0, 1024, 90.0, 90.0) == pytest.approx(45.0, abs=0.01)
        assert bearing_from_pixel(1024, 1024, 90.0, 90.0) == pytest.approx(135.0, abs=0.01)

    def test_it_wraps_through_north(self):
        assert bearing_from_pixel(0, 1024, 10.0, 90.0) == pytest.approx(325.0, abs=0.01)

    def test_it_is_not_linear_in_x(self):
        """The pinhole relation bows away from the linear approximation."""
        quarter = bearing_from_pixel(256, 1024, 90.0, 90.0)
        linear = 90.0 - 0.25 * 90.0
        assert quarter != pytest.approx(linear, abs=0.5)
        # tan(theta) = -0.5 * tan(45 deg) -> theta = -26.565 degrees
        assert quarter == pytest.approx(90.0 - 26.565, abs=0.01)

    def test_round_trip(self):
        for x in (1, 100, 512, 900, 1023):
            bearing = bearing_from_pixel(x, 1024, 137.0, 62.0)
            assert pixel_from_bearing(bearing, 1024, 137.0, 62.0) == pytest.approx(x, abs=0.01)

    def test_off_frame_bearing_has_no_pixel(self):
        assert pixel_from_bearing(200.0, 1024, 20.0, 60.0) is None

    def test_a_zero_width_image_is_an_error(self):
        with pytest.raises(ValueError, match="image_width"):
            bearing_from_pixel(10, 0, 0.0, 90.0)

    def test_an_impossible_field_of_view_is_an_error(self):
        for hfov in (0.0, -10.0, 400.0):
            with pytest.raises(ValueError, match="hfov_deg"):
                bearing_from_pixel(10, 1024, 0.0, hfov)

    def test_a_panoramic_head_uses_the_equiangular_model(self):
        """A 180 degree stitched panorama is not a pinhole, and the pinhole
        relation does not merely lose accuracy on it, it diverges."""
        assert bearing_from_pixel(0, 1024, 90.0, 180.0) == pytest.approx(0.0, abs=0.01)
        assert bearing_from_pixel(512, 1024, 90.0, 180.0) == pytest.approx(90.0, abs=0.01)
        assert bearing_from_pixel(1024, 1024, 90.0, 180.0) == pytest.approx(180.0, abs=0.01)
        assert bearing_from_pixel(256, 1024, 90.0, 180.0) == pytest.approx(45.0, abs=0.01)

    def test_a_panoramic_round_trip_also_holds(self):
        for x in (1, 300, 512, 900):
            bearing = bearing_from_pixel(x, 1024, 40.0, 180.0)
            assert pixel_from_bearing(bearing, 1024, 40.0, 180.0) == pytest.approx(x, abs=0.01)


class TestAngles:
    def test_signed_delta_takes_the_short_way(self):
        assert signed_delta(10.0, 350.0) == pytest.approx(20.0)
        assert signed_delta(350.0, 10.0) == pytest.approx(-20.0)

    def test_coverage_respects_the_field_of_view(self):
        assert covers_bearing(40.0, 0.0, 90.0)
        assert not covers_bearing(50.0, 0.0, 90.0)
        assert covers_bearing(350.0, 0.0, 90.0), "coverage must wrap through north"

    def test_a_negative_margin_shrinks_the_frame(self):
        assert covers_bearing(44.0, 0.0, 90.0)
        assert not covers_bearing(44.0, 0.0, 90.0, margin_deg=-2.0)


class TestGeodesy:
    def test_a_degree_of_latitude_is_about_111_km(self):
        assert haversine_m(33.0, -116.0, 34.0, -116.0) == pytest.approx(111_195, rel=0.001)

    def test_projecting_then_measuring_returns_the_distance(self):
        lat, lon = project_bearing(33.3, -116.85, 47.0, 12_000.0)
        assert haversine_m(33.3, -116.85, lat, lon) == pytest.approx(12_000.0, rel=0.002)

    def test_projecting_then_bearing_returns_the_bearing(self):
        lat, lon = project_bearing(33.3, -116.85, 47.0, 12_000.0)
        assert bearing_between(33.3, -116.85, lat, lon) == pytest.approx(47.0, abs=0.1)

    def test_due_north_is_zero_and_due_east_is_ninety(self):
        assert bearing_between(33.0, -116.0, 34.0, -116.0) == pytest.approx(0.0, abs=0.01)
        assert bearing_between(33.0, -116.0, 33.0, -115.0) == pytest.approx(90.0, abs=0.5)


class TestCrossRays:
    @staticmethod
    def rays_at(fire: tuple[float, float], cameras: list[tuple[float, float]], sigma=1.0):
        return [
            Ray(f"cam{i}", lat, lon, bearing_between(lat, lon, *fire), sigma_deg=sigma)
            for i, (lat, lon) in enumerate(cameras)
        ]

    def test_two_perfect_bearings_recover_the_point(self):
        fire = (33.30, -116.85)
        # Deliberately not opposite each other, or the rays would be collinear.
        cameras = [(33.20, -116.95), (33.38, -116.80)]
        fix = cross_rays(self.rays_at(fire, cameras))
        assert haversine_m(fix.lat, fix.lon, *fire) < 5.0
        assert fix.residual_m < 1.0
        assert set(fix.range_m) == {"cam0", "cam1"}

    def test_three_bearings_still_recover_the_point(self):
        fire = (33.30, -116.85)
        cameras = [(33.20, -116.95), (33.38, -116.80), (33.28, -116.70)]
        fix = cross_rays(self.rays_at(fire, cameras))
        # Ten metres, not five. With three rays the tangent-plane approximation
        # cannot satisfy all of them exactly and leaves a few metres of residual
        # curvature error. That is two orders of magnitude below the bearing
        # error a real camera contributes, so it is not worth a proper geodesic
        # solver, but it is worth writing down rather than hiding in a tolerance.
        assert haversine_m(fix.lat, fix.lon, *fire) < 10.0
        assert fix.residual_m < 5.0
        assert len(fix.rays) == 3

    def test_a_bearing_error_moves_the_fix_by_range_over_the_crossing_angle(self):
        """One degree of bearing error is range x 1 degree across the line of
        sight, then divided by the sine of the crossing angle along it. A shallow
        cross multiplies the error, which is why cross_rays refuses below eight
        degrees and why the ellipse is reported."""
        fire = (33.30, -116.85)
        cameras = [(33.20, -116.95), (33.38, -116.80)]
        rays = self.rays_at(fire, cameras)
        clean = cross_rays(rays)
        nudged = cross_rays([
            Ray(rays[0].camera_id, rays[0].lat, rays[0].lon, rays[0].bearing_deg + 1.0),
            rays[1],
        ])
        error = haversine_m(nudged.lat, nudged.lon, *fire)
        distance = haversine_m(cameras[0][0], cameras[0][1], *fire)
        expected = distance * math.radians(1.0) / math.sin(math.radians(clean.crossing_angle_deg))
        assert error == pytest.approx(expected, rel=0.25)

    def test_the_uncertainty_grows_as_the_crossing_gets_shallow(self):
        fire = (33.40, -116.85)
        wide = cross_rays(self.rays_at(fire, [(33.30, -116.95), (33.30, -116.75)]))
        narrow = cross_rays(self.rays_at(fire, [(33.30, -116.89), (33.30, -116.81)]))
        assert narrow.crossing_angle_deg < wide.crossing_angle_deg
        assert narrow.semi_major_m > wide.semi_major_m

    def test_one_ray_is_refused(self):
        with pytest.raises(CrossingRefused) as exc:
            cross_rays([Ray("a", 33.2, -116.9, 45.0)])
        assert exc.value.code == "TOO_FEW_RAYS"

    def test_near_parallel_rays_are_refused(self):
        rays = [
            Ray("a", 33.20, -116.90, 1.0),
            Ray("b", 33.20, -116.89, 2.0),
        ]
        with pytest.raises(CrossingRefused) as exc:
            cross_rays(rays)
        assert exc.value.code == "RAYS_TOO_PARALLEL"

    def test_a_crossing_behind_the_cameras_is_refused(self):
        """Both looking away from each other: the lines meet, the sightlines do not."""
        rays = [
            Ray("a", 33.20, -116.90, 200.0),
            Ray("b", 33.30, -116.80, 110.0),
        ]
        with pytest.raises(CrossingRefused) as exc:
            cross_rays(rays)
        assert exc.value.code in {"BEHIND_CAMERA", "RAYS_TOO_PARALLEL"}

    def test_a_crossing_past_the_useful_range_is_refused(self):
        fire = (34.40, -116.85)
        cameras = [(33.20, -116.95), (33.22, -116.60)]
        rays = [
            Ray(f"cam{i}", lat, lon, bearing_between(lat, lon, *fire), max_range_m=20_000.0)
            for i, (lat, lon) in enumerate(cameras)
        ]
        with pytest.raises(CrossingRefused) as exc:
            cross_rays(rays)
        assert exc.value.code == "BEYOND_RANGE"

    def test_a_tighter_sigma_pulls_the_answer_toward_that_camera(self):
        """Weighting only bites with three or more rays.

        Two rays are exactly determined: they meet where they meet, and the
        residual is zero whatever the sigmas say. With three, one of which is
        wrong by two degrees, down-weighting the wrong one should move the
        answer back toward the truth."""
        fire = (33.30, -116.85)
        cameras = [(33.20, -116.95), (33.38, -116.80), (33.26, -116.70)]
        rays = self.rays_at(fire, cameras)
        wrong = Ray(rays[0].camera_id, rays[0].lat, rays[0].lon, rays[0].bearing_deg + 2.0)

        even = cross_rays([
            Ray(wrong.camera_id, wrong.lat, wrong.lon, wrong.bearing_deg, sigma_deg=1.0),
            Ray(rays[1].camera_id, rays[1].lat, rays[1].lon, rays[1].bearing_deg, sigma_deg=1.0),
            Ray(rays[2].camera_id, rays[2].lat, rays[2].lon, rays[2].bearing_deg, sigma_deg=1.0),
        ])
        distrusted = cross_rays([
            Ray(wrong.camera_id, wrong.lat, wrong.lon, wrong.bearing_deg, sigma_deg=8.0),
            Ray(rays[1].camera_id, rays[1].lat, rays[1].lon, rays[1].bearing_deg, sigma_deg=0.4),
            Ray(rays[2].camera_id, rays[2].lat, rays[2].lon, rays[2].bearing_deg, sigma_deg=0.4),
        ])
        assert haversine_m(distrusted.lat, distrusted.lon, *fire) < haversine_m(
            even.lat, even.lon, *fire
        )
