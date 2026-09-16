"""Each impostor, and the rule that is supposed to catch it.

Every rule gets two tests: it fires on the thing it is for, and it does not fire
on a plume. The second half matters more. A rejector that also rejects smoke is
not a precision improvement, it is a missed fire, and the only way to know is to
check each rule against a known-good column.
"""

from __future__ import annotations

import numpy as np
import pytest

from firstsmoke.rejectors import (
    Rejection,
    blind_reason,
    reject_airborne,
    reject_cloud,
    reject_dust,
    reject_erratic,
    reject_flare,
    reject_lens_artefact,
    reject_ridge_registration,
    reject_too_brief,
    reject_weather_front,
)
from firstsmoke.scene import Alignment, assess_scene
from firstsmoke.tracks import Growth

FRAME_HEIGHT = 683


def growth(**overrides) -> Growth:
    """A healthy smoke column, unless a test says otherwise."""
    base = dict(
        frames=8, span_s=420.0, area_first=300, area_last=2400,
        area_slope_px_per_min=300.0, area_ratio=8.0, monotonic_fraction=1.0,
        top_rise_px_per_min=13.0, base_drift_px_per_min=1.2,
        centroid_drift_px_per_min=4.0, horizontal_drift_px_per_min=1.0,
        width_growth_px_per_min=4.0, aspect_first=1.4, aspect_last=2.6,
        base_jitter_px=3.0, anchor_score=0.9,
    )
    base.update(overrides)
    return Growth(**base)


class FakeRegion:
    """Only the fields the rules read. A real Region needs a full mask."""

    def __init__(self, **kwargs):
        self.w = kwargs.get("w", 40)
        self.h = kwargs.get("h", 160)
        self.area = self.w * self.h
        self.base_below_horizon = kwargs.get("base_below_horizon", 8.0)
        self.edge_softness = kwargs.get("edge_softness", 0.7)
        self.greyness = kwargs.get("greyness", 0.8)
        self.texture_ratio = kwargs.get("texture_ratio", 0.5)
        self.mask = kwargs.get("mask", np.zeros((FRAME_HEIGHT, 1024), np.uint8))
        # Empty by default, so the flare rule exercises its fallback path. Tests
        # that want the fast path pass metrics explicitly.
        self.metrics = kwargs.get("metrics", {})


def plume_region(**kwargs) -> FakeRegion:
    return FakeRegion(**kwargs)


CORRECTED = Alignment(0.4, 0.2, 0.9, np.zeros((4, 4, 3), np.uint8), corrected=False)
SHAKEN = Alignment(9.0, 2.0, 0.05, np.zeros((4, 4, 3), np.uint8), corrected=False)


class TestCloud:
    def test_it_fires_on_something_that_translates_without_growing(self):
        rejection = reject_cloud(
            growth(anchor_score=0.1, centroid_drift_px_per_min=14.0, area_ratio=1.1,
                   top_rise_px_per_min=0.5, base_drift_px_per_min=14.0),
            plume_region(base_below_horizon=-60.0),
            FRAME_HEIGHT,
        )
        assert rejection is not None
        assert rejection.code == "CLOUD_TRANSLATION"
        assert rejection.impostor == "cloud"

    def test_it_does_not_fire_on_a_plume(self):
        assert reject_cloud(growth(), plume_region(), FRAME_HEIGHT) is None

    def test_something_airborne_is_rejected_more_confidently(self):
        moving = dict(anchor_score=0.1, centroid_drift_px_per_min=14.0, area_ratio=1.1,
                      top_rise_px_per_min=0.5, base_drift_px_per_min=14.0)
        high = reject_cloud(growth(**moving), plume_region(base_below_horizon=-90.0), FRAME_HEIGHT)
        low = reject_cloud(growth(**moving), plume_region(base_below_horizon=6.0), FRAME_HEIGHT)
        assert high.confidence > low.confidence


class TestAirborne:
    def test_a_shape_floating_above_the_ridge_is_rejected(self):
        rejection = reject_airborne(
            plume_region(base_below_horizon=-80.0), FRAME_HEIGHT, growth(top_rise_px_per_min=0.4)
        )
        assert rejection is not None
        assert rejection.code == "AIRBORNE"

    def test_a_plume_attached_to_the_ground_is_not(self):
        assert reject_airborne(plume_region(), FRAME_HEIGHT, growth()) is None

    def test_a_detached_shape_that_is_climbing_hard_is_spared(self):
        """A plume behind a ridge is genuinely detached in the image. If it is
        rising fast, that is not a cloud."""
        assert reject_airborne(
            plume_region(base_below_horizon=-80.0), FRAME_HEIGHT, growth(top_rise_px_per_min=13.0)
        ) is None


class TestDust:
    def test_a_brown_plume_running_along_the_ground_is_rejected(self):
        rejection = reject_dust(
            growth(top_rise_px_per_min=0.6, horizontal_drift_px_per_min=11.0),
            plume_region(base_below_horizon=30.0, greyness=0.2),
        )
        assert rejection is not None
        assert rejection.code == "GROUND_DRIFT"

    def test_a_grey_column_that_rises_is_not_dust(self):
        assert reject_dust(growth(), plume_region()) is None

    def test_a_neutral_plume_moving_sideways_is_not_called_dust(self):
        """Greyness is the deciding evidence; without it the rule holds off."""
        assert reject_dust(
            growth(top_rise_px_per_min=0.6, horizontal_drift_px_per_min=11.0),
            plume_region(base_below_horizon=30.0, greyness=0.85),
        ) is None


class TestLensArtefact:
    def test_a_motionless_hard_edged_blob_is_rejected(self):
        rejection = reject_lens_artefact(
            growth(centroid_drift_px_per_min=0.05, area_ratio=1.0, base_jitter_px=0.1),
            plume_region(edge_softness=0.05),
        )
        assert rejection is not None
        assert rejection.code == "STATIC_ARTEFACT"

    def test_a_plume_is_not_mistaken_for_a_droplet(self):
        assert reject_lens_artefact(growth(), plume_region()) is None

    def test_a_still_but_breathing_blob_is_spared(self):
        """A plume that has stopped growing still breathes. A droplet does not."""
        assert reject_lens_artefact(
            growth(centroid_drift_px_per_min=0.05, area_ratio=1.0, base_jitter_px=6.0),
            plume_region(edge_softness=0.05),
        ) is None


class TestErratic:
    def test_a_base_that_runs_around_is_rejected(self):
        rejection = reject_erratic(growth(base_drift_px_per_min=30.0, anchor_score=0.2))
        assert rejection is not None
        assert rejection.code == "ERRATIC_BASE"

    def test_a_plume_base_is_not_erratic(self):
        assert reject_erratic(growth()) is None

    def test_a_fast_riser_with_a_moving_base_is_spared_if_the_anchor_holds(self):
        assert reject_erratic(growth(base_drift_px_per_min=30.0, anchor_score=0.8)) is None


class TestFlare:
    @staticmethod
    def scene_of(image):
        return assess_scene(image)

    def test_a_blown_out_blob_is_rejected(self, clear_frame):
        image = clear_frame.copy()
        mask = np.zeros(image.shape[:2], np.uint8)
        mask[100:200, 300:400] = 255
        image[mask > 0] = 255
        region = plume_region(mask=mask)
        rejection = reject_flare(region, image, self.scene_of(image))
        assert rejection is not None
        assert rejection.code == "FLARE"

    def test_a_grey_plume_is_not_flare(self, clear_frame):
        image = clear_frame.copy()
        mask = np.zeros(image.shape[:2], np.uint8)
        mask[100:200, 300:400] = 255
        image[mask > 0] = 175
        assert reject_flare(plume_region(mask=mask), image, self.scene_of(image)) is None

    def test_an_empty_mask_is_not_an_error(self, clear_frame):
        region = plume_region(mask=np.zeros(clear_frame.shape[:2], np.uint8))
        assert reject_flare(region, clear_frame, self.scene_of(clear_frame)) is None


class TestRidgeRegistration:
    def test_a_thin_sliver_on_a_shaken_frame_is_rejected(self):
        rejection = reject_ridge_registration(
            plume_region(w=300, h=8, base_below_horizon=2.0), SHAKEN, FRAME_HEIGHT
        )
        assert rejection is not None
        assert rejection.code == "RIDGE_REGISTRATION"

    def test_the_same_sliver_on_a_steady_frame_is_kept(self):
        assert reject_ridge_registration(
            plume_region(w=300, h=8, base_below_horizon=2.0), CORRECTED, FRAME_HEIGHT
        ) is None

    def test_a_tall_column_on_a_shaken_frame_is_kept(self):
        assert reject_ridge_registration(plume_region(), SHAKEN, FRAME_HEIGHT) is None


class TestWeatherFront:
    def test_a_frame_wide_contrast_collapse_is_rejected(self, foggy_frames):
        scene = assess_scene(foggy_frames[-1])
        rejection = reject_weather_front(plume_region(texture_ratio=0.9), scene, 900.0)
        assert rejection is not None
        assert rejection.code == "WEATHER_FRONT"

    def test_a_clear_frame_is_not_a_weather_front(self, clear_frame):
        scene = assess_scene(clear_frame)
        assert reject_weather_front(plume_region(), scene, 700.0) is None

    def test_with_no_baseline_the_rule_holds_off(self, foggy_frames):
        scene = assess_scene(foggy_frames[-1])
        assert reject_weather_front(plume_region(), scene, None) is None

    def test_a_patch_much_softer_than_the_rest_still_counts(self, foggy_frames):
        """Fog everywhere, but this patch is far softer than the fog: that is a
        local event inside bad weather, not the weather itself."""
        scene = assess_scene(foggy_frames[-1])
        assert reject_weather_front(plume_region(texture_ratio=0.01), scene, 900.0) is None


class TestTooBrief:
    def test_one_frame_is_not_a_column(self):
        rejection = reject_too_brief(growth(frames=1), minimum_frames=3)
        assert rejection is not None
        assert rejection.code == "TOO_BRIEF"
        assert rejection.confidence < 0.5, "being early is not the same as being wrong"

    def test_enough_frames_passes(self):
        assert reject_too_brief(growth(frames=5), minimum_frames=3) is None


class TestBlindReason:
    def test_each_blind_state_becomes_a_rejection(self, night_frames, foggy_frames):
        for frame, code in ((night_frames[0], "CAMERA_NIGHT"), (foggy_frames[-1], "CAMERA_FOG")):
            rejection = blind_reason(assess_scene(frame))
            assert rejection is not None
            assert rejection.code == code
            assert rejection.confidence == 1.0

    def test_a_usable_camera_produces_no_rejection(self, clear_frame):
        assert blind_reason(assess_scene(clear_frame)) is None


def test_every_rejection_serialises_with_its_numbers():
    rejection = Rejection("X", "thing", "because", {"a": 1.23456, "b": "text"})
    payload = rejection.to_dict()
    assert payload["details"]["a"] == pytest.approx(1.235)
    assert payload["details"]["b"] == "text"
    assert set(payload) == {"code", "impostor", "message", "confidence", "details"}
