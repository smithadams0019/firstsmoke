"""Background modelling, region measurement and column growth.

The interesting tests here are the ones with an analytic answer: a known gain
and offset that the photometric fit has to recover, a known shading gradient it
has to ignore, and a track constructed by hand with a known rise and a known
base drift.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import cv2
import numpy as np
import pytest

from firstsmoke.background import BackgroundModel, _low_frequency, photometric_fit, tod_bucket
from firstsmoke.candidates import Region, _hysteresis, extract_regions
from firstsmoke.scene import find_horizon
from firstsmoke.tracks import Tracker

START = datetime(2026, 9, 16, 19, 0, tzinfo=UTC)


class TestPhotometricFit:
    def test_it_recovers_a_gain_and_offset_we_applied(self, clear_frame):
        gray = cv2.cvtColor(clear_frame, cv2.COLOR_BGR2GRAY)
        brighter = cv2.convertScaleAbs(gray, alpha=1.15, beta=12.0)
        gain, offset, residual = photometric_fit(brighter, gray)
        assert gain == pytest.approx(1.15, abs=0.02)
        assert offset == pytest.approx(12.0, abs=2.0)
        assert residual < 3.0

    def test_an_unchanged_frame_gives_unity(self, clear_frame):
        gray = cv2.cvtColor(clear_frame, cv2.COLOR_BGR2GRAY)
        gain, offset, _ = photometric_fit(gray, gray)
        assert gain == pytest.approx(1.0, abs=0.01)
        assert offset == pytest.approx(0.0, abs=0.5)

    def test_a_small_bright_patch_does_not_drag_the_fit(self, clear_frame):
        """The robust trim exists so a plume cannot rewrite the exposure model."""
        gray = cv2.cvtColor(clear_frame, cv2.COLOR_BGR2GRAY)
        changed = gray.copy()
        changed[200:320, 400:520] = 250
        gain, offset, _ = photometric_fit(changed, gray)
        assert gain == pytest.approx(1.0, abs=0.05)
        assert offset == pytest.approx(0.0, abs=4.0)

    def test_a_flat_background_falls_back_to_an_offset(self):
        flat = np.full((200, 200), 100, np.uint8)
        gain, offset, _ = photometric_fit(np.full((200, 200), 130, np.uint8), flat)
        assert gain == pytest.approx(1.0)
        assert offset == pytest.approx(30.0, abs=1.0)


class TestLowFrequencyShading:
    def test_a_smooth_gradient_is_removed(self):
        h, w = 400, 600
        gradient = np.tile(np.linspace(-8, 8, w, dtype=np.float32), (h, 1))
        corrected = gradient - _low_frequency(gradient)
        assert np.abs(corrected).mean() < 0.35 * np.abs(gradient).mean()

    def test_a_compact_plume_survives_the_correction(self):
        h, w = 400, 600
        field = np.zeros((h, w), np.float32)
        field[180:240, 280:330] = 25.0
        corrected = field - _low_frequency(field)
        assert corrected[200, 300] > 0.9 * 25.0, "the correction must not eat a plume"

    def test_the_correction_is_clamped(self):
        huge = np.full((300, 300), 400.0, np.float32)
        assert np.abs(_low_frequency(huge)).max() <= 10.0 + 1e-6


class TestBackgroundModel:
    def test_it_reports_itself_cold_until_it_has_seen_enough(self, clean_sequence):
        model = BackgroundModel("test")
        assert not model.warmed
        for frame in clean_sequence.frames[:3]:
            change = model.update(frame.image, frame.hour_utc)
            assert not change.warmed
        for frame in clean_sequence.frames[3:6]:
            change = model.update(frame.image, frame.hour_utc)
        assert change.warmed
        assert model.warmed

    def test_a_plume_shows_up_in_the_signed_difference(self, plume_sequence):
        model = BackgroundModel("test")
        for frame in plume_sequence.frames[:6]:
            model.update(frame.image, frame.hour_utc)
        quiet = model.update(plume_sequence.frames[5].image, plume_sequence.frames[5].hour_utc)
        late = model.update(plume_sequence.frames[-1].image, plume_sequence.frames[-1].hour_utc)
        assert np.abs(late.signed).max() > np.abs(quiet.signed).max()
        assert float(np.mean(late.foreground > 0)) > 0.0

    def test_freezing_the_model_stops_it_learning(self, plume_sequence):
        """The plume must not be allowed to become the background."""
        learning = BackgroundModel("learning")
        frozen = BackgroundModel("frozen")
        for frame in plume_sequence.frames[:6]:
            learning.update(frame.image, frame.hour_utc)
            frozen.update(frame.image, frame.hour_utc)
        seen_learning = learning.frames_seen
        for frame in plume_sequence.frames[6:]:
            learning.update(frame.image, frame.hour_utc)
            frozen.update(frame.image, frame.hour_utc, learn=False)
        assert learning.frames_seen > seen_learning
        assert frozen.frames_seen == seen_learning

    def test_time_of_day_buckets_wrap(self):
        assert tod_bucket(0.0) == tod_bucket(24.0)
        assert tod_bucket(1.0) != tod_bucket(13.0)

    def test_a_reference_is_returned_for_a_nearby_hour(self, clean_sequence):
        model = BackgroundModel("test")
        for frame in clean_sequence.frames[:6]:
            model.update(frame.image, frame.hour_utc)
        assert model.reference_for(clean_sequence.frames[0].hour_utc) is not None
        assert model.reference_for(3.0) is not None, "fall back to the nearest bucket we have"

    def test_an_unseen_camera_has_no_reference(self):
        assert BackgroundModel("test").reference_for(12.0) is None


class TestHysteresis:
    """The thresholds are set from the field's own noise, so the tests supply
    noise. On a noiseless field everything is a strong seed and the test proves
    nothing."""

    @staticmethod
    def noisy(seed: int = 3) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.normal(0.0, 1.5, (200, 200)).astype(np.float32)

    def test_a_faint_tail_joined_to_a_strong_core_is_kept(self):
        field = self.noisy()
        field[100:110, 95:105] = 40.0  # a confident core
        field[60:100, 97:103] = 4.0  # a faint tail, touching the core
        mask = _hysteresis(field, floor=2.0)
        assert mask[105, 100] == 255
        assert mask[70, 100] == 255, "the faint tail should survive because it joins the core"

    def test_the_same_faint_patch_on_its_own_is_dropped(self):
        field = self.noisy()
        field[60:100, 97:103] = 4.0
        assert _hysteresis(field, floor=2.0)[70, 100] == 0

    def test_a_noisier_frame_raises_its_own_bar(self):
        """Same signal, ten times the noise: it should stop being a detection."""
        quiet = self.noisy()
        quiet[100:110, 95:105] = 9.0
        loud = (self.noisy() * 10.0).astype(np.float32)
        loud[100:110, 95:105] = 9.0
        assert _hysteresis(quiet, floor=2.0)[105, 100] == 255
        assert _hysteresis(loud, floor=2.0)[105, 100] == 0

    def test_an_empty_field_gives_an_empty_mask(self):
        assert not _hysteresis(np.zeros((50, 50), np.float32), floor=2.0).any()


class TestRegionMeasurement:
    @staticmethod
    def measure(plume_sequence, index: int):
        model = BackgroundModel("test")
        for frame in plume_sequence.frames[:6]:
            model.update(frame.image, frame.hour_utc)
        frame = plume_sequence.frames[index]
        change = model.update(frame.image, frame.hour_utc)
        horizon = find_horizon(frame.image)
        return extract_regions(
            change.alignment.image, change, horizon,
            reference=model.reference_for(frame.hour_utc), frame_index=index,
        )

    def test_the_plume_is_found_and_sits_on_the_skyline(self, plume_sequence):
        regions = self.measure(plume_sequence, len(plume_sequence) - 1)
        assert regions, "no candidate region at all on a frame with an obvious plume"
        plume = max(regions, key=lambda r: r.area)
        assert plume.area > 400
        assert abs(plume.base_below_horizon) < 90, "the base should be near the ridge"
        assert plume.h > plume.w * 0.5, "a column is taller than it is wide, or close to it"

    def test_a_region_carries_every_measurement_it_promises(self, plume_sequence):
        regions = self.measure(plume_sequence, len(plume_sequence) - 1)
        payload = max(regions, key=lambda r: r.area).to_dict()
        for key in (
            "bbox", "area", "texture_ratio", "saturation_ratio", "edge_softness",
            "base_below_horizon", "base_point", "greyness",
        ):
            assert key in payload, f"{key} missing from the region record"

    def test_the_base_point_is_at_the_bottom_of_the_mask(self, plume_sequence):
        plume = max(self.measure(plume_sequence, len(plume_sequence) - 1), key=lambda r: r.area)
        assert plume.base_cy > plume.cy, "the base must sit below the centroid"
        assert plume.y <= plume.base_cy <= plume.base_y + 1


def make_region(x: int, y: int, w: int, h: int, *, shape=(400, 600)) -> Region:
    mask = np.zeros(shape, np.uint8)
    mask[y : y + h, x : x + w] = 255
    return Region(
        x=x, y=y, w=w, h=h, area=w * h, cx=x + w / 2, cy=y + h / 2, mask=mask,
        base_cx=x + w / 2, base_cy=y + h,
    )


class TestGrowth:
    def test_a_rising_column_with_a_fixed_base(self):
        """Top climbs 10 px a minute, base does not move. Anchor must be high."""
        tracker = Tracker("cam", frame_width=600)
        for i in range(6):
            top = 200 - 10 * i
            tracker.update(i, START + timedelta(minutes=i), [make_region(290, top, 20, 200 - top + 20)])
        track = tracker.tracks[0]
        growth = track.growth()
        assert growth.frames == 6
        assert growth.top_rise_px_per_min == pytest.approx(10.0, abs=0.5)
        assert growth.base_drift_px_per_min == pytest.approx(0.0, abs=0.5)
        assert growth.anchor_score > 0.9
        assert growth.area_ratio > 2.0

    def test_a_cloud_translating_bodily(self):
        """Everything moves together, area constant. Anchor must be low."""
        tracker = Tracker("cam", frame_width=600)
        for i in range(6):
            tracker.update(i, START + timedelta(minutes=i), [make_region(100 + 12 * i, 80, 60, 30)])
        growth = tracker.tracks[0].growth()
        assert growth.top_rise_px_per_min == pytest.approx(0.0, abs=0.5)
        assert growth.base_drift_px_per_min == pytest.approx(12.0, abs=1.0)
        assert growth.anchor_score < 0.1
        assert growth.area_ratio == pytest.approx(1.0)

    def test_jitter_is_not_counted_as_travel(self):
        """The reason base drift is a net displacement and not a path length."""
        tracker = Tracker("cam", frame_width=600)
        for i in range(9):
            wobble = 8 if i % 2 else -8
            tracker.update(i, START + timedelta(minutes=i), [make_region(300 + wobble, 100, 40, 60)])
        growth = tracker.tracks[0].growth()
        assert growth.base_drift_px_per_min < 3.0, "a blob wobbling in place has not travelled"
        assert growth.base_jitter_px > 10.0, "but the wobble itself must still be measured"

    def test_a_single_observation_reads_zero_rather_than_exploding(self):
        tracker = Tracker("cam", frame_width=600)
        tracker.update(0, START, [make_region(100, 100, 20, 20)])
        growth = tracker.tracks[0].growth()
        assert growth.frames == 1
        assert growth.anchor_score == 0.0
        assert growth.area_ratio == 1.0


class TestTracker:
    def test_a_growing_region_stays_on_one_track(self):
        tracker = Tracker("cam", frame_width=600)
        for i in range(5):
            tracker.update(i, START + timedelta(minutes=i), [make_region(280, 180 - 12 * i, 30 + 4 * i, 40 + 12 * i)])
        assert len(tracker.tracks) == 1
        assert len(tracker.tracks[0]) == 5

    def test_two_separate_things_get_two_tracks(self):
        tracker = Tracker("cam", frame_width=600)
        for i in range(4):
            tracker.update(
                i, START + timedelta(minutes=i),
                [make_region(60, 100, 30, 30), make_region(450, 250, 30, 30)],
            )
        assert len([t for t in tracker.tracks if len(t) > 1]) == 2

    def test_a_track_survives_a_single_missed_frame(self):
        tracker = Tracker("cam", frame_width=600)
        tracker.update(0, START, [make_region(300, 100, 40, 40)])
        tracker.update(1, START + timedelta(minutes=1), [])
        tracker.update(2, START + timedelta(minutes=2), [make_region(300, 100, 40, 40)])
        assert len(tracker.tracks) == 1
        assert len(tracker.tracks[0]) == 2

    def test_a_track_closes_after_too_many_misses(self):
        tracker = Tracker("cam", frame_width=600)
        tracker.update(0, START, [make_region(300, 100, 40, 40)])
        for i in range(1, 5):
            tracker.update(i, START + timedelta(minutes=i), [])
        assert not tracker.tracks[0].alive
