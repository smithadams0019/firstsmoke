"""Horizon finding, usability verdicts and camera-shake correction."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from firstsmoke.scene import (
    MAX_CORRECTABLE_SHIFT_PX,
    Usability,
    align_to,
    assess_scene,
    find_horizon,
)
from firstsmoke.synth import SequenceSpec, ViewSpec, camera_wobble, render_sequence, render_view


class TestHorizon:
    def test_it_finds_the_ridge_we_drew(self):
        spec = ViewSpec(seed=21)
        image = render_view(spec)
        horizon = find_horizon(image)
        truth = spec.ridge()
        error = np.abs(horizon.boundary.astype(float) - truth)
        assert np.median(error) < 2.0, f"median boundary error {np.median(error):.1f} px"
        assert abs(float(np.mean(horizon.boundary.astype(float) - truth))) < 1.0, (
            "the boundary must not be biased high or low; a systematic offset moves every "
            "attachment measurement in the same direction"
        )
        assert not horizon.flat
        assert horizon.confidence > 0.3

    def test_it_follows_the_relief_rather_than_flattening_it(self):
        spec = ViewSpec(seed=21, relief_fraction=0.14)
        horizon = find_horizon(render_view(spec))
        drawn_relief = float(spec.ridge().max() - spec.ridge().min())
        found_relief = float(horizon.boundary.max() - horizon.boundary.min())
        assert found_relief > 0.6 * drawn_relief

    def test_a_featureless_frame_is_reported_as_flat(self):
        blank = np.full((400, 600, 3), 190, np.uint8)
        horizon = find_horizon(blank)
        assert horizon.flat
        assert horizon.confidence == 0.0

    def test_the_sky_mask_is_above_the_boundary(self):
        spec = ViewSpec(seed=7)
        image = render_view(spec)
        horizon = find_horizon(image)
        mask = horizon.sky_mask(image.shape[:2])
        column = image.shape[1] // 2
        boundary = int(horizon.boundary[column])
        assert mask[max(boundary - 20, 0), column] == 255
        assert mask[min(boundary + 20, image.shape[0] - 1), column] == 0

    def test_the_band_mask_straddles_the_boundary(self):
        spec = ViewSpec(seed=7)
        image = render_view(spec)
        horizon = find_horizon(image)
        band = horizon.band_mask(image.shape[:2], above_px=10, below_px=10)
        column = image.shape[1] // 2
        boundary = int(horizon.boundary[column])
        assert band[boundary, column] == 255
        assert band[max(boundary - 40, 0), column] == 0

    def test_height_above_is_signed(self):
        spec = ViewSpec(seed=7)
        horizon = find_horizon(render_view(spec))
        boundary = float(horizon.boundary[500])
        assert horizon.height_above(500, boundary - 30) == pytest.approx(30.0, abs=1.0)
        assert horizon.height_above(500, boundary + 30) == pytest.approx(-30.0, abs=1.0)


class TestUsability:
    def test_a_clear_frame_is_usable(self, clear_frame):
        state = assess_scene(clear_frame)
        assert state.usability in (Usability.USABLE, Usability.DEGRADED)
        assert not state.blind

    def test_night_is_recognised_and_is_not_clear(self, night_frames):
        state = assess_scene(night_frames[0])
        assert state.usability is Usability.NIGHT
        assert state.blind
        assert "night" in state.reason or "brightness" in state.reason

    def test_fog_is_recognised(self, foggy_frames):
        state = assess_scene(foggy_frames[-1])
        assert state.usability is Usability.FOG
        assert state.blind

    def test_fog_is_judged_against_this_camera_own_normal_when_we_have_one(self, clear_frame):
        """A camera whose clear view measures 700 and a camera whose clear view
        measures 9,000 both go blind when their own detail collapses. An absolute
        threshold cannot express that, so the baseline is what the test uses."""
        state = assess_scene(clear_frame, baseline_contrast=40_000.0)
        assert state.usability is Usability.DEGRADED, (
            "a frame at a fiftieth of its camera's normal contrast is not 'clear'"
        )
        assert assess_scene(clear_frame).usability is Usability.USABLE

    def test_a_bright_clipped_sky_is_not_glare(self, clear_frame):
        """Real HPWREN frames clip up to nine percent of their pixels on an
        ordinary clear afternoon. A glare test tuned below that declares a
        working camera blind every day."""
        speckled = clear_frame.copy()
        h = speckled.shape[0]
        speckled[: int(h * 0.09), :] = 255
        assert assess_scene(speckled).usability is not Usability.GLARE

    def test_a_frozen_feed_is_recognised_after_a_couple_of_repeats(self, frozen_frames):
        state = None
        repeats = 0
        previous = None
        for frame in frozen_frames:
            state = assess_scene(frame, previous=previous, repeat_count=repeats)
            repeats = state.repeat_count
            previous = frame
        assert state is not None
        assert state.usability is Usability.FROZEN
        assert state.blind
        assert state.repeat_count >= 2

    def test_a_live_feed_is_never_called_frozen(self, lone_camera):
        """A live sensor always carries read noise, even on a still scene."""
        frames = render_sequence(
            SequenceSpec(camera=lone_camera, view=ViewSpec(seed=4), frames=6, shake_px=0.0)
        )
        repeats = 0
        previous = None
        for frame in frames:
            state = assess_scene(frame, previous=previous, repeat_count=repeats)
            repeats = state.repeat_count
            previous = frame
            assert state.usability is not Usability.FROZEN

    def test_glare_is_recognised(self, clear_frame):
        blown = clear_frame.copy()
        cv2.circle(blown, (400, 180), 260, (255, 255, 255), -1)
        blown = cv2.add(blown, 80)
        state = assess_scene(blown)
        assert state.usability is Usability.GLARE
        assert state.blind

    def test_every_blind_state_carries_a_sentence_a_person_can_read(
        self, night_frames, foggy_frames
    ):
        for frame in (night_frames[0], foggy_frames[-1]):
            state = assess_scene(frame)
            assert len(state.reason) > 20
            assert state.reason[0].islower() or state.reason[0].isdigit()

    def test_the_state_serialises(self, clear_frame):
        payload = assess_scene(clear_frame).to_dict()
        assert set(payload) >= {"usability", "reason", "mean_luma", "contrast", "horizon"}


class TestAlignment:
    def test_a_small_shift_is_measured_and_undone(self, clear_frame):
        moved = camera_wobble(clear_frame, 6.0, -3.0)
        alignment = align_to(moved, clear_frame)
        assert alignment.corrected
        assert alignment.dx == pytest.approx(6.0, abs=1.0)
        assert alignment.dy == pytest.approx(-3.0, abs=1.0)
        before = float(cv2.absdiff(moved, clear_frame).mean())
        after = float(cv2.absdiff(alignment.image, clear_frame).mean())
        assert after < before * 0.6, f"alignment made it worse: {before:.2f} -> {after:.2f}"

    def test_a_shift_too_large_to_trust_is_left_alone(self, clear_frame):
        moved = camera_wobble(clear_frame, MAX_CORRECTABLE_SHIFT_PX + 25.0, 0.0)
        alignment = align_to(moved, clear_frame)
        assert not alignment.corrected
        assert alignment.shift_px > MAX_CORRECTABLE_SHIFT_PX

    def test_an_unmoved_frame_is_not_warped(self, clear_frame):
        alignment = align_to(clear_frame, clear_frame)
        assert not alignment.corrected
        assert alignment.shift_px < 0.5
        assert alignment.image is clear_frame
