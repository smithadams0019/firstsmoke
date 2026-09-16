"""One camera watching one sequence: does it find the column, and does it explain itself."""

from __future__ import annotations

import pytest

from firstsmoke.detector import CONFIRM_AT, SUSPECT_AT, CameraWatch, Verdict
from firstsmoke.geometry import signed_delta


def watch_through(camera, sequence, **kwargs):
    watch = CameraWatch(camera, **kwargs)
    return watch, [watch.observe(frame) for frame in sequence]


class TestPlumeSequence:
    @pytest.fixture(scope="class")
    @staticmethod
    def readings(lone_camera, plume_sequence):
        return watch_through(lone_camera, plume_sequence)

    def test_it_says_nothing_before_the_plume_starts(self, readings):
        _, all_readings = readings
        for reading in all_readings[:6]:
            assert reading.verdict in (Verdict.CLEAR, Verdict.SUSPECT)
            assert reading.confidence < CONFIRM_AT

    def test_it_reaches_at_least_suspicion_once_the_plume_is_established(self, readings):
        _, all_readings = readings
        late = [r.confidence for r in all_readings[-5:]]
        assert max(late) >= SUSPECT_AT, f"best late confidence was only {max(late):.2f}"

    def test_the_confidence_rises_as_the_column_grows(self, readings):
        _, all_readings = readings
        early = max((r.confidence for r in all_readings[6:9]), default=0.0)
        late = max((r.confidence for r in all_readings[-4:]), default=0.0)
        assert late > early

    def test_the_bearing_is_inside_the_camera_field_of_view(self, readings, lone_camera):
        _, all_readings = readings
        best = max(all_readings, key=lambda r: r.confidence).best
        assert abs(signed_delta(best.bearing_deg, lone_camera.azimuth_deg)) < lone_camera.hfov_deg / 2

    def test_the_bearing_carries_an_uncertainty_that_grows_with_the_plume(self, readings):
        _, all_readings = readings
        detections = [r.best for r in all_readings if r.best]
        assert all(d.bearing_sigma_deg >= 0.5 for d in detections), "mounting error is a floor"
        widest = max(detections, key=lambda d: d.region.w)
        narrowest = min(detections, key=lambda d: d.region.w)
        assert widest.bearing_sigma_deg >= narrowest.bearing_sigma_deg

    def test_every_detection_lists_its_reasons_with_numbers(self, readings):
        _, all_readings = readings
        best = max(all_readings, key=lambda r: r.confidence).best
        assert best.reasons
        names = {r.name for r in best.reasons}
        assert {"growth", "rise", "anchored", "veiling", "attachment"} <= names
        for reason in best.reasons:
            assert 0.0 <= reason.value <= 1.0
            assert len(reason.text) > 15
            assert any(ch.isdigit() for ch in reason.text), "a reason must quote its measurement"

    def test_the_weighted_reasons_add_up_to_the_support(self, readings):
        _, all_readings = readings
        best = max(all_readings, key=lambda r: r.confidence).best
        total = sum(r.contribution for r in best.reasons)
        weights = sum(r.weight for r in best.reasons)
        assert total / weights == pytest.approx(best.support, abs=0.01)

    def test_the_summary_is_a_sentence(self, readings):
        _, all_readings = readings
        best = max(all_readings, key=lambda r: r.confidence).best
        assert best.summary().endswith(".")
        assert "bearing" in best.summary()

    def test_a_reading_serialises(self, readings):
        _, all_readings = readings
        payload = max(all_readings, key=lambda r: r.confidence).to_dict()
        assert set(payload) >= {"camera_id", "verdict", "confidence", "usability", "detections"}


class TestImpostors:
    def test_a_cloud_alone_never_reaches_confirmation(self, lone_camera, cloud_sequence):
        _, readings = watch_through(lone_camera, cloud_sequence)
        assert max(r.confidence for r in readings) < CONFIRM_AT

    def test_a_dust_plume_alone_never_reaches_confirmation(self, lone_camera, dust_sequence):
        _, readings = watch_through(lone_camera, dust_sequence)
        assert max(r.confidence for r in readings) < CONFIRM_AT

    def test_a_clean_hillside_stays_clear(self, lone_camera, clean_sequence):
        _, readings = watch_through(lone_camera, clean_sequence)
        assert max(r.confidence for r in readings) < CONFIRM_AT
        assert all(r.verdict is not Verdict.COLUMN for r in readings)


class TestBlindCamera:
    def test_night_is_reported_as_blind_not_clear(self, lone_camera, blind_incident):
        camera = blind_incident.network.get("syn2-mobo-c")
        watch = CameraWatch(camera)
        reading = watch.observe(blind_incident.sequences["syn2-mobo-c"].frames[0])
        assert reading.verdict is Verdict.BLIND
        assert reading.scene.blind
        assert not reading.detections, "a blind camera must not produce detections"

    def test_a_blind_reading_still_carries_its_reason(self, blind_incident):
        camera = blind_incident.network.get("syn1-mobo-c")
        watch = CameraWatch(camera)
        reading = watch.observe(blind_incident.sequences["syn1-mobo-c"].frames[0])
        assert len(reading.scene.reason) > 20
        assert reading.to_dict()["usability_reason"] == reading.scene.reason


class TestMonochromeCamera:
    def test_a_mono_camera_is_not_punished_for_having_no_colour(self, lone_camera, plume_sequence):
        """Dropping the desaturation term must renormalise, not score it zero."""
        from dataclasses import replace

        mono = replace(lone_camera, imager="monochrome")
        _, colour_readings = watch_through(lone_camera, plume_sequence)
        _, mono_readings = watch_through(mono, plume_sequence)
        colour_best = max(r.confidence for r in colour_readings)
        mono_best = max(r.confidence for r in mono_readings)
        assert mono_best > colour_best * 0.7

    def test_a_mono_camera_does_not_report_a_desaturation_reason(self, lone_camera, plume_sequence):
        from dataclasses import replace

        mono = replace(lone_camera, imager="monochrome")
        _, readings = watch_through(mono, plume_sequence)
        best = max(readings, key=lambda r: r.confidence).best
        assert "desaturation" not in {r.name for r in best.reasons}


class TestWarmUp:
    def test_no_tracks_are_seeded_before_the_model_is_warm(self, lone_camera, plume_sequence):
        _, readings = watch_through(lone_camera, plume_sequence)
        cold = [r for r in readings if not r.warmed]
        assert cold, "the model should report itself cold at the start"
        for reading in cold:
            assert not reading.detections
