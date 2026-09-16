"""The escalation loop, end to end, against fires we placed ourselves.

These are the tests that matter most, because they are the only ones that can
check the whole chain: a latitude and longitude we chose becomes a bearing,
becomes a pixel column, becomes a rendered plume, and the agent has to get back
to the coordinates without ever being told them.

They are slow — each one renders and then analyses a few dozen frames — so they
carry the ``slow`` marker and the incidents are session-scoped fixtures.
"""

from __future__ import annotations

import pytest

from firstsmoke.agent import Lookout, ReplaySource, State
from firstsmoke.detector import CameraWatch, Verdict
from firstsmoke.geometry import haversine_m, signed_delta

pytestmark = pytest.mark.slow


def run(incident, **kwargs):
    events: list[dict] = []
    lookout = Lookout(ReplaySource(incident), on_event=events.append, **kwargs)
    alert = lookout.run()
    return lookout, alert, events


class TestConfirmedFire:
    """Three cameras ringed around one fire. The known-answer case."""

    @pytest.fixture(scope="class")
    @staticmethod
    def result(confirmed_incident):
        return run(confirmed_incident)

    def test_it_reaches_an_alert(self, result):
        lookout, alert, _ = result
        assert lookout.state is State.ALERTED
        assert alert is not None

    def test_the_fix_lands_where_we_put_the_fire(self, result, confirmed_incident):
        _, alert, _ = result
        truth = confirmed_incident.truth
        assert alert.fix is not None, f"no position: {alert.fix_refusal}"
        error = haversine_m(alert.fix.lat, alert.fix.lon, truth["fire_lat"], truth["fire_lon"])
        assert error < 600, f"fix was {error:.0f} m from the fire we rendered"

    def test_the_reported_uncertainty_covers_the_real_error(self, result, confirmed_incident):
        """An error bar that does not contain the truth is worse than none."""
        _, alert, _ = result
        truth = confirmed_incident.truth
        error = haversine_m(alert.fix.lat, alert.fix.lon, truth["fire_lat"], truth["fire_lon"])
        assert error <= alert.fix.semi_major_m * 3.0

    def test_the_bearings_match_each_camera_true_line_of_sight(self, result, confirmed_incident):
        _, alert, _ = result
        network = confirmed_incident.network
        truth = confirmed_incident.truth
        from firstsmoke.geometry import bearing_between

        for ray in alert.rays:
            camera = network.get(ray["camera_id"])
            expected = bearing_between(camera.lat, camera.lon, truth["fire_lat"], truth["fire_lon"])
            assert abs(signed_delta(ray["bearing_deg"], expected)) < 4.0

    def test_it_consulted_cameras_it_chose_from_the_bearing(self, result):
        lookout, _, _ = result
        assert lookout.consultations, "the agent never asked a neighbour anything"
        supported = [c for c in lookout.consultations if c.outcome == "supported"]
        assert supported, "no neighbour ever corroborated"
        for consultation in supported:
            offset = abs(
                signed_delta(
                    consultation.detection.bearing_deg, consultation.expected_bearing_deg
                )
            )
            assert offset < 7.0, "a supporting camera must have seen it where geometry predicted"

    def test_every_camera_it_chose_to_read_overlooks_the_bearing(self, result, confirmed_incident):
        """The selection is geometric, so every chosen camera must actually have
        the suspect bearing inside its own field of view."""
        lookout, _, _ = result
        network = confirmed_incident.network
        from firstsmoke.geometry import covers_bearing

        checked = 0
        for transition in lookout.transitions:
            if transition.trigger != "bearing_selected_cameras":
                continue
            for candidate in transition.data.get("candidates", []):
                camera = network.get(candidate["camera_id"])
                assert covers_bearing(
                    candidate["expected_bearing_deg"], camera.azimuth_deg, camera.hfov_deg
                )
                checked += 1
        assert checked > 0, "no camera selection was ever recorded"

    def test_the_transition_log_tells_the_whole_story(self, result):
        lookout, _, _ = result
        triggers = [t.trigger for t in lookout.transitions]
        assert "weak_detection" in triggers
        assert "bearing_selected_cameras" in triggers
        assert "crossed_bearings" in triggers
        assert "alert_raised" in triggers
        for transition in lookout.transitions:
            assert len(transition.detail) > 15, f"{transition.trigger} has no explanation"

    def test_the_alert_explains_itself_in_sentences(self, result):
        _, alert, _ = result
        assert len(alert.reasoning) >= 2
        assert alert.headline
        assert any("cross" in line or "degrees" in line for line in alert.reasoning)

    def test_it_serialises_whole(self, result):
        _, alert, _ = result
        payload = alert.to_dict()
        assert payload["fix"]["lat"]
        assert payload["consultations"]
        assert payload["transitions"]
        assert payload["state"] == "alerted"

    def test_progress_events_are_emitted_for_the_ui(self, result):
        _, _, events = result
        kinds = {event["type"] for event in events}
        assert {"tick", "reading", "transition", "consultation", "alert"} <= kinds


class TestStandDown:
    """A cloud and nothing else. The loop must end quiet."""

    @pytest.fixture(scope="class")
    @staticmethod
    def result(quiet_incident):
        return run(quiet_incident)

    def test_no_alert_is_raised(self, result):
        _, alert, _ = result
        assert alert is None

    def test_it_does_not_end_in_an_alerting_state(self, result):
        lookout, _, _ = result
        assert lookout.state in (State.WATCH, State.STOOD_DOWN, State.CONSULT, State.SUSPECT)

    def test_if_it_stood_down_it_said_why(self, result):
        lookout, _, _ = result
        stand_downs = [t for t in lookout.transitions if t.to_state is State.STOOD_DOWN]
        for transition in stand_downs:
            assert transition.trigger in {"not_corroborated", "faded"}
            assert len(transition.detail) > 20

    def test_a_stand_down_returns_to_watching(self, quiet_incident):
        """Standing down must not end the watch. A lookout that saw a cloud,
        checked and was satisfied goes back to watching the hill."""
        lookout, _, _ = run(quiet_incident)
        assert State.ALERTED not in {t.to_state for t in lookout.transitions}
        assert lookout.readings, "the agent stopped reading frames"
        last_tick = max(r.frame_index for r in lookout.readings)
        assert last_tick >= len(quiet_incident.timeline()) - 3


class TestBlindNeighbours:
    """A fire one camera can see, and neighbours that are fogged and dark.

    The wrong answer is a clean stand-down, because that reads as "nothing
    there". The right answer is to say the cover is not there.
    """

    @pytest.fixture(scope="class")
    @staticmethod
    def result(blind_incident):
        return run(blind_incident)

    def test_the_blind_cameras_are_named_as_unusable(self, result):
        lookout, _, _ = result
        assert lookout.unusable, "no camera was reported unusable"
        for reason in lookout.unusable.values():
            assert len(reason) > 20

    def test_it_does_not_quietly_stand_down(self, result):
        lookout, alert, _ = result
        if alert is not None:
            assert alert.state in (State.ALERTED, State.NEEDS_HUMAN)
        blind_states = [t for t in lookout.transitions if t.trigger == "neighbours_blind"]
        stood_down = [t for t in lookout.transitions if t.trigger == "not_corroborated"]
        assert blind_states or alert is not None or not stood_down

    def test_a_blind_camera_is_never_reported_as_clear(self, blind_incident):
        camera = blind_incident.network.get("syn1-mobo-c")
        watch = CameraWatch(camera)
        for frame in blind_incident.sequences["syn1-mobo-c"].frames[:6]:
            reading = watch.observe(frame)
            assert reading.verdict is not Verdict.CLEAR
            assert reading.verdict is Verdict.BLIND
            assert reading.scene.blind


class TestNeighbourSelection:
    def test_the_bearing_decides_which_cameras_are_read(self, confirmed_incident):
        """The claim under test: different pixels select different cameras.

        A bearing pointing at the middle of the ring reaches cameras that
        overlook it; a bearing pointing away from every other summit reaches
        none. Nothing about this is a fixed neighbour list.
        """
        network = confirmed_incident.network
        origin = network.get("syn0-mobo-c")
        inward = network.consultable(origin, origin.azimuth_deg)
        outward = network.consultable(origin, (origin.azimuth_deg + 180.0) % 360.0)
        assert inward, "looking into the ring should find neighbours"
        assert len(outward) < len(inward)

    def test_a_neighbour_is_told_where_in_its_frame_to_look(self, confirmed_incident):
        network = confirmed_incident.network
        origin = network.get("syn0-mobo-c")
        for consultation in network.consultable(origin, origin.azimuth_deg):
            assert consultation.expected_x is not None
            assert 0 <= consultation.expected_x <= 1024
            assert consultation.crossing_angle_deg >= 10.0

    def test_cameras_on_the_same_mast_are_never_consulted(self, confirmed_incident):
        network = confirmed_incident.network
        origin = network.get("syn0-mobo-c")
        for consultation in network.consultable(origin, origin.azimuth_deg):
            assert consultation.camera.site_id != origin.site_id
