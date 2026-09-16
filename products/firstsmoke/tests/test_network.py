"""The camera registry and the incident bundle format.

The HPWREN parsing tests run against the real ``sites.js`` shipped in
``data/network/``, because a parser tested only against a hand-written sample
passes right up until it meets the file it was written for.
"""

from __future__ import annotations

import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from firstsmoke.cameras import Camera, Network, parse_sites_js
from firstsmoke.frames import (
    SequenceError,
    load_bundle,
    prepare,
    timestamp_from_name,
    write_bundle,
)
from firstsmoke.synth import synthetic_incident

SITES_JS = Path(__file__).resolve().parents[1] / "data" / "network" / "hpwren_sites.js"


class TestHpwrenMetadata:
    @pytest.fixture(scope="class")
    @staticmethod
    def sites():
        return parse_sites_js(SITES_JS.read_text(encoding="utf-8"))

    def test_the_published_metadata_parses(self, sites):
        assert len(sites) > 20
        first = next(iter(sites.values()))
        assert {"lat", "long", "cams"} <= set(first)

    def test_a_network_is_built_with_real_geometry(self, sites):
        network = Network.from_hpwren_sites(sites, name="hpwren")
        assert len(network) > 80
        for camera in network:
            assert -90 <= camera.lat <= 90
            assert -180 <= camera.lon <= 180
            assert 0 <= camera.azimuth_deg < 360
            assert 0 < camera.hfov_deg <= 180
            assert camera.imager in {"color", "monochrome"}

    def test_pan_tilt_heads_are_excluded_by_default(self, sites):
        """Their published azimuth is a home position, not where they point now."""
        default = Network.from_hpwren_sites(sites)
        with_ptz = Network.from_hpwren_sites(sites, include_ptz=True)
        assert len(with_ptz) > len(default)
        assert all(c.imager != "ptz" for c in default)

    def test_the_still_url_template_is_applied(self, sites):
        network = Network.from_hpwren_sites(
            sites, still_url_template="https://example.invalid/{camera_id}.jpg"
        )
        camera = next(iter(network))
        assert camera.still_url == f"https://example.invalid/{camera.camera_id}.jpg"

    def test_unusual_imagers_are_left_out_unless_asked_for(self, sites):
        """The real metadata carries VNIR, SWIR, infrared and experimental heads.
        Their radiometry is nothing like a visible camera's, so every threshold
        in the detector would be wrong on them."""
        default = Network.from_hpwren_sites(sites)
        everything = Network.from_hpwren_sites(sites, include_ptz=True, include_unusual=True)
        assert {c.imager for c in default} == {"color", "monochrome"}
        assert {"VNIR", "SWIR"} <= {c.imager for c in everything}

    def test_panoramic_heads_do_not_break_the_bearing_maths(self, sites):
        """Seven cameras in the published metadata declare a 180 degree field of
        view. The pinhole relation does not lose accuracy on those, it diverges,
        so they take the equiangular model instead."""
        from firstsmoke.geometry import bearing_from_pixel

        everything = Network.from_hpwren_sites(sites, include_ptz=True, include_unusual=True)
        wide = [c for c in everything if c.hfov_deg >= 180]
        assert wide, "the real metadata contains 180 degree cameras"
        for camera in wide:
            left = bearing_from_pixel(0, 1024, camera.azimuth_deg, camera.hfov_deg)
            right = bearing_from_pixel(1024, 1024, camera.azimuth_deg, camera.hfov_deg)
            assert 0.0 <= left < 360.0
            assert 0.0 <= right < 360.0

    def test_monochrome_cameras_are_flagged(self, sites):
        network = Network.from_hpwren_sites(sites)
        mono = [c for c in network if c.imager == "monochrome"]
        assert mono, "the real network has monochrome imagers"
        assert all(not c.has_colour for c in mono)

    def test_real_neighbours_can_be_found_for_a_real_bearing(self, sites):
        """The selection has to work on the actual network, not only the ring."""
        network = Network.from_hpwren_sites(sites)
        found = 0
        for camera in list(network)[:40]:
            if network.consultable(camera, camera.azimuth_deg, assumed_range_m=12_000.0):
                found += 1
        assert found > 5, "no camera in the real network overlooks any other camera's view"


class TestConsultable:
    @staticmethod
    def ring():
        from firstsmoke.synth import ring_network

        return ring_network(count=4)

    def test_it_returns_cameras_that_can_see_the_point(self):
        network = self.ring()
        origin = network.get("syn0-mobo-c")
        results = network.consultable(origin, origin.azimuth_deg)
        assert results
        for consultation in results:
            assert consultation.camera.site_id != origin.site_id
            assert consultation.crossing_angle_deg >= 10.0
            assert consultation.baseline_m >= 1500.0
            assert "km" in consultation.why

    def test_the_results_are_sorted_by_how_squarely_they_cut(self):
        network = self.ring()
        results = network.consultable(network.get("syn0-mobo-c"), 180.0)
        angles = [min(c.crossing_angle_deg, 90.0) for c in results]
        assert angles == sorted(angles, reverse=True)

    def test_a_bearing_nobody_overlooks_returns_nothing(self):
        network = self.ring()
        origin = network.get("syn0-mobo-c")
        assert network.consultable(origin, (origin.azimuth_deg + 180) % 360) == []

    def test_the_limit_is_respected(self):
        network = self.ring()
        origin = network.get("syn0-mobo-c")
        assert len(network.consultable(origin, origin.azimuth_deg, limit=1)) <= 1

    def test_an_inactive_camera_is_never_offered(self):
        from dataclasses import replace

        network = self.ring()
        origin = network.get("syn0-mobo-c")
        before = len(network.consultable(origin, origin.azimuth_deg))
        for camera in list(network):
            if camera.camera_id != origin.camera_id:
                network.add(replace(camera, active=False))
        assert before > 0
        assert network.consultable(origin, origin.azimuth_deg) == []


class TestNetworkBasics:
    def test_an_unknown_camera_raises_with_its_name(self):
        network = Network()
        with pytest.raises(KeyError, match="nope"):
            network.get("nope")

    def test_a_camera_label_names_the_site_and_the_direction(self):
        camera = Camera("x-n-mobo-c", "x", "Boucher Hill", 33.3, -116.9, 1600, 0.0, 90.0)
        assert camera.label == "Boucher Hill N"
        assert Camera("y", "y", "Red Mountain", 33.3, -116.9, 1600, 270.0, 90.0).label.endswith("W")

    def test_a_network_round_trips_through_its_own_spec(self):
        original = self.__class__ and __import__("firstsmoke.synth", fromlist=["ring_network"])
        network = original.ring_network(count=3)
        rebuilt = Network.from_spec(network.to_dict())
        assert len(rebuilt) == len(network)
        for camera in network:
            twin = rebuilt.get(camera.camera_id)
            assert twin.lat == pytest.approx(camera.lat)
            assert twin.azimuth_deg == pytest.approx(camera.azimuth_deg)

    def test_bounds_cover_every_camera(self):
        from firstsmoke.synth import ring_network

        network = ring_network(count=4)
        bounds = network.bounds()
        for camera in network:
            assert bounds["min_lat"] <= camera.lat <= bounds["max_lat"]
            assert bounds["min_lon"] <= camera.lon <= bounds["max_lon"]


class TestTimestamps:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("20260916_143000.jpg", datetime(2026, 9, 16, 14, 30, 0, tzinfo=UTC)),
            ("frame_20260916T143000.jpg", datetime(2026, 9, 16, 14, 30, 0, tzinfo=UTC)),
            ("1569959956_-02400.jpg", datetime(2019, 10, 1, 19, 59, 16, tzinfo=UTC)),
        ],
    )
    def test_it_reads_the_formats_camera_archives_use(self, name, expected):
        assert timestamp_from_name(name) == expected

    def test_an_unreadable_name_returns_the_fallback_not_now(self):
        """Defaulting to 'now' would make every gap zero and every growth rate
        meaningless, which is worse than refusing."""
        assert timestamp_from_name("holiday-snap.jpg") is None
        fallback = datetime(2026, 1, 1, tzinfo=UTC)
        assert timestamp_from_name("holiday-snap.jpg", fallback) == fallback


class TestPrepare:
    def test_it_scales_to_the_working_width_and_reports_the_factor(self):
        image = np.zeros((1536, 2048, 3), np.uint8)
        prepared, scale, original = prepare(image, width=1024)
        assert prepared.shape[1] == 1024
        assert prepared.shape[0] == 768
        assert scale == pytest.approx(0.5)
        assert original == (1536, 2048)

    def test_a_grayscale_image_becomes_three_channel(self):
        prepared, _, _ = prepare(np.zeros((200, 1024), np.uint8))
        assert prepared.ndim == 3

    def test_an_empty_image_is_an_error(self):
        with pytest.raises(SequenceError):
            prepare(np.zeros((0, 0, 3), np.uint8))


class TestBundles:
    @pytest.fixture(scope="class")
    @staticmethod
    def bundle(tmp_path_factory):
        incident = synthetic_incident(frames=4, plume_at=2, cameras=2, name="round trip")
        path = tmp_path_factory.mktemp("bundle") / "incident.zip"
        write_bundle(incident, path)
        return incident, path

    def test_a_bundle_round_trips(self, bundle):
        original, path = bundle
        loaded = load_bundle(path)
        assert loaded.name == original.name
        assert loaded.camera_ids() == original.camera_ids()
        assert len(loaded.sequences[original.camera_ids()[0]]) == 4

    def test_the_ground_truth_survives_the_round_trip(self, bundle):
        original, path = bundle
        loaded = load_bundle(path)
        assert loaded.truth["fire_lat"] == pytest.approx(original.truth["fire_lat"])

    def test_the_timestamps_survive_the_round_trip(self, bundle):
        original, path = bundle
        loaded = load_bundle(path)
        camera_id = original.camera_ids()[0]
        for before, after in zip(
            original.sequences[camera_id].frames, loaded.sequences[camera_id].frames, strict=True
        ):
            assert before.timestamp == after.timestamp

    def test_a_bundle_without_a_manifest_is_refused(self, tmp_path):
        path = tmp_path / "empty.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("frames/a.jpg", b"not an image")
        with pytest.raises(SequenceError, match="manifest"):
            load_bundle(path)

    def test_a_manifest_naming_an_unknown_camera_is_refused(self, tmp_path):
        path = tmp_path / "bad.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr(
                "manifest.json",
                json.dumps({"network": {"name": "n", "cameras": []}, "sequences": {"ghost": []}}),
            )
        with pytest.raises(SequenceError, match="unknown camera"):
            load_bundle(path)

    def test_something_that_is_not_a_bundle_is_refused(self, tmp_path):
        path = tmp_path / "notes.txt"
        path.write_text("hello")
        with pytest.raises(SequenceError, match="neither"):
            load_bundle(path)


class TestIncident:
    def test_asking_a_neighbour_for_the_same_moment_finds_its_nearest_frame(self):
        incident = synthetic_incident(frames=5, plume_at=2, cameras=2)
        camera_id = incident.camera_ids()[1]
        when = incident.sequences[incident.camera_ids()[0]].frames[2].timestamp
        assert incident.at(camera_id, when) is not None

    def test_a_moment_nothing_is_near_returns_nothing(self):
        incident = synthetic_incident(frames=5, plume_at=2, cameras=2)
        far = datetime(2030, 1, 1, tzinfo=UTC)
        assert incident.at(incident.camera_ids()[0], far) is None

    def test_the_timeline_is_ordered_and_deduplicated(self):
        incident = synthetic_incident(frames=5, plume_at=2, cameras=3)
        timeline = incident.timeline()
        assert timeline == sorted(timeline)
        assert len(timeline) == len(set(timeline))
