"""Uploaded footage: one camera, no survey, and a result that says so."""

from __future__ import annotations

import io
import json
import zipfile

import cv2
import numpy as np
import pytest

from firstsmoke.agent import Lookout, ReplaySource, State
from firstsmoke.frames import SequenceError
from firstsmoke.uploads import (
    MAX_ANALYSED,
    MIN_ANALYSED,
    is_bundle,
    load_upload,
    plan,
    stills_to_zip,
)


def jpeg(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    return buf.tobytes()


class TestPlan:
    def test_a_lookout_rate_clip_is_read_frame_by_frame(self):
        assert plan(30, 60.0) == (1, 30)

    def test_a_fast_timelapse_is_thinned_but_not_below_the_floor(self):
        stride, analysed = plan(885, 1.0)
        assert analysed >= MIN_ANALYSED
        assert stride == 885 // MIN_ANALYSED

    def test_a_long_clip_is_read_whole_before_it_is_cut(self):
        stride, analysed = plan(1200, 21.6)
        assert analysed == MAX_ANALYSED
        assert stride * MAX_ANALYSED >= 1200

    def test_a_clip_too_long_to_thin_is_cut(self):
        stride, analysed = plan(5000, 60.0)
        assert analysed == MAX_ANALYSED
        assert stride * analysed < 5000


class TestStills:
    @pytest.fixture(scope="class")
    @staticmethod
    def zipped(tmp_path_factory, plume_sequence):
        data = stills_to_zip([(f"{i:03d}.jpg", jpeg(f.image)) for i, f in enumerate(plume_sequence)])
        path = tmp_path_factory.mktemp("stills") / "stills.zip"
        path.write_bytes(data)
        return path

    def test_a_zip_of_stills_is_not_mistaken_for_a_bundle(self, zipped):
        assert not is_bundle(zipped)

    def test_the_camera_has_no_position_and_no_aim(self, zipped):
        incident, report = load_upload(zipped, {"interval_s": 60})
        camera = next(iter(incident.network))
        assert not camera.position_known
        assert not camera.aim_known
        assert report.analysed_frames == report.total_frames == 16
        assert not report.interval_assumed

    def test_an_assumed_interval_is_reported(self, zipped):
        _, report = load_upload(zipped, {})
        assert report.interval_assumed
        assert any("assumed" in note for note in report.notes)

    def test_a_single_still_is_refused_with_a_reason(self, tmp_path, plume_sequence):
        path = tmp_path / "one.jpg"
        path.write_bytes(jpeg(plume_sequence[0].image))
        with pytest.raises(SequenceError, match="one still"):
            load_upload(path, {})

    def test_a_bad_bearing_is_refused(self, zipped):
        with pytest.raises(SequenceError, match="bearing_deg"):
            load_upload(zipped, {"bearing_deg": 400})

    def test_the_watch_never_reports_a_location(self, zipped):
        incident, _ = load_upload(zipped, {"interval_s": 60})
        lookout = Lookout(ReplaySource(incident), keep_watching=True, suspect_at=0.3)
        lookout.run()
        assert lookout.alerts, "the plume should raise at least one flag"
        for alert in lookout.alerts:
            assert alert.fix is None
            assert alert.rays == []
            assert alert.fix_refusal["code"] == "NO_SECOND_VIEW"
            assert "bearing" not in alert.headline
        assert lookout.state is State.NEEDS_HUMAN


def test_a_video_is_sampled_and_its_coverage_stated(tmp_path, plume_sequence):
    path = tmp_path / "clip.avi"
    first = plume_sequence[0].image
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
                             (first.shape[1], first.shape[0]))
    if not writer.isOpened():
        pytest.skip("this OpenCV build cannot write MJPG")
    for frame in plume_sequence:
        for _ in range(3):
            writer.write(frame.image)
    writer.release()
    incident, report = load_upload(path, {"interval_s": 20, "bearing_deg": 90, "hfov_deg": 50})
    assert report.kind == "video"
    assert report.total_frames == 48
    assert report.stride == 1
    assert report.coverage.startswith("analysed 48 of 48")
    camera = next(iter(incident.network))
    assert camera.aim_known and not camera.position_known


def test_the_stills_zip_names_every_still(plume_sequence):
    data = stills_to_zip([("a.jpg", jpeg(plume_sequence[0].image))] * 3)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
        assert json.loads(zf.read("UPLOAD.json")) == {"stills": 3}
    assert sum(n.endswith(".jpg") for n in names) == 3
