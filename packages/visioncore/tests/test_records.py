from __future__ import annotations

import json
import threading
import time

import pytest
from visioncore import records, timing


class TestRunRecord:
    def test_serialises_to_json_with_everything_a_judge_needs(self):
        record = records.RunRecord(product="demo", input={"filename": "a.mp4"})
        record.add_stage("decode", 12.5)
        record.metrics["crack_width_mm"] = 0.8
        record.add_evidence(records.Evidence(label="frame 12", frame_index=12, uri="/e/0"))
        payload = json.loads(record.to_json())
        assert payload["product"] == "demo"
        assert payload["schema_version"] == records.SCHEMA_VERSION
        assert payload["timings"]["stages"][0]["name"] == "decode"
        assert payload["metrics"]["crack_width_mm"] == 0.8
        assert payload["evidence"][0]["frame_index"] == 12
        assert payload["env"]["opencv_version"].startswith("5.")
        assert payload["refused"] is False

    def test_repeated_stages_merge_into_a_total_and_a_call_count(self):
        record = records.RunRecord(product="demo")
        for _ in range(4):
            record.add_stage("per_frame", 10.0)
        stage = record.stage("per_frame")
        assert stage.calls == 4
        assert stage.ms == pytest.approx(40.0)
        assert stage.ms_per_call == pytest.approx(10.0)
        assert record.total_ms == pytest.approx(40.0)

    def test_refusals_are_output_not_exceptions(self):
        record = records.RunRecord(product="demo")
        record.refuse("TOO_OBLIQUE", "re-shoot square to the surface", obliquity_deg=52.0)
        assert record.refused
        payload = record.to_dict()
        assert payload["refusals"][0]["code"] == "TOO_OBLIQUE"
        assert payload["refusals"][0]["details"]["obliquity_deg"] == 52.0

    def test_numpy_values_survive_json_encoding(self):
        import numpy as np

        record = records.RunRecord(product="demo")
        record.metrics["widths"] = np.array([1.0, 2.0])
        record.metrics["count"] = np.int64(7)
        payload = json.loads(record.to_json())
        assert payload["metrics"]["widths"] == [1.0, 2.0]
        assert payload["metrics"]["count"] == 7

    def test_run_ids_are_unique(self):
        ids = {records.RunRecord(product="d").run_id for _ in range(50)}
        assert len(ids) == 50


class TestTiming:
    def test_stage_context_manager_records_into_the_active_record(self):
        record = records.RunRecord(product="demo")
        with timing.recording(record), timing.stage("work") as timer:
            time.sleep(0.01)
        assert timer.ms >= 9.0
        assert record.stage("work").ms == pytest.approx(timer.ms)

    def test_stage_without_a_record_still_measures(self):
        with timing.stage("orphan") as timer:
            time.sleep(0.005)
        assert timer.ms >= 4.0
        assert timing.active_record() is None

    def test_timed_decorator_names_the_stage_after_the_function(self):
        @timing.timed()
        def analyse():
            time.sleep(0.005)
            return 42

        record = records.RunRecord(product="demo")
        with timing.recording(record):
            assert analyse() == 42
        assert record.stage("analyse") is not None
        assert analyse.last_ms >= 4.0

    def test_timed_records_even_when_the_function_raises(self):
        @timing.timed("boom")
        def explode():
            raise ValueError("nope")

        record = records.RunRecord(product="demo")
        with timing.recording(record), pytest.raises(ValueError):
            explode()
        assert record.stage("boom") is not None

    def test_the_active_record_is_per_thread(self):
        outer = records.RunRecord(product="outer")
        seen: list[object] = []

        def worker():
            seen.append(timing.active_record())

        with timing.recording(outer):
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()
        assert seen == [None], "a background thread must not inherit another run's record"

    def test_nested_recording_restores_the_outer_record(self):
        outer, inner = records.RunRecord(product="a"), records.RunRecord(product="b")
        with timing.recording(outer):
            with timing.recording(inner), timing.stage("x"):
                pass
            with timing.stage("y"):
                pass
        assert inner.stage("x") is not None and inner.stage("y") is None
        assert outer.stage("y") is not None and outer.stage("x") is None
