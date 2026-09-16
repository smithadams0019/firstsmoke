"""YOLOX decode tested against arrays whose right answer is known by construction.

No model file is downloaded here: the decode and post-process are pure functions,
so a synthetic raw output with one planted detection proves the maths end to end.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from visioncore import models


class TestLetterbox:
    @pytest.mark.parametrize("shape", [(480, 640), (1080, 1920), (900, 600), (416, 416)])
    def test_output_is_exactly_the_target_size(self, shape):
        image = np.zeros((*shape, 3), np.uint8)
        canvas, ratio = models.letterbox(image, (416, 416))
        assert canvas.shape == (416, 416, 3)
        assert ratio == pytest.approx(min(416 / shape[1], 416 / shape[0]))

    def test_padding_is_bottom_right_only_so_unscaling_needs_no_offset(self):
        image = np.full((100, 400, 3), 255, np.uint8)
        canvas, ratio = models.letterbox(image, (416, 416))
        assert canvas[0, 0].tolist() == [255, 255, 255], "top-left must be image, not pad"
        assert canvas[-1, -1].tolist() == [114, 114, 114], "bottom-right must be pad"
        assert canvas[int(100 * ratio) + 2, 0].tolist() == [114, 114, 114]

    def test_aspect_ratio_is_preserved(self):
        image = np.zeros((200, 800, 3), np.uint8)
        canvas, ratio = models.letterbox(image, (416, 416))
        assert round(800 * ratio) == 416
        assert np.all(canvas[int(200 * ratio) + 1 :, :] == 114)


class TestGrids:
    def test_anchor_count_matches_the_measured_yolox_tiny_output(self):
        """measured out=(1, 3549, 85) at 416x416: 52^2 + 26^2 + 13^2."""
        grid, strides = models.yolox_grids((416, 416))
        assert grid.shape == (1, 3549, 2)
        assert strides.shape == (1, 3549, 1)
        assert 52**2 + 26**2 + 13**2 == 3549

    def test_strides_are_assigned_in_blocks(self):
        _, strides = models.yolox_grids((416, 416))
        flat = strides[0, :, 0]
        assert set(flat[: 52**2]) == {8}
        assert set(flat[52**2 : 52**2 + 26**2]) == {16}
        assert set(flat[52**2 + 26**2 :]) == {32}


class TestDecode:
    @staticmethod
    def _raw(anchor: int, dx: float, dy: float, lw: float, lh: float,
             obj: float = 0.9, cls: int = 0, cls_score: float = 0.95) -> np.ndarray:
        raw = np.zeros((1, 3549, 85), np.float32)
        raw[0, :, 4] = 0.001  # everything else is background
        raw[0, anchor, :4] = (dx, dy, lw, lh)
        raw[0, anchor, 4] = obj
        raw[0, anchor, 5 + cls] = cls_score
        return raw

    def test_centre_is_grid_offset_times_stride(self):
        """Anchor 0 is grid cell (0,0) at stride 8; dx=0.5 means centre x = 0.5*8 = 4."""
        decoded = models.decode_yolox(self._raw(0, 0.5, 0.5, 0.0, 0.0), (416, 416))
        assert decoded[0, 0] == pytest.approx(4.0)
        assert decoded[0, 1] == pytest.approx(4.0)

    def test_size_is_exponentiated_then_scaled_by_stride(self):
        decoded = models.decode_yolox(self._raw(0, 0.0, 0.0, np.log(3.0), np.log(5.0)),
                                      (416, 416))
        assert decoded[0, 2] == pytest.approx(24.0, rel=1e-4)  # 3 * 8
        assert decoded[0, 3] == pytest.approx(40.0, rel=1e-4)  # 5 * 8

    def test_a_stride_32_anchor_uses_the_coarse_grid(self):
        anchor = 52**2 + 26**2 + 5  # grid cell (5, 0) at stride 32
        decoded = models.decode_yolox(self._raw(anchor, 0.0, 0.0, 0.0, 0.0), (416, 416))
        assert decoded[anchor, 0] == pytest.approx(160.0)  # 5 * 32

    def test_wrong_anchor_count_is_rejected_loudly(self):
        with pytest.raises(ValueError, match="anchors"):
            models.decode_yolox(np.zeros((1, 100, 85), np.float32), (416, 416))

    def test_decode_does_not_mutate_its_input(self):
        raw = self._raw(0, 0.5, 0.5, 0.0, 0.0)
        before = raw.copy()
        models.decode_yolox(raw, (416, 416))
        assert np.array_equal(raw, before)


class TestPostprocess:
    def _planted(self, cx: float, cy: float, w: float, h: float, cls: int = 2):
        """Plant one detection at an exact position by inverting the decode."""
        stride, grid_w = 32, 13
        gx, gy = int(cx // stride), int(cy // stride)
        anchor = 52**2 + 26**2 + gy * grid_w + gx
        raw = np.zeros((1, 3549, 85), np.float32)
        raw[0, :, 4] = 0.0
        raw[0, anchor, 0] = cx / stride - gx
        raw[0, anchor, 1] = cy / stride - gy
        raw[0, anchor, 2] = np.log(w / stride)
        raw[0, anchor, 3] = np.log(h / stride)
        raw[0, anchor, 4] = 1.0
        raw[0, anchor, 5 + cls] = 1.0
        return raw

    def test_recovers_the_planted_box_in_input_pixels(self):
        raw = self._planted(200.0, 160.0, 64.0, 96.0)
        dets = models.postprocess_yolox(raw, ratio=1.0, score_threshold=0.5)
        assert len(dets) == 1
        x1, y1, x2, y2 = dets[0].bbox
        assert (x1, y1, x2, y2) == pytest.approx((168.0, 112.0, 232.0, 208.0), abs=1.0)
        assert dets[0].class_id == 2
        assert dets[0].class_name == "car"
        assert dets[0].score == pytest.approx(1.0)

    def test_ratio_unscales_the_box_back_to_original_pixels(self):
        raw = self._planted(200.0, 160.0, 64.0, 96.0)
        dets = models.postprocess_yolox(raw, ratio=0.5, score_threshold=0.5)
        x1, y1, x2, y2 = dets[0].bbox
        assert (x1, y1, x2, y2) == pytest.approx((336.0, 224.0, 464.0, 416.0), abs=2.0)

    def test_score_threshold_suppresses_weak_detections(self):
        raw = self._planted(200.0, 160.0, 64.0, 96.0)
        raw[0, :, 4] *= 0.2
        assert models.postprocess_yolox(raw, 1.0, score_threshold=0.5) == []
        assert len(models.postprocess_yolox(raw, 1.0, score_threshold=0.1)) == 1

    def test_no_detections_returns_an_empty_list_not_none(self):
        assert models.postprocess_yolox(np.zeros((1, 3549, 85), np.float32), 1.0) == []

    def test_nms_collapses_overlapping_duplicates(self):
        raw = self._planted(200.0, 160.0, 64.0, 96.0)
        neighbour = self._planted(208.0, 160.0, 64.0, 96.0)
        raw = np.maximum(raw, neighbour)
        dets = models.postprocess_yolox(raw, 1.0, score_threshold=0.5, nms_threshold=0.45)
        assert len(dets) == 1


    def test_detections_are_sorted_by_score(self):
        far_apart = np.maximum(
            self._planted(96.0, 96.0, 32.0, 32.0, cls=0),
            self._planted(320.0, 320.0, 32.0, 32.0, cls=1),
        )
        far_apart[0, :, 4] = np.where(far_apart[0, :, 4] > 0, far_apart[0, :, 4], 0.0)
        dets = models.postprocess_yolox(far_apart, 1.0, score_threshold=0.5)
        assert len(dets) == 2
        assert dets[0].score >= dets[1].score

    def test_detection_serialises(self):
        det = models.Detection((1.234, 2.0, 3.0, 4.0), 0.98765, 0, "person")
        assert det.to_dict() == {
            "bbox": [1.23, 2.0, 3.0, 4.0],
            "score": 0.9877,
            "class_id": 0,
            "class_name": "person",
        }


class TestModelSpecAndFetch:
    def test_default_detector_is_apache_licensed_yolox_not_ultralytics(self):
        """Ultralytics YOLO is AGPL-3.0; section 13 makes a hosted demo a
        source-disclosure event. This assertion is the guard rail."""
        assert models.YOLOX_TINY.licence == "Apache-2.0"
        assert "ultralytics" not in models.YOLOX_TINY.uri.lower()

    def test_local_path_resolves_without_a_download(self, tmp_path):
        onnx = tmp_path / "fake.onnx"
        onnx.write_bytes(b"not really onnx but non-empty")
        spec = models.ModelSpec(name="fake", uri=str(onnx))
        assert models.fetch_model(spec) == onnx

    def test_missing_local_file_raises_model_error(self, tmp_path):
        spec = models.ModelSpec(name="fake", uri=str(tmp_path / "absent.onnx"))
        with pytest.raises(models.ModelError, match="no such file"):
            models.fetch_model(spec)

    def test_empty_file_is_rejected(self, tmp_path):
        onnx = tmp_path / "empty.onnx"
        onnx.touch()
        with pytest.raises(models.ModelError, match="empty"):
            models.fetch_model(models.ModelSpec(name="fake", uri=str(onnx)))

    def test_sha256_mismatch_is_rejected(self, tmp_path):
        onnx = tmp_path / "f.onnx"
        onnx.write_bytes(b"abc")
        spec = models.ModelSpec(name="f", uri=str(onnx), sha256="0" * 64)
        with pytest.raises(models.ModelError, match="sha256 mismatch"):
            models.fetch_model(spec)

    def test_s3_uri_downloads_through_boto3(self, tmp_path, monkeypatch):
        calls = {}

        class FakeS3:
            def download_file(self, bucket, key, dest):
                calls["args"] = (bucket, key)
                pathlib.Path(dest).write_bytes(b"onnx bytes")

        fake_boto3 = type("m", (), {"client": staticmethod(lambda _svc: FakeS3())})
        monkeypatch.setitem(__import__("sys").modules, "boto3", fake_boto3)
        spec = models.ModelSpec(name="s3model", uri="s3://bucket-x/models/m.onnx")
        path = models.fetch_model(spec, cache_dir=tmp_path)
        assert path.read_bytes() == b"onnx bytes"
        assert calls["args"] == ("bucket-x", "models/m.onnx")

    def test_s3_download_is_cached_on_the_second_call(self, tmp_path, monkeypatch):
        hits = {"n": 0}

        class FakeS3:
            def download_file(self, bucket, key, dest):
                hits["n"] += 1
                pathlib.Path(dest).write_bytes(b"onnx bytes")

        monkeypatch.setitem(
            __import__("sys").modules, "boto3",
            type("m", (), {"client": staticmethod(lambda _s: FakeS3())}),
        )
        spec = models.ModelSpec(name="cached", uri="s3://b/k.onnx")
        models.fetch_model(spec, cache_dir=tmp_path)
        models.fetch_model(spec, cache_dir=tmp_path)
        assert hits["n"] == 1

    def test_unknown_scheme_is_rejected(self, tmp_path):
        spec = models.ModelSpec(name="x", uri="ftp://host/m.onnx")
        with pytest.raises(models.ModelError, match="unsupported URI scheme"):
            models.fetch_model(spec, cache_dir=tmp_path)

    def test_unknown_engine_names_the_ort_trap(self, tmp_path):
        onnx = tmp_path / "f.onnx"
        onnx.write_bytes(b"x")
        with pytest.raises(models.ModelError, match="ONNX Runtime: NO"):
            models.DnnRunner(models.ModelSpec(name="f", uri=str(onnx)), engine="ort")

    def test_engine_names_map_to_opencv_constants(self):
        import cv2

        assert models.ENGINES["new"] == cv2.dnn.ENGINE_NEW
        assert models.ENGINES["classic"] == cv2.dnn.ENGINE_CLASSIC
        assert models.ENGINES["auto"] == cv2.dnn.ENGINE_AUTO
