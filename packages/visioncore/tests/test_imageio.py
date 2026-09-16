from __future__ import annotations

import cv2
import numpy as np
import pytest
from visioncore import imageio as vio


@pytest.fixture
def synthetic_video(tmp_path):
    """30 frames at 10 fps, each frame's blue channel equal to its index.

    That makes every frame self-identifying, so decimation and ordering are
    checkable by value rather than by count alone.
    """
    path = tmp_path / "counter.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    assert writer.isOpened()
    for i in range(30):
        frame = np.zeros((48, 64, 3), np.uint8)
        frame[:, :, 0] = i * 8
        writer.write(frame)
    writer.release()
    return path


class TestVideo:
    def test_info_reports_geometry_and_duration(self, synthetic_video):
        info = vio.video_info(synthetic_video)
        assert (info.width, info.height) == (64, 48)
        assert info.fps == pytest.approx(10.0, rel=0.05)
        assert info.frame_count == 30
        assert info.duration_ms == pytest.approx(3000.0, rel=0.05)
        assert "path" in info.to_dict()

    def test_iterates_every_frame_in_order(self, synthetic_video):
        frames = list(vio.iter_video(synthetic_video))
        assert len(frames) == 30
        assert [f.index for f in frames] == list(range(30))
        assert frames[0].timestamp_ms == pytest.approx(0.0, abs=1.0)

    def test_timestamps_advance_at_the_frame_rate(self, synthetic_video):
        frames = list(vio.iter_video(synthetic_video, max_frames=5))
        gaps = np.diff([f.timestamp_ms for f in frames])
        assert np.allclose(gaps, 100.0, atol=5.0)

    @pytest.mark.parametrize("stride,expected", [(1, 30), (2, 15), (3, 10), (10, 3)])
    def test_decimation_keeps_one_frame_in_stride(self, synthetic_video, stride, expected):
        frames = list(vio.iter_video(synthetic_video, stride=stride))
        assert len(frames) == expected
        assert all(f.index % stride == 0 for f in frames)

    def test_max_frames_caps_the_kept_count_not_the_decoded_count(self, synthetic_video):
        frames = list(vio.iter_video(synthetic_video, stride=3, max_frames=4))
        assert [f.index for f in frames] == [0, 3, 6, 9]

    def test_end_ms_stops_early(self, synthetic_video):
        frames = list(vio.iter_video(synthetic_video, end_ms=450.0))
        assert len(frames) == 5
        assert max(f.timestamp_ms for f in frames) <= 450.0

    def test_max_side_downscales_preserving_aspect(self, synthetic_video):
        frame = next(vio.iter_video(synthetic_video, max_side=32))
        assert max(frame.image.shape[:2]) == 32
        assert frame.image.shape[:2] == (24, 32)

    def test_resize_to_forces_exact_dimensions(self, synthetic_video):
        frame = next(vio.iter_video(synthetic_video, resize_to=(20, 10)))
        assert frame.image.shape[:2] == (10, 20)

    def test_stride_zero_is_a_programming_error(self, synthetic_video):
        with pytest.raises(ValueError):
            list(vio.iter_video(synthetic_video, stride=0))

    def test_missing_file_raises_decode_error_not_a_silent_empty_iterator(self, tmp_path):
        with pytest.raises(vio.DecodeError):
            list(vio.iter_video(tmp_path / "nope.mp4"))

    def test_unsupported_property_reads_as_missing_not_minus_one(self):
        """OpenCV 5 returns -1 where 4.x returned 0; `_prop` must normalise it."""

        class FakeCap:
            def get(self, _prop):
                return -1.0

        assert vio._prop(FakeCap(), cv2.CAP_PROP_FPS, default=0.0) == 0.0
        assert vio._prop(FakeCap(), cv2.CAP_PROP_FPS, default=25.0) == 25.0


class TestImages:
    def test_encode_decode_png_round_trips_exactly(self):
        image = np.random.default_rng(0).integers(0, 255, (32, 48, 3), dtype=np.uint8)
        assert np.array_equal(vio.decode_image(vio.encode_png(image)), image)

    def test_jpeg_round_trips_approximately(self):
        image = np.full((32, 48, 3), 128, np.uint8)
        decoded = vio.decode_image(vio.encode_jpeg(image, quality=95))
        assert decoded.shape == image.shape
        assert np.abs(decoded.astype(int) - image.astype(int)).max() < 8

    def test_empty_payload_raises(self):
        with pytest.raises(vio.DecodeError, match="empty"):
            vio.decode_image(b"")

    def test_garbage_payload_raises_instead_of_returning_none(self):
        with pytest.raises(vio.DecodeError):
            vio.decode_image(b"this is not an image")

    def test_read_image_raises_on_a_missing_file(self, tmp_path):
        with pytest.raises(vio.DecodeError):
            vio.read_image(tmp_path / "absent.png")

    def test_write_then_read_round_trips(self, tmp_path):
        image = np.full((16, 16, 3), 200, np.uint8)
        path = vio.write_image(tmp_path / "sub" / "out.png", image)
        assert path.is_file()
        assert np.array_equal(vio.read_image(path), image)


class TestSharpness:
    def test_a_blurred_image_scores_lower_than_its_sharp_original(self):
        sharp = np.zeros((100, 100), np.uint8)
        sharp[::4] = 255
        blurred = cv2.GaussianBlur(sharp, (9, 9), 4)
        assert vio.sharpness(sharp) > vio.sharpness(blurred) * 3

    def test_pick_sharpest_returns_k_frames_in_original_order(self):
        frames = []
        for i in range(5):
            image = np.zeros((60, 60), np.uint8)
            image[:: (i + 2)] = 255
            frames.append(vio.Frame(index=i, timestamp_ms=float(i), image=image))
        picked = vio.pick_sharpest(frames, k=2)
        assert len(picked) == 2
        assert [f.index for f in picked] == sorted(f.index for f in picked)


class TestFrame:
    def test_with_image_preserves_provenance(self):
        frame = vio.Frame(index=3, timestamp_ms=99.0, image=np.zeros((4, 5, 3), np.uint8),
                          source="a.mp4")
        derived = frame.with_image(np.zeros((8, 8, 3), np.uint8))
        assert (derived.index, derived.timestamp_ms, derived.source) == (3, 99.0, "a.mp4")
        assert derived.shape == (8, 8)
