from __future__ import annotations

import re

import cv2
import pytest
from visioncore import version as v


def test_opencv_is_version_5_not_4():
    """The single most important assertion in the repo.

    opencv-python 4.14.0 shipped after 5.0.0, so an unpinned install silently
    resolves to 4.x and fails the competition's core requirement.
    """
    assert cv2.__version__.startswith("5."), cv2.__version__
    v.assert_opencv5()


def test_assert_opencv5_raises_on_a_four_x_version(monkeypatch):
    monkeypatch.setattr(cv2, "__version__", "4.14.0")
    with pytest.raises(v.OpenCVVersionError, match=re.escape("4.14.0")):
        v.assert_opencv5()


def test_environment_is_serialisable_and_complete():
    env = v.environment().to_dict()
    assert env["opencv_version"].startswith("5.")
    for key in ("numpy_version", "python_version", "platform", "machine", "git_sha", "cpu"):
        assert key in env
    assert env["cpu"]["threads"] >= 1


def test_cpu_features_reports_the_hal_that_decides_benchmark_meaning():
    cpu = v.cpu_features()
    # On aarch64 this must be True: the stock PyPI wheel already ships KleidiCV,
    # which is exactly the trap in the COOL benchmark.
    assert isinstance(cpu["kleidicv"], bool)
    assert "custom_hal" in cpu or "baseline" in cpu
