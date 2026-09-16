"""Shared fixtures for the servicekit suite. Builders live in helpers.py."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from helpers import build_app
from visioncore import Evidence, encode_png


@pytest.fixture
def client(tmp_path):
    with TestClient(build_app(tmp_path=tmp_path)) as c:
        yield c


@pytest.fixture
def sample_png() -> bytes:
    """Two white squares on black: contour_count is 2 by construction."""
    image = np.zeros((120, 200, 3), np.uint8)
    cv2.rectangle(image, (20, 20), (60, 60), (255, 255, 255), -1)
    cv2.rectangle(image, (120, 40), (170, 90), (255, 255, 255), -1)
    return encode_png(image)


@pytest.fixture
def evidence_item() -> Evidence:
    return Evidence(label="x", uri="/e/0")
