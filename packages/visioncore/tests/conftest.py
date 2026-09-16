"""Shared fixtures for the visioncore suite. Scene builders live in synthetic.py."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def aruco_dict_name() -> str:
    return "DICT_4X4_50"
