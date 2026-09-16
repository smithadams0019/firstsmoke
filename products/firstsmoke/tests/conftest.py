"""Shared fixtures.

The synthetic incidents are expensive to render (a few seconds each) and
completely deterministic, so they are session-scoped. Tests must not mutate
them; anything that needs to change a frame copies it first.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from firstsmoke.cameras import Camera
from firstsmoke.synth import (
    SequenceSpec,
    ViewSpec,
    render_sequence,
    sequence_from_spec,
    synthetic_incident,
)

START = datetime(2026, 9, 16, 19, 0, tzinfo=UTC)


@pytest.fixture(scope="session")
def lone_camera() -> Camera:
    return Camera(
        camera_id="test-n-mobo-c",
        site_id="test",
        site_name="Test Ridge",
        lat=33.30,
        lon=-116.85,
        elevation_m=1200.0,
        azimuth_deg=0.0,
        hfov_deg=90.0,
    )


@pytest.fixture(scope="session")
def plume_sequence(lone_camera):
    """One camera, clean for six frames, then a growing column."""
    return sequence_from_spec(
        SequenceSpec(camera=lone_camera, view=ViewSpec(seed=5), frames=16, plume_at=6, start=START)
    )


@pytest.fixture(scope="session")
def clean_sequence(lone_camera):
    return sequence_from_spec(
        SequenceSpec(camera=lone_camera, view=ViewSpec(seed=5), frames=14, plume_at=None, start=START)
    )


@pytest.fixture(scope="session")
def cloud_sequence(lone_camera):
    """A cloud crossing the sky, and nothing else. The stand-down case."""
    return sequence_from_spec(
        SequenceSpec(
            camera=lone_camera, view=ViewSpec(seed=5), frames=16, plume_at=None,
            cloud=True, start=START,
        )
    )


@pytest.fixture(scope="session")
def dust_sequence(lone_camera):
    return sequence_from_spec(
        SequenceSpec(
            camera=lone_camera, view=ViewSpec(seed=5), frames=16, plume_at=None,
            dust=True, start=START,
        )
    )


@pytest.fixture(scope="session")
def confirmed_incident():
    """Three cameras ringed around a fire we placed. The known-answer case."""
    return synthetic_incident(frames=22, plume_at=6, seed=11, start=START)


@pytest.fixture(scope="session")
def quiet_incident():
    """The same network with no fire at all, only a cloud on one camera."""
    return synthetic_incident(
        frames=18, plume_at=99, seed=11, impostors={"syn0-mobo-c": "cloud"},
        name="quiet", start=START,
    )


@pytest.fixture(scope="session")
def blind_incident():
    """A fire visible to one camera, with every neighbour unable to see."""
    return synthetic_incident(
        frames=20, plume_at=6, seed=11,
        blind={"syn1-mobo-c": "fog", "syn2-mobo-c": "night"},
        name="blind neighbours", start=START,
    )


@pytest.fixture(scope="session")
def frozen_frames(lone_camera) -> list[np.ndarray]:
    return render_sequence(
        SequenceSpec(camera=lone_camera, view=ViewSpec(seed=3), frames=8, frozen_from=3, start=START)
    )


@pytest.fixture(scope="session")
def foggy_frames(lone_camera) -> list[np.ndarray]:
    return render_sequence(
        SequenceSpec(camera=lone_camera, view=ViewSpec(seed=3), frames=5, fog_from=0, start=START)
    )


@pytest.fixture(scope="session")
def night_frames(lone_camera) -> list[np.ndarray]:
    return render_sequence(
        SequenceSpec(camera=lone_camera, view=ViewSpec(seed=3), frames=5, night=True, start=START)
    )


@pytest.fixture(scope="session")
def clear_frame(lone_camera) -> np.ndarray:
    return render_sequence(
        SequenceSpec(camera=lone_camera, view=ViewSpec(seed=3), frames=1, shake_px=0.0, start=START)
    )[0]
