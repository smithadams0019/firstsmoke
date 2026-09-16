"""Firstsmoke - a lookout for mountain-top camera networks.

Importing this checks that OpenCV 5 is what got installed, by way of
``visioncore``, which raises on a 4.x wheel.
"""

from __future__ import annotations

from visioncore import assert_opencv5

from .agent import Alert, ConsultationRecord, Lookout, ReplaySource, State, Transition
from .background import BackgroundModel, ChangeMap, NetworkBackground
from .cameras import Camera, Consultation, Network, parse_sites_js
from .candidates import Region, extract_regions
from .confirm import Confirmation, SmokeConfirmer, load_confirmer
from .detector import CameraReading, CameraWatch, Detection, Reason, Verdict
from .frames import CameraFrame, Incident, Sequence_, load_bundle, write_bundle
from .geometry import CrossingRefused, Fix, Ray, bearing_from_pixel, cross_rays
from .rejectors import Rejection
from .scene import Horizon, SceneState, Usability, assess_scene, find_horizon
from .tracks import Growth, Track, Tracker

assert_opencv5()

__version__ = "0.1.0"

__all__ = [
    "Alert", "BackgroundModel", "Camera", "CameraFrame", "CameraReading", "CameraWatch",
    "ChangeMap", "Confirmation", "Consultation", "ConsultationRecord", "CrossingRefused",
    "Detection", "Fix", "Growth", "Horizon", "Incident", "Lookout", "Network",
    "NetworkBackground", "Ray", "Reason", "Region", "Rejection", "ReplaySource",
    "SceneState", "Sequence_", "SmokeConfirmer", "State", "Track", "Tracker", "Transition",
    "Usability", "Verdict", "__version__", "assess_scene", "bearing_from_pixel", "cross_rays",
    "extract_regions", "find_horizon", "load_bundle", "load_confirmer", "parse_sites_js",
    "write_bundle",
]
