"""The camera network: who is on which summit, where each one is pointed.

The schema mirrors HPWREN's public ``sites.js``, because that is the metadata a
real mountain-top network publishes: a site with a latitude, longitude and
elevation, and under it a set of fixed cameras each with an azimuth and a
horizontal field of view. Keeping the same shape means a real network file drops
straight in without a translation layer.

The one thing this module exists to answer is the question the agent asks when a
detection is weak: *given a bearing from camera A, which other cameras can see
that same patch of ground, and where in their frames should it be?* That is
:meth:`Network.consultable`, and it is the geometry that makes the escalation
loop a loop rather than a slogan.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .geometry import (
    bearing_between,
    covers_bearing,
    haversine_m,
    pixel_from_bearing,
    project_bearing,
    signed_delta,
)

DEFAULT_RANGE_M = 40_000.0

FIXED_IMAGERS = frozenset({"color", "monochrome"})
"""The imager types this product's thresholds were built for. The published
HPWREN metadata also contains VNIR, SWIR, infrared and experimental heads."""


@dataclass(frozen=True)
class Camera:
    """One fixed camera on one summit."""

    camera_id: str
    site_id: str
    site_name: str
    lat: float
    lon: float
    elevation_m: float
    azimuth_deg: float
    hfov_deg: float
    imager: str = "color"
    """``color``, ``monochrome`` or ``ptz``. A monochrome imager has no colour
    evidence, so the detector drops the saturation tests rather than pretending."""
    active: bool = True
    range_m: float = DEFAULT_RANGE_M
    still_url: str | None = None
    """Where a live still comes from. ``None`` means this camera is replay-only."""
    position_known: bool = True
    """False for footage someone uploaded from a camera nobody has surveyed. The
    latitude and longitude are then placeholders and must never reach a map."""
    aim_known: bool = True
    """False when nobody told us which way the camera faces. A bearing computed
    from a guessed azimuth is a number with nothing behind it, so it is not shown."""

    @property
    def has_colour(self) -> bool:
        return self.imager != "monochrome"

    @property
    def label(self) -> str:
        if not self.aim_known:
            return self.site_name
        cardinal = _cardinal(self.azimuth_deg)
        return f"{self.site_name} {cardinal}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "site_id": self.site_id,
            "site_name": self.site_name,
            "label": self.label,
            "lat": self.lat,
            "lon": self.lon,
            "elevation_m": self.elevation_m,
            "azimuth_deg": self.azimuth_deg,
            "hfov_deg": self.hfov_deg,
            "imager": self.imager,
            "active": self.active,
            "range_m": self.range_m,
            "position_known": self.position_known,
            "aim_known": self.aim_known,
        }


def _cardinal(azimuth_deg: float) -> str:
    points = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
    return points[round((azimuth_deg % 360) / 45.0) % 8]


@dataclass(frozen=True)
class Consultation:
    """A neighbour worth asking, and where in its frame to look."""

    camera: Camera
    expected_bearing_deg: float
    """Bearing from the neighbour to the suspect point, not to the first camera."""
    expected_x: float | None
    """Image column the candidate should appear in, for a 1024 px wide frame."""
    baseline_m: float
    """Distance between the two lookouts. A short baseline means a shallow cross."""
    crossing_angle_deg: float
    """How squarely this neighbour's line of sight cuts the first camera's."""
    why: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera.camera_id,
            "label": self.camera.label,
            "expected_bearing_deg": round(self.expected_bearing_deg, 2),
            "expected_x_frac": (
                None if self.expected_x is None else round(self.expected_x / 1024.0, 4)
            ),
            "baseline_km": round(self.baseline_m / 1000.0, 2),
            "crossing_angle_deg": round(self.crossing_angle_deg, 1),
            "why": self.why,
        }


@dataclass
class Network:
    """A set of cameras, and the geometry questions you can ask of them."""

    cameras: dict[str, Camera] = field(default_factory=dict)
    name: str = "network"
    attribution: str = ""

    def __len__(self) -> int:
        return len(self.cameras)

    def __iter__(self) -> Iterator[Camera]:
        return iter(self.cameras.values())

    def __contains__(self, camera_id: object) -> bool:
        return camera_id in self.cameras

    def get(self, camera_id: str) -> Camera:
        try:
            return self.cameras[camera_id]
        except KeyError:
            raise KeyError(f"no camera {camera_id!r} in network {self.name!r}") from None

    def add(self, camera: Camera) -> Camera:
        self.cameras[camera.camera_id] = camera
        return camera

    def active(self) -> list[Camera]:
        return [c for c in self.cameras.values() if c.active]

    def sites(self) -> dict[str, list[Camera]]:
        out: dict[str, list[Camera]] = {}
        for cam in self.cameras.values():
            out.setdefault(cam.site_id, []).append(cam)
        return out

    # -- the question the agent actually asks --------------------------------
    def consultable(
        self,
        origin: Camera,
        bearing_deg: float,
        *,
        assumed_range_m: float = 12_000.0,
        min_crossing_angle_deg: float = 10.0,
        min_baseline_m: float = 1_500.0,
        limit: int = 4,
        frame_width: int = 1024,
    ) -> list[Consultation]:
        """Which cameras overlook the point ``origin`` is pointing at?

        We do not know the range yet, so we take a working guess at where along
        the ray the smoke is (``assumed_range_m``) and ask which other cameras
        have that point in frame. The answer is not very sensitive to the guess:
        a camera on a different summit that covers 12 km along the ray almost
        always covers 8 km and 20 km too, because its field of view is wide and
        the ray is nearly radial from it.

        Cameras on the *same* summit are excluded no matter what they can see.
        Two cameras a metre apart give two parallel rays and no fix at all, and
        an agent that "confirmed" from its own mast would be fooling itself.
        """
        target_lat, target_lon = project_bearing(
            origin.lat, origin.lon, bearing_deg, assumed_range_m
        )
        out: list[Consultation] = []
        for cam in self.cameras.values():
            if cam.camera_id == origin.camera_id or not cam.active:
                continue
            if cam.site_id == origin.site_id:
                continue
            baseline = haversine_m(origin.lat, origin.lon, cam.lat, cam.lon)
            if baseline < min_baseline_m:
                continue
            to_target = bearing_between(cam.lat, cam.lon, target_lat, target_lon)
            if not covers_bearing(to_target, cam.azimuth_deg, cam.hfov_deg, margin_deg=-2.0):
                continue
            if haversine_m(cam.lat, cam.lon, target_lat, target_lon) > cam.range_m:
                continue
            crossing = abs(signed_delta(to_target, bearing_deg)) % 180.0
            crossing = min(crossing, 180.0 - crossing)
            if crossing < min_crossing_angle_deg:
                continue
            out.append(
                Consultation(
                    camera=cam,
                    expected_bearing_deg=to_target,
                    expected_x=pixel_from_bearing(
                        to_target, frame_width, cam.azimuth_deg, cam.hfov_deg
                    ),
                    baseline_m=baseline,
                    crossing_angle_deg=crossing,
                    why=(
                        f"{baseline / 1000:.1f} km away on {cam.site_name}, cuts the bearing at "
                        f"{crossing:.0f} degrees"
                    ),
                )
            )
        # A square cut is worth more than a long baseline, so sort on the angle
        # first and use the baseline only to break ties.
        out.sort(key=lambda c: (-min(c.crossing_angle_deg, 90.0), -c.baseline_m))
        return out[:limit]

    # -- serialisation -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "attribution": self.attribution,
            "cameras": [c.to_dict() for c in self.cameras.values()],
        }

    def bounds(self) -> dict[str, float]:
        lats = [c.lat for c in self.cameras.values()]
        lons = [c.lon for c in self.cameras.values()]
        if not lats:
            return {"min_lat": 0.0, "max_lat": 0.0, "min_lon": 0.0, "max_lon": 0.0}
        return {
            "min_lat": min(lats),
            "max_lat": max(lats),
            "min_lon": min(lons),
            "max_lon": max(lons),
        }

    @classmethod
    def from_cameras(
        cls, cameras: Iterable[Camera], *, name: str = "network", attribution: str = ""
    ) -> Network:
        return cls({c.camera_id: c for c in cameras}, name=name, attribution=attribution)

    @classmethod
    def from_json(cls, path: str | Path) -> Network:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_spec(data)

    @classmethod
    def from_spec(cls, data: dict[str, Any]) -> Network:
        cameras = [
            Camera(
                camera_id=c["camera_id"],
                site_id=c.get("site_id", c["camera_id"].split("-")[0]),
                site_name=c.get("site_name", c.get("site_id", "")),
                lat=float(c["lat"]),
                lon=float(c["lon"]),
                elevation_m=float(c.get("elevation_m", 0.0)),
                azimuth_deg=float(c["azimuth_deg"]),
                hfov_deg=float(c.get("hfov_deg", 90.0)),
                imager=c.get("imager", "color"),
                active=bool(c.get("active", True)),
                range_m=float(c.get("range_m", DEFAULT_RANGE_M)),
                still_url=c.get("still_url"),
                position_known=bool(c.get("position_known", True)),
                aim_known=bool(c.get("aim_known", True)),
            )
            for c in data["cameras"]
        ]
        return cls.from_cameras(
            cameras,
            name=data.get("name", "network"),
            attribution=data.get("attribution", ""),
        )

    @classmethod
    def from_hpwren_sites(
        cls,
        sites: dict[str, Any],
        *,
        include_ptz: bool = False,
        include_unusual: bool = False,
        still_url_template: str | None = None,
        name: str = "hpwren",
        attribution: str = "",
    ) -> Network:
        """Build a network from HPWREN's published ``sites.js`` object.

        ``sites.js`` is a ``var sites = {...}`` assignment; strip the assignment
        and hand the object in. PTZ heads are excluded by default: their azimuth
        in the metadata is the home position, not where they are pointed right
        now, and a bearing computed from a stale azimuth is worse than no
        bearing at all.
        """
        cameras: list[Camera] = []
        for site_id, site in sites.items():
            for cam_id, cam in (site.get("cams") or {}).items():
                imager = cam.get("imager", "color")
                if imager == "ptz":
                    if not include_ptz:
                        continue
                elif imager not in FIXED_IMAGERS and not include_unusual:
                    # VNIR, SWIR, thermal and experimental heads appear in the
                    # published metadata. Their radiometry is nothing like a
                    # visible-light camera's, so every threshold in the detector
                    # would be wrong on them, and reporting a confident bearing
                    # from one would be worse than leaving it out.
                    continue
                cameras.append(
                    Camera(
                        camera_id=cam_id,
                        site_id=site_id,
                        site_name=site.get("name", site_id),
                        lat=float(site["lat"]),
                        lon=float(site["long"]),
                        elevation_m=float(site.get("elev", 0.0)),
                        azimuth_deg=float(cam.get("azimuth", 0.0)),
                        hfov_deg=float(cam.get("horizontalView", 90.0)),
                        imager=imager,
                        active=str(cam.get("active", "y")).lower().startswith("y"),
                        still_url=(
                            still_url_template.format(camera_id=cam_id)
                            if still_url_template
                            else None
                        ),
                    )
                )
        return cls.from_cameras(cameras, name=name, attribution=attribution)


def parse_sites_js(text: str) -> dict[str, Any]:
    """Pull the JSON object out of a ``var sites = {...};`` file."""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found in sites.js")
    return json.loads(text[start : end + 1])


def great_circle_arc(
    lat: float, lon: float, bearing_deg: float, length_m: float, steps: int = 2
) -> list[list[float]]:
    """Points along a bearing ray, for drawing it on a map."""
    return [
        list(project_bearing(lat, lon, bearing_deg, length_m * i / max(steps - 1, 1)))
        for i in range(steps)
    ]


def sun_azimuth_deg(hour_utc: float, day_of_year: int, lat: float, lon: float) -> float:
    """A rough solar azimuth, used only to flag lens flare near the sun.

    This is a low-precision almanac formula. It is good to a few degrees, which
    is all a flare test needs: flare is a big bright blob, and we only ask
    whether the blob sits roughly where the sun is.
    """
    decl = math.radians(-23.44 * math.cos(math.radians(360.0 / 365.0 * (day_of_year + 10))))
    hour_angle = math.radians(15.0 * (hour_utc + lon / 15.0 - 12.0))
    phi = math.radians(lat)
    elevation = math.asin(
        math.sin(phi) * math.sin(decl) + math.cos(phi) * math.cos(decl) * math.cos(hour_angle)
    )
    y = -math.sin(hour_angle)
    x = math.tan(decl) * math.cos(phi) - math.sin(phi) * math.cos(hour_angle)
    azimuth = math.degrees(math.atan2(y, x)) % 360.0
    return azimuth if elevation > 0 else -1.0
