"""Bearings, rays and where two lookouts think the same smoke is.

A lookout camera is a fixed pinhole on a known summit. Three numbers turn a
pixel column into a compass bearing: the camera's azimuth, its horizontal field
of view, and the image width. Two bearings from two summits cross at a point.

Everything here is plain trigonometry on a local tangent plane. Over the tens of
kilometres a camera network spans, the flat-earth error is far smaller than the
bearing error, and the bearing error is what actually limits the fix. The
functions below carry that uncertainty through rather than hiding it.

Conventions
-----------
* Bearings are degrees clockwise from true north, in [0, 360).
* Latitude/longitude are WGS84 decimal degrees.
* Image x runs left to right. A camera at azimuth A with horizontal field of
  view F looks at A, sees A - F/2 at the left edge and A + F/2 at the right.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

EARTH_RADIUS_M = 6_371_008.8


# --------------------------------------------------------------------------- #
# pixel -> bearing
# --------------------------------------------------------------------------- #
PANORAMIC_HFOV_DEG = 120.0
"""At or above this, treat the image as equiangular rather than rectilinear.

HPWREN's published metadata contains seven cameras declaring a 180 degree
horizontal field of view: two multispectral heads and five units whose names end
in ``-u180``. None of those is a single rectilinear pinhole — they are stitched
panoramas and fisheyes — and the pinhole relation does not merely lose accuracy
on them, it diverges, because ``tan(90 degrees)`` is infinite. A stitched
panorama maps angle to column linearly, so that is the model used above this
threshold. Below it, the pinhole relation is right and the linear one is wrong by
over a degree at the edges of a 90 degree lens."""


def bearing_from_pixel(
    x: float,
    image_width: int,
    azimuth_deg: float,
    hfov_deg: float,
) -> float:
    """Compass bearing of the ray through image column ``x``.

    For an ordinary lens the naive version, ``azimuth + (x/w - 0.5) * hfov``, is
    linear in x and is wrong at the edges by more than a degree. A 1 degree error
    at 20 km is 350 m on the ground, so we use the pinhole relation instead:

        tan(theta) = (2x/w - 1) * tan(hfov / 2)

    For a panoramic head the linear version is the correct one. See
    :data:`PANORAMIC_HFOV_DEG`.
    """
    if image_width <= 0:
        raise ValueError("image_width must be positive")
    if not 0 < hfov_deg <= 360:
        raise ValueError(f"hfov_deg out of range: {hfov_deg}")
    offset = (2.0 * float(x) / image_width) - 1.0
    if hfov_deg >= PANORAMIC_HFOV_DEG:
        theta_deg = offset * hfov_deg / 2.0
    else:
        half = math.radians(hfov_deg) / 2.0
        theta_deg = math.degrees(math.atan(offset * math.tan(half)))
    return normalise_bearing(azimuth_deg + theta_deg)


def pixel_from_bearing(
    bearing_deg: float,
    image_width: int,
    azimuth_deg: float,
    hfov_deg: float,
) -> float | None:
    """Inverse of :func:`bearing_from_pixel`. ``None`` if the bearing is off-frame.

    This is the function that lets one camera tell another where to look.
    """
    delta = signed_delta(bearing_deg, azimuth_deg)
    if abs(delta) >= hfov_deg / 2.0:
        return None
    if hfov_deg >= PANORAMIC_HFOV_DEG:
        offset = delta / (hfov_deg / 2.0)
    else:
        half = math.radians(hfov_deg) / 2.0
        offset = math.tan(math.radians(delta)) / math.tan(half)
    return (offset + 1.0) * image_width / 2.0


def normalise_bearing(deg: float) -> float:
    return float(deg) % 360.0


def signed_delta(bearing_deg: float, reference_deg: float) -> float:
    """Signed angle from ``reference`` to ``bearing``, in (-180, 180]."""
    delta = (float(bearing_deg) - float(reference_deg) + 180.0) % 360.0 - 180.0
    return 180.0 if delta == -180.0 else delta


def covers_bearing(
    bearing_deg: float, azimuth_deg: float, hfov_deg: float, margin_deg: float = 0.0
) -> bool:
    """Would a camera at this azimuth have the bearing inside its frame?"""
    return abs(signed_delta(bearing_deg, azimuth_deg)) <= (hfov_deg / 2.0) + margin_deg


# --------------------------------------------------------------------------- #
# geodesy, on a local tangent plane
# --------------------------------------------------------------------------- #
def enu_offset(lat: float, lon: float, ref_lat: float, ref_lon: float) -> tuple[float, float]:
    """Metres east and north of a reference point."""
    lat_rad = math.radians((lat + ref_lat) / 2.0)
    east = math.radians(lon - ref_lon) * EARTH_RADIUS_M * math.cos(lat_rad)
    north = math.radians(lat - ref_lat) * EARTH_RADIUS_M
    return east, north


def latlon_from_enu(
    east: float, north: float, ref_lat: float, ref_lon: float
) -> tuple[float, float]:
    lat = ref_lat + math.degrees(north / EARTH_RADIUS_M)
    lat_rad = math.radians((lat + ref_lat) / 2.0)
    lon = ref_lon + math.degrees(east / (EARTH_RADIUS_M * max(math.cos(lat_rad), 1e-9)))
    return lat, lon


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def bearing_between(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing from point 1 to point 2."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return normalise_bearing(math.degrees(math.atan2(y, x)))


def project_bearing(
    lat: float, lon: float, bearing_deg: float, distance_m: float
) -> tuple[float, float]:
    """Where you end up walking ``distance_m`` along ``bearing_deg``."""
    theta = math.radians(bearing_deg)
    east = math.sin(theta) * distance_m
    north = math.cos(theta) * distance_m
    return latlon_from_enu(east, north, lat, lon)


# --------------------------------------------------------------------------- #
# crossing the bearings
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Ray:
    """One lookout's line of sight to a candidate."""

    camera_id: str
    lat: float
    lon: float
    bearing_deg: float
    sigma_deg: float = 1.5
    max_range_m: float = 40_000.0


@dataclass(frozen=True)
class Fix:
    """Where the rays say the smoke is, and how much to trust it."""

    lat: float
    lon: float
    rays: tuple[str, ...]
    residual_m: float
    """Root-mean-square perpendicular distance from the fix to each ray."""
    semi_major_m: float
    """Long axis of the 1-sigma uncertainty ellipse, from the bearing sigmas."""
    semi_minor_m: float
    crossing_angle_deg: float
    """Smallest angle between any pair of rays. Below ~15 degrees the fix is soft."""
    range_m: dict[str, float]
    """Distance from each camera to the fix."""

    def to_dict(self) -> dict[str, object]:
        return {
            "lat": round(self.lat, 6),
            "lon": round(self.lon, 6),
            "rays": list(self.rays),
            "residual_m": round(self.residual_m, 1),
            "semi_major_m": round(self.semi_major_m, 1),
            "semi_minor_m": round(self.semi_minor_m, 1),
            "crossing_angle_deg": round(self.crossing_angle_deg, 1),
            "range_m": {k: round(v, 1) for k, v in self.range_m.items()},
        }


class CrossingRefused(ValueError):
    """Raised when the rays do not determine a point worth reporting."""

    def __init__(self, code: str, message: str, **details: object) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


@dataclass(frozen=True)
class NearestApproach:
    """How close two bearings came to meeting, when they did not meet.

    A refusal that says only "no crossing" tells an operator nothing. Saying the
    two lines passed 3.1 km apart, and that the combined uncertainty at that
    range is 780 m, tells them the two cameras are looking at two different
    things rather than at one thing measured badly — which is a different
    problem with a different answer.
    """

    distance_m: float
    lat: float
    lon: float
    """Midpoint of the shortest segment between the two rays."""
    corridor_m: float
    """The two rays' combined 1-sigma width where they pass. If the distance is
    inside this, the bearings are consistent and the geometry is merely soft."""
    consistent: bool
    cameras: tuple[str, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "distance_m": round(self.distance_m, 1),
            "lat": round(self.lat, 6),
            "lon": round(self.lon, 6),
            "corridor_m": round(self.corridor_m, 1),
            "consistent": self.consistent,
            "cameras": list(self.cameras),
        }


def nearest_approach(a: Ray, b: Ray) -> NearestApproach:
    """The closest the two sightlines come to each other, in front of both."""
    ref_lat = (a.lat + b.lat) / 2.0
    ref_lon = (a.lon + b.lon) / 2.0
    points = []
    directions = []
    for ray in (a, b):
        ox, oy = enu_offset(ray.lat, ray.lon, ref_lat, ref_lon)
        theta = math.radians(ray.bearing_deg - grid_convergence(ray.lat, ray.lon, ref_lon))
        points.append((ox, oy))
        directions.append((math.sin(theta), math.cos(theta)))

    (px, py), (qx, qy) = points
    (ux, uy), (vx, vy) = directions
    wx, wy = px - qx, py - qy
    uu = ux * ux + uy * uy
    uv = ux * vx + uy * vy
    vv = vx * vx + vy * vy
    uw = ux * wx + uy * wy
    vw = vx * wx + vy * wy
    denom = uu * vv - uv * uv
    if abs(denom) < 1e-9:
        s = t = 0.0
    else:
        s = max(0.0, (uv * vw - vv * uw) / denom)
        t = max(0.0, (uu * vw - uv * uw) / denom)
    ax, ay = px + ux * s, py + uy * s
    bx, by = qx + vx * t, qy + vy * t
    distance = math.hypot(ax - bx, ay - by)
    lat, lon = latlon_from_enu((ax + bx) / 2.0, (ay + by) / 2.0, ref_lat, ref_lon)
    corridor = math.tan(math.radians(a.sigma_deg)) * s + math.tan(math.radians(b.sigma_deg)) * t
    return NearestApproach(
        distance_m=distance, lat=lat, lon=lon, corridor_m=corridor,
        consistent=distance <= corridor, cameras=(a.camera_id, b.camera_id),
    )


def closest_pair(rays: list[Ray]) -> NearestApproach | None:
    """The pair of rays that came nearest to meeting."""
    best: NearestApproach | None = None
    for i, a in enumerate(rays):
        for b in rays[i + 1 :]:
            candidate = nearest_approach(a, b)
            if best is None or candidate.distance_m < best.distance_m:
                best = candidate
    return best


def grid_convergence(lat: float, lon: float, ref_lon: float) -> float:
    """Degrees between true north at this point and "up" on the tangent plane.

    The plane we solve on has north pointing the same way everywhere. The real
    world does not: meridians converge, so a camera east of the reference sees
    true north tilted relative to the plane's north by ``(lon - ref_lon) *
    sin(lat)``. Over a network 30 km across at 33 degrees north that is about
    0.06 degrees, which sounds ignorable and is not — at 15 km range it moves
    the fix by 16 m, and it is a systematic bias rather than noise, so it does
    not average out across cameras.

    Correcting it costs one sine per ray and removes the whole effect.
    """
    return (lon - ref_lon) * math.sin(math.radians(lat))


MIN_CROSSING_ANGLE_DEG = 8.0


def cross_rays(rays: list[Ray], *, min_crossing_angle_deg: float = MIN_CROSSING_ANGLE_DEG) -> Fix:
    """Least-squares intersection of two or more bearing rays.

    Each ray contributes the linear constraint "the fix lies on this line".
    Writing the ray direction as ``d`` and its left normal as ``n``, a point ``p``
    is on the line when ``n . (p - origin) = 0``. Stack those rows and solve.

    Refuses rather than guesses when:

    * fewer than two rays survive;
    * the rays are near-parallel, where a small bearing error throws the fix
      tens of kilometres along the line of sight;
    * the fix falls behind a camera, which means the bearings never met in
      front of the lookouts and the "intersection" is a mathematical artefact.
    """
    if len(rays) < 2:
        raise CrossingRefused(
            "TOO_FEW_RAYS", "a fix needs bearings from at least two cameras", count=len(rays)
        )

    ref_lat = sum(r.lat for r in rays) / len(rays)
    ref_lon = sum(r.lon for r in rays) / len(rays)

    crossing = min(
        abs(signed_delta(a.bearing_deg, b.bearing_deg)) % 180.0
        for i, a in enumerate(rays)
        for b in rays[i + 1 :]
    )
    crossing = min(crossing, 180.0 - crossing)
    if crossing < min_crossing_angle_deg:
        raise CrossingRefused(
            "RAYS_TOO_PARALLEL",
            f"the bearings cross at {crossing:.1f} degrees; below {min_crossing_angle_deg:.0f} the "
            "fix slides along the line of sight",
            crossing_angle_deg=round(crossing, 2),
        )

    # Build the normal equations. Weight each row by 1/sigma so a camera with a
    # wide lens and a fuzzy bearing pulls the answer less than a tight one.
    ata = [[0.0, 0.0], [0.0, 0.0]]
    atb = [0.0, 0.0]
    for ray in rays:
        ox, oy = enu_offset(ray.lat, ray.lon, ref_lat, ref_lon)
        theta = math.radians(ray.bearing_deg - grid_convergence(ray.lat, ray.lon, ref_lon))
        dx, dy = math.sin(theta), math.cos(theta)
        nx, ny = dy, -dx  # left normal
        w = 1.0 / max(ray.sigma_deg, 0.05)
        c = nx * ox + ny * oy
        ata[0][0] += w * nx * nx
        ata[0][1] += w * nx * ny
        ata[1][0] += w * ny * nx
        ata[1][1] += w * ny * ny
        atb[0] += w * nx * c
        atb[1] += w * ny * c

    det = ata[0][0] * ata[1][1] - ata[0][1] * ata[1][0]
    if abs(det) < 1e-9:
        raise CrossingRefused("SINGULAR", "the bearing geometry is singular", determinant=det)

    east = (atb[0] * ata[1][1] - ata[0][1] * atb[1]) / det
    north = (ata[0][0] * atb[1] - atb[0] * ata[1][0]) / det

    residuals: list[float] = []
    ranges: dict[str, float] = {}
    for ray in rays:
        ox, oy = enu_offset(ray.lat, ray.lon, ref_lat, ref_lon)
        theta = math.radians(ray.bearing_deg - grid_convergence(ray.lat, ray.lon, ref_lon))
        dx, dy = math.sin(theta), math.cos(theta)
        vx, vy = east - ox, north - oy
        along = vx * dx + vy * dy
        if along <= 0:
            raise CrossingRefused(
                "BEHIND_CAMERA",
                f"the crossing point is behind {ray.camera_id}; these bearings never met in front "
                "of the lookouts",
                camera=ray.camera_id,
                along_m=round(along, 1),
            )
        if along > ray.max_range_m:
            raise CrossingRefused(
                "BEYOND_RANGE",
                f"the crossing point is {along / 1000:.1f} km from {ray.camera_id}, past the "
                f"{ray.max_range_m / 1000:.0f} km this camera can usefully see",
                camera=ray.camera_id,
                range_m=round(along, 1),
            )
        residuals.append(abs(vx * (dy) + vy * (-dx)))
        ranges[ray.camera_id] = along

    rms = math.sqrt(sum(r * r for r in residuals) / len(residuals))
    lat, lon = latlon_from_enu(east, north, ref_lat, ref_lon)

    # Uncertainty. Each ray's angular sigma becomes a cross-range sigma of
    # range * tan(sigma). The along-range sigma is that divided by sin(crossing),
    # which is exactly why a shallow crossing angle is punished.
    cross_sigmas = [math.tan(math.radians(r.sigma_deg)) * ranges[r.camera_id] for r in rays]
    semi_minor = min(cross_sigmas)
    semi_major = max(cross_sigmas) / max(math.sin(math.radians(crossing)), 1e-3)

    return Fix(
        lat=lat,
        lon=lon,
        rays=tuple(r.camera_id for r in rays),
        residual_m=rms,
        semi_major_m=semi_major,
        semi_minor_m=semi_minor,
        crossing_angle_deg=crossing,
        range_m=ranges,
    )
