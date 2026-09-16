"""Synthetic lookout scenes, so the tests have an answer to check against.

Recorded imagery tells you whether the detector works. It cannot tell you
whether the *geometry* works, because nobody published the true coordinates of
the fires in FIgLib. So this module renders a camera network looking at a fire
whose latitude and longitude we chose, which makes the whole chain testable end
to end: a known point on the ground becomes a known bearing, becomes a known
pixel column, becomes a rendered plume — and the agent has to get back to within
a few hundred metres of where we put it.

It also gives the deployed service something to show a judge in three seconds
from a cold start, with no licence attached to it, and it gives the rejectors
their negatives: the same generator makes clouds that translate, dust that
crawls along the ground, fog that eats the whole frame, a droplet stuck on the
glass and a feed that has frozen solid.

Nothing here pretends to be photoreal. It is built to have the *statistics* the
detector keys on — a textured ridge under a smooth sky, a plume that veils
rather than replaces what is behind it, and impostors that get one of the four
behavioural tests right and the others wrong.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import cv2
import numpy as np

from .cameras import Camera, Network
from .frames import CameraFrame, Incident, Sequence_, prepare
from .geometry import bearing_between, haversine_m, pixel_from_bearing

DEFAULT_SIZE = (1024, 683)


# --------------------------------------------------------------------------- #
# terrain
# --------------------------------------------------------------------------- #
def value_noise(
    shape: tuple[int, int], octaves: int, rng: np.random.Generator, *, persistence: float = 0.55
) -> np.ndarray:
    """Multi-octave value noise in 0..1. Cheap, and rough in the right way.

    ``persistence`` is how much of each octave survives into the next. The
    default 0.55 gives a smooth, cloud-like field. Terrain wants much more, for
    the reason in :func:`render_view`.
    """
    h, w = shape
    total = np.zeros((h, w), np.float32)
    amplitude = 1.0
    norm = 0.0
    size = 4
    for _ in range(octaves):
        grid = rng.random((max(size, 2), max(size, 2))).astype(np.float32)
        total += amplitude * cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
        norm += amplitude
        amplitude *= persistence
        size = min(size * 2, max(h, w))
    return np.clip(total / max(norm, 1e-6), 0.0, 1.0)


@dataclass
class ViewSpec:
    """A camera's fixed view of the world: its ridge and its ground texture."""

    seed: int
    size: tuple[int, int] = DEFAULT_SIZE
    horizon_fraction: float = 0.52
    relief_fraction: float = 0.10
    haze: float = 0.18

    def ridge(self) -> np.ndarray:
        """Per-column skyline row. The same every frame, because a hill is."""
        rng = np.random.default_rng(self.seed)
        w, h = self.size
        x = np.linspace(0.0, 1.0, w)
        profile = np.zeros(w, np.float32)
        for harmonic, amplitude in ((1.0, 0.55), (2.3, 0.25), (5.1, 0.13), (11.7, 0.07)):
            profile += amplitude * np.sin(2 * np.pi * harmonic * x + rng.random() * 2 * np.pi)
        profile = profile / (np.abs(profile).max() + 1e-6)
        rows = self.horizon_fraction * h + profile * self.relief_fraction * h
        return np.clip(rows, 8, h - 8).astype(np.float32)


def render_view(spec: ViewSpec, *, sun: float = 0.5, exposure: float = 1.0) -> np.ndarray:
    """The static scene: sky above the ridge, textured terrain below it.

    ``sun`` runs 0 to 1 across the day and drives both the sky colour and the
    shading on the slopes, which is exactly the nuisance signal the background
    model's photometric fit has to cancel.
    """
    w, h = spec.size
    rng = np.random.default_rng(spec.seed)
    ridge = spec.ridge()
    rows = np.arange(h, dtype=np.float32)[:, None]
    sky_mask = rows < ridge[None, :]

    # Sky: a vertical gradient, bluer at the top, milkier at the horizon.
    vertical = np.repeat((rows / h), w, axis=1)
    warmth = 0.5 + 0.5 * math.cos((sun - 0.5) * math.pi)
    sky_b = 215 - 60 * vertical + 25 * spec.haze
    sky_g = 195 - 35 * vertical + 30 * spec.haze - 18 * warmth
    sky_r = 175 - 15 * vertical + 35 * spec.haze - 32 * warmth
    sky = np.stack([sky_b, sky_g, sky_r], axis=2)

    # Terrain: noise, shaded by a slope estimate so the sun angle bites.
    #
    # The texture here is not decoration. Measured on real HPWREN frames, a
    # hillside has a Laplacian variance between about 1,000 and 9,600. The first
    # version of this generator produced 180, and every absolute threshold tuned
    # against it was calibrated on a scene an order of magnitude smoother than
    # anything a real camera sees. So the terrain carries chaparral-scale detail:
    # nine octaves out to pixel scale, high persistence, plus a per-pixel speckle
    # standing in for individual shrubs.
    relief = value_noise((h, w), 5, rng)
    detail = value_noise((h, w), 9, rng, persistence=0.82)
    speckle = rng.random((h, w)).astype(np.float32)
    texture = np.clip(0.45 * relief + 0.40 * detail + 0.15 * speckle, 0.0, 1.0)
    slope = cv2.Sobel(relief, cv2.CV_32F, 1, 0, ksize=3)
    shade = 1.0 + 0.55 * slope * (sun - 0.5) * 2.0
    # Vegetation is not a single colour: the channels get their own weightings so
    # the ground is not a grey ramp with a tint.
    base = np.stack(
        [
            56 + 46 * texture + 14 * detail,
            72 + 64 * texture + 10 * relief,
            64 + 54 * texture + 18 * speckle,
        ],
        axis=2,
    ) * shade[:, :, None]

    image = np.where(sky_mask[:, :, None], sky, base)

    # Aerial perspective: things near the ridge are hazier than the foreground.
    depth = np.clip((ridge[None, :] + 0.16 * h - rows) / (0.32 * h), 0.0, 1.0)
    haze_colour = np.array([205.0, 200.0, 196.0], np.float32)
    veil = spec.haze * depth[:, :, None]
    image = image * (1 - veil) + haze_colour * veil

    return np.clip(image * exposure, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# the things that appear in front of it
# --------------------------------------------------------------------------- #
def _blob(shape: tuple[int, int], cx: float, cy: float, rx: float, ry: float,
          rng: np.random.Generator, roughness: float = 0.55) -> np.ndarray:
    """A soft, irregular alpha blob."""
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    radial = ((xs - cx) / max(rx, 1e-3)) ** 2 + ((ys - cy) / max(ry, 1e-3)) ** 2
    alpha = np.clip(1.4 - radial, 0.0, 1.0)
    alpha *= 0.55 + roughness * value_noise((h, w), 5, rng)
    return cv2.GaussianBlur(alpha, (0, 0), max(rx, ry) * 0.12 + 1.0)


def draw_plume(
    image: np.ndarray,
    base_x: float,
    base_y: float,
    minutes: float,
    *,
    rng: np.random.Generator,
    rise_px_per_min: float = 13.0,
    spread_px_per_min: float = 4.5,
    wind_px_per_min: float = 2.0,
    density: float = 0.72,
) -> np.ndarray:
    """A column rising from a fixed base, widening as it climbs.

    Drawn as a stack of blobs along a slightly wind-sheared axis, with the alpha
    falling off toward the top. The key property for the detector is that this
    *veils*: it is alpha-composited over the terrain with a grey that keeps some
    of the background luminance, so the texture underneath is attenuated rather
    than replaced.
    """
    if minutes <= 0:
        return image
    h, w = image.shape[:2]
    height = rise_px_per_min * minutes
    half_width = 3.0 + spread_px_per_min * minutes

    alpha = np.zeros((h, w), np.float32)
    steps = max(4, int(height / 6))
    for i in range(steps):
        t = i / max(steps - 1, 1)
        y = base_y - height * t
        x = base_x + wind_px_per_min * minutes * t * t
        rx = (2.5 + half_width * (0.35 + 0.65 * t)) * 0.5
        ry = max(4.0, height / steps * 1.9)
        alpha = np.maximum(alpha, _blob((h, w), x, y, rx, ry, rng) * (1.0 - 0.45 * t))

    alpha = np.clip(alpha * density, 0.0, 0.92)[:, :, None]
    # Smoke grey, slightly warm, and lifted toward the local sky brightness so it
    # reads against terrain and sky alike.
    smoke = np.full_like(image, 176, dtype=np.float32)
    smoke[:, :, 0] += 8  # a little blue in the shadowed side
    smoke[:, :, 2] += 4
    blended = image.astype(np.float32) * (1 - alpha) + smoke * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def draw_cloud(
    image: np.ndarray, cx: float, cy: float, minutes: float, *, rng: np.random.Generator,
    drift_px_per_min: float = 11.0, rx: float = 70.0, ry: float = 26.0,
) -> np.ndarray:
    """A cloud: same colour, same softness, but it translates and does not grow."""
    h, w = image.shape[:2]
    alpha = _blob((h, w), cx + drift_px_per_min * minutes, cy, rx, ry, rng, roughness=0.4)
    alpha = np.clip(alpha * 0.82, 0.0, 0.9)[:, :, None]
    white = np.full_like(image, 232, dtype=np.float32)
    return np.clip(image.astype(np.float32) * (1 - alpha) + white * alpha, 0, 255).astype(np.uint8)


def draw_dust(
    image: np.ndarray, x: float, y: float, minutes: float, *, rng: np.random.Generator,
    drift_px_per_min: float = 14.0,
) -> np.ndarray:
    """Dust: the right colour family, below the ridge, moving sideways, not rising."""
    h, w = image.shape[:2]
    alpha = _blob((h, w), x + drift_px_per_min * minutes, y, 30.0 + 3.0 * minutes, 13.0, rng)
    alpha = np.clip(alpha * 0.6, 0.0, 0.75)[:, :, None]
    dust = np.zeros_like(image, dtype=np.float32)
    dust[:, :, 0], dust[:, :, 1], dust[:, :, 2] = 140, 162, 196  # brown, in BGR
    return np.clip(image.astype(np.float32) * (1 - alpha) + dust * alpha, 0, 255).astype(np.uint8)


def draw_droplet(image: np.ndarray, x: float, y: float, radius: float = 13.0) -> np.ndarray:
    """A water drop on the glass: fixed, hard-edged, and slightly magnifying."""
    out = image.copy()
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.circle(mask, (int(x), int(y)), int(radius), 255, -1)
    blurred = cv2.GaussianBlur(image, (0, 0), 4.0)
    out[mask > 0] = cv2.addWeighted(blurred, 0.8, np.full_like(image, 210), 0.2, 0)[mask > 0]
    cv2.circle(out, (int(x), int(y)), int(radius), (198, 200, 205), 1, cv2.LINE_AA)
    return out


def apply_fog(image: np.ndarray, strength: float) -> np.ndarray:
    milk = np.full_like(image, 208, dtype=np.float32)
    blurred = cv2.GaussianBlur(image, (0, 0), 2.0 + 5.0 * strength).astype(np.float32)
    return np.clip(blurred * (1 - strength) + milk * strength, 0, 255).astype(np.uint8)


def apply_night(image: np.ndarray, level: float = 0.09) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dark = np.stack([gray * level * 1.15, gray * level, gray * level * 0.85], axis=2)
    return np.clip(dark, 0, 255).astype(np.uint8)


def camera_wobble(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    matrix = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def sensor_noise(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, sigma, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def jpeg_roundtrip(image: np.ndarray, quality: int = 86) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else image


# --------------------------------------------------------------------------- #
# whole sequences
# --------------------------------------------------------------------------- #
@dataclass
class SequenceSpec:
    """How one camera's recorded run should look."""

    camera: Camera
    view: ViewSpec
    frames: int = 22
    interval_s: float = 60.0
    start: datetime = field(default_factory=lambda: datetime(2026, 9, 16, 19, 0, tzinfo=UTC))
    plume_at: int | None = None
    """Frame index where the plume becomes visible. ``None`` for a clean camera."""
    plume_x: float | None = None
    plume_rise: float = 13.0
    plume_density: float = 0.72
    cloud: bool = False
    dust: bool = False
    droplet: bool = False
    fog_from: int | None = None
    night: bool = False
    frozen_from: int | None = None
    shake_px: float = 1.1
    noise_sigma: float = 2.4


def render_sequence(spec: SequenceSpec) -> list[np.ndarray]:
    """Render the frames for one camera."""
    rng = np.random.default_rng(spec.view.seed + 7919)
    ridge = spec.view.ridge()
    w, h = spec.view.size
    plume_x = spec.plume_x if spec.plume_x is not None else w * 0.5
    plume_base_y = float(ridge[int(np.clip(plume_x, 0, w - 1))]) + 4.0

    images: list[np.ndarray] = []
    frozen: np.ndarray | None = None
    for i in range(spec.frames):
        if spec.frozen_from is not None and i >= spec.frozen_from and frozen is not None:
            # A frozen feed repeats the *encoded bytes*, so there is not even
            # sensor noise between frames. That is the tell we detect on.
            images.append(frozen.copy())
            continue

        sun = 0.5 + 0.34 * (i / max(spec.frames - 1, 1))
        exposure = 1.0 - 0.06 * (i / max(spec.frames - 1, 1))
        image = render_view(spec.view, sun=sun, exposure=exposure)

        if spec.cloud:
            image = draw_cloud(image, w * 0.24, ridge.min() * 0.55, float(i), rng=rng)
        if spec.dust:
            image = draw_dust(image, w * 0.7, float(ridge[int(w * 0.7)]) + 16.0, float(i), rng=rng)
        if spec.plume_at is not None and i >= spec.plume_at:
            image = draw_plume(
                image, plume_x, plume_base_y, float(i - spec.plume_at),
                rng=np.random.default_rng(spec.view.seed + 13),
                rise_px_per_min=spec.plume_rise, density=spec.plume_density,
            )
        if spec.fog_from is not None and i >= spec.fog_from:
            image = apply_fog(image, min(0.85, 0.3 + 0.15 * (i - spec.fog_from)))
        if spec.droplet:
            image = draw_droplet(image, w * 0.33, h * 0.28)
        if spec.night:
            image = apply_night(image)

        if spec.shake_px:
            image = camera_wobble(
                image, rng.normal(0, spec.shake_px), rng.normal(0, spec.shake_px * 0.6)
            )
        image = sensor_noise(image, spec.noise_sigma, rng)
        image = jpeg_roundtrip(image)
        images.append(image)
        if spec.frozen_from is not None and i == spec.frozen_from - 1:
            frozen = image.copy()
    return images


def sequence_from_spec(spec: SequenceSpec) -> Sequence_:
    images = render_sequence(spec)
    seq = Sequence_(camera=spec.camera)
    for index, image in enumerate(images):
        prepared, scale, original = prepare(image)
        seq.frames.append(
            CameraFrame(
                camera_id=spec.camera.camera_id,
                index=index,
                image=prepared,
                timestamp=spec.start + timedelta(seconds=index * spec.interval_s),
                source=f"synthetic#{index}",
                scale=scale,
                original_shape=original,
            )
        )
    return seq


# --------------------------------------------------------------------------- #
# a whole network looking at one fire we placed
# --------------------------------------------------------------------------- #
def ring_network(
    centre: tuple[float, float] = (33.30, -116.85),
    radius_km: float = 11.0,
    count: int = 3,
    *,
    name: str = "synthetic ring",
) -> Network:
    """Cameras spaced around a circle, each looking inward at the centre.

    Looking inward matters: it guarantees the bearings cross at a sensible angle
    rather than running nearly parallel, which is the geometry a real network of
    ridge-top lookouts tends to have anyway.
    """
    from .geometry import project_bearing

    cameras = []
    for i in range(count):
        bearing_out = 360.0 * i / count
        lat, lon = project_bearing(centre[0], centre[1], bearing_out, radius_km * 1000.0)
        inward = (bearing_out + 180.0) % 360.0
        cameras.append(
            Camera(
                camera_id=f"syn{i}-mobo-c",
                site_id=f"syn{i}",
                site_name=f"Ridge {chr(ord('A') + i)}",
                lat=lat,
                lon=lon,
                elevation_m=1200.0 + 80 * i,
                azimuth_deg=inward,
                hfov_deg=90.0,
                imager="color",
                range_m=30_000.0,
            )
        )
    return Network.from_cameras(
        cameras, name=name, attribution="synthetic, generated by firstsmoke"
    )


def place_fire(
    network: Network,
    fire: tuple[float, float],
    *,
    size: tuple[int, int] = DEFAULT_SIZE,
) -> dict[str, float | None]:
    """Which pixel column each camera should see a fire at that point in.

    Returns ``None`` for a camera that cannot see it, which is itself useful:
    the test for "the agent does not consult a camera that is looking the other
    way" needs a camera that is looking the other way.
    """
    out: dict[str, float | None] = {}
    for camera in network:
        bearing = bearing_between(camera.lat, camera.lon, fire[0], fire[1])
        out[camera.camera_id] = pixel_from_bearing(
            bearing, size[0], camera.azimuth_deg, camera.hfov_deg
        )
    return out


def synthetic_incident(
    *,
    seed: int = 11,
    fire: tuple[float, float] | None = None,
    centre: tuple[float, float] = (33.30, -116.85),
    radius_km: float = 11.0,
    cameras: int = 3,
    frames: int = 22,
    plume_at: int = 6,
    blind: dict[str, str] | None = None,
    impostors: dict[str, str] | None = None,
    name: str = "synthetic incident",
    start: datetime | None = None,
) -> Incident:
    """A whole network watching one fire at a location we chose.

    ``blind`` maps camera id to one of ``night``, ``fog``, ``frozen`` or
    ``droplet``; ``impostors`` maps camera id to ``cloud`` or ``dust``. Both are
    how the tests build the awkward cases.
    """
    network = ring_network(centre, radius_km, cameras)
    fire = fire or centre
    columns = place_fire(network, fire)
    blind = blind or {}
    impostors = impostors or {}
    start = start or datetime(2026, 9, 16, 19, 0, tzinfo=UTC)

    incident = Incident(
        network=network,
        name=name,
        notes="Rendered by firstsmoke.synth. No real imagery, no identifiable people.",
        truth={
            "fire_lat": fire[0],
            "fire_lon": fire[1],
            "plume_visible_from_frame": plume_at,
            "plume_visible_at": (start + timedelta(seconds=plume_at * 60)).isoformat(),
            "expected_columns": {
                k: (round(v, 1) if v is not None else None) for k, v in columns.items()
            },
            "ranges_m": {
                c.camera_id: round(haversine_m(c.lat, c.lon, fire[0], fire[1]), 1) for c in network
            },
        },
    )

    for index, camera in enumerate(network):
        column = columns[camera.camera_id]
        condition = blind.get(camera.camera_id, "")
        impostor = impostors.get(camera.camera_id, "")
        spec = SequenceSpec(
            camera=camera,
            view=ViewSpec(seed=seed + 101 * index),
            frames=frames,
            start=start,
            plume_at=plume_at if column is not None else None,
            plume_x=column,
            cloud=impostor == "cloud",
            dust=impostor == "dust",
            droplet=condition == "droplet",
            fog_from=0 if condition == "fog" else None,
            night=condition == "night",
            frozen_from=3 if condition == "frozen" else None,
        )
        incident.sequences[camera.camera_id] = sequence_from_spec(spec)
    return incident
