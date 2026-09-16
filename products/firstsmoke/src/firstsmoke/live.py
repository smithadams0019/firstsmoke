"""The live path: real cameras, on demand, behind a switch.

Everything else in this product runs against recorded frames. This module is the
one place that reaches out to a real camera network, and it is off unless
``FIRSTSMOKE_LIVE=1`` is set, because a service that quietly starts pulling from
somebody else's research CDN the moment it is deployed is not a good neighbour.

Source and terms
----------------
HPWREN publishes a current still for every camera at

    https://cdn.hpwren.ucsd.edu/RT/<camera_id>.jpg

verified reachable without authentication, returning a 3072x2048 JPEG. The
camera metadata comes from the same project's ``sites.js``. HPWREN's data usage
page (https://hpwren.ucsd.edu/cc.html) states the program's data is licensed
CC BY-NC-ND 4.0, and its acknowledgement page asks for a credit reference, with
the minimum credit line ``http://hpwren.ucsd.edu``.

So: fetches are rate-limited and cached, the credit line is shown in the
interface and returned on every response, the use is non-commercial, and no
fetched frame is redistributed — the annotated overlays a live run produces stay
in that job's own evidence directory and expire with the job.

Cadence
-------
A lookout network updates about once a minute, which is the right cadence for
the detector and the wrong cadence for a demonstration. The poller therefore
takes a configurable interval, defaulting to 30 seconds, and says plainly in its
own output that a short interval gives the growth analysis less to work with
than a real deployment would have.
"""

from __future__ import annotations

import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .cameras import Camera, Network
from .frames import CameraFrame, Incident, Sequence_, decode, prepare

STILL_URL = "https://cdn.hpwren.ucsd.edu/RT/{camera_id}.jpg"
THUMB_URL = "https://cdn.hpwren.ucsd.edu/RTS/{camera_id}-640.jpg"
ATTRIBUTION = (
    "Live imagery: HPWREN, University of California San Diego. http://hpwren.ucsd.edu "
    "Licensed CC BY-NC-ND 4.0. Fetched on demand, not redistributed."
)
USER_AGENT = "firstsmoke/0.1 (OpenCV AI Competition 2026 entry; research use)"

DEFAULT_INTERVAL_S = 30.0
DEFAULT_ROUNDS = 8
MAX_ROUNDS = 40
MIN_INTERVAL_S = 15.0


class LiveDisabled(RuntimeError):
    """The live path is switched off."""


def live_enabled() -> bool:
    return os.environ.get("FIRSTSMOKE_LIVE", "").strip() in {"1", "true", "yes", "on"}


def live_status() -> dict[str, Any]:
    return {
        "enabled": live_enabled(),
        "source": "HPWREN",
        "still_url_pattern": STILL_URL,
        "attribution": ATTRIBUTION,
        "licence": "CC BY-NC-ND 4.0 (https://hpwren.ucsd.edu/cc.html)",
        "default_interval_s": DEFAULT_INTERVAL_S,
        "default_rounds": DEFAULT_ROUNDS,
        "switch": "set FIRSTSMOKE_LIVE=1 to enable",
        "note": (
            "A real lookout network updates about once a minute. Polling faster than that gives "
            "the growth analysis less history than a deployment would have, which makes the live "
            "demonstration harder than the recorded one, not easier."
        ),
    }


def fetch_still(camera_id: str, *, timeout: float = 20.0, thumbnail: bool = False) -> bytes:
    url = (THUMB_URL if thumbnail else STILL_URL).format(camera_id=camera_id)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def live_network(base: Network, camera_ids: list[str]) -> Network:
    """Cut a live-capable sub-network out of the published metadata."""
    cameras: list[Camera] = []
    for camera_id in camera_ids:
        camera = base.cameras.get(camera_id)
        if camera is None:
            continue
        cameras.append(
            Camera(
                camera_id=camera.camera_id, site_id=camera.site_id, site_name=camera.site_name,
                lat=camera.lat, lon=camera.lon, elevation_m=camera.elevation_m,
                azimuth_deg=camera.azimuth_deg, hfov_deg=camera.hfov_deg, imager=camera.imager,
                active=camera.active, range_m=camera.range_m,
                still_url=STILL_URL.format(camera_id=camera.camera_id),
            )
        )
    if not cameras:
        raise LiveDisabled("none of the requested cameras is in the published metadata")
    return Network.from_cameras(cameras, name="HPWREN live", attribution=ATTRIBUTION)


def poll(
    network: Network,
    *,
    rounds: int = DEFAULT_ROUNDS,
    interval_s: float = DEFAULT_INTERVAL_S,
    on_round=None,
    workers: int = 4,
) -> Incident:
    """Collect ``rounds`` frames from every camera, ``interval_s`` apart.

    Failures are not fatal. A camera that times out simply contributes fewer
    frames, and the detector already treats a short sequence as undecidable
    rather than as clear, so a flaky camera degrades into silence rather than
    into a false all-clear.
    """
    if not live_enabled():
        raise LiveDisabled("the live path is off; set FIRSTSMOKE_LIVE=1 to enable it")
    rounds = max(1, min(int(rounds), MAX_ROUNDS))
    interval_s = max(float(interval_s), MIN_INTERVAL_S)

    incident = Incident(
        network=network,
        name=f"live {datetime.now(UTC):%Y-%m-%d %H:%M} UTC",
        notes=ATTRIBUTION,
    )
    for camera in network:
        incident.sequences[camera.camera_id] = Sequence_(camera=camera)

    def grab(camera: Camera) -> tuple[str, bytes | None]:
        try:
            return camera.camera_id, fetch_still(camera.camera_id)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
            return camera.camera_id, None

    for round_index in range(rounds):
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            grabbed = list(pool.map(grab, list(network)))
        stamp = datetime.now(UTC)
        ok = 0
        for camera_id, data in grabbed:
            if data is None:
                continue
            try:
                image, scale, original = prepare(decode(data))
            except Exception:
                continue
            sequence = incident.sequences[camera_id]
            sequence.frames.append(
                CameraFrame(
                    camera_id=camera_id,
                    index=len(sequence.frames),
                    image=image,
                    timestamp=stamp,
                    source=STILL_URL.format(camera_id=camera_id),
                    scale=scale,
                    original_shape=original,
                )
            )
            ok += 1
        if on_round:
            on_round(round_index + 1, rounds, ok, len(network))
        if round_index + 1 < rounds:
            time.sleep(max(0.0, interval_s - (time.monotonic() - started)))
    return incident


def save_live_bundle(incident: Incident, path: Path) -> Path:
    """Write a live pull out as a bundle, for replaying it later.

    Local use only. The frames are HPWREN's and CC BY-NC-ND; keeping a working
    copy is fine, publishing one is not.
    """
    from .frames import write_bundle

    return write_bundle(incident, path)
