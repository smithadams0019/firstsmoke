"""Footage someone brings: a video, or a run of stills, from one camera.

The bundled incidents come with a surveyed network. An upload almost never does.
A clip off the internet has no known summit, often no known heading, and a
capture interval its author may never have published. This module turns that
into an :class:`~firstsmoke.frames.Incident` the ordinary loop can run, and keeps
a plain record of every assumption it had to make, so the result can say what it
was given rather than implying more.

Three rules follow from that.

* **No invented location.** The camera is marked ``position_known=False``. The
  agent then refuses a map fix with ``NO_SECOND_VIEW`` and says why, instead of
  drawing a bearing out of a placeholder coordinate.
* **Time is the user's to state.** The detector measures growth in pixels per
  minute, so a time-lapse at 20 s a frame and a real-time video at 1/25 s a frame
  are different evidence even when the pixels match. The capture interval is a
  parameter, and when it is left out the default is written into the result.
* **Length is capped out loud.** A long clip is sampled and, if still too long,
  cut, and the result says "analysed X of Y" rather than stopping silently.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import cv2

from .cameras import Camera, Network
from .frames import (
    IMAGE_SUFFIXES,
    MANIFEST_NAME,
    VIDEO_SUFFIXES,
    WORKING_WIDTH,
    CameraFrame,
    Incident,
    Sequence_,
    SequenceError,
    decode,
    prepare,
    timestamp_from_name,
)

UPLOAD_CAMERA_ID = "upload"

TARGET_SPACING_S = 60.0
"""HPWREN's cadence, and the one the growth thresholds were set on. A clip is
sampled to roughly one analysed frame a minute of real time where it can be."""

MIN_ANALYSED = 40
"""Sampling never thins a clip below this many frames. The detector needs a few
frames of history before it will speak, and a one-minute stride on a 90-second
time-lapse would leave it nothing to track."""

MAX_ANALYSED = 240
"""The most frames one upload is analysed over. At the 1024 px working width
that is about 430 MB held in memory, which two concurrent jobs on a 4 GB
instance can afford."""

MAX_SPACING_S = 4 * TARGET_SPACING_S
"""A long clip is first thinned further so the whole of it is read. Past this
spacing, frames are too far apart for growth to mean anything, so the clip is
cut instead, and the cut is reported."""

MIN_FRAMES = 6
"""Fewer than this and there is no growth to measure at all."""

DEFAULT_STILLS_INTERVAL_S = 60.0
DEFAULT_VIDEO_INTERVAL_S = 1.0
"""Used only when the caller gives no interval, and always reported as assumed.
One second a frame is a common time-lapse rate; a real-time video is 1/25 s."""

START = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


@dataclass
class UploadReport:
    """What was analysed, and every assumption it took to analyse it."""

    kind: str
    """``video`` or ``stills``."""
    total_frames: int
    analysed_frames: int
    stride: int
    interval_s: float
    interval_assumed: bool
    spacing_s: float
    """Real seconds between two analysed frames."""
    capped: bool
    timestamps: str
    """``interval``, or ``filenames`` when every still carried its own time."""
    aim_known: bool
    fps: float = 0.0
    """Playback rate of an uploaded video, so a moment can be named as a point in
    the clip a person can scrub to. Zero for stills."""
    notes: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> str:
        every = f", one in every {self.stride}" if self.stride > 1 else ""
        return f"analysed {self.analysed_frames} of {self.total_frames} frames{every}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "total_frames": self.total_frames,
            "analysed_frames": self.analysed_frames,
            "stride": self.stride,
            "interval_s": self.interval_s,
            "interval_assumed": self.interval_assumed,
            "spacing_s": round(self.spacing_s, 2),
            "capped": self.capped,
            "timestamps": self.timestamps,
            "aim_known": self.aim_known,
            "coverage": self.coverage,
            "fps": round(self.fps, 3),
            "start": START.isoformat(),
            "notes": list(self.notes),
        }


def is_bundle(path: Path) -> bool:
    """A zip with a manifest is an incident bundle; one without is a run of stills."""
    if not zipfile.is_zipfile(path):
        return False
    with zipfile.ZipFile(path) as zf:
        return any(n == MANIFEST_NAME or n.endswith("/" + MANIFEST_NAME) for n in zf.namelist())


def _number(params: dict[str, Any], name: str, low: float, high: float) -> float | None:
    value = params.get(name)
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise SequenceError(f"{name} must be a number, not {value!r}") from None
    if not low <= number <= high:
        raise SequenceError(f"{name} must be between {low:g} and {high:g}, not {number:g}")
    return number


def upload_camera(params: dict[str, Any]) -> Camera:
    """The one camera an upload came from, with only what the caller told us."""
    azimuth = _number(params, "bearing_deg", 0.0, 360.0)
    hfov = _number(params, "hfov_deg", 1.0, 180.0)
    return Camera(
        camera_id=UPLOAD_CAMERA_ID,
        site_id=UPLOAD_CAMERA_ID,
        site_name="Uploaded camera",
        lat=0.0,
        lon=0.0,
        elevation_m=0.0,
        azimuth_deg=(azimuth or 0.0) % 360.0,
        hfov_deg=hfov or 60.0,
        position_known=False,
        aim_known=azimuth is not None,
    )


def plan(total: int, interval_s: float) -> tuple[int, int]:
    """Choose a stride and a frame count. Returns (stride, analysed)."""
    if total <= 0:
        return 1, 0
    stride = max(1, round(TARGET_SPACING_S / interval_s))
    # Never thin below MIN_ANALYSED when the clip has that many to give.
    stride = max(1, min(stride, total // MIN_ANALYSED)) if total >= MIN_ANALYSED else 1
    if -(-total // stride) > MAX_ANALYSED:
        # Too long: read the whole clip more sparsely, as long as that stays
        # close enough together for growth to be measured.
        wider = -(-total // MAX_ANALYSED)
        stride = max(stride, min(wider, max(1, int(MAX_SPACING_S // interval_s))))
    available = -(-total // stride)
    return stride, min(available, MAX_ANALYSED)


def load_upload(
    path: str | Path, params: dict[str, Any], *, width: int = WORKING_WIDTH
) -> tuple[Incident, UploadReport]:
    """Read an uploaded video, zip of stills, or single still into a one-camera incident."""
    path = Path(path)
    suffix = path.suffix.lower()
    camera = upload_camera(params)
    given = _number(params, "interval_s", 0.01, 86_400.0)

    if suffix in VIDEO_SUFFIXES:
        interval = given or DEFAULT_VIDEO_INTERVAL_S
        sequence, report = _video(path, camera, interval, given is None, width)
    elif suffix == ".zip":
        interval = given or DEFAULT_STILLS_INTERVAL_S
        sequence, report = _stills_zip(path, camera, interval, given is None, width)
    elif suffix in IMAGE_SUFFIXES:
        raise SequenceError(
            "one still cannot show smoke growing; upload a video, or several stills from the "
            f"same camera (at least {MIN_FRAMES})"
        )
    else:
        raise SequenceError(f"{suffix or 'that file'} is not a video, a zip of stills or a bundle")

    if len(sequence) < MIN_FRAMES:
        raise SequenceError(
            f"only {len(sequence)} usable frames; growth needs at least {MIN_FRAMES}"
        )

    report.aim_known = camera.aim_known
    if report.interval_assumed:
        report.notes.append(
            f"No capture interval was given, so {report.interval_s:g} s a frame was assumed. "
            "Growth is measured per minute, so a wrong interval changes the scores."
        )
    if not camera.aim_known:
        report.notes.append(
            "No camera bearing was given, so positions are reported as a place across the "
            "frame, not a compass bearing."
        )
    report.notes.append(
        "The camera's position is not known, so no second view can be consulted and no "
        "location is reported."
    )
    if report.capped:
        report.notes.append(
            f"The footage was too long to read whole at a usable spacing, so it was cut after "
            f"{report.analysed_frames} analysed frames ({report.coverage}); the rest was not read."
        )

    network = Network.from_cameras([camera], name="uploaded footage")
    incident = Incident(
        network=network,
        sequences={camera.camera_id: sequence},
        name=path.name,
        notes="; ".join(report.notes),
    )
    return incident, report


def _video(
    path: Path, camera: Camera, interval: float, assumed: bool, width: int
) -> tuple[Sequence_, UploadReport]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SequenceError(
            f"could not open {path.name}; H.264 MP4 is the safest format for this decoder"
        )
    try:
        # OpenCV 5 reports -1 for a property the container does not carry, so an
        # unknown count is found by reading, not trusted from the header.
        counted = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        total = counted if counted > 0 else _count(path)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        stride, wanted = plan(total, interval)
        seq = Sequence_(camera=camera)
        raw = 0
        while len(seq.frames) < wanted:
            if raw % stride == 0:
                ok, image = cap.read()
                if not ok:
                    break
                prepared, scale, original = prepare(image, width)
                seq.frames.append(
                    CameraFrame(
                        camera.camera_id, len(seq.frames), prepared,
                        START + timedelta(seconds=raw * interval),
                        f"{path.name}#{raw}", scale, original,
                    )
                )
            elif not cap.grab():
                break
            raw += 1
    finally:
        cap.release()
    if not seq.frames:
        raise SequenceError(f"{path.name} decoded to zero frames")
    available = -(-total // stride)
    report = UploadReport(
        kind="video", total_frames=total, analysed_frames=len(seq.frames), stride=stride,
        interval_s=interval, interval_assumed=assumed, spacing_s=stride * interval,
        capped=available > len(seq.frames), timestamps="interval", aim_known=camera.aim_known,
        fps=fps if fps > 0 else 0.0,
    )
    return seq, report


def _count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = 0
    try:
        while cap.grab():
            n += 1
    finally:
        cap.release()
    return n


def _stills_zip(
    path: Path, camera: Camera, interval: float, assumed: bool, width: int
) -> tuple[Sequence_, UploadReport]:
    with zipfile.ZipFile(path) as zf:
        names = sorted(
            n for n in zf.namelist()
            if Path(n).suffix.lower() in IMAGE_SUFFIXES
            and not Path(n).name.startswith(".")
            and "__MACOSX" not in n
        )
        if not names:
            raise SequenceError("the zip holds no images")
        stamps = [timestamp_from_name(Path(n).name) for n in names]
        from_names = all(s is not None for s in stamps) and len(set(stamps)) == len(stamps)
        # Times in the filenames, when every still has one, beat a stated
        # interval: they are what the camera recorded.
        if from_names:
            order = sorted(range(len(names)), key=lambda i: stamps[i])
            names = [names[i] for i in order]
            stamps = [stamps[i] for i in order]
        stride, wanted = plan(len(names), interval)
        seq = Sequence_(camera=camera)
        for raw in range(0, len(names), stride):
            if len(seq.frames) >= wanted:
                break
            try:
                image = decode(zf.read(names[raw]))
            except SequenceError:
                continue
            prepared, scale, original = prepare(image, width)
            stamp = stamps[raw] if from_names else START + timedelta(seconds=raw * interval)
            seq.frames.append(
                CameraFrame(camera.camera_id, len(seq.frames), prepared, stamp,
                            Path(names[raw]).name, scale, original)
            )
    spacing = stride * interval
    if from_names and len(seq.frames) > 1:
        spacing = seq.interval_s
    available = -(-len(names) // stride)
    report = UploadReport(
        kind="stills", total_frames=len(names), analysed_frames=len(seq.frames), stride=stride,
        interval_s=interval, interval_assumed=assumed and not from_names, spacing_s=spacing,
        capped=available > len(seq.frames),
        timestamps="filenames" if from_names else "interval", aim_known=camera.aim_known,
    )
    if from_names:
        report.notes.append("Every still carried a time in its filename, so those times were used.")
    return seq, report


def stills_to_zip(files: list[tuple[str, bytes]]) -> bytes:
    """Pack several uploaded stills into the zip form the job path accepts."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as zf:
        for index, (name, data) in enumerate(files):
            safe = Path(name or f"{index:04d}.jpg").name
            zf.writestr(f"{index:04d}_{safe}", data)
        zf.writestr("UPLOAD.json", json.dumps({"stills": len(files)}))
    return buffer.getvalue()
