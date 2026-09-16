"""The web service: servicekit's shell, plus the routes this product needs.

What the shell gives us for free is the whole job lifecycle — upload, a worker
thread, a Server-Sent Events progress stream, the JSON result and an evidence
endpoint. What Firstsmoke adds is three things the shell cannot know about:

* **`GET /api/network`** — the camera geometry, so the map has something to draw
  before any job has run.
* **`GET /api/scenarios`** and **`POST /api/scenarios/{name}`** — the bundled
  incidents. A judge should see a detection within seconds of the page loading,
  without finding a file to upload, and the scenario route feeds a bundled
  incident through exactly the same job machinery as an upload so there is no
  second code path to get out of step.
* **`POST /api/stills`** — several stills from one camera, packed into the zip
  form the job path already accepts. A single video or zip goes through the
  shell's own ``POST /api/jobs``; see :mod:`firstsmoke.uploads` for what an
  upload with no surveyed camera can and cannot conclude.
* **`GET /api/live`** — the live-camera switch, off by default. See
  :mod:`firstsmoke.live`.

The analyzer itself is a thin adapter: it loads the upload as an incident,
hands it to :class:`~firstsmoke.agent.Lookout`, and turns the lookout's events
into progress messages and its evidence frames into files the UI can fetch.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Any

import cv2
from fastapi import File, Form, Request, UploadFile
from servicekit import ProductInfo, ServiceConfig, create_app
from servicekit.errors import ServiceError
from servicekit.jobs import JobContext
from visioncore import Evidence, RunRecord

from . import __version__
from .agent import Lookout, ReplaySource, State
from .calibration import CalibrationSet
from .candidates import draw_horizon, draw_regions
from .confirm import load_confirmer
from .detector import CONFIRM_AT, SUSPECT_AT, CameraReading
from .evaluate import default_network
from .frames import IMAGE_SUFFIXES, VIDEO_SUFFIXES, Incident, SequenceError, load_bundle
from .paths import calibration_file, scenario_dir, static_dir
from .scenarios import SCENARIOS, ensure_scenario, scenario_catalogue
from .uploads import UploadReport, UploadSource, is_bundle, open_upload, stills_to_zip

PRODUCT = ProductInfo(
    slug="firstsmoke",
    title="Firstsmoke",
    tagline="A lookout for mountain-top camera networks",
    description=(
        "Finds the first smoke column on a wildfire camera network, asks the neighbouring "
        "cameras when it is unsure, and crosses their bearings on a map."
    ),
    accent="#2B5D8A",
    version=__version__,
    repo_url="https://github.com/smithadams0019/firstsmoke",
)

STATIC_DIR = static_dir()
SCENARIO_DIR = scenario_dir()


def _overlay(reading: CameraReading, ctx: JobContext, label: str) -> str | None:
    """Save an annotated frame and return the URI the UI fetches it from."""
    image = reading.alignment.image
    if image is None or image.size == 0:
        return None
    canvas = draw_horizon(image, reading.scene.horizon)
    regions = [d.region for d in reading.detections[:3]]
    if regions:
        canvas = draw_regions(
            canvas,
            regions,
            colour=(26, 48, 180),
            labels=[f"{d.confidence:.2f}" for d in reading.detections[:3]],
        )
    ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        return None
    return ctx.save_evidence(f"{label}.jpg", buf.tobytes())


def _threshold(value: Any) -> float:
    """The suspicion threshold a caller asked for, or the shipped one."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return SUSPECT_AT
    return min(max(number, 0.1), CONFIRM_AT - 0.01)


UPLOAD_MAX_FLAGS = 24
"""Uploaded footage keeps watching after a flag. This bounds how many it raises,
so a clip full of cloud cannot produce an unreadable list; reaching it stops the
watch and the result says so."""


def _save_flag(lookout: Lookout, alert: dict[str, Any], ctx: JobContext, record: RunRecord) -> None:
    """Annotate the frame that raised a flag on uploaded footage, as it is raised.

    An upload keeps only each camera's latest reading, so this has to happen now:
    by the end of the run the frame is gone, which is the point."""
    reading = lookout.latest.get(alert["origin_camera"])
    if reading is None:
        return
    number = len(lookout.alerts)
    uri = _overlay(reading, ctx, f"flag-{number:02d}-{reading.frame_index:03d}")
    if not uri:
        return
    record.add_evidence(
        Evidence(
            label=lookout.network.get(reading.camera_id).label,
            kind="overlay",
            uri=uri,
            frame_index=reading.frame_index,
            caption=alert["headline"],
            metrics={
                "confidence": round(float(alert["confidence"]), 3),
                "usability": reading.scene.usability.value,
                "flag": number,
                "at": reading.timestamp.isoformat(),
            },
        )
    )


def analyze(ctx: JobContext) -> RunRecord:
    """Run the escalation loop over an uploaded or bundled incident."""
    record = ctx.record
    upload: UploadReport | None = None
    source: ReplaySource | UploadSource
    try:
        if is_bundle(ctx.input_path):
            ctx.progress(4, "reading the incident bundle")
            incident: Incident = load_bundle(ctx.input_path)
            source = ReplaySource(incident)
            frames = sum(len(s) for s in incident.sequences.values())
            name, truth = incident.name, incident.truth
        else:
            ctx.progress(4, "opening the uploaded footage")
            source, upload = open_upload(ctx.input_path, ctx.params)
            frames = upload.analysed_frames
            name, truth = str(record.input.get("filename") or source.name), {}
    except (SequenceError, KeyError, ValueError) as exc:
        raise ServiceError("BAD_REQUEST", str(exc)) from exc
    try:
        return _watch(ctx, record, source, upload, name, frames, truth)
    finally:
        if isinstance(source, UploadSource):
            source.close()
        # The lookout and its event callback refer to each other, so its
        # background models and tracks wait for the cycle collector. Run it now
        # rather than whenever the next allocation happens to trigger it.
        del source
        gc.collect()


def _watch(
    ctx: JobContext,
    record: RunRecord,
    source: ReplaySource | UploadSource,
    upload: UploadReport | None,
    name: str,
    frames: int,
    truth: dict[str, Any],
) -> RunRecord:
    network = source.network
    if upload is not None:
        record.input["upload"] = upload.to_dict()
        for note in upload.notes:
            record.warn(note)
        ctx.note(f"footage opened: {upload.coverage}", upload=upload.to_dict())

    record.input.update(
        {
            "incident": name,
            "cameras": len(network),
            "frames": frames,
            "attribution": network.attribution,
        }
    )
    record.params["threshold"] = _threshold(ctx.params.get("threshold"))

    confirmer = load_confirmer()
    if confirmer is None:
        record.warn(
            "The learned confirmation model was not found, so the detector ran on its classical "
            "evidence alone. Run scripts/train_confirmer.py to build it."
        )

    total = max(len(source.timeline()), 1)

    def on_event(event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "tick":
            step = event.get("step", 0)
            ctx.progress(
                5 + 80.0 * step / total,
                f"{event['at'][11:16]} watching {len(network)} cameras",
                state=event.get("state"),
            )
        elif kind == "transition":
            transition = event["transition"]
            ctx.note(f"{transition['to']}: {transition['detail']}", transition=transition)
        elif kind == "consultation":
            consultation = event["consultation"]
            ctx.note(
                f"asked {consultation['label']}: {consultation['answer']}",
                consultation=consultation,
            )
        elif kind == "re_examine":
            ctx.note(
                f"re-read {event['camera_id']} over {event['frames']} earlier frames "
                f"({event['confidence_before']:.2f} to {event['confidence_after']:.2f})"
            )
        elif kind == "alert" and upload is not None:
            _save_flag(lookout, event["alert"], ctx, record)

    calibrations = load_calibrations()
    if calibrations is not None:
        record.params["calibrated_cameras"] = sum(
            1 for camera in network if calibrations.get(camera.camera_id)
        )
    lookout = Lookout(
        source, confirmer=confirmer, calibrations=calibrations,
        on_event=on_event, suspect_at=_threshold(ctx.params.get("threshold")),
        keep_watching=upload is not None, max_alerts=UPLOAD_MAX_FLAGS,
        keep_readings=upload is None,
    )
    alert = lookout.run()

    ctx.progress(88, "collecting the frames behind the decision")
    saved: set[str] = set()
    for reading in lookout.readings:
        if reading.camera_id in saved:
            continue
        if reading.detections or reading.scene.blind:
            uri = _overlay(reading, ctx, f"{reading.camera_id}-{reading.frame_index:03d}")
            if uri:
                saved.add(reading.camera_id)
                record.add_evidence(
                    Evidence(
                        label=network.get(reading.camera_id).label,
                        kind="overlay",
                        uri=uri,
                        frame_index=reading.frame_index,
                        caption=reading.scene.reason,
                        metrics={
                            "confidence": round(reading.confidence, 3),
                            "usability": reading.scene.usability.value,
                        },
                    )
                )

    summary = lookout.summary()
    record.results.append(summary)
    record.metrics.update(
        {
            "state": lookout.state.value,
            "cameras": len(network),
            "frames_read": summary["frames_read"],
            "cameras_consulted": len({c["camera_id"] for c in summary["consultations"]}),
            "unusable_cameras": len(summary["unusable"]),
            "alerts": len(summary["alerts"]),
        }
    )
    if alert is not None:
        record.metrics["confidence"] = round(alert.confidence, 3)
        record.metrics["headline"] = alert.headline
        if alert.fix:
            record.metrics["fix_lat"] = round(alert.fix.lat, 5)
            record.metrics["fix_lon"] = round(alert.fix.lon, 5)
            record.metrics["fix_uncertainty_m"] = round(alert.fix.semi_major_m)
        elif alert.fix_refusal:
            record.refuse(
                alert.fix_refusal["code"], alert.fix_refusal["message"],
                **alert.fix_refusal.get("details", {}),
            )
    elif lookout.state in (State.WATCH, State.STOOD_DOWN):
        record.metrics["headline"] = "Nothing raised. The watch continues."

    if upload is not None and isinstance(source, UploadSource):
        # What was actually decoded, which is less than planned if the watch
        # stopped early or the file ended before its header said it would.
        upload.analysed_frames = source.decoded
        notes = record.input["upload"]["notes"]
        record.input["upload"] = upload.to_dict() | {"notes": notes}
        record.metrics["coverage"] = upload.coverage
        ended = lookout.transitions and lookout.transitions[-1].trigger == "sequence_ended"
        if not ended:
            last = lookout.alerts[-1].raised_at if lookout.alerts else None
            stopped = (
                f"The watch stopped at its {len(lookout.alerts)}th flag, the most one upload may "
                "raise, before the end of the footage"
            )
            record.metrics["stopped_early"] = stopped
            record.input["upload"]["notes"].append(stopped + ".")
            record.input["upload"]["stopped_at"] = last.isoformat() if last else None
            record.warn(stopped + ".")

    if truth:
        record.results.append({"truth": truth})

    ctx.progress(100, "done")
    return record


def load_calibrations() -> CalibrationSet | None:
    """Per-camera calibrations, if the operator has fitted any.

    A camera with no history has nothing to be calibrated against, so the
    rendered demonstration incidents run without one; a real network fitted with
    scripts/evaluate.py writes docs/calibration.json, which is picked up here.
    """
    path = calibration_file()
    if not path.is_file():
        return None
    try:
        return CalibrationSet.load(path)
    except (ValueError, KeyError, OSError):
        return None


def build_config() -> ServiceConfig:
    return ServiceConfig(
        product=PRODUCT,
        allowed_suffixes=(".zip", *sorted(VIDEO_SUFFIXES), *sorted(IMAGE_SUFFIXES)),
        max_upload_bytes=int(os.environ.get("FIRSTSMOKE_MAX_UPLOAD", 400 * 1024 * 1024)),
        static_dir=STATIC_DIR,
        max_concurrent_jobs=int(os.environ.get("FIRSTSMOKE_MAX_CONCURRENT", 2)),
        params_schema=[
            {
                "name": "threshold",
                "type": "number",
                "label": "Suspicion threshold",
                "default": SUSPECT_AT,
                "min": 0.1,
                "max": 0.9,
                "step": 0.05,
                "help": (
                    "Below this a camera stays quiet. "
                    f"Between this and {CONFIRM_AT:.2f} it consults."
                ),
            },
            {
                "name": "interval_s",
                "type": "number",
                "label": "Seconds between frames",
                "default": None,
                "min": 0.01,
                "max": 86400,
                "help": "Real time between two frames of the footage. A lookout camera is 60.",
            },
            {
                "name": "bearing_deg",
                "type": "number",
                "label": "Camera bearing, degrees",
                "default": None,
                "min": 0,
                "max": 360,
                "help": "Optional. Which way the centre of the frame faces.",
            },
            {
                "name": "hfov_deg",
                "type": "number",
                "label": "Field of view, degrees",
                "default": None,
                "min": 1,
                "max": 180,
                "help": "Optional. Horizontal field of view.",
            },
        ],
    )


def create() -> Any:
    config = build_config()
    app = create_app(config, analyze)

    @app.get("/api/network")
    async def network() -> dict[str, Any]:
        """The real camera geometry, so the map can draw itself immediately."""
        net = default_network()
        return {
            "name": net.name,
            "attribution": net.attribution,
            "bounds": net.bounds(),
            "cameras": [c.to_dict() for c in net],
        }

    @app.get("/api/scenarios")
    async def scenarios() -> dict[str, Any]:
        return {"scenarios": scenario_catalogue()}

    @app.post("/api/scenarios/{name}", status_code=202)
    async def run_scenario(request: Request, name: str) -> dict[str, Any]:
        """Start a job from a bundled incident, through the ordinary job path."""
        if name not in SCENARIOS:
            raise ServiceError("NOT_FOUND", f"no bundled scenario called {name!r}",
                               available=sorted(SCENARIOS))
        path = ensure_scenario(name, SCENARIO_DIR)
        store = request.app.state.store
        job = store.create(f"{name}.zip", path.read_bytes(), {"scenario": name})
        store.start(job)
        return {
            "job_id": job.job_id,
            "status": job.status,
            "scenario": name,
            "events_url": f"/api/jobs/{job.job_id}/events",
        }

    @app.post("/api/stills", status_code=202)
    async def run_stills(
        request: Request,
        files: list[UploadFile] = File(...),
        params: str = Form("{}"),
    ) -> dict[str, Any]:
        """Several stills from one camera, run as one job."""
        try:
            parsed = json.loads(params) if params else {}
        except json.JSONDecodeError as exc:
            raise ServiceError("BAD_REQUEST", f"params is not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ServiceError("BAD_REQUEST", "params must be a JSON object")
        limit = request.app.state.config.max_upload_bytes
        stills: list[tuple[str, bytes]] = []
        total = 0
        for upload in files:
            name = upload.filename or ""
            if Path(name).suffix.lower() not in IMAGE_SUFFIXES:
                raise ServiceError("UNSUPPORTED_MEDIA", f"{name or 'a file'} is not a still image",
                                   accepted=sorted(IMAGE_SUFFIXES))
            data = await upload.read()
            total += len(data)
            if total > limit:
                raise ServiceError("TOO_LARGE", f"stills exceed {limit // (1024 * 1024)} MB",
                                   max_bytes=limit)
            stills.append((name, data))
        if len(stills) < 2:
            raise ServiceError("BAD_REQUEST", "send several stills from the same camera")
        store = request.app.state.store
        job = store.create("stills.zip", stills_to_zip(stills), parsed)
        store.start(job)
        return {"job_id": job.job_id, "status": job.status, "stills": len(stills),
                "events_url": f"/api/jobs/{job.job_id}/events"}

    @app.get("/api/live")
    async def live() -> dict[str, Any]:
        from .live import live_status

        return live_status()

    return app


app = create()


def main() -> None:  # pragma: no cover - entry point
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", 8080)),
        log_config=None,
    )


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["PRODUCT", "analyze", "app", "build_config", "create"]
