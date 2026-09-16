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
* **`GET /api/live`** — the live-camera switch, off by default. See
  :mod:`firstsmoke.live`.

The analyzer itself is a thin adapter: it loads the upload as an incident,
hands it to :class:`~firstsmoke.agent.Lookout`, and turns the lookout's events
into progress messages and its evidence frames into files the UI can fetch.
"""

from __future__ import annotations

import os
from typing import Any

import cv2
from fastapi import Request
from servicekit import ProductInfo, ServiceConfig, create_app
from servicekit.errors import ServiceError
from servicekit.jobs import JobContext
from visioncore import Evidence, RunRecord

from . import __version__
from .agent import Lookout, ReplaySource, State
from .candidates import draw_horizon, draw_regions
from .confirm import load_confirmer
from .detector import CameraReading
from .evaluate import default_network
from .frames import Incident, load_bundle
from .paths import scenario_dir, static_dir
from .scenarios import SCENARIOS, ensure_scenario, scenario_catalogue

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


def analyze(ctx: JobContext) -> RunRecord:
    """Run the escalation loop over an uploaded or bundled incident."""
    record = ctx.record
    ctx.progress(4, "reading the incident bundle")
    try:
        incident: Incident = load_bundle(ctx.input_path)
    except Exception as exc:  # the loader raises SequenceError with a readable message
        raise ServiceError("BAD_REQUEST", str(exc)) from exc

    record.input.update(
        {
            "incident": incident.name,
            "cameras": len(incident.network),
            "frames": sum(len(s) for s in incident.sequences.values()),
            "attribution": incident.network.attribution,
        }
    )
    record.params.setdefault("threshold", ctx.params.get("threshold"))

    confirmer = load_confirmer()
    if confirmer is None:
        record.warn(
            "The learned confirmation model was not found, so the detector ran on its classical "
            "evidence alone. Run scripts/train_confirmer.py to build it."
        )

    timeline = incident.timeline()
    total = max(len(timeline), 1)
    saved: set[str] = set()

    def on_event(event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "tick":
            step = event.get("step", 0)
            ctx.progress(
                5 + 80.0 * step / total,
                f"{event['at'][11:16]} watching {len(incident.network)} cameras",
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

    lookout = Lookout(ReplaySource(incident), confirmer=confirmer, on_event=on_event)
    alert = lookout.run()

    ctx.progress(88, "collecting the frames behind the decision")
    for reading in lookout.readings:
        if reading.camera_id in saved:
            continue
        if reading.detections or reading.scene.blind:
            uri = _overlay(reading, ctx, f"{reading.camera_id}-{reading.frame_index:03d}")
            if uri:
                saved.add(reading.camera_id)
                record.add_evidence(
                    Evidence(
                        label=incident.network.get(reading.camera_id).label,
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
            "cameras": len(incident.network),
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

    if incident.truth:
        record.results.append({"truth": incident.truth})

    ctx.progress(100, "done")
    return record


def build_config() -> ServiceConfig:
    return ServiceConfig(
        product=PRODUCT,
        allowed_suffixes=(".zip",),
        max_upload_bytes=int(os.environ.get("FIRSTSMOKE_MAX_UPLOAD", 400 * 1024 * 1024)),
        static_dir=STATIC_DIR,
        max_concurrent_jobs=int(os.environ.get("FIRSTSMOKE_MAX_CONCURRENT", 2)),
        params_schema=[
            {
                "name": "threshold",
                "type": "number",
                "label": "Suspicion threshold",
                "default": 0.35,
                "min": 0.1,
                "max": 0.9,
                "step": 0.05,
                "help": (
                    "Below this a camera stays quiet. Between this and 0.68 it consults."
                ),
            }
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
