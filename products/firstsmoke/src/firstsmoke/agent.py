"""The lookout: a state machine whose next move is decided by what it just saw.

This is the part of Firstsmoke that is an agent rather than a detector, and the
distinction is worth being precise about, because it is easy to claim and easy
to fake.

The loop is:

    WATCH ──weak detection──> SUSPECT ──bearing──> CONSULT ──> CONFIRMED ──> ALERT
                                  ^                   │
                                  └───inconclusive────┤
                                                      └──────> STOOD DOWN
                                                      └──────> NEEDS A HUMAN

What makes it a loop rather than a pipeline is the arrow from the image back to
the input. When one camera produces an ambiguous detection, the **pixel column
of that detection** is converted to a compass bearing; the bearing is projected
onto the map; and the map decides which other cameras are read next and, to
within a few degrees, where in each of their frames to look. Different pixels
give different neighbours. Nothing in the loop is a fixed list.

The system is allowed three kinds of action and no others:

1. **Read a camera it was not otherwise reading**, chosen by geometry.
2. **Re-read a camera it has already read**, at a higher effective resolution
   and without letting the background model learn, when a detection is on the
   edge of the threshold.
3. **Ask a human**, when the evidence is real but will not resolve.

It cannot pan a camera, dispatch anything, or notify anyone outside the system.
That ceiling is deliberate: the consequence of a false alarm here is a person
looking at a picture, and the consequence of a missed fire is not, so the
autonomy is spent on looking harder and never on acting.

Every transition is recorded with the frames that caused it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol

from .calibration import CalibrationSet
from .cameras import Camera, Consultation, Network
from .confirm import SmokeConfirmer
from .detector import CONFIRM_AT, SUSPECT_AT, CameraReading, CameraWatch, Detection, Verdict
from .frames import CameraFrame, Incident
from .geometry import CrossingRefused, Fix, Ray, closest_pair, cross_rays, signed_delta


class State(StrEnum):
    WATCH = "watch"
    SUSPECT = "suspect"
    CONSULT = "consult"
    CONFIRMED = "confirmed"
    STOOD_DOWN = "stood_down"
    NEEDS_HUMAN = "needs_human"
    ALERTED = "alerted"


TERMINAL = (State.ALERTED, State.NEEDS_HUMAN)
"""Standing down is not terminal. A lookout that saw something, checked, and was
satisfied goes back to watching; it does not go home. Treating a stand-down as
the end of the run meant one warm-up artefact at 19:03 could close the watch
before the real ignition at 19:05."""


class FrameSource(Protocol):
    """Where frames come from. Replay and live differ only in this object."""

    network: Network

    def read(self, camera_id: str, when: datetime) -> CameraFrame | None: ...

    def history(self, camera_id: str, when: datetime, count: int) -> list[CameraFrame]: ...

    def timeline(self) -> list[datetime]: ...

    def describe(self) -> dict[str, Any]: ...


class ReplaySource:
    """Frames from a recorded incident bundle."""

    def __init__(self, incident: Incident) -> None:
        self.incident = incident
        self.network = incident.network
        self.reads = 0

    def read(self, camera_id: str, when: datetime) -> CameraFrame | None:
        frame = self.incident.at(camera_id, when)
        if frame is not None:
            self.reads += 1
        return frame

    def history(self, camera_id: str, when: datetime, count: int) -> list[CameraFrame]:
        seq = self.incident.sequences.get(camera_id)
        if not seq:
            return []
        earlier = [f for f in seq.frames if f.timestamp < when]
        self.reads += min(count, len(earlier))
        return earlier[-count:]

    def timeline(self) -> list[datetime]:
        return self.incident.timeline()

    def describe(self) -> dict[str, Any]:
        return {"kind": "replay", "incident": self.incident.name, "reads": self.reads}


@dataclass
class Transition:
    """One state change, with the evidence that caused it."""

    at: datetime
    from_state: State
    to_state: State
    trigger: str
    detail: str
    cameras: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "at": self.at.isoformat(),
            "from": self.from_state.value,
            "to": self.to_state.value,
            "trigger": self.trigger,
            "detail": self.detail,
            "cameras": list(self.cameras),
            "evidence": list(self.evidence),
            "data": dict(self.data),
        }


@dataclass
class ConsultationRecord:
    """What happened when the agent went and asked a neighbour."""

    camera_id: str
    label: str
    asked_at: datetime
    expected_bearing_deg: float
    crossing_angle_deg: float
    baseline_km: float
    why_chosen: str
    outcome: str
    """``supported``, ``silent``, ``blind``, ``no_frame`` or ``off_bearing``."""
    answer: str
    confidence_before: float
    confidence_after: float
    detection: Detection | None = None
    reading: CameraReading | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "label": self.label,
            "asked_at": self.asked_at.isoformat(),
            "expected_bearing_deg": round(self.expected_bearing_deg, 2),
            "crossing_angle_deg": round(self.crossing_angle_deg, 1),
            "baseline_km": round(self.baseline_km, 2),
            "why_chosen": self.why_chosen,
            "outcome": self.outcome,
            "answer": self.answer,
            "confidence_before": round(self.confidence_before, 4),
            "confidence_after": round(self.confidence_after, 4),
            "detection": self.detection.to_dict() if self.detection else None,
        }


@dataclass
class Alert:
    """What a human is handed. Everything needed to agree or disagree in seconds."""

    alert_id: str
    raised_at: datetime
    state: State
    confidence: float
    origin_camera: str
    origin_bearing_deg: float
    fix: Fix | None
    fix_refusal: dict[str, Any] | None
    rays: list[dict[str, Any]]
    consultations: list[ConsultationRecord]
    transitions: list[Transition]
    headline: str
    reasoning: list[str]
    unusable: list[dict[str, str]]
    evidence: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "raised_at": self.raised_at.isoformat(),
            "state": self.state.value,
            "confidence": round(self.confidence, 4),
            "origin_camera": self.origin_camera,
            "origin_bearing_deg": round(self.origin_bearing_deg, 2),
            "headline": self.headline,
            "reasoning": list(self.reasoning),
            "fix": self.fix.to_dict() if self.fix else None,
            "fix_refusal": self.fix_refusal,
            "rays": list(self.rays),
            "consultations": [c.to_dict() for c in self.consultations],
            "transitions": [t.to_dict() for t in self.transitions],
            "unusable": list(self.unusable),
            "evidence": list(self.evidence),
        }


BEARING_TOLERANCE_DEG = 6.0
"""How far a neighbour's detection may sit from where the geometry says it should
be and still count as the same thing. Six degrees is generous: it covers the
origin camera's own bearing error, the assumed-range guess, and a plume that is
genuinely a kilometre wide."""

MAX_CONSULT_ROUNDS = 3
RE_EXAMINE_FRAMES = 4

MIN_FRAMES_TO_ESCALATE = 4
"""How many frames a candidate must have been tracked for before the agent will
spend a consultation on it.

The score alone is not enough. A two-frame blob can reach 0.4 on a lucky
combination of veiling and attachment, and two such blobs on two cameras can sit
within a few degrees of each other by coincidence — which, on the synthetic
network, produced a confident confirmed alert three kilometres from the fire.
Four frames is the minimum from which a growth rate means anything, and the
whole argument of this product is that growth is what identifies a column."""

SUPPORTER_WEIGHT = 0.7
"""How much a neighbour's own confidence counts toward the joint confidence.

Less than one, because a neighbour is answering a leading question: it was told
where to look, so its agreement is worth real but not equal weight against the
camera that raised the alarm unprompted."""

MIN_FRAMES_TO_SUPPORT = 3
"""The same bar for a neighbour, one frame lower: a neighbour is answering a
specific question about a specific bearing, so it starts with more context than
the camera that raised the alarm."""


class Lookout:
    """Runs the escalation loop over a frame source."""

    def __init__(
        self,
        source: FrameSource,
        *,
        confirmer: SmokeConfirmer | None = None,
        calibrations: CalibrationSet | None = None,
        suspect_at: float = SUSPECT_AT,
        confirm_at: float = CONFIRM_AT,
        max_consult_rounds: int = MAX_CONSULT_ROUNDS,
        on_event=None,
        keep_watching: bool = False,
        max_alerts: int = 8,
        keep_readings: bool = True,
    ) -> None:
        self.source = source
        self.network = source.network
        self.confirmer = confirmer
        self.calibrations = calibrations
        self.suspect_at = suspect_at
        self.confirm_at = confirm_at
        self.max_consult_rounds = max_consult_rounds
        self.on_event = on_event
        self.keep_readings = keep_readings
        """Hold every reading, image and masks included, for the interface. An
        upload turns this off and keeps only each camera's latest reading: a
        long clip's readings were most of the memory an upload used."""
        self.latest: dict[str, CameraReading] = {}
        self.keep_watching = keep_watching
        """Carry on after a flag instead of holding at it. A network holds, because
        a person is about to look. A single uploaded clip has nobody coming, and
        holding at a cloud in the third second means never reaching the smoke."""
        self.max_alerts = max_alerts
        self._flagged: dict[tuple[str, int], int] = {}
        """Which tracks have already raised something, and how loudly: 1 for a
        flag a person must settle, 2 for an assertion. A track is not raised
        again at the same level, or one cloud would fill the list."""
        self.state = State.WATCH
        self.transitions: list[Transition] = []
        self.consultations: list[ConsultationRecord] = []
        self.alerts: list[Alert] = []
        self.watches: dict[str, CameraWatch] = {}
        self.readings: list[CameraReading] = []
        self.unusable: dict[str, str] = {}
        self._rounds = 0
        self._consulted: set[str] = set()
        self._pursuing: tuple[str, int] | None = None
        """Which camera and track the current escalation is about. When the
        strongest candidate changes to a different object, the consultation
        budget starts again — otherwise a transient early in the sequence spends
        the rounds that the real column needed."""

    # -- plumbing ------------------------------------------------------------
    def watch_for(self, camera: Camera) -> CameraWatch:
        watch = self.watches.get(camera.camera_id)
        if watch is None:
            watch = CameraWatch(
                camera, confirmer=self.confirmer,
                calibration=self.calibrations.get(camera.camera_id) if self.calibrations else None,
                suspect_at=self.suspect_at, confirm_at=self.confirm_at,
                keep_readings=self.keep_readings,
            )
            self.watches[camera.camera_id] = watch
        return watch

    def _emit(self, kind: str, **payload: Any) -> None:
        if self.on_event:
            self.on_event({"type": kind, **payload})

    def _go(
        self,
        to_state: State,
        at: datetime,
        trigger: str,
        detail: str,
        *,
        cameras: Iterable[str] = (),
        evidence: Iterable[str] = (),
        **data: Any,
    ) -> Transition:
        transition = Transition(
            at=at, from_state=self.state, to_state=to_state, trigger=trigger, detail=detail,
            cameras=list(cameras), evidence=list(evidence), data=data,
        )
        self.transitions.append(transition)
        self.state = to_state
        self._emit("transition", transition=transition.to_dict())
        return transition

    # -- the loop ------------------------------------------------------------
    def run(self, *, limit: int | None = None) -> Alert | None:
        """Walk the timeline until the loop reaches a terminal state or runs out."""
        timeline = self.source.timeline()
        if limit:
            timeline = timeline[:limit]
        if not timeline:
            return None

        for step, when in enumerate(timeline):
            self._emit("tick", at=when.isoformat(), step=step, total=len(timeline),
                       state=self.state.value)
            alert = self.step(when)
            if alert is not None and self.keep_watching and len(self.alerts) < self.max_alerts:
                track = self._pursuing[1] if self._pursuing else -1
                level = 2 if alert.confidence >= self.confirm_at else 1
                self._flagged[(alert.origin_camera, track)] = level
                self._go(State.WATCH, when, "kept_watching",
                         "nobody else can settle this flag, so it stays open and the watch "
                         "carries on through the rest of the footage")
                self._rounds = 0
                self._consulted.clear()
                self._pursuing = None
                continue
            if alert is not None:
                return alert
            if self.state in TERMINAL:
                break
        if self.alerts and self.keep_watching:
            count = len(self.alerts)
            self._go(State.NEEDS_HUMAN, timeline[-1], "sequence_ended",
                     f"the footage ended with {count} flag{'s' if count != 1 else ''} still open, "
                     "and none of them can be settled from one camera")
            return max(self.alerts, key=lambda a: a.confidence)
        if self.state not in TERMINAL:
            self._go(State.WATCH, timeline[-1], "sequence_ended",
                     "the recording ended with nothing that met the threshold")
        return None

    def step(self, when: datetime) -> Alert | None:
        """One tick: sweep the watch set, then act on the strongest thing seen."""
        sweep = self._sweep(when)
        if self._flagged:
            sweep = [
                d for d in sweep
                if self._flagged.get((d.camera_id, d.track_id), 0)
                < (2 if d.confidence >= self.confirm_at else 1)
            ]
        if not sweep:
            return None

        strongest = max(sweep, key=lambda d: d.confidence)
        key = (strongest.camera_id, strongest.track_id)
        if key != self._pursuing:
            self._pursuing = key
            self._rounds = 0
            self._consulted.clear()
        if strongest.confidence < self.suspect_at:
            if self.state is not State.WATCH:
                self._go(State.WATCH, when, "faded",
                         f"the best candidate fell to {strongest.confidence:.2f}, below the "
                         f"{self.suspect_at:.2f} worth looking at")
                self._rounds = 0
                self._consulted.clear()
                self._pursuing = None
            return None

        if self.state in (State.WATCH, State.STOOD_DOWN):
            self._go(
                State.SUSPECT, when, "weak_detection",
                f"{self.network.get(strongest.camera_id).label} scored "
                f"{strongest.confidence:.2f}: {strongest.summary()}",
                cameras=[strongest.camera_id],
                bearing_deg=round(strongest.bearing_deg, 2),
                confidence=round(strongest.confidence, 4),
            )

        # Action 2: look harder at the camera that raised it, before bothering
        # the neighbours. Cheaper than a consultation and often decisive.
        strongest = self._re_examine(strongest, when)

        if strongest.growth.frames < MIN_FRAMES_TO_ESCALATE:
            self._emit(
                "holding",
                camera_id=strongest.camera_id,
                frames=strongest.growth.frames,
                needed=MIN_FRAMES_TO_ESCALATE,
                confidence=round(strongest.confidence, 4),
            )
            return None

        return self._consult(strongest, when)

    def _sweep(self, when: datetime) -> list[Detection]:
        """Read every active camera once and collect their best detections."""
        found: list[Detection] = []
        for camera in self.network.active():
            frame = self.source.read(camera.camera_id, when)
            if frame is None:
                continue
            reading = self.watch_for(camera).observe(frame)
            self.latest[camera.camera_id] = reading
            if self.keep_readings:
                self.readings.append(reading)
            self._emit("reading", reading=reading.to_dict())
            if reading.verdict is Verdict.BLIND:
                self.unusable[camera.camera_id] = reading.scene.reason
                continue
            self.unusable.pop(camera.camera_id, None)
            best = reading.best
            if best is not None:
                found.append(best)
        return found

    def _re_examine(self, detection: Detection, when: datetime) -> Detection:
        """Re-run the suspect camera's recent frames without letting it learn.

        A background model that keeps learning will absorb a slow plume: by the
        tenth frame the smoke *is* the background and the camera reports clear
        while the hillside burns. Freezing the model and replaying the recent
        history gives the growth analysis the full span it needs, and it is the
        one action the agent takes on itself rather than on a neighbour.
        """
        camera = self.network.get(detection.camera_id)
        history = self.source.history(camera.camera_id, when, RE_EXAMINE_FRAMES)
        if len(history) < 2:
            return detection
        watch = self.watch_for(camera)
        before = detection.confidence
        best = detection
        for frame in history:
            reading = watch.observe(frame, learn=False)
            # Only the same track counts. Taking the best detection of any kind
            # let a re-examination hand back a different object on the other side
            # of the frame, and the agent then consulted neighbours about a
            # bearing that nothing had ever been seen on.
            for candidate in reading.detections:
                if candidate.track_id != detection.track_id:
                    continue
                if candidate.confidence > best.confidence:
                    best = candidate
        self._emit(
            "re_examine",
            camera_id=camera.camera_id,
            frames=len(history),
            confidence_before=round(before, 4),
            confidence_after=round(best.confidence, 4),
        )
        if abs(best.confidence - before) > 1e-6:
            self.transitions.append(
                Transition(
                    at=when, from_state=self.state, to_state=self.state,
                    trigger="re_examined",
                    detail=(
                        f"replayed {len(history)} earlier frames from {camera.label} with the "
                        f"background model frozen; confidence moved {before:.2f} to "
                        f"{best.confidence:.2f}"
                    ),
                    cameras=[camera.camera_id],
                    data={"frames": len(history), "before": round(before, 4),
                          "after": round(best.confidence, 4)},
                )
            )
        return best

    # -- action 1: ask the neighbours ---------------------------------------
    def _consult(self, detection: Detection, when: datetime) -> Alert | None:
        camera = self.network.get(detection.camera_id)
        candidates = self.network.consultable(camera, detection.bearing_deg)
        fresh = [c for c in candidates if c.camera.camera_id not in self._consulted]

        if not candidates:
            if len(self.network) == 1 and not camera.position_known:
                why = (
                    "this footage comes from one camera whose position is not known, so no "
                    "second view can cross it and no location on a map is possible"
                )
            else:
                why = "no camera overlooks that bearing"
            return self._resolve_without_neighbours(detection, when, why)

        self._go(
            State.CONSULT, when, "bearing_selected_cameras",
            f"bearing {detection.bearing_deg:.1f} degrees from {camera.label} puts the candidate "
            f"where {len(candidates)} other camera{'s' if len(candidates) != 1 else ''} should see "
            "it; reading them now",
            cameras=[c.camera.camera_id for c in candidates],
            bearing_deg=round(detection.bearing_deg, 2),
            candidates=[c.to_dict() for c in candidates],
        )
        self._rounds += 1

        supporting: list[ConsultationRecord] = []
        for consultation in fresh or candidates:
            record = self._ask(consultation, detection, when)
            self.consultations.append(record)
            self._consulted.add(consultation.camera.camera_id)
            self._emit("consultation", consultation=record.to_dict())
            if record.outcome == "supported":
                supporting.append(record)

        if supporting:
            combined = detection.confidence + sum(
                0.5 * (r.confidence_after or 0.0) for r in supporting
            )
            if combined >= self.confirm_at:
                return self._confirm(detection, supporting, when)
            self._emit(
                "corroborated_but_weak",
                camera_id=detection.camera_id,
                combined=round(combined, 4),
                needed=self.confirm_at,
            )

        if self._rounds >= self.max_consult_rounds:
            return self._stand_down_or_escalate(detection, when)
        return None

    def _ask(
        self, consultation: Consultation, origin: Detection, when: datetime
    ) -> ConsultationRecord:
        """Read one neighbour and decide whether it saw the same thing."""
        neighbour = consultation.camera
        base = ConsultationRecord(
            camera_id=neighbour.camera_id,
            label=neighbour.label,
            asked_at=when,
            expected_bearing_deg=consultation.expected_bearing_deg,
            crossing_angle_deg=consultation.crossing_angle_deg,
            baseline_km=consultation.baseline_m / 1000.0,
            why_chosen=consultation.why,
            outcome="no_frame",
            answer="no frame available from this camera at that time",
            confidence_before=0.0,
            confidence_after=0.0,
        )

        watch = self.watch_for(neighbour)
        # Prime the neighbour's background model on its own recent past. A camera
        # the agent has not been reading has no model, and a cold model cannot
        # tell us anything, so the consultation has to pay for its own warm-up.
        for frame in self.source.history(neighbour.camera_id, when, RE_EXAMINE_FRAMES + 2):
            watch.observe(frame)

        frame = self.source.read(neighbour.camera_id, when)
        if frame is None:
            return base

        reading = watch.observe(frame)
        if self.keep_readings:
            self.readings.append(reading)
        base.reading = reading

        if reading.verdict is Verdict.BLIND:
            self.unusable[neighbour.camera_id] = reading.scene.reason
            base.outcome = "blind"
            base.answer = f"cannot answer: {reading.scene.reason}"
            return base

        on_bearing = [
            d for d in reading.detections
            if abs(signed_delta(d.bearing_deg, consultation.expected_bearing_deg))
            <= BEARING_TOLERANCE_DEG
        ]
        off_bearing = [d for d in reading.detections if d not in on_bearing]

        if not on_bearing:
            if off_bearing:
                best_off = max(off_bearing, key=lambda d: d.confidence)
                expected = consultation.expected_bearing_deg
                base.outcome = "off_bearing"
                base.answer = (
                    f"saw something at {best_off.bearing_deg:.1f} degrees, which is "
                    f"{abs(signed_delta(best_off.bearing_deg, expected)):.0f} "
                    f"degrees from where this candidate should be; that is a different thing"
                )
                base.confidence_after = best_off.confidence
            else:
                base.outcome = "silent"
                base.answer = (
                    f"clear at {consultation.expected_bearing_deg:.1f} degrees, where the "
                    "candidate should have been"
                )
            return base

        best = max(on_bearing, key=lambda d: d.confidence)
        base.detection = best
        base.confidence_after = best.confidence
        if best.confidence >= self.suspect_at and best.growth.frames < MIN_FRAMES_TO_SUPPORT:
            base.outcome = "silent"
            base.answer = (
                f"has something on that bearing at {best.confidence:.2f}, but has only seen it in "
                f"{best.growth.frames} frame{'s' if best.growth.frames != 1 else ''}; not yet a "
                "second opinion"
            )
        elif best.confidence >= self.suspect_at:
            base.outcome = "supported"
            base.answer = (
                f"sees it too, at {best.bearing_deg:.1f} degrees "
                f"({abs(signed_delta(best.bearing_deg, consultation.expected_bearing_deg)):.1f} "
                f"degrees from predicted), confidence {best.confidence:.2f}: {best.summary()}"
            )
        else:
            base.outcome = "silent"
            reason = best.rejections[0].message if best.rejections else "nothing above threshold"
            base.answer = (
                f"found a faint change on that bearing but scored it "
                f"{best.confidence:.2f}: {reason}"
            )
        return base

    # -- outcomes ------------------------------------------------------------
    def _confirm(
        self, origin: Detection, supporting: list[ConsultationRecord], when: datetime
    ) -> Alert:
        rays = [
            Ray(
                camera_id=origin.camera_id,
                lat=self.network.get(origin.camera_id).lat,
                lon=self.network.get(origin.camera_id).lon,
                bearing_deg=origin.bearing_deg,
                sigma_deg=origin.bearing_sigma_deg,
                max_range_m=self.network.get(origin.camera_id).range_m,
            )
        ]
        for record in supporting:
            detection = record.detection
            if detection is None:
                continue
            camera = self.network.get(record.camera_id)
            rays.append(
                Ray(
                    camera_id=camera.camera_id, lat=camera.lat, lon=camera.lon,
                    bearing_deg=detection.bearing_deg,
                    sigma_deg=detection.bearing_sigma_deg,
                    max_range_m=camera.range_m,
                )
            )

        fix: Fix | None = None
        refusal: dict[str, Any] | None = None
        try:
            fix = cross_rays(rays)
        except CrossingRefused as exc:
            refusal = {"code": exc.code, "message": exc.message, "details": dict(exc.details)}
            # A refusal that says only "no crossing" tells nobody anything. How
            # far apart the two sightlines passed, against how wide their
            # combined uncertainty was at that range, says whether these are two
            # cameras looking at one thing badly or at two different things.
            approach = closest_pair(rays)
            if approach is not None:
                refusal["nearest_approach"] = approach.to_dict()

        # Independent cameras, so the confidences combine as a noisy OR rather
        # than a sum. A sum reached 0.99 from one 0.60 and two 0.5s, which is
        # not what three moderate opinions are worth; the noisy OR gives 0.84
        # for the same evidence and still cannot exceed 1.
        confidence = 1.0 - (1.0 - origin.confidence)
        for record in supporting:
            confidence = 1.0 - (1.0 - confidence) * (
                1.0 - SUPPORTER_WEIGHT * (record.confidence_after or 0.0)
            )
        names = ", ".join(self.network.get(r.camera_id).label for r in supporting)
        origin_label = self.network.get(origin.camera_id).label

        self._go(
            State.CONFIRMED, when, "crossed_bearings",
            f"{origin_label} and {names} are looking at the same bearing; "
            + (
                f"they cross {fix.range_m[origin.camera_id] / 1000:.1f} km out"
                if fix else f"the crossing was refused: {refusal['message'] if refusal else ''}"
            ),
            cameras=[origin.camera_id] + [r.camera_id for r in supporting],
            fix=fix.to_dict() if fix else None,
            refusal=refusal,
        )
        return self._raise(origin, confidence, fix, refusal, rays, when, State.ALERTED)

    def _resolve_without_neighbours(
        self, detection: Detection, when: datetime, why: str
    ) -> Alert | None:
        """A bearing with nobody to cross it against.

        We still alert if the single camera is confident, because one lookout
        seeing a clear column is worth telling someone about. But the alert says
        "bearing only" and carries no position, which is the truth.
        """
        if detection.confidence < self.confirm_at:
            self._go(
                State.NEEDS_HUMAN, when, "no_second_view",
                f"{why}, and one camera alone scored {detection.confidence:.2f}, which is not "
                "enough to assert on its own",
                cameras=[detection.camera_id],
            )
            return self._raise(detection, detection.confidence, None,
                               {"code": "NO_SECOND_VIEW", "message": why, "details": {}},
                               self._origin_ray(detection), when, State.NEEDS_HUMAN)
        self._go(
            State.CONFIRMED, when, "single_camera_confident",
            f"{why}, but {self.network.get(detection.camera_id).label} scored "
            f"{detection.confidence:.2f} on its own",
            cameras=[detection.camera_id],
        )
        rays = self._origin_ray(detection)
        return self._raise(detection, detection.confidence, None,
                           {"code": "NO_SECOND_VIEW", "message": why, "details": {}},
                           rays, when, State.ALERTED)

    def _origin_ray(self, detection: Detection) -> list[Ray]:
        """The one bearing we do have.

        An unresolved alert used to carry no rays at all, so the map that was
        supposed to show a person *what* was unresolved showed them an empty
        sheet. One ray with no crossing is the honest picture."""
        camera = self.network.get(detection.camera_id)
        if not (camera.position_known and camera.aim_known):
            return []
        return [
            Ray(camera.camera_id, camera.lat, camera.lon, detection.bearing_deg,
                detection.bearing_sigma_deg, camera.range_m)
        ]

    def _stand_down_or_escalate(self, detection: Detection, when: datetime) -> Alert | None:
        """Nobody backed it up. Either it was never there, or nobody could see."""
        # One entry per camera, not one per consultation round. A neighbour asked
        # in three successive rounds appeared three times, and the sentence a
        # person actually reads turned into a stuck record.
        blind = list(
            {
                c.camera_id: c
                for c in self.consultations
                if c.outcome == "blind" and c.camera_id in self._consulted
            }.values()
        )
        silent = list(
            {
                c.camera_id: c
                for c in self.consultations
                if c.outcome in ("silent", "off_bearing")
            }.values()
        )

        if blind and not silent:
            detail = (
                "every camera that overlooks that bearing is unusable: "
                + "; ".join(f"{c.label} {c.answer}" for c in blind)
                + ". That is not a stand-down, it is a gap in cover."
            )
            self._go(State.NEEDS_HUMAN, when, "neighbours_blind", detail,
                     cameras=[c.camera_id for c in blind])
            return self._raise(detection, detection.confidence, None,
                               {"code": "NEIGHBOURS_BLIND", "message": detail, "details": {}},
                               self._origin_ray(detection), when, State.NEEDS_HUMAN)

        if detection.confidence >= self.confirm_at:
            detail = (
                f"{self.network.get(detection.camera_id).label} is still at "
                f"{detection.confidence:.2f} after {self._rounds} rounds and no neighbour agrees; "
                "a person should look at this"
            )
            self._go(State.NEEDS_HUMAN, when, "unresolved", detail, cameras=[detection.camera_id])
            return self._raise(detection, detection.confidence, None,
                               {"code": "UNRESOLVED", "message": detail, "details": {}},
                               self._origin_ray(detection), when, State.NEEDS_HUMAN)

        names = "; ".join(f"{c.label} {c.answer}" for c in silent) or "no neighbour saw anything"
        self._go(
            State.STOOD_DOWN, when, "not_corroborated",
            f"stood down after {self._rounds} rounds: {names}",
            cameras=[c.camera_id for c in silent],
            origin_confidence=round(detection.confidence, 4),
        )
        self._rounds = 0
        self._consulted.clear()
        return None

    def _raise(
        self,
        detection: Detection,
        confidence: float,
        fix: Fix | None,
        refusal: dict[str, Any] | None,
        rays: list[Ray],
        when: datetime,
        state: State,
    ) -> Alert:
        camera = self.network.get(detection.camera_id)
        supporting = [c for c in self.consultations if c.outcome == "supported"]

        if not camera.aim_known:
            prefix = "Unresolved: possible smoke" if state is State.NEEDS_HUMAN else "Smoke"
            headline = f"{prefix} {detection.where}, one camera, no location"
        elif state is State.NEEDS_HUMAN:
            headline = (
                f"Unresolved: possible smoke on bearing {detection.bearing_deg:.0f} from "
                f"{camera.label}"
            )
        elif fix is not None:
            headline = (
                f"Smoke {fix.range_m[camera.camera_id] / 1000:.1f} km from {camera.site_name}, "
                f"confirmed by {len(supporting) + 1} cameras"
            )
        else:
            headline = f"Smoke on bearing {detection.bearing_deg:.0f} from {camera.label}"

        # One line per camera, in the order they were asked. A camera that was
        # asked twice, or that is also in the unusable list, must not appear
        # twice: the first draft printed every blind neighbour once per
        # consultation round and read like a stuck record.
        reasoning = [detection.summary()]
        seen: set[str] = set()
        for record in self.consultations:
            if record.camera_id in seen:
                continue
            seen.add(record.camera_id)
            reasoning.append(f"{record.label}: {record.answer}")
        if fix is not None:
            reasoning.append(
                f"The bearings cross at {fix.crossing_angle_deg:.0f} degrees, giving a position "
                f"good to about {fix.semi_major_m:.0f} m along the long axis."
            )
        elif refusal:
            message = refusal["message"].rstrip()
            if not message.endswith((".", "!", "?")):
                message += "."
            reasoning.append(f"No position: {message}")
            approach = refusal.get("nearest_approach")
            if approach:
                reasoning.append(
                    f"The bearings pass {approach['distance_m'] / 1000:.1f} km apart, against a "
                    f"combined uncertainty of {approach['corridor_m']:.0f} m at that range, so "
                    + (
                        "they are consistent and it is the geometry that is soft."
                        if approach["consistent"]
                        else "they are not looking at the same thing."
                    )
                )
        for camera_id, reason in self.unusable.items():
            if camera_id in seen:
                continue
            reasoning.append(f"{self.network.get(camera_id).label} could not help: {reason}")

        alert = Alert(
            alert_id=f"{camera.camera_id}-{int(when.timestamp())}",
            raised_at=when,
            state=state,
            confidence=float(min(confidence, 0.99)),
            origin_camera=camera.camera_id,
            origin_bearing_deg=detection.bearing_deg,
            fix=fix,
            fix_refusal=refusal,
            rays=[
                {
                    "camera_id": r.camera_id, "lat": r.lat, "lon": r.lon,
                    "bearing_deg": round(r.bearing_deg, 2),
                    "sigma_deg": round(r.sigma_deg, 2),
                    "length_m": r.max_range_m,
                }
                for r in rays
            ],
            consultations=list(self.consultations),
            transitions=list(self.transitions),
            headline=headline,
            reasoning=reasoning,
            unusable=[
                {"camera_id": cid, "label": self.network.get(cid).label, "reason": reason}
                for cid, reason in self.unusable.items()
            ],
        )
        self.alerts.append(alert)
        self._go(state, when, "alert_raised", headline, cameras=[camera.camera_id])
        self._emit("alert", alert=alert.to_dict())
        return alert

    # -- reporting -----------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "source": self.source.describe(),
            "cameras": len(self.network),
            # The map draws from this, not from the service's global camera
            # list. An uploaded bundle carries its own network, and drawing the
            # published HPWREN geometry instead put markers on the wrong
            # continent while the rays were computed from the right one.
            "network": self.network.to_dict(),
            "frames_read": sum(w.observations for w in self.watches.values()),
            "transitions": [t.to_dict() for t in self.transitions],
            "consultations": [c.to_dict() for c in self.consultations],
            "alerts": [a.to_dict() for a in self.alerts],
            "unusable": [
                {"camera_id": cid, "label": self.network.get(cid).label, "reason": reason}
                for cid, reason in self.unusable.items()
            ],
        }


def utcnow() -> datetime:
    return datetime.now(UTC)
