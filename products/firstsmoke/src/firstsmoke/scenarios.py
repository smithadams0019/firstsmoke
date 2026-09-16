"""The incidents the deployed service ships with, and why they are rendered.

A judge should see the whole loop run within seconds of opening the page, from a
cold container, with nothing to download and nothing to configure. That means
the frames have to be in the image.

**They are synthetic, deliberately, and it is a licence decision rather than a
shortcut.** The recorded imagery this product is evaluated against is HPWREN's
FIgLib, which is CC BY-NC-ND 4.0. The NoDerivatives term is the problem: the
interface's whole point is to show an annotated frame with the detected region
drawn on it, and an annotated frame is a derivative work. Publishing those from
a hosted endpoint would breach the licence of the people whose cameras made this
possible. So the bundled scenarios are rendered by :mod:`firstsmoke.synth`,
which is our own code and our own pixels, and they are labelled as rendered
everywhere they appear.

Real imagery is still where the numbers come from. `docs/evaluation.md` reports
65 recorded FIgLib sequences from 20 real cameras, cached locally by whoever
runs the evaluation and never redistributed by us. And the live switch
(:mod:`firstsmoke.live`) fetches current stills from the HPWREN CDN on demand,
with the credit line the project asks for, which is use rather than
redistribution.

Four scenarios, chosen so the set contains one of each ending the state machine
can reach. A demo that only shows the success is not a demo of an honest system.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .frames import write_bundle
from .synth import synthetic_incident

START = datetime(2026, 9, 16, 19, 0, tzinfo=UTC)


@dataclass(frozen=True)
class Scenario:
    name: str
    title: str
    blurb: str
    expect: str
    """What the agent should end up doing. Shown in the interface before the run,
    so a judge can check the claim against the outcome rather than being told
    afterwards what they just saw."""
    build: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "blurb": self.blurb,
            "expect": self.expect,
            "synthetic": True,
        }


def _confirmed():
    return synthetic_incident(
        frames=22, plume_at=6, seed=11, cameras=3, name="Crossed bearings", start=START
    )


def _stand_down():
    return synthetic_incident(
        frames=18, plume_at=99, seed=11, cameras=3,
        impostors={"syn0-mobo-c": "cloud", "syn1-mobo-c": "dust"},
        name="Cloud and dust", start=START,
    )


def _blind():
    return synthetic_incident(
        frames=20, plume_at=6, seed=11, cameras=3,
        blind={"syn1-mobo-c": "fog", "syn2-mobo-c": "night"},
        name="Nobody else can see", start=START,
    )


def _frozen():
    return synthetic_incident(
        frames=20, plume_at=7, seed=23, cameras=3,
        blind={"syn1-mobo-c": "frozen", "syn2-mobo-c": "droplet"},
        name="A dead feed and a wet lens", start=START,
    )


SCENARIOS: dict[str, Scenario] = {
    "crossed-bearings": Scenario(
        name="crossed-bearings",
        title="A column two cameras agree on",
        blurb=(
            "Three lookouts on a ridge. One sees a weak change, converts its pixel column to a "
            "bearing, and reads the two cameras that overlook that bearing."
        ),
        expect="Should end in an alert with a position and an uncertainty ellipse.",
        build=_confirmed,
    ),
    "cloud-and-dust": Scenario(
        name="cloud-and-dust",
        title="A cloud and a dust plume",
        blurb=(
            "The same network with no fire. A cloud crosses one camera and a vehicle throws dust "
            "on another: both are grey, both are new, neither is a column."
        ),
        expect="Should stand down, naming what each impostor did that smoke does not.",
        build=_stand_down,
    ),
    "nobody-else-can-see": Scenario(
        name="nobody-else-can-see",
        title="A fire nobody can corroborate",
        blurb=(
            "One camera sees a plume. Of the two that overlook it, one is fogged in and the other "
            "has no night capability."
        ),
        expect=(
            "Should refuse to stand down and ask for a human, because silence from a blind camera "
            "is not evidence of an empty hillside."
        ),
        build=_blind,
    ),
    "dead-feed-wet-lens": Scenario(
        name="dead-feed-wet-lens",
        title="A frozen feed and a wet lens",
        blurb=(
            "One neighbour is repeating the same picture and the other has water on the glass. "
            "Both are up; neither is watching anything."
        ),
        expect="Should mark both cameras unusable and say which fault each one has.",
        build=_frozen,
    ),
}


def scenario_catalogue() -> list[dict[str, Any]]:
    return [scenario.to_dict() for scenario in SCENARIOS.values()]


def ensure_scenario(name: str, directory: Path) -> Path:
    """Return the bundle for a scenario, rendering it once and caching it.

    Rendering three cameras of twenty frames takes about twelve seconds, which
    is far too long to do while a judge waits. The container build runs
    :func:`build_all`, so on a deployed instance this only ever finds the file
    already there; the on-demand path exists so a developer who cloned the repo
    and started the server gets the same behaviour without a build step.
    """
    scenario = SCENARIOS[name]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.zip"
    if path.is_file() and path.stat().st_size > 1024:
        return path
    write_bundle(scenario.build(), path)
    return path


def build_all(directory: Path) -> list[Path]:
    """Render every scenario. Called at container build time."""
    return [ensure_scenario(name, directory) for name in SCENARIOS]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
