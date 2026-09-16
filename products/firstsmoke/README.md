# Firstsmoke

A lookout for mountain-top camera networks. It finds the first smoke column of a
wildfire, asks the neighbouring cameras when it is unsure, and crosses their
bearings on a map.

**Live: https://afynmqkk9e.eu-west-1.awsapprunner.com**

Built for the OpenCV AI Competition 2026. OpenCV 5.0.0, pinned.

---

## What it is

Most wildfire camera systems are classifiers: a frame goes in, a probability
comes out. That design cannot separate a smoke column from a cloud, because in a
single frame they look the same. Both are grey, both are soft-edged, both
desaturate what is behind them.

Firstsmoke behaves like a lookout instead. It watches how a shape *behaves*
over minutes, and when it is unsure it goes and looks somewhere else:

1. **Detect the change.** A background model per camera, with the sun's movement
   fitted out rather than learned away, and candidate regions grown by hysteresis
   so a faint plume top stays attached to its dense base.
2. **Escalate when unsure.** A weak detection's *pixel column* becomes a compass
   bearing. The bearing decides which other cameras overlook that patch of
   ground, and to within a few degrees where in each of their frames to look.
   Different pixels select different cameras. This is the loop.
3. **Localise.** Two or more confirmed bearings are crossed on the map, with an
   uncertainty ellipse from each camera's own angular error.
4. **Report.** A timestamped alert carrying the frames, the bearings, which
   cameras were consulted, what each one answered, and the measurement behind
   every step.

It also says plainly when a camera cannot be believed: night without
illumination, fog, rain on the lens, direct sun, or a feed that has frozen. A
blind camera's silence is never counted as evidence of an empty hillside.

## Honest summary of the results

On 64 recorded sequences from 44 real HPWREN cameras, at the shipped operating
point: **90.5% of fires detected, a median of 60 seconds after the human who
labelled them, and 487 false positives per camera-day.** The second number is
far too high to deploy. Requiring a second camera to agree cuts it by 60% and
costs 24 points of detection. On recorded data the crossing is refused more
often than it succeeds, and the system says so rather than guessing; on
synthetic incidents where we placed the fire ourselves, the fix lands 13 m from
the truth.

Full numbers, failure cases and what they mean: **[docs/evaluation.md](docs/evaluation.md)**.

No field deployment trial exists for this system or, as far as we could find, for
any comparable one.

## Pinned dependencies

| | |
|---|---|
| `opencv-python==5.0.0.93` | local development, with GUI support |
| `opencv-python-headless==5.0.0.93` | the container |
| `numpy==2.5.3` | |
| `fastapi==0.141.1`, `uvicorn[standard]==0.53.0` | |
| `pydantic==2.13.5`, `python-multipart==0.0.32` | |
| `boto3==1.42.6` | |
| `onnx==1.20.0` | **training only**; the runtime loads ONNX through `cv2.dnn` |

OpenCV 4.14.0 was released *after* 5.0.0, so an unpinned `pip install
opencv-python` resolves to 4.x. The pin appears in `pyproject.toml`, in the
`Dockerfile`, and as an assertion inside `visioncore` that raises on import. The
running version is printed at `/version` and in the interface.

## Run it

```bash
# from the repository root
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python -e packages/visioncore -e packages/servicekit
uv pip install --python .venv/bin/python -e products/firstsmoke

# the four bundled incidents, rendered once (about 100 seconds)
.venv/bin/python -c "from pathlib import Path; from firstsmoke.scenarios import build_all; \
    build_all(Path('products/firstsmoke/data/scenarios'))"

FIRSTSMOKE_SCENARIOS=$PWD/products/firstsmoke/data/scenarios \
  .venv/bin/python -m uvicorn firstsmoke.service:app --port 8099
```

Open http://127.0.0.1:8099 and press one of the four incidents.

### Test

```bash
.venv/bin/python -m pytest products/firstsmoke/tests -q          # 173 tests
.venv/bin/python -m pytest products/firstsmoke/tests -q -m "not slow"   # skip the agent runs
.venv/bin/ruff check products/firstsmoke
```

The interesting ones are in `tests/test_agent.py`: they render a camera network
looking at a fire at a latitude and longitude we chose, and assert the agent
recovers the position to within 600 m without ever being told it.

### Evaluate against real fires

```bash
cd products/firstsmoke
python scripts/fetch_figlib.py     # ~5,000 frames from HPWREN, 770 MB, ~25 min
python scripts/train_confirmer.py  # optional ONNX confirmer, ~5 min, numpy only
python scripts/evaluate.py         # ~12 min, writes docs/evaluation.json
```

`fetch_figlib.py` caches outside the repository. HPWREN imagery is CC BY-NC-ND
4.0 and is not redistributed here.

### Watch live cameras

Off by default, deliberately.

```bash
FIRSTSMOKE_LIVE=1 .venv/bin/python -m uvicorn firstsmoke.service:app --port 8099
curl localhost:8099/api/live
```

Live stills come from `https://cdn.hpwren.ucsd.edu/RT/<camera_id>.jpg`. See
`src/firstsmoke/live.py` for the terms this honours.

## Deploy

```bash
infra/ecr.sh firstsmoke --context . --dockerfile products/firstsmoke/Dockerfile
infra/apprunner.sh firstsmoke --cpu 2 --memory 4 --port 8080
infra/apprunner.sh firstsmoke --status
```

Prefix both with `AWS_REGION=eu-west-1` to reproduce the current deployment.
**[docs/costs.md](docs/costs.md)** explains why it is not in us-east-1, lists every
resource created, and gives the hourly cost.

Build the image alone:

```bash
docker build -f products/firstsmoke/Dockerfile -t firstsmoke:local .
docker run --rm -p 8098:8080 firstsmoke:local
```

## Documentation

| | |
|---|---|
| [docs/report.md](docs/report.md) | the technical report: problem, users, architecture, OpenCV 5 implementation, AWS, evaluation, limitations, responsible use |
| [docs/evaluation.md](docs/evaluation.md) | the numbers, the failure cases, and what the evaluation does not establish |
| [docs/architecture.md](docs/architecture.md) | Mermaid diagrams of the pipeline, the agent loop and the AWS components |
| [docs/costs.md](docs/costs.md) | what is running on AWS and what it costs |
| [docs/devpost.md](docs/devpost.md) | the submission text |
| [docs/narration.md](docs/narration.md) | the video script |
| [docs/screens/](docs/screens/) | screenshots taken from the running service, not from a mockup |

## Layout

```
src/firstsmoke/
  geometry.py     bearings, ray crossing, uncertainty, nearest approach
  cameras.py      the network, and which cameras overlook a given bearing
  frames.py       sequences, incident bundles, video and directory inputs
  scene.py        horizon finding, usability verdicts, shake correction
  background.py   per-camera background modelling with time-of-day handling
  candidates.py   change map to measured regions
  tracks.py       our own tracker, and the growth analysis
  rejectors.py    the seven impostors and how each gives itself away
  confirm.py      the ONNX confirmation head, through cv2.dnn
  detector.py     one camera's opinion, as a weighted sum of named reasons
  agent.py        the escalation state machine
  synth.py        rendered scenes, so the tests have a known answer
  evaluate.py     the FIgLib harness
  service.py      the web service on servicekit
  live.py         the live camera path, behind a switch
```

## Licences

| | |
|---|---|
| This code | ours |
| HPWREN imagery and camera metadata | CC BY-NC-ND 4.0, http://hpwren.ucsd.edu, not redistributed here |
| `pyronear/pyro-sdis` (confirmer training data) | Apache-2.0 |
| The trained confirmer | ours, trained only on Apache-2.0 data |
| OpenCV 5.0.0.93 | Apache-2.0 |

No AGPL dependency, anywhere. In particular nothing imports `ultralytics`:
AGPL-3.0 section 13 makes a hosted demo API a source-disclosure event, and this
is a hosted demo API.
