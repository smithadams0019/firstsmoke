# Firstsmoke

A lookout for mountain-top camera networks. It finds the first smoke column of a
wildfire, corroborates it against the cameras that overlook the same bearing, and
ranks what it finds so the person on the desk opens the right flag first.

**Live: https://afynmqkk9e.eu-west-1.awsapprunner.com** (AWS App Runner, eu-west-1)
**Repository: https://github.com/smithadams0019/firstsmoke**

OpenCV 5.0.0, pinned.

---

## The result

On the 130-sequence held-out FIgLib test set, with the threshold and calibration frozen
before a single test sequence was downloaded, at the shipped operating point of 0.45:

| At 0.45, map calibration | Test set, 130 held out | Development set, 64 |
|---|---|---|
| One camera: fires found | **100 of 129 — 77.5%** | 50 of 63 — 79.4% |
| One camera: false-positive frames per camera-day | **154.2** | 150.3 |
| One camera: median alert after the human mark | **+240 s** | +438 s |
| Alerts ahead of the human mark | **0 of 100** | 0 |
| Triangulation: fixes / median spread between pair fixes | **5 of 17 / 6.3 km** | 2 of 8 / 7.4 km |

154.2 false-positive frames a camera-day is about **one clear frame in nine**. That is
the number this product is organised around. Finding smoke is not what defeats a camera
network; a duty officer handed a hundred flags an hour, with nothing to say which to
open, is.

**The second camera is not a gate, and we measured why.** On the 67 held-out fires that
two or more summits recorded, requiring two cameras to agree finds 29.9% of them at
19.6 false-positive frames a camera-day; raising one camera's own bar to 0.55 instead
finds 43.3% at 16.2 — more fires for fewer false alarms, from one camera. So
corroboration orders the queue and attaches a position, rather than deciding whether
anything is raised at all. `docs/evaluation.md` §1 has the full curve.

Two more things the test set settled: the per-camera calibration made no measurable
difference (154.2 false alarms a camera-day with or without it, and the same 100 fires
found), and triangulation on real data lands kilometres apart, though on rendered
incidents where we placed the fire the fix lands 13 m from the truth.

Full numbers and every failure case: **[docs/evaluation.md](docs/evaluation.md)**.

## What it is

Most wildfire camera systems are classifiers: a frame goes in, a probability comes out.
That design cannot separate a smoke column from a cloud, because in a single frame they
look the same. Both are grey, both are soft-edged, both desaturate what is behind them.

Firstsmoke behaves like a lookout instead. It watches how a shape *behaves* over
minutes, and when it is unsure it goes and looks somewhere else:

1. **Detect the change.** A background model per camera, with the sun's movement fitted
   out rather than learned away, and candidate regions grown by hysteresis so a faint
   plume top stays attached to its dense base.
2. **Turn a pixel into a bearing.** A weak detection's *pixel column* becomes a compass
   bearing. The bearing decides which other cameras overlook that patch of ground, and
   to within a few degrees where in each of their frames to look. Different pixels
   select different cameras. This is the loop.
3. **Rank the queue.** Agreeing bearings are crossed on the map, with an uncertainty
   ellipse from each camera's own angular error, and that flag sorts first with a
   position. A confident camera with nobody to cross against still reaches a person,
   marked `NO_SECOND_VIEW` and carrying no position.
4. **Report.** A timestamped flag carrying the frames, the bearings, which cameras were
   consulted, what each answered, and the measurement behind every step.

It also says plainly when a camera cannot be believed: night without illumination, fog,
rain on the lens, direct sun, or a feed that has frozen. A blind camera's silence is
never counted as evidence of an empty hillside.

On real footage through the upload page, a Waldo Canyon time-lapse raised 27 flags, 9 of
them on cloud before any smoke appeared, with the highest score on the smoke column. A
Grand Canyon cloud time-lapse raised 5 flags with no fire present. A faint
prescribed-burn wisp in North Derby Gulch was missed.

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
uv pip install --python .venv/bin/python -e "products/firstsmoke[dev]"

# the four bundled incidents, rendered once (about 100 seconds)
.venv/bin/python -c "from pathlib import Path; from firstsmoke.scenarios import build_all; \
    build_all(Path('products/firstsmoke/data/scenarios'))"

FIRSTSMOKE_SCENARIOS=$PWD/products/firstsmoke/data/scenarios \
  .venv/bin/python -m uvicorn firstsmoke.service:app --port 8099
```

Open http://127.0.0.1:8099 and press one of the four incidents, or upload a
video or stills from one camera under "Run your own footage". An upload has no
surveyed camera, so it runs on one camera, keeps watching after each flag, and
reports every flag with `NO_SECOND_VIEW` and no location. A long clip is sampled
to about one frame a minute of real time and the result says "analysed X of Y".

### Test

```bash
# if you installed without [dev]: uv pip install --python .venv/bin/python pytest==9.1.1 ruff==0.16.7
.venv/bin/python -m pytest products/firstsmoke/tests -q          # 196 tests
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
python scripts/evaluate.py         # ~40 min, baseline and calibration variants
python scripts/fetch_figlib.py $(cat data/figlib_test_sequences.txt)   # the frozen test set
python scripts/evaluate.py --test data/figlib_test_sequences.txt
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
| [docs/devpost.md](docs/devpost.md) | the submission text |
| [docs/narration.md](docs/narration.md) | the video script, in three acts |
| [docs/screens/](docs/screens/) | screenshots taken from the running service, not from a mockup |

## Layout

```
src/firstsmoke/
  geometry.py     bearings, ray crossing, uncertainty, nearest approach
  cameras.py      the network, and which cameras overlook a given bearing
  frames.py       sequences, incident bundles, video and directory inputs
  uploads.py      one-camera footage, decoded on demand
  calibration.py  per-camera nuisance maps from clear frames on other dates
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
