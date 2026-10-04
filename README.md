# Firstsmoke

A lookout for mountain-top camera networks. It finds the first smoke column of a
wildfire, asks the neighbouring cameras when it is unsure, and crosses their
bearings on a map.

**Live: https://afynmqkk9e.eu-west-1.awsapprunner.com**

Entry for the OpenCV AI Competition 2026, targeting **Best Agentic Vision**.
OpenCV 5.0.0, pinned.

---

## Start here

| | |
|---|---|
| **[products/firstsmoke/README.md](products/firstsmoke/README.md)** | what it is, how to run it, how to test it, how to deploy it |
| [products/firstsmoke/docs/report.md](products/firstsmoke/docs/report.md) | the technical report |
| [products/firstsmoke/docs/evaluation.md](products/firstsmoke/docs/evaluation.md) | the numbers, and the failure cases |
| [products/firstsmoke/docs/architecture.md](products/firstsmoke/docs/architecture.md) | diagrams of the pipeline, the agent loop and the AWS components |

## Why the repository has this shape

`products/firstsmoke` is the entry. `packages/visioncore` and
`packages/servicekit` are two small shared libraries — an OpenCV 5 primitives
layer that refuses to import on a 4.x wheel, and a FastAPI service shell — and
`infra/` holds the deployment scripts. They are included here so the repository
is self-contained and every command in the documentation works verbatim from a
clean clone.

`docs/` carries the design atlas and this product's design spec, plus the two
research documents the evidence and the technology choices came from.

## Thirty seconds

```bash
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python -e packages/visioncore -e packages/servicekit
uv pip install --python .venv/bin/python -e products/firstsmoke
.venv/bin/python -m pytest products/firstsmoke/tests -q
```

## The honest summary

On the 130-sequence held-out FIgLib test set, configuration frozen before download, at
the shipped operating point of 0.45: **one camera finds 100 of 129 fires — 77.5% — at
154.2 false-positive frames per camera-day, a median of 240 s after the human who
labelled them, and never earlier.** On the 64 development sequences where the threshold
was chosen: 79.4% at 150.3, +438 s. 154.2 frames a camera-day is about one clear frame
in nine, and that volume, not detection, is what the design after the detector is for.

**The second camera is not a gate, and we measured why.** On the 67 held-out fires that
two or more summits recorded, requiring both to agree finds 29.9% of them at 19.6
false-positive frames a camera-day, while raising one camera's own bar to 0.55 finds
43.3% at 16.2 — more fires for fewer false alarms, from one camera. So corroboration
orders the queue and attaches a position rather than deciding whether anything is
raised at all.

Calibration made no measurable difference on test (154.2 false alarms a camera-day with
or without it, and the same 100 fires found), and triangulated fixes from independent
pairs land a median 6.3 km apart on real cameras.

There is no field deployment trial for this system, and we could not find a
published outcome study for any comparable camera-based safety product in any
domain.

Full detail, including every failure case, is in
[the evaluation](products/firstsmoke/docs/evaluation.md).
