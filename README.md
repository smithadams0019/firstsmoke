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
| [products/firstsmoke/docs/costs.md](products/firstsmoke/docs/costs.md) | what is running on AWS and what it costs |
| [products/firstsmoke/docs/devpost.md](products/firstsmoke/docs/devpost.md) | the submission text |

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

On 64 recorded sequences from 44 real HPWREN cameras: **90.5% of fires found, a
median of 60 seconds after the human who labelled them, and 487 false positives
per camera-day.** The second number is far too high to deploy, and requiring a
second camera to agree cuts it by 60% at the cost of 24 points of detection.
On recorded data the bearing crossing is refused more often than it succeeds,
and the system says so rather than guessing.

There is no field deployment trial for this system, and we could not find a
published outcome study for any comparable camera-based safety product in any
domain.

Full detail, including every failure case, is in
[the evaluation](products/firstsmoke/docs/evaluation.md).
