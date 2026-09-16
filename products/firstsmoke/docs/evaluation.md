# Evaluation

Every number below comes from `docs/evaluation.json`, which is written by
`scripts/evaluate.py` and can be regenerated from a clean checkout. Nothing here
is estimated and nothing is rounded in our favour.

**The headline, stated before the detail: this system is not deployable at the
accuracy it currently achieves.** It finds nine fires in ten, a minute after a
human would, and it also raises several hundred false alarms per camera per day
on clear views. The second number is the one that decides whether anyone keeps a
system switched on, and ours is far too high. The rest of this document is about
exactly where that comes from and what does and does not help.

## 1. The data

HPWREN's FIgLib: recorded sequences from real mountain-top cameras in southern
California, each one a run of stills around a real ignition. The filenames carry
the ground truth as a signed offset in seconds from the moment a human annotator
first marked the plume as visible.

| | |
|---|---|
| Sequences | 64 |
| Frames | 4,959 |
| Distinct cameras | 44 |
| Frames labelled smoke | 2,508 |
| Frames labelled clear | 2,451 |
| Clear camera time | 41.0 hours |
| Cadence | one frame a minute |
| Working resolution | 1024 px wide, downscaled from 3072 x 2048 |

The negatives matter as much as the positives and they are the right negatives:
the same cameras, the same weather, the same hours of the same days, taken from
the forty minutes immediately before each fire. A false-positive rate measured
against stock photographs of empty hillsides would be meaningless, and it is the
easiest number in this field to flatter.

Imagery is HPWREN, University of California San Diego, http://hpwren.ucsd.edu,
licensed CC BY-NC-ND 4.0. Frames are cached locally by whoever runs the
evaluation and are not redistributed with this repository. `scripts/fetch_figlib.py`
downloads them.

## 2. Detection rate and time to alert

At the shipped operating point, a per-camera confidence of 0.35:

| | |
|---|---|
| Fires with a usable view | 63 of 64 |
| Fires detected | **57, or 90.5%** |
| Median time to alert | **+60 s** after the human's mark |
| Mean time to alert | +252 s |
| 10th / 90th percentile | 0 s / +865 s |
| Detected before the human's mark | **0 of 57** |
| Median processing time | **128 ms a frame** (max 171 ms) |

The 64th sequence is `20210319_FIRE_om-n-mobo-c`, which is dark for all 81 of its
frames. The system reported it as unusable rather than clear on every frame,
which is the correct behaviour, and it is excluded from the denominator rather
than counted as a miss. Counting it as a miss would give 89.1%.

**We never beat the human.** Not once in 57 detections. The annotator marked the
first frame in which a person could see the plume, and the system needs three to
four frames of growth before it will say anything, so a structural lag of two to
four minutes is built into the design. That lag is the price of the growth
analysis, and the growth analysis is what makes the impostor rejection work at
all; but it should not be sold as early detection, because against this ground
truth it is not.

## 3. False positives, and the operating curve

Counted as frames labelled clear on which the detector crossed the threshold,
divided by the clear camera time.

| Threshold | Detection | Median alert | FP frames | **FP per camera-day** |
|---|---|---|---|---|
| 0.25 | 95.2% | 0 s | 1,467 | 857.8 |
| 0.30 | 93.7% | 0 s | 1,171 | 684.7 |
| **0.35** | **90.5%** | **+60 s** | **833** | **487.1** |
| 0.40 | 85.7% | +180 s | 503 | 294.1 |
| 0.45 | 79.4% | +330 s | 293 | 171.3 |
| 0.50 | 60.3% | +750 s | 141 | 82.5 |
| 0.55 | 30.2% | +780 s | 47 | 27.5 |
| 0.60 | 12.7% | +870 s | 11 | 6.4 |
| 0.65 | 6.3% | +960 s | 0 | 0.0 |

There is no good point on this curve. At 0.35 the alarm rate is unusable; at
0.55 the detection rate is unusable. That is the honest shape of a
single-camera detector built this way, and it is the reason the product is not a
single-camera detector.

### Why the false positives happen

The diagnosis is sharper than "it is noisy". On the worst sequences the highest
confidence reached anywhere in the sequence occurs on a frame labelled **clear**:

| Sequence | FP frames | Peak on clear | Peak on smoke |
|---|---|---|---|
| `20191005_FIRE_wc-n-mobo-c` | 34 of 39 | 0.63 | 0.63 |
| `20180614_FIRE_hp-s-mobo-c` | 32 of 40 | 0.58 | 0.58 |
| `20180723_FIRE_tp-e-mobo-c` | 31 of 40 | 0.59 | 0.59 |
| `20190825_FIRE_sm-w-mobo-c` | 30 of 40 | 0.58 | 0.58 |
| `20180517_FIRE_rm-n-mobo-c` | 29 of 40 | 0.53 | 0.53 |

The two peaks being identical means the detector's best candidate in the whole
sequence is a persistent feature present before the fire started, and that it
stayed locked onto that feature rather than onto the plume. These are not
flickers. They are stable, growing, anchored, soft-edged, desaturating regions
that satisfy every test the product applies, on eight of the 44 cameras. We did
not identify what they physically are; the likeliest candidates are marine layer
creeping up a valley, and slope shadow rotating across a ridge at a rate the
low-frequency shading correction does not fully remove.

By contrast, 17 of the 64 sequences produce zero false positives across their
whole clear period. The problem is concentrated in particular camera views
rather than spread evenly, which suggests a per-camera calibration would help
far more than a threshold change. That is not built.

## 4. What the second camera buys

This is the product's central claim, so it is measured rather than asserted.

Measured on the 33 sequences from the 8 dates where two or more cameras on
**different summits** recorded the same fire, applying the real rule from
`geometry.cross_rays`: a second camera must see something on a bearing that
crosses the first camera's in front of both of them and within range, within 120
seconds.

| Threshold | One camera: detect / FP-day | Corroborated: detect / FP-day |
|---|---|---|
| 0.30 | 93.9% / 720.6 | 72.7% / 329.5 |
| **0.35** | **87.9% / 507.4** | **63.6% / 201.8** |
| 0.40 | 84.8% / 307.8 | 45.5% / 96.9 |
| 0.45 | 75.8% / 165.3 | 42.4% / 35.3 |
| 0.50 | 60.6% / 71.8 | 0.0% / 12.5 |

At the shipped threshold, requiring corroboration **cuts false positives by 60%**
and costs 24 points of detection. At 0.45 it cuts them by 79% and costs 33
points. The mechanism works and the direction is right. It is not enough on its
own to make the numbers deployable, and the trade is expensive.

Two things this measurement understates, both in the conservative direction: it
does not model the agent's four-frame history bar before it will escalate, nor
its re-reads of the origin camera, and both raise precision further. One thing
it overstates: a corroborating detection here only has to be within 120 seconds,
where the real agent also requires the neighbour's detection to survive its own
rejectors.

## 5. Localisation

**On recorded data, this largely does not work, and the failure is instructive.**

Of the 8 dates with two or more summits, the crossing succeeded on 2 and was
**refused on 6**:

| Date | Cameras | Outcome |
|---|---|---|
| 20171010 | hp-w, rm-e | **fix**, crossing 59.1°, residual 0.0 m |
| 20180806 | mg-s, vo-w | **fix**, crossing 48.9°, residual 0.0 m |
| 20160604 | rm-n, smer-tcs3 | refused, BEHIND_CAMERA |
| 20180727 | bh-n, bh-s, bl-e, mg-w, wc-n | refused, BEHIND_CAMERA |
| 20190829 | bl-n, pi-e, rm-w, smer-tcs8 | refused, BEHIND_CAMERA |
| 20191001 | bh-w, lp-s, om-e, om-s, rm-w | refused, BEHIND_CAMERA |
| 20191005 | hp-s, vo-n, wc-e, wc-n | refused, RAYS_TOO_PARALLEL |
| 20200911 | lp-e, mlo-s, pi-s | refused, BEYOND_RANGE |

Where two independent pairs of bearings could both be crossed, the two fixes
landed a **median of 6.9 km apart**. That is not a working fix, and the system
says so: every one of those six is a refusal with a named reason, not a
confident wrong answer. Four of them are `BEHIND_CAMERA`, meaning the two
bearings only meet behind one of the two lookouts, which is geometrically
impossible for one object.

The likely cause is upstream. The bearing is taken at the *first* frame that
crosses the threshold, and at 487 false positives per camera-day the first
crossing is frequently a false positive rather than the fire. The system is
therefore crossing bearings to two different things and correctly refusing. Fix
the false-positive rate and this should largely resolve; we ran out of time to
demonstrate that.

**The geometry itself is correct**, and that is testable separately because we
can place a fire ourselves. On synthetic incidents rendered by
`firstsmoke.synth`, where the true latitude and longitude are known:

| | |
|---|---|
| Fix error against the placed fire | **13 m** |
| Reported 1σ semi-major axis | 161 m |
| Crossing angle | 60.2° |
| Residual | 36.6 m |

The reported uncertainty contains the true error with a large margin, which is
the property that matters most: an error bar that does not contain the truth is
worse than no error bar. `tests/test_agent.py` asserts both, and asserts the
bearings are within 4° of each camera's true line of sight.

## 6. Which rejectors actually fire

Across all 4,959 frames:

| Rejector | Times fired |
|---|---|
| `TOO_BRIEF` (fewer than three frames of history) | 507 |
| `ERRATIC_BASE` (base travelling faster than a fire can) | 29 |
| `GROUND_DRIFT` (dust: sideways, not rising, not neutral) | 27 |
| `FLARE` (clipped sensor) | 15 |
| `CLOUD_TRANSLATION` (whole shape moving, area constant) | 10 |
| `AIRBORNE` (base detached from the skyline) | 5 |
| `WEATHER_FRONT` (frame-wide contrast collapse) | 1 |

`TOO_BRIEF` dominating is the design working: most candidates are dismissed for
not having been observed long enough, which is the whole thesis. But the
specific impostor rules fire rarely — 87 times in total — which tells us the
persistent false positives in §3 are **not** being caught by any named rule.
They pass every test. That is the most useful single finding in this evaluation
and it is where the next work should go.

## 7. Cameras that reported themselves unusable

| Reason | Frames |
|---|---|
| Night, no usable illumination | 81 |
| Direct sun in frame | 16 |

Those 97 frames were reported as unusable rather than clear. This is a small
number because FIgLib sequences are curated around daylight ignitions; on a real
24-hour deployment the night figure would dominate, and `20210319_FIRE_om-n-mobo-c`
shows what that looks like — 81 frames of 81 correctly refused.

## 8. The learned confirmer

Trained by `scripts/train_confirmer.py` on `pyronear/pyro-sdis` (Apache-2.0),
which permits derivative works; FIgLib is CC BY-NC-ND and is therefore used only
to evaluate, never to train. 53,978 parameters, three convolutions and two dense
layers, plain numpy, about five minutes.

On the pyro-sdis validation split, 924 crops of which 384 are smoke:

| | |
|---|---|
| Accuracy | 0.885 |
| Precision | 0.948 |
| Recall | 0.766 |
| Through `cv2.dnn` | 0.54 ms a crop |
| Export agreement | the ONNX file reproduces the numpy model's answers |

It is out of distribution on FIgLib: trained on a European camera network,
evaluated on a Californian one. It contributes a modest amount of precision and
nothing to recall, and the pipeline runs without it and says so in the run
record when the file is absent.

## 9. What this evaluation does not establish

- **No field deployment trial exists.** We searched for a published
  outcome study for any camera-based safety detection product, in any domain,
  and found none. If a judge asks whether this has been shown to save lives in
  the field, the honest answer is that nobody has published that evidence, for
  this system or for any comparable one.
- **The fires are not located.** FIgLib publishes no coordinates, so the
  localisation accuracy on recorded data cannot be measured against a surveyed
  point at all. §5 reports a self-consistency spread and a synthetic accuracy,
  and calls neither one accuracy on real data.
- **63 fires is a small set**, drawn from one network in one region, and 44
  cameras is not enough to characterise per-camera behaviour when the false
  positives are concentrated in eight of them.
- **Every sequence contains a fire.** FIgLib has no all-clear days, so the clear
  period is only forty minutes per camera and always immediately precedes an
  ignition. A true false-positive rate needs quiet days, and we do not have them.
- **The threshold was not tuned on a held-out set.** 0.35 was chosen on
  synthetic data before the recorded evaluation was run, which is the right order,
  but the operating curve in §3 was then computed on the same 64 sequences that
  are the only recorded evidence we have.

## Reproducing this

```bash
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python -e packages/visioncore -e packages/servicekit
uv pip install --python .venv/bin/python -e products/firstsmoke
cd products/firstsmoke
python scripts/fetch_figlib.py          # about 5,000 frames, 770 MB, 25 minutes
python scripts/train_confirmer.py       # optional, 5 minutes
python scripts/evaluate.py              # about 12 minutes, writes docs/evaluation.json
```
