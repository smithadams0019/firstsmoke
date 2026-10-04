# Evaluation

Every number here comes from a JSON file written by `scripts/evaluate.py`: the
held-out test set in `docs/evaluation-test.json` (130 sequences) and the
development set in `docs/evaluation-dev-round1.json` (64). Nothing is estimated.

## 0. The held-out result

The threshold (0.45) and the calibration variant (`map`) were chosen on 64
development sequences and committed as `62063f7` **before any test sequence was
downloaded**. The test set is the 130 FIgLib sequences frozen in
`data/figlib_test_sequences.txt`. All 130 are scored.

| At the shipped point: 0.45, `map` calibration | Test set, 130 held out | Development set, where 0.45 was chosen |
|---|---|---|
| One camera: fires found | **100 of 129 — 77.5%** | 50 of 63 — 79.4% |
| One camera: false-positive frames per camera-day | **154.2** | 150.3 |
| One camera: median time to alert, after the human mark | **+240 s** | +438 s |
| Alerts ahead of the human mark | **0 of 100** | 0 |
| Triangulation: fixes on multi-summit dates | **5 of 17** | 2 of 8 |
| Triangulation: median spread between pair fixes | **6.3 km** | 7.4 km |
| Median processing time | 246.8 ms a frame | 137 ms a frame |

One-camera detection and false alarms barely moved from development to test —
79.4% at 150.3 against 77.5% at 154.2 — which is the main thing a held-out set is
for.

Two notes on the run. 33 individual frames timed out during the test-set download and
are missing from the cache, so the run scored the 10,107 frames that arrived. And
`20210302_FIRE_lp-e-mobo-c` is dark on all 81 frames and was correctly reported
unusable; it is excluded from the 129, and the headline reads 76.9% of 130 if it is
counted as a miss. An interim run over the first 94 sequences is kept as
`docs/evaluation-test-94.json`; nothing below quotes it.

537 of the 5,012 clear frames were flagged: **about one clear frame in nine**. That
is the number the rest of this document is about. Finding smoke is not what defeats
a camera network; the volume of flags is.

## 1. The second camera is not a gate

The product was built around corroboration: one camera becomes suspicious, the
geometry names the cameras that overlook the same bearing, and they are asked
whether they see it too. The original claim was that this *gates* the alarm — that
nothing is raised until a second summit agrees.

The held-out set does not support that claim. `corroboration_curve` in
`evaluation-test.json` measures the gate directly, on the 67 fires where two or
more summits recorded the same ignition, with the one-camera figures on **the same
67 fires** beside it:

| Threshold | One camera: found / FP per camera-day | Two cameras agreeing: found / FP per camera-day |
|---|---|---|
| 0.35 | 61 of 67 — 91.0% / 499.0 | 43 — 64.2% / 236.9 |
| 0.40 | 60 — 89.6% / 304.0 | 39 — 58.2% / 110.1 |
| **0.45** | 53 — 79.1% / 136.9 | **20 — 29.9% / 19.6** |
| 0.50 | 42 — 62.7% / 46.9 | 9 — 13.4% / 0.0 |
| **0.55** | **29 — 43.3% / 16.2** | 6 — 9.0% / 0.0 |

Read the two bold rows against each other. Requiring a second camera to agree, at
the shipped threshold, finds **29.9% of those fires at 19.6 false-positive frames
a camera-day**. Raising one camera's own bar to 0.55 instead finds **43.3% at
16.2** — half again as many fires, for fewer false alarms, from one camera. The
gate is dominated on both axes. It does not move the detection-versus-false-alarm
curve; it only moves the operating point along it, and further than you want.

The same reading on the full 130-sequence set: one camera at 0.55 finds 48 of 129
(37.2%) at 19.2 false-positive frames a camera-day.

**So corroboration is not a gate here, and the code never used it as one.**
`Lookout._resolve_without_neighbours` raises an alert from a single camera above
`CONFIRM_AT`, and `_stand_down_or_escalate` sends a confident uncorroborated
detection to `NEEDS_HUMAN` rather than dismissing it. What the second camera earns is
the order of the queue:

- a crossed bearing arrives with a position and an uncertainty ellipse, and sorts
  first;
- a confident camera with nobody to cross against goes to a person marked
  `NO_SECOND_VIEW`, carrying no position;
- a crossing that cannot be trusted is refused by name, with the nearest approach
  of the rays;
- a camera that cannot see reports which fault it has, so its silence is never
  read as an empty hillside;
- only a weak detection that a neighbour actively contradicted is logged as a
  stand-down, and a stand-down returns the camera to watching rather than ending
  the run.

That is triage, and it is what the agent loop is for. It is not a filter on the
alarm, and this document no longer reports it as one.

**One caveat, in our disfavour.** The 0.55 reading above is a measurement on the
test set, so it is a finding and not a new frozen choice. The shipped threshold
stays at 0.45, where it was fixed on development data, and every held-out figure
in section 0 is reported at it.

## 2. The data

HPWREN's FIgLib: recorded sequences from real mountain-top cameras in southern
California, each a run of stills around a real ignition. The filenames carry the
ground truth as a signed offset in seconds from the moment a human annotator first
marked the plume as visible.

| | |
|---|---|
| Development sequences | 64 |
| Frames | 4,959 |
| Distinct cameras | 44 |
| Frames labelled smoke / clear | 2,508 / 2,451 |
| Clear camera time | 41.0 hours |
| Cadence | one frame a minute |
| Working resolution | 1024 px wide, downscaled from 3072 x 2048 |
| Held-out test sequences | 130, frozen; all scored (10,107 frames) |

The negatives are the right negatives: the same cameras, the same weather, the same
hours of the same days, taken from the forty minutes immediately before each fire.

Imagery is HPWREN, University of California San Diego, http://hpwren.ucsd.edu,
licensed CC BY-NC-ND 4.0. Frames are cached locally by whoever runs the evaluation
and are not redistributed with this repository.

## 3. How the shipped configuration was chosen

On the development set only. It is recorded because it is how the frozen choice was
made.

| | Before: 0.35, no calibration | After: 0.45, `map` calibration |
|---|---|---|
| One camera: fires detected | 57 of 63 — 90.5% | 50 of 63 — 79.4% |
| One camera: false positives per camera-day | 485 | 150 |
| One camera: median time to alert | +60 s | +438 s |
| Median processing time | 137 ms a frame | 137 ms a frame |

The 64th sequence, `20210319_FIRE_om-n-mobo-c`, is dark for all 81 frames; the
system reported it unusable on every one, which is correct, and it is excluded from
the denominator.

### 3.1 Per-camera calibration

The false positives in the first evaluation were concentrated: on eight of 44 cameras
the highest score in the whole sequence sat on a frame labelled clear. That points at
persistent features of particular views, not at noise, so the fix is learnt from each
camera's own clear frames. `calibration.py` keeps two things per camera: **a nuisance
map**, a 24 by 32 grid of how often each cell held a candidate scoring 0.25 or more on
a clear frame, rejecting habitual candidates as `HABITUAL_REGION`; and **a confidence
ceiling**, the 95th percentile of that camera's best clear-frame score, rejecting
anything below it as `BELOW_CAMERA_BASELINE`.

A camera needs 25 clear frames to be calibrated at all, and a sequence is never scored
with a calibration that saw it — each camera's calibration is pooled from its clear
frames **on other dates only**. On the 64 development sequences that left 30 sequences
on 16 cameras, a median of 36 clear frames each.

### 3.2 Map or ceiling, decided on the curves

Three variants were run — the map alone, the ceiling alone, and both — swept across
all 64 development sequences. Each cell is fires detected / false positives per
camera-day:

| Threshold | Uncalibrated | map | ceiling | map + ceiling |
|---|---|---|---|---|
| 0.35 | 90.5% / 485.3 | 90.5% / 464.9 | 81.0% / 346.2 | 81.0% / 340.9 |
| 0.40 | 85.7% / 293.0 | 85.7% / 271.3 | 76.2% / 218.7 | 76.2% / 213.4 |
| 0.45 | 79.4% / 171.3 | 79.4% / 150.3 | 73.0% / 132.7 | 73.0% / 128.7 |
| 0.50 | 60.3% / 82.5 | 60.3% / 70.8 | 55.6% / 71.3 | 55.6% / 67.3 |
| 0.55 | 30.2% / 27.5 | 30.2% / 21.6 | 30.2% / 24.0 | 30.2% / 19.9 |

**The ceiling does not move the curve.** map + ceiling at 0.35 gives 81.0% at 340.9;
the uncalibrated detector at 0.40 already gives 85.7% at 293.0 — more detection for
fewer false alarms, and the same holds up to 0.50. A ceiling is a per-camera
threshold increase, so it buys nothing a higher global threshold would not. Its six
extra misses are lost outright rather than delayed: swept down to 0.25 the gap stays
at six.

**The map does move it, a little.** It detects exactly the same fires as the
uncalibrated detector at every threshold and raises fewer false positives at every
threshold: 4% fewer at 0.35, 12% at 0.45, 21% at 0.55. The gain is small because
only 16 of 44 cameras had another date to learn from.

So the map ships and the ceiling does not. **On the test set the map made no
measurable difference**: 154.2 false positives per camera-day on all 130 with or
without it, and the same 100 fires found either way, on the 110 of 130 sequences
that had a calibration available. The small development gain did not reproduce.

### 3.3 The threshold

0.45 was taken as the knee of the development **corroborated** curve, where the
corroborated false-positive rate fell from 71.8 to 21.7 for the cost of one fire in
33, and corroborated detection reached zero by 0.50. Section 1 is what the held-out
set then said about reading a threshold off that curve. The variant and the threshold
were written into `SHIPPED_VARIANT`, `SHIPPED_THRESHOLD` and `SUSPECT_AT`, and
committed as `62063f7` before a single test sequence had been downloaded.

## 4. What "+240 seconds" means

**The comparison is against a hindsight annotator, not a live watcher.** FIgLib's
offset zero is the first frame in which an expert, reviewing the whole recorded
sequence afterwards and knowing where the plume would appear, could see it. A person
watching a live wall of cameras has no such advantage, so +240 s is a lag behind the
best a human could possibly do on these frames, not behind a human on shift. It
still means the system was **never earlier** than that mark, on any of the 100 fires
it found, at any threshold.

**Published detectors on the same library are faster.** On FIgLib, SmokeyNet reports
a mean time to detection of 3.12 minutes (Dewangan et al., *Remote Sensing*
14(4):1007, 2022, arXiv:2112.08598), a re-run of SmokeyNet 4.70 ± 0.90 minutes and
3.66 with weather data (arXiv:2212.14143), and ContrastSwin 2.26 minutes
(arXiv:2311.10116). Those are means on their own splits and operating points, so the
comparison is loose. Our mean on the test set is 8.8 minutes (528.2 s), slower than
every figure above.

**There is no sourced live comparator in minutes.** ALERTCalifornia says its cameras
beat 911 calls "over 30% of the time" (alertcalifornia.org), which gives no minutes,
and we found no other figure we could verify. So no claim is made about whether this
system would be earlier or later than people in the field.

## 5. Localisation

**Triangulation does not work on real data yet.** On the test set the crossing gave a
fix on 5 of 17 multi-summit dates; the other 12 were refused — 8 with rays too
parallel, 3 beyond range, 1 behind the camera. Where two independent camera pairs
could both be crossed, their fixes landed a median **6.3 km** apart (7.4 km on
development). A position that uncertain is not somewhere anyone could send a crew.

The refusals are the right failure: named, with a reason and a nearest-approach
measurement, instead of a confident wrong point. The likely cause is upstream — the
bearing is taken from the first threshold crossing, and at these false-positive rates
that is often not the fire. FIgLib publishes no coordinates, so accuracy against a
surveyed point cannot be measured at all.

**The geometry itself is correct** where it can be checked. On rendered incidents
where we placed the fire, the fix lands **13 m** from the truth inside a reported
161 m 1σ ellipse, and `tests/test_agent.py` asserts it.

## 6. Which rejectors fire

Across the 10,107 test frames, with the map calibration, at 0.45:

| Rejector | Times fired |
|---|---|
| `TOO_BRIEF` (fewer than three frames of history) | 1,024 |
| `ERRATIC_BASE` (base travelling faster than a fire can) | 70 |
| `GROUND_DRIFT` (dust: sideways, not rising, not neutral) | 57 |
| `FLARE` (clipped sensor) | 41 |
| `CLOUD_TRANSLATION` (whole shape moving, area constant) | 23 |
| `AIRBORNE` (base detached from the skyline) | 12 |
| `WEATHER_FRONT` (frame-wide contrast collapse) | 2 |
| `HABITUAL_REGION` (on this camera's learnt nuisance map) | 1 |

`TOO_BRIEF` dominating is the design working: most candidates are dismissed for not
having been observed long enough, which is the whole thesis. The hand-written
impostor rules fire rarely — 205 times across 10,107 frames — and the learnt nuisance
map fired once, which is consistent with it making no measurable difference on test.
The persistent false positives pass every named rule, and we did not identify what
they physically are.

## 7. Cameras that reported themselves unusable

81 frames as night with no usable illumination, 44 as direct sun in frame: 125 frames
reported unusable rather than clear. The number is small because FIgLib is curated
around daylight ignitions. On a real 24-hour deployment night would dominate, and
`20210302_FIRE_lp-e-mobo-c` shows what that looks like — 81 frames of 81 refused.

## 8. The learned confirmer

Trained by `scripts/train_confirmer.py` on `pyronear/pyro-sdis` (Apache-2.0, which
permits derivative works); FIgLib is CC BY-NC-ND and is therefore used only to
evaluate, never to train. 53,978 parameters, three convolutions and two dense
layers, plain numpy, about five minutes.

On the pyro-sdis validation split, 924 crops of which 384 are smoke: accuracy 0.885,
precision 0.948, recall 0.766, 0.54 ms a crop through `cv2.dnn`. The training script
verifies its own export by re-running the split through `cv2.dnn` and checking the
answers match. It is out of distribution on FIgLib — a European camera network
evaluated on a Californian one — contributes a modest amount of precision and
nothing to recall, and the pipeline runs without it and says so in the run record
when the file is absent.

## 9. Footage from outside the network

The service also accepts a video or stills from any single camera. None of that
footage has a surveyed camera, so none of it can be corroborated or located, and
every flag it raises carries `NO_SECOND_VIEW`. At the shipped threshold:

| Clip | What it contains | What Firstsmoke did |
|---|---|---|
| `waldo-canyon-clear-to-onset.mp4` (Steve Moraco, CC BY 3.0), 21.6 s a frame | clear sky, then a smoke column from about 0:20 | 240 of 1,200 frames read; 27 flags. Nine, from 0:03 to 0:19, came before any smoke was in frame. The highest score, 0.75 at 0:43, is on the smoke column |
| `grand-canyon-clouds-timelapse.mp4` (NPS, CC BY 4.0), 1 s a frame assumed | cloud over the canyon, no smoke | 42 of 1,108 frames read; 5 flags, all false, highest 0.51 |
| `north-derby-gulch-rx-timelapse.mp4` (USDA Forest Service, public domain) | a faint wisp of smoke on a far ridge during a prescribed burn | missed. The highest box, 0.58, sat on a wind-blown bush in the foreground; the wisp itself scored 0.37 to 0.43, under the threshold (`media/real/firstsmoke/PROVENANCE.md` §3) |

The Waldo and Grand Canyon rows are the deployed service's own results, on image
`fs-21772f1`. A local run of the Waldo clip gave 22 flags with the same peak, so small
decoder or CPU differences move the weakest flags either side of 0.45. All three are
what the held-out numbers predict: it flags cloud, and it does find a real column.

## 10. What this evaluation does not establish

- **No field deployment trial exists.** We searched for a published outcome study
  for any camera-based safety detection product, in any domain, and found none. If a
  judge asks whether this has been shown to save lives in the field, the honest
  answer is that nobody has published that evidence, for this system or for any
  comparable one.
- **The fires are not located.** FIgLib publishes no coordinates, so localisation
  accuracy on real data cannot be measured against a surveyed point. Section 5
  reports a self-consistency spread and a synthetic accuracy, and calls neither one
  accuracy on real data.
- **63 development fires and 130 test fires is a small set**, from one network in
  one region, and 44 cameras is not enough to characterise per-camera behaviour when
  the false positives are concentrated in eight of them.
- **Every sequence contains a fire.** FIgLib has no all-clear days, so the clear
  period is only forty minutes per camera and always immediately precedes an
  ignition. A true false-positive rate needs quiet days, and we do not have them.
- **The threshold and calibration were chosen on development data.** Sections 3 to 8
  are on that set. Sections 0 and 1 are on the 130 held-out sequences.
- **Only 16 of 44 cameras had a second date** to learn a calibration from, so the
  calibration's effect was measured on 30 sequences.

## Reproducing this

```bash
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python -e packages/visioncore -e packages/servicekit
uv pip install --python .venv/bin/python -e products/firstsmoke
cd products/firstsmoke
python scripts/fetch_figlib.py          # about 5,000 frames, 770 MB, 25 minutes
python scripts/train_confirmer.py       # optional, 5 minutes
python scripts/evaluate.py              # about 40 minutes: baseline and three calibration variants
python scripts/evaluate.py --test data/figlib_test_sequences.txt   # the frozen configuration, once
```
