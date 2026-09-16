# Evaluation

Every number below comes from a JSON file written by `scripts/evaluate.py`: the
held-out test set from `docs/evaluation-test-94.json`, the development set from
`docs/evaluation-dev-round1.json`. Nothing is estimated or rounded up.

**This system is not deployable at these false-alarm rates.**

## 0. The headline: the held-out test set

The threshold (0.45) and the calibration variant (`map`) were chosen on 64
development sequences and committed as `62063f7` before any test sequence was
downloaded. The test set is 130 FIgLib sequences frozen in
`data/figlib_test_sequences.txt`. **94 of the 130 are scored so far.** A full
130-sequence run is in progress, and these figures will be replaced when it
finishes.

| At the shipped point (0.45, map calibration) | Test set, held out (94 sequences) | Development set, where the threshold was chosen (64) |
|---|---|---|
| One camera: fires found | **78.7%** (74 of 94) | 79.4% (50 of 63) |
| One camera: false positives per camera-day | **151.6** | 150.3 |
| One camera: median time to alert, after the human mark | **+210 s** | +438 s |
| Two cameras agreeing: fires found | **28.6%** (14 of 49) | 42.4% (of 33) |
| Two cameras agreeing: false positives per camera-day | **18.4** | 21.7 |
| Alerts ahead of the human mark | **0** | 0 |
| Triangulation: fixes / multi-summit dates | **4 of 13** | 2 of 8 |
| Triangulation: median spread between pair fixes | **3.8 km** | 7.4 km |
| Median processing time | 174.8 ms a frame | 137 ms a frame |

What the test set says, plainly:

- **One camera held up; two cameras did not.** One-camera detection and false
  alarms barely moved from dev to test. Two-camera detection fell from 42.4% on
  dev to 28.6% on test. The threshold was picked at the knee of the dev
  two-camera curve, and that knee did not carry over.
- **Calibration made no measurable difference on test.** Without calibration the
  one-camera rate is 151.2 false positives per camera-day; with the `map`
  calibration it is 151.6. The same 74 fires are found either way. 78 of the 94
  sequences had a calibration available. The small gain on dev (171.3 to 150.3 at
  0.45) did not reproduce.
- **It never alerted ahead of the human mark,** on any of the 74 fires it found.
  The test mean is +467 s; the 90th percentile is +1,320 s.
- **Triangulation lands kilometres apart.** Where two independent camera pairs
  could both be crossed, their fixes were a median 3.8 km apart. Nine of the 13
  multi-summit dates were refused (6 with rays too parallel, 2 beyond range, 1
  behind the camera).
- **It is not deployable.** 151.6 false alarms a camera-day on one camera, or
  28.6% of fires found with a second camera, is far from anything a lookout
  service would switch on.

The processing time on test ran on a busier machine than dev; the uncalibrated
test pass on the same frames measured 104.2 ms a frame.

Everything from section 2 on is the **development set**, where the choices were
made. It is kept because it records how the shipped configuration was chosen.

## 1. The data

HPWREN's FIgLib: recorded sequences from real mountain-top cameras in southern
California, each one a run of stills around a real ignition. The filenames carry
the ground truth as a signed offset in seconds from the moment a human annotator
first marked the plume as visible.

| | |
|---|---|
| Development sequences | 64 |
| Frames | 4,959 |
| Distinct cameras | 44 |
| Frames labelled smoke | 2,508 |
| Frames labelled clear | 2,451 |
| Clear camera time | 41.0 hours |
| Cadence | one frame a minute |
| Working resolution | 1024 px wide, downscaled from 3072 x 2048 |
| Held-out test sequences | 130, frozen; 94 scored (7,272 frames) |

The negatives are the right negatives: the same cameras, the same weather, the
same hours of the same days, taken from the forty minutes immediately before
each fire.

Imagery is HPWREN, University of California San Diego, http://hpwren.ucsd.edu,
licensed CC BY-NC-ND 4.0. Frames are cached locally by whoever runs the
evaluation and are not redistributed with this repository.

## 2. Before and after, development set

| | Before: 0.35, no calibration | After: 0.45, map calibration |
|---|---|---|
| One camera: fires detected | 57 of 63, **90.5%** | 50 of 63, **79.4%** |
| One camera: false positives per camera-day | **485** | **150** |
| One camera: median time to alert | +60 s | **+438 s** |
| Corroborated (33 multi-summit sequences): detected | 63.6% | **42.4%** |
| Corroborated: false positives per camera-day | 199.5 | **21.7** |
| Detected before the human's mark | 0 | 0 |
| Median processing time | 137 ms a frame | 137 ms a frame |

The 64th sequence, `20210319_FIRE_om-n-mobo-c`, is dark for all 81 frames; the
system reported it unusable on every frame, which is correct, and it is excluded
from the denominator.

## 3. How the shipped configuration was chosen, on the development set only

### 3.1 Per-camera calibration

The false positives in the first evaluation were concentrated: on eight of 44
cameras the highest score in the whole sequence sat on a frame labelled clear.
That points at persistent features of particular views, not at noise, so the fix
is learnt from each camera's own clear frames.

For every camera, `calibration.py` keeps two things from frames labelled clear:

- **a nuisance map**, a 24 by 32 grid of how often each cell was covered by a
  candidate scoring 0.25 or more on a clear frame. A new candidate that sits
  mostly on habitual cells (habituation 0.35 or more) is rejected as
  `HABITUAL_REGION`, with a confidence that rises the more habitual it is;
- **a confidence ceiling**, the 95th percentile of the camera's best score on
  clear frames. A candidate at or below it is rejected as `BELOW_CAMERA_BASELINE`.

A camera needs at least 25 clear frames before it is calibrated at all.

**Which frames trained it.** A sequence is never scored with a calibration that
saw it. Each camera's calibration is pooled from that camera's clear frames on
*other dates only*, and a camera that appears on one date gets no calibration and
is reported as uncalibrated. On the 64 development sequences that left 30
sequences on 16 cameras with a held-out calibration, a median of 36 clear frames
each.

### 3.2 The ablation, and the rule we declared and then did not trust

Three variants were run: the map alone, the ceiling alone, and both. Before
running them we wrote down the selection rule: at the then-shipped threshold of
0.35, on the 30 calibrated sequences, pick the variant with the fewest false
positives per camera-day among those losing no more than two of the fires the
uncalibrated detector found.

| Variant | Sequences | Fires found before | After | Lost | FP/camera-day before | After |
|---|---|---|---|---|---|---|
| map | 30 | 25 | 25 | 0 | 456.6 | 412.7 |
| ceiling | 30 | 25 | 19 | **6** | 456.6 | 158.0 |
| map + ceiling | 30 | 25 | 19 | **6** | 456.6 | 146.8 |

By the declared rule the map wins. But the rule compares variants at a single
threshold, and that flatters anything that simply raises a camera's bar: a
ceiling is a per-camera threshold increase, so of course it cuts false positives
at a fixed global threshold. The fair question is whether a variant moves the
detection-versus-false-positive curve, or only moves the operating point along
it. So the choice was made on the curves, on all 64 sequences:

| Threshold | Uncalibrated | map | ceiling | map + ceiling |
|---|---|---|---|---|
| 0.25 | 95.2% / 856.7 | 95.2% / 854.9 | 85.7% / 576.0 | 87.3% / 591.8 |
| 0.30 | 93.7% / 682.4 | 93.7% / 667.8 | 85.7% / 481.8 | 85.7% / 482.4 |
| 0.35 | 90.5% / 485.3 | 90.5% / 464.9 | 81.0% / 346.2 | 81.0% / 340.9 |
| 0.40 | 85.7% / 293.0 | 85.7% / 271.3 | 76.2% / 218.7 | 76.2% / 213.4 |
| 0.45 | 79.4% / 171.3 | 79.4% / 150.3 | 73.0% / 132.7 | 73.0% / 128.7 |
| 0.50 | 60.3% / 82.5 | 60.3% / 70.8 | 55.6% / 71.3 | 55.6% / 67.3 |
| 0.55 | 30.2% / 27.5 | 30.2% / 21.6 | 30.2% / 24.0 | 30.2% / 19.9 |

Each cell is fires detected / false positives per camera-day.

- **The ceiling does not move the curve.** map + ceiling at 0.35 gives 81.0% at
  340.9; the uncalibrated detector at 0.40 already gives 85.7% at 293.0, more
  detection for fewer false alarms. The same holds at every threshold from 0.25
  to 0.50. Only at 0.55, where every variant finds 30% of fires, does the ceiling
  save a few false positives at equal detection, and nobody would run there.
- **The map does move it, a little.** It detects exactly the same fires as the
  uncalibrated detector at every threshold and raises fewer false positives at
  every threshold: 4% fewer at 0.35, 12% at 0.45, 21% at 0.55. The gain is
  small because only 16 of 44 cameras had another date to learn from.

So the map is shipped and the ceiling is not.

### 3.3 The ceiling's six extra misses: never, not late

All six fires the ceiling lost are lost outright, not delayed. Swept all the way
down to 0.25, the ceiling variant finds 54 fires against the uncalibrated
detector's 60: the gap stays at six at every threshold on the curve.

| Sequence | Uncalibrated, at 0.35 | Ceiling, at 0.35 |
|---|---|---|
| `20160604_FIRE_rm-n-mobo-c` | alert at +900 s, peak 0.51 | never; peak 0.35 |
| `20170807_FIRE_bh-n-mobo-c` | alert at +120 s, peak 0.38 | never; peak 0.30 |
| `20180919_FIRE_rm-e-mobo-c` | alert at +1320 s, peak 0.37 | never; peak 0.33 |
| `20190829_FIRE_pi-e-mobo-c` | alert at +1621 s, peak 0.45 | never; peak 0.16 |
| `20191005_FIRE_hp-s-mobo-c` | alert at +240 s, peak 0.57 | never; the 0.57 peak is on a clear frame |
| `20191005_FIRE_wc-n-mobo-c` | alert at 0 s, peak 0.63 | never; the 0.63 peak is on a clear frame |

The last one needs a caveat in our disfavour. That sequence's uncalibrated peak
was on a clear frame as well (0.63 on clear and on smoke frames alike), so its "alert at 0 s" was probably
the nuisance feature rather than the plume, and the map alone still finds it, at
+480 s with 2 false-positive frames instead of 34.

The mechanism is plain. A camera whose clear afternoons regularly score 0.45
somewhere in frame has a ceiling near 0.45, and a real plume that scores 0.45 on
that camera is then indistinguishable from its ordinary afternoon. The ceiling
trades those fires for fewer alarms, which a lower-threshold curve already
offers. The median alert with the ceiling also moved from +60 s to +120 s.

### 3.4 The threshold, chosen on the corroborated curve

The product does not alert on one camera: it consults. So the threshold was
chosen on the curve that includes the second camera, on the 33 development
sequences from the 8 dates where two or more summits saw the same fire, with the
map calibration:

| Threshold | One camera: detect / FP-day | Corroborated: detect / FP-day |
|---|---|---|
| 0.25 | 97.0% / 872.2 | 72.7% / 449.2 |
| 0.30 | 93.9% / 687.5 | 72.7% / 296.4 |
| 0.35 | 87.9% / 465.2 | 63.6% / 157.3 |
| 0.40 | 84.9% / 264.5 | 45.5% / 71.8 |
| **0.45** | **75.8% / 124.3** | **42.4% / 21.7** |
| 0.50 | 60.6% / 49.0 | 0.0% / 12.5 |

From 0.35 to 0.40 the corroborated false-positive rate halves and costs six
fires in 33. From 0.40 to 0.45 it falls by a further 70%, from 71.8 to 21.7, and
costs one. From 0.45 to 0.50 corroborated detection goes to zero. 0.45 is the
knee, so 0.45 is shipped, and it was 0.35 before only because 0.35 had been
picked on rendered data before any recorded evaluation existed.

The calibration variant and the threshold were then written into the code
(`SHIPPED_VARIANT`, `SHIPPED_THRESHOLD`, `SUSPECT_AT`) and committed as
`62063f7`, before a single test sequence had been downloaded.

## 4. What "+210 seconds" (test) and "+438 seconds" (development) mean

**The comparison is against a hindsight annotator, not a live watcher.** FIgLib's
offset zero is the first frame in which an expert, reviewing the whole recorded
sequence afterwards, could see the plume, knowing where it would appear. A person
watching a live wall of cameras does not have that advantage, so "+438 s after the
mark" is a lag behind the best a human could possibly do on these frames, not a
lag behind a human on shift. It still means the system was **never earlier** than
that mark, on any fire, at any threshold.

**Published detectors on the same library are faster.** On FIgLib, SmokeyNet
reports a mean time to detection of 3.12 minutes (Dewangan et al., *Remote
Sensing* 14(4):1007, 2022, arXiv:2112.08598), a re-run of SmokeyNet 4.70 ± 0.90
minutes and 3.66 with weather data (arXiv:2212.14143), and ContrastSwin 2.26
minutes (arXiv:2311.10116). Those are means on their own splits and operating
points, so the comparison is loose. Our own mean on the test set is 7.8 minutes
(466.7 s), and on the development set the median alone is 7.3 minutes. Both are
slower than every published figure above.

**There is no sourced live comparator in minutes.** We looked for a published
figure for how long after ignition a fire is first reported by a 911 call or by
people watching cameras, and did not find one we could verify. ALERTCalifornia
says its cameras beat 911 calls "over 30% of the time" (alertcalifornia.org),
which gives no minutes. We do not put a number on how much earlier or later this
system would be than people in the field, because we cannot source one.

## 5. Localisation on recorded data: not useful yet (development set; test set in section 0)

Plainly: **triangulation does not work on real data yet.** Of the 8 dates with
two or more summits, the crossing succeeded on 2 and was refused on 6. Where two
independent pairs could both be crossed, the fixes landed a median of **7.4 km
apart** with the map calibration (6.9 km without). A position that uncertain is
not a location anyone could send a crew to.

| Date | Cameras | Outcome |
|---|---|---|
| 20171010 | hp-w, rm-e | fix, crossing 59.1° |
| 20180806 | mg-s, vo-w | fix, crossing 48.9° |
| 20160604 | rm-n, smer-tcs3 | refused, BEHIND_CAMERA |
| 20180727 | bh-n, bh-s, bl-e, mg-w, wc-n | refused, BEHIND_CAMERA |
| 20190829 | bl-n, pi-e, rm-w, smer-tcs8 | refused, BEHIND_CAMERA |
| 20191001 | bh-w, lp-s, om-e, om-s, rm-w | refused, BEHIND_CAMERA |
| 20191005 | hp-s, vo-n, wc-e, wc-n | refused, RAYS_TOO_PARALLEL |
| 20200911 | lp-e, mlo-s, pi-s | refused, BEYOND_RANGE |

The refusals are the right failure: named, with a reason, instead of a confident
wrong point. The likely cause is upstream: the bearing is taken from the first
threshold crossing, and at these false-positive rates that is often not the
fire. FIgLib publishes no fire coordinates, so accuracy against a surveyed point
cannot be measured at all.

**The geometry itself is correct** where it can be checked. On rendered incidents
where we placed the fire, the fix lands **13 m** from the truth inside a reported
161 m 1σ ellipse, and `tests/test_agent.py` asserts it.

## 6. Which rejectors actually fire

Across all 4,959 frames, with the map calibration, at 0.35:

| Rejector | Times fired |
|---|---|
| `TOO_BRIEF` (fewer than three frames of history) | 506 |
| `HABITUAL_REGION` (on this camera's learnt nuisance map) | 63 |
| `ERRATIC_BASE` (base travelling faster than a fire can) | 29 |
| `GROUND_DRIFT` (dust: sideways, not rising, not neutral) | 27 |
| `FLARE` (clipped sensor) | 15 |
| `CLOUD_TRANSLATION` (whole shape moving, area constant) | 10 |
| `AIRBORNE` (base detached from the skyline) | 5 |
| `WEATHER_FRONT` (frame-wide contrast collapse) | 1 |

`TOO_BRIEF` dominating is the design working: most candidates are dismissed for
not having been observed long enough, which is the whole thesis. But the
hand-written impostor rules fire rarely, 87 times in total. The persistent false
positives passed every named rule, which is why the calibration exists; the
learnt map now catches 63 more, and most still get through.

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

## 9. Real footage from outside the network

The service also accepts a video or stills from any single camera (see the
README). None of that footage has a surveyed camera, so none of it can be
corroborated or located, and every flag it raises carries `NO_SECOND_VIEW`. On
two public clips, at the shipped threshold:

| Clip | What it contains | What Firstsmoke did |
|---|---|---|
| `waldo-canyon-clear-to-onset.mp4` (Steve Moraco, CC BY 3.0), 21.6 s a frame | clear sky, then a smoke column from about 0:20 | 240 of 1,200 frames read; 27 flags. Nine of them, 0:03 to 0:19, come before any smoke is in frame. The highest score, 0.75 at 0:43, is on the smoke column |
| `grand-canyon-clouds-timelapse.mp4` (NPS, CC BY 4.0), 1 s a frame assumed | cloud over the canyon, no smoke | 42 of 1,108 frames read; 5 flags, all false, highest 0.51 |

A third clip, `north-derby-gulch-rx-timelapse.mp4` (USDA Forest Service, public
domain), shows a faint wisp of smoke on a far ridge during a prescribed burn.
Firstsmoke missed it: its highest box (0.58) sat on a wind-blown bush in the
foreground, and the wisp itself scored 0.37 to 0.43, under the 0.45 threshold
(`media/real/firstsmoke/PROVENANCE.md` §3).

The Waldo and Grand Canyon rows are the deployed service's own results, on image
`fs-21772f1` (commit `21772f1`). A local run of the Waldo clip gave
22 flags with the same peak, so small decoder or CPU differences move the weakest
flags either side of 0.45.

Both are what the numbers above predict: on one camera it cries wolf on cloud,
and it does find a real column eventually.

## 10. What this evaluation does not establish

- **No field deployment trial exists.** We searched for a published
  outcome study for any camera-based safety detection product, in any domain,
  and found none. If a judge asks whether this has been shown to save lives in
  the field, the honest answer is that nobody has published that evidence, for
  this system or for any comparable one.
- **The fires are not located.** FIgLib publishes no coordinates, so the
  localisation accuracy on recorded data cannot be measured against a surveyed
  point at all. §5 reports a self-consistency spread and a synthetic accuracy,
  and calls neither one accuracy on real data.
- **63 development fires and 94 test fires is a small set**, drawn from one network in one region, and 44
  cameras is not enough to characterise per-camera behaviour when the false
  positives are concentrated in eight of them.
- **Every sequence contains a fire.** FIgLib has no all-clear days, so the clear
  period is only forty minutes per camera and always immediately precedes an
  ignition. A true false-positive rate needs quiet days, and we do not have them.
- **The shipped threshold and calibration were chosen on the development set.**
  Sections 2 to 8 are on that set. Section 0 is on 94 of the 130 held-out test
  sequences; the remaining 36 are being scored.
- **Only 16 of 44 cameras had a second date** to learn a calibration from, so the
  calibration's effect is measured on 30 sequences.

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
