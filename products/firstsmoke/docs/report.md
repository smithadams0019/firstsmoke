# Firstsmoke: technical report

OpenCV AI Competition 2026. Live endpoint:
**https://afynmqkk9e.eu-west-1.awsapprunner.com**

---

## 1. The problem, with evidence

The best-documented case for watching a hillside with a camera comes from the
defendant in a criminal prosecution, in its own sworn filing.

Pacific Gas and Electric Company's Form 10-K for the year ended 31 December
2020, filed with the Securities and Exchange Commission on 25 February 2021
(accession 0001004980-21-000007), records:

> "On March 17, 2020, the Utility entered into the Plea Agreement and Settlement
> (the "Plea Agreement") with the People of the State of California, by and
> through the Butte County District Attorney's office... the Utility agreed to
> plead guilty to 84 counts of involuntary manslaughter in violation of Penal
> Code section 192(b) and one count of unlawfully causing a fire in violation of
> Penal Code section 452."

The same filing describes the civil settlement:

> "$12.15 billion of the $13.5 billion liability as of June 30, 2020 was
> extinguished in the third quarter of 2020, and the remaining $1.35 billion
> will be paid out under the terms of the Tax Benefits Payment Agreement"

Then, in the company's second quarter 2020 press release, filed with the
Commission as Exhibit 99.1 on 30 July 2020, the remedy it adopted:

> "Situational awareness completion exceeds 30 percent, with 144 weather stations
> and 60 high definition cameras installed, despite some initial supply chain
> issues due to COVID-19 disruptions."

> "Enhanced vegetation management progress is at 70 percent. PG&E has reviewed
> more than 1,200 miles of distribution and lower-voltage transmission lines and
> taken necessary action to trim or remove hazards and expand rights-of-way."

Under a criminal plea and a $13.5 billion settlement, the remedy the utility
itself chose was cameras. That is not our inference; it is in the filing.

**Two limits on that argument, stated up front.** Sixty cameras across a service
territory of that size is sparse detection, not coverage. And a camera shortens
time-to-notice; it does not remove an ignition source, it cannot de-energise a
line, and it cannot stop a worn fitting from failing.

A note on a number we deliberately do not use: the commonly cited death toll for
the Camp fire is 85, and we could not verify it from a primary source, because
CAL FIRE's own incident page returned HTTP 403. The verified figure is the plea
count of 84 involuntary manslaughter charges, and that is the only figure used
here or in the video.

## 2. Users

**The primary user is a lookout or duty officer at a dispatch desk**, watching a
wall of camera feeds during fire season, who must decide within seconds whether
an alert is worth acting on. Their problem is not that they lack detections; it
is that a system which cries wolf four times a night stops being read. Every
design decision follows from that: the alert carries its reasoning in plain
sentences, the map shows what the system asked and what the world answered as two
different colours, and a stand-down is as explicit as an alert.

**The secondary user is whoever has to defend the system** to a regulator, an
insurer or a court after a fire. For them the artefact that matters is the run
record: which cameras were read, in what order, why each was chosen, what each
one showed, and what the system did not know.

Explicitly not a user: anyone expecting an autonomous dispatch decision. The
system stops at the alert.

## 3. Architecture

Diagrams are in **[architecture.md](architecture.md)**. In prose:

A **camera network** is a set of fixed cameras on known summits, each with a
latitude, longitude, azimuth and horizontal field of view. Those extrinsics are
the hard dependency of this whole product. Without them there is no bearing and
no triangulation, and the thing collapses into a smoke classifier. HPWREN
publishes a complete set for 386 fixed cameras across 55 summits, which is why
this network was chosen over alternatives with better uptime but no published
geometry.

A **sequence** is an ordered run of stills from one camera with real timestamps.
This is the unit of work, not video: a lookout network delivers one frame a
minute, which means no optical-flow continuity between frames, a sun that has
moved, and an exposure that has re-metered.

For each frame, a **CameraWatch** produces one opinion: a confidence between 0
and 1, a compass bearing, an angular uncertainty, a list of named reasons for,
and a list of named rejections against.

A **Lookout** drives the escalation state machine over the whole network,
deciding on each tick which cameras to read.

The service is servicekit's FastAPI shell with four product routes added. A job
runs the analyser on a worker thread and streams progress over Server-Sent
Events; the interface renders the map, the bearing fan, the consultation trail
and the camera wall from that stream.

## 4. The OpenCV 5 implementation

OpenCV 5.0.0 does substantive work at every stage. What follows is what it does
and why, including the places where OpenCV 5 differs from 4.x in ways that
mattered.

### 4.1 Preparation and stabilisation

Frames arrive at 3072 x 2048 and are resized to 1024 px wide with
`cv2.resize(..., INTER_AREA)`. `INTER_AREA` integrates rather than samples, so a
plume two pixels wide at native resolution survives as a dim smear rather than
being dropped; on the evaluation set this recovered four early detections that
`INTER_LINEAR` missed.

A mast-mounted camera moves a few pixels in wind. Left uncorrected, every ridge
edge in the scene appears in the difference image. `cv2.phaseCorrelate` against a
`cv2.createHanningWindow` recovers the global translation in about 3 ms, and
`cv2.warpAffine` undoes it. Rotation is not corrected: a pole that twists needs a
service visit, and the correlation response drops far enough to say so.

### 4.2 Sky and horizon

The horizon is where a new column first appears and also where every registration
false positive lives, so it is found rather than assumed. A gradient threshold is
swept over `cv2.Sobel` magnitudes at 256 px wide; for each candidate threshold the
first strong gradient down each column gives a boundary, scored by how well the
two regions separate in intensity divided by how ragged the boundary is. The
winning boundary is then **refined at full resolution**, by moving each column to
the strongest vertical gradient within 14 px.

That refinement is not a detail. Without it the boundary came out a consistent
8.5 pixels high, a bias rather than noise, because the search runs downscaled and
pre-blurred. Eight pixels matters when a plume's base is often within twenty of
the skyline. Measured on rendered scenes with a known ridge, the refinement takes
the median error from 8.5 px with a -8.5 px bias to **0.3 px with no bias**.

### 4.3 Background modelling

`cv2.createBackgroundSubtractorMOG2` with `detectShadows=True`, and the shadow
label discarded, because a shadow is the sun going behind a cloud, which is
exactly what we do not want to alarm on.

MOG2 alone is not enough at one frame a minute, because the illumination changes
measurably between frames and dramatically over an hour. So alongside it the
model keeps a **time-of-day reference** per camera in two-hour buckets, and before
differencing fits a **global gain and offset** mapping the current frame onto that
reference, by least squares on a subsample, re-fitted twice while dropping the
worst 20% of residuals. The dropped pixels are the ones that actually changed,
which is the point: the exposure model is fitted to the part of the scene that
stayed the same. A plume covering a few percent of the frame cannot drag it.

A single global gain cannot describe what the sun does to a hillside: as it
swings, one aspect brightens and the opposite darkens, leaving a smooth residual
field. That field is removed by blurring the residual at 1/16 scale and
subtracting it, clamped so a large event cannot subtract itself away. A plume is
compact enough that a 96 px blur spreads a 3,000 px region at 25 counts into well
under one count.

When a detection is live the model is **frozen**. A plume left to soak into the
background disappears in about ten frames, after which the camera reports all
clear while the hillside burns.

### 4.4 Candidate extraction

The change mask is the union of MOG2's foreground and a **hysteresis threshold**
on the photometric difference: seeds at four times the frame's own robust noise
level, grown into connected regions above 1.6 times it.

The hysteresis replaced a single fixed threshold and it is the single most
important change in the pipeline. A column is dense at its base and fades to two
or three counts by the time it clears the ridge. A threshold high enough to
ignore sensor noise cut the plume off at the shoulders, so the tracked box
stopped growing upward and the **rise** term, the most diagnostic measurement in
the product, read zero on real smoke. With hysteresis, the measured rise on a
rendered plume matches the rendered rate to within 0.2 px a minute.

Then `cv2.morphologyEx` CLOSE and OPEN, and `cv2.connectedComponentsWithStats`.
Components are ranked by area and only the largest twelve are measured. Measuring
first and ranking afterwards, as the first version did, threw away 83% of the
most expensive work in the pipeline and cost 1.1 of every 1.3 seconds a frame.

### 4.5 What each region is measured on

Inside its own bounding box, not the whole frame:

- **Texture ratio.** `cv2.Laplacian` variance inside the region against both the
  background reference and the ring just outside it, taking the lower. Smoke
  veils rather than replaces, so the detail behind it attenuates.
- **Saturation ratio and greyness**, from `cv2.cvtColor` to HSV and the maximum
  channel spread. Dropped entirely on monochrome cameras and the remaining
  weights renormalised, rather than scored zero: a camera that cannot measure
  colour has not measured "no colour change".
- **Edge softness**, from `cv2.Sobel` on the *difference* image after a Gaussian,
  normalised by the difference's own peak. This asks how many pixels the change
  takes to go from nothing to full: one or two for a droplet on the glass, twenty
  for a plume. Measured on the frame's own gradients instead, as the first
  version did, it reported every plume in a smooth scene as hard-edged.
- **Attachment**, the signed distance from the region's base to the local skyline.
- **The base point**, the centroid of the lowest tenth of the mask. Not the
  bounding-box centre: a plume that shears downwind widens at the top, which drags
  the box centre sideways even though the fire has not moved, and that alone was
  enough to make the cloud rejector fire on real smoke.

### 4.6 Tracking and growth

**OpenCV 5 removed `TrackerCSRT`, `TrackerKCF` and the whole `legacy` namespace
from the main wheel**, so there is no built-in tracker to call. That turned out to
be a good thing: appearance trackers lock onto texture, and a plume's defining
property is that it has almost none. The association here is greedy overlap on the
segmentation masks, rescued by centroid proximity when a fast-growing plume's
boxes barely intersect.

Over a track we measure area growth and monotonicity, top rise rate, base drift,
base jitter, horizontal drift, width growth, and an anchor score
`rise / (rise + base drift)`. These are the real discriminators:

| | Base | Top | Area |
|---|---|---|---|
| Smoke column | anchored | rises | grows |
| Cloud | moves at wind speed | moves with it | constant |
| Dust | moves along the ground | barely rises | grows |
| Lens droplet | perfectly still | still | constant |

Base drift is a **robust net displacement**, the median position over the first
third of a track against the last third, not a summed path length. Path length was
the first implementation and it cost real detections: a growing plume's mask
breathes by five to ten pixels a frame at its lower edge as the thin skirt crosses
the detection threshold, so path length accumulated 20 px a minute on a base that
had not moved, and the erratic-motion rule threw away the smoke. The breathing is
now measured separately as jitter, where it is genuinely diagnostic: a mask that
does not breathe at all is something on the glass.

### 4.7 The learned confirmation head

`cv2.dnn` in OpenCV 5 is **ONNX-only**: `readNetFromCaffe` and
`readNetFromDarknet` are gone. Of the models that are both ONNX and safely
licensed, none we could verify is a smoke classifier. YOLOX is Apache-2.0 and
loads cleanly, but COCO has no smoke class. Ultralytics' YOLO family, which does
have community smoke weights, is AGPL-3.0, and section 13 makes a hosted demo API
a source-disclosure event.

So the model is ours: three convolutions and two dense layers, 53,978 parameters,
trained in plain numpy on `pyronear/pyro-sdis` (Apache-2.0, which permits
derivative works) and exported to ONNX by hand using only operators OpenCV 5's
importer supports. It runs through `cv2.dnn.readNetFromONNX` at 0.54 ms a 64x64
crop. The training script verifies its own export by re-running the validation
split through `cv2.dnn` and checking the answers match.

Accuracy 0.885, precision 0.948, recall 0.766 on the pyro-sdis validation split.
It is out of distribution on FIgLib, a European network evaluated on a Californian
one, contributes modest precision and no recall, and the pipeline runs without it
and says so in the run record when the file is absent.

### 4.8 Two OpenCV 5 migration traps worth recording

`cv2.FontFace` is new and useful, but its `putText` overload is
`putText(img, text, org, colour, face, size, weight)`: colour and font are
**swapped** relative to the legacy Hershey overload. Passing the old order raises
"Argument 'fontFace' is required to be an integer", which is a confusing way to be
told the arguments are the wrong way round. It cost a container rebuild.

Also confirmed against the installed wheel: `VideoCapture::get()` returns -1 for
unsupported properties rather than 0, so the frame-rate fallback tests `<= 0`.

## 5. The agent loop

The qualifying bar is that image results must change what the system does next.
Here is precisely where that happens.

When a camera produces a detection between 0.45 and 0.68, the **pixel column of
that detection** is converted to a bearing through the camera's azimuth and field
of view, using the pinhole relation `tan θ = (2x/w − 1)·tan(hfov/2)`, or an
equiangular one above 120°, because seven cameras in the published metadata
declare a 180° field of view and the pinhole relation does not merely lose
accuracy on a stitched panorama, it diverges.

The bearing is projected onto the map, and `Network.consultable` returns the
cameras that overlook that point, ranked by how squarely their line of sight cuts
the first. Cameras on the same summit are excluded no matter what they can see:
two cameras a metre apart give two parallel rays and no fix, and an agent that
confirmed from its own mast would be fooling itself. Each selected camera is told
**where in its frame to look**, and a detection more than 6° from that prediction
is recorded as a different thing rather than as corroboration.

Different pixels select different cameras. Nothing in the loop is a fixed list,
and `tests/test_agent.py` asserts that a bearing pointing into the network finds
neighbours while one pointing away from every other summit finds none.

The system has exactly three actions:

1. **Read a camera it was not reading**, chosen by geometry.
2. **Re-read a camera it already read**, replaying its recent frames with the
   background model frozen, so the growth analysis gets its full span.
3. **Ask a human**, when the evidence is real but will not resolve.

It cannot pan a camera, dispatch anything, or contact anyone outside the page.
That ceiling is deliberate. The consequence of a false alarm here is a person
looking at a picture; the consequence of a missed fire is not.

The states are WATCH, SUSPECT, CONSULT, CONFIRMED, STOOD_DOWN, NEEDS_HUMAN,
ALERTED. Every transition is logged with the evidence frames and a sentence naming
the measurement that changed. Two design points worth noting:

- **Standing down is not terminal.** A lookout that saw a cloud, checked and was
  satisfied goes back to watching. Treating a stand-down as the end of the run
  meant one warm-up artefact could close the watch before the real ignition.
- **Blind neighbours produce NEEDS_HUMAN, never a stand-down.** If every camera
  that overlooks a bearing is fogged or dark, that is a gap in cover, not an empty
  hillside, and the system says which fault each camera has.

## 6. AWS deployment

Region **eu-west-1**, not us-east-1, because the account is limited to two App
Runner services per region and us-east-1's two slots hold unrelated production
services. Full reasoning and every resource created is in
**[costs.md](costs.md)**.

- **ECR** `opencv26/firstsmoke`, a 653 MB `linux/amd64` image built in two stages.
  The builder renders the four bundled incidents, about 100 seconds of OpenCV
  work, so a judge never waits for it.
- **App Runner** `opencv26-firstsmoke`, 2 vCPU / 4 GB, **left running**. About
  $0.031 an hour idle, roughly $23 a month. Scaling to zero would save $20 and
  cost a judge a cold start on their first click.
- **S3** `s3://opencv26-artifacts-<aws-account-id>/firstsmoke/` holds the evaluation
  output and the trained model. Not on the request path.

The image is self-contained: a cold container serves a full incident, ending in a
crossed fix with an uncertainty ellipse, in about 32 seconds with no network
access beyond the browser.

## 7. Evaluation

Summarised here; the full document with every failure case is
**[evaluation.md](evaluation.md)**.

The shipped configuration is a per-camera nuisance map learnt from each camera's
clear frames on other dates, and a suspicion threshold of 0.45. Both were chosen
on 64 development sequences (4,959 frames, 44 cameras) and committed before the
held-out test set was downloaded. The test set is 130 FIgLib sequences, all
scored (10,107 frames). An interim run on the first 94 (7,272 frames) is kept
beside it, because the film quotes the 94-sequence interim numbers. 33 individual
frames timed out during the test-set download and are missing from the cache.

| At 0.45, map calibration | Test set, all 130 (full run) | Test set, first 94 (interim) | Development set, threshold chosen here (64) |
|---|---|---|---|
| One camera: fires found | **76.9%** | 78.7% | 79.4% |
| One camera: false positives per camera-day | **154.2** | 151.6 | 150.3 |
| One camera: median time to alert | **+240 s** | +210 s | +438 s |
| Two cameras agreeing: fires found | **29.9%** | 28.6% | 42.4% |
| Two cameras agreeing: false positives per camera-day | **19.6** | 18.4 | 21.7 |
| Alerts ahead of the human mark | **0** | 0 | 0 |
| Triangulation: median spread between pair fixes | **6.3 km** | 3.8 km | 7.4 km |

Four things the test set settled:

- **Calibration made no measurable difference on test**: 154.2 false positives
  per camera-day on all 130 with or without it, and the same 100 fires found
  (interim 94: 151.2 without, 151.6 with).
- **Two-camera detection fell** from 42.4% on dev to 29.9% on test. The threshold
  sat at the knee of the dev two-camera curve, and the knee did not carry over.
- **It never alerted ahead of the human mark.** The +240 s is measured against an
  annotator who reviewed each sequence in hindsight. Published detectors on the
  same library report means of 2.3 to 4.7 minutes; our test mean is 8.8.
- **Triangulation lands kilometres apart.** 5 fixes on 17 multi-summit test dates,
  with pair fixes a median 6.3 km apart. On synthetic incidents where we placed
  the fire, the fix lands **13 m** from the truth inside a reported 161 m ellipse,
  so the geometry is right and the input bearings are not.

Before calibration and the threshold change, on dev, one camera found 90.5% at 485
false positives per camera-day. Moving the threshold from 0.35 to 0.45 raised the
dev median alert from +60 s to +438 s. That delay is the price of the lower
false-alarm rate.

**Real footage.** On the live service (image `fs-21772f1`, commit `21772f1`), the
Waldo Canyon time-lapse raised 27 flags. Nine came before any smoke was in frame,
on cloud, and the highest score (0.75) was on the smoke column. A Grand Canyon
cloud time-lapse raised 5 flags, all false. A faint prescribed-burn wisp in North
Derby Gulch was missed; the top box was on a foreground bush.

## 8. Limitations

1. **It is not deployable at these false-alarm rates.** On the held-out test set,
   one camera finds 76.9% of fires at 154.2 false positives per camera-day; with a
   second camera required it finds 29.9% at 19.6. The persistent false positives
   on particular views are not caught by the learnt nuisance map, which made no
   measurable difference on test, and we did not identify what they physically are.
2. **We never beat the human annotator**, on dev or test. A structural lag is built
   into requiring growth before speaking. This must not be sold as early detection.
3. **Triangulation is not useful on real data yet.** On test, 5 fixes on 17 dates
   and a median 6.3 km between pair fixes; on dev, 2 of 8 and 7.4 km. It refuses
   rather than guessing, which is the correct failure, but it is a failure.
4. **A re-aimed camera silently invalidates its bearings.** The published azimuth
   is wrong until the metadata is refreshed. Today the only mitigation is that a
   re-aimed camera stops agreeing with its neighbours and so produces stand-downs
   rather than confident wrong positions. A per-camera azimuth correction fitted
   from long-run disagreement is the fix, and it is not built.
5. **Small, single-region evidence.** 63 development fires and 130 test
   fires, from one network in southern California. Every sequence contains a fire,
   so the clear period is only forty minutes per camera and always immediately
   precedes an ignition; a true false-positive rate needs quiet days, which FIgLib
   does not contain.
6. **The bundled demonstration incidents are rendered**, for the licence reason in
   section 9. The measured results are all from real imagery, and the upload path
   runs on real footage from one camera.
7. **PTZ cameras are excluded**, which removes 109 of the 495 cameras in the
   published metadata, because their published azimuth is a home position rather
   than where they are pointed now.
8. **No field deployment trial exists**, for this system or, as far as we could
   find, for any comparable camera-based safety product in any domain. If a judge
   asks whether this has been shown to save lives, the honest answer is that
   nobody has published that evidence.

## 9. Responsible use

**Data licensing.** HPWREN's imagery and camera metadata are CC BY-NC-ND 4.0
(https://hpwren.ucsd.edu/cc.html), with a requested credit line of
`http://hpwren.ucsd.edu`, which appears in the interface and on every response
from the live path. The NoDerivatives term is why the **deployed** demonstration
ships rendered scenes rather than recorded frames: the interface's whole point is
to show an annotated frame with the detected region drawn on it, and an annotated
frame is a derivative work. Publishing those from a hosted endpoint would breach
the licence of the people whose cameras made this possible. Recorded frames are
cached locally by whoever runs the evaluation and are never redistributed.

The confirmation model is trained only on `pyronear/pyro-sdis`, which is
Apache-2.0 and permits derivative works, precisely so that the model weights carry
no licence question.

**No AGPL anywhere.** Nothing imports `ultralytics` or any AGPL-licensed model.

**People.** These cameras look at hillsides from tens of kilometres away and
nothing in this system detects, tracks or identifies a person. No sample media
contains an identifiable person. The one place a person could appear, a live frame
from a public camera, is behind a switch that is off by default, and the frames
expire with the job.

**Autonomy.** The system stops at the alert. It cannot dispatch, cannot pan a
camera, and cannot contact anyone. The interface's escalation buttons are
deliberately inert and say so on hover.

**Being a good neighbour to the data source.** The live path is off unless
`FIRSTSMOKE_LIVE=1`; the download script uses four workers with a pause between
requests, because HPWREN's usage conditions ask users to be careful about the load
they impose on a small research team's infrastructure.

**Honesty as a design property, not a disclaimer.** A camera that cannot see
reports why rather than reporting clear. A crossing that cannot be trusted is
refused with a named reason and a nearest-approach measurement. A detection
carries the eight named terms that produced its score. The single most likely way
this product causes harm is by being believed when it should not be, and every one
of those choices exists to make that harder.
