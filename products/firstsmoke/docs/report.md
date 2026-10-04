# Firstsmoke: technical report

Live endpoint: **https://afynmqkk9e.eu-west-1.awsapprunner.com**

---

## 1. The result

Firstsmoke watches mountain-top cameras for the first smoke column of a wildfire and
ranks what it finds, so that the person on the desk opens the right flag first.

On the 130-sequence held-out FIgLib test set, with the threshold and calibration
frozen before a single test sequence was downloaded, at the shipped operating point
of 0.45:

| | |
|---|---|
| Fires found, one camera | **100 of 129 — 77.5%** |
| False-positive frames per camera-day | **154.2** — about one clear frame in nine |
| Median time to alert, after the human mark | **+240 s**, and never earlier |
| Triangulation fixes on multi-summit dates | 5 of 17, pair fixes a median 6.3 km apart |

Finding smoke is not what defeats a camera network. One clear frame in nine is: a duty
officer handed a hundred flags an hour, with nothing to say which to open, stops reading
them. Everything after the detector exists to order that queue.

## 2. The problem, with evidence

The best-documented case for watching a hillside with a camera comes from a defendant
in a criminal prosecution, in its own sworn filing.

Pacific Gas and Electric Company's Form 10-K for the year ended 31 December 2020,
filed with the Securities and Exchange Commission on 25 February 2021 (accession
0001004980-21-000007), records:

> "the Utility agreed to plead guilty to 84 counts of involuntary manslaughter in
> violation of Penal Code section 192(b) and one count of unlawfully causing a fire in
> violation of Penal Code section 452."

The same filing accounts for a $13.5 billion liability to the victims. Then, in the
company's second quarter 2020 press release, filed with the Commission as Exhibit 99.1
on 30 July 2020, the remedy it adopted:

> "Situational awareness completion exceeds 30 percent, with 144 weather stations and
> 60 high definition cameras installed, despite some initial supply chain issues due
> to COVID-19 disruptions."

Under a criminal plea and a $13.5 billion settlement, the remedy the utility itself
chose was cameras. That is not our inference; it is in the filing.

**Two limits on that argument, stated up front.** Sixty cameras across a service
territory of that size is sparse detection, not coverage. And a camera shortens
time-to-notice; it does not remove an ignition source, de-energise a line, or stop a
worn fitting from failing. (The commonly cited Camp fire death toll of 85 is not used
anywhere here: CAL FIRE's incident page returned HTTP 403, so the only figure we can
source is the plea count of 84.)

## 3. Users

**The primary user is a lookout or duty officer at a dispatch desk**, watching a wall
of camera feeds during fire season, who must decide within seconds whether a flag is
worth acting on. Their problem is not that they lack detections. It is that a system
which cries wolf four times a night stops being read. Every design decision follows
from that: a flag carries its reasoning in plain sentences, the map shows what the
system asked and what the world answered as two different colours, and a stand-down
is as explicit as an alert.

**The secondary user is whoever has to defend the system** to a regulator, an insurer
or a court after a fire. For them the artefact that matters is the run record: which
cameras were read, in what order, why each was chosen, what each showed, and what the
system did not know. Explicitly not a user: anyone expecting an autonomous dispatch
decision. The system stops at the alert.

## 4. Architecture

Diagrams are in **[architecture.md](architecture.md)**. In prose:

A **camera network** is a set of fixed cameras on known summits, each with a latitude,
longitude, azimuth and horizontal field of view. Those extrinsics are the hard
dependency of this whole product: without them there is no bearing and no
triangulation, and the thing collapses into a smoke classifier. HPWREN publishes a
complete set for 386 fixed cameras across 55 summits, which is why this network was
chosen over alternatives with better uptime but no published geometry.

A **sequence** is an ordered run of stills from one camera with real timestamps — the
unit of work, not video, because a lookout network delivers one frame a minute. For
each frame a **CameraWatch** produces one opinion (a confidence, a compass bearing, an
angular uncertainty, named reasons for and named rejections against), and a **Lookout**
drives the escalation state machine over the network, deciding on each tick which
cameras to read. The service is servicekit's FastAPI shell with four product routes
added: a job runs the analyser on a worker thread and streams progress over Server-Sent
Events, and the interface renders the map, the bearing fan, the consultation trail and
the camera wall from that stream.

## 5. The OpenCV 5 implementation

OpenCV 5.0.0 does substantive work at every stage. What follows is what it does and
why, including the places where OpenCV 5 differs from 4.x in ways that mattered.

### 5.1 Preparation and stabilisation

Frames arrive at 3072 x 2048 and are resized to 1024 px wide with
`cv2.resize(..., INTER_AREA)`, which integrates rather than samples, so a plume two
pixels wide at native resolution survives as a dim smear instead of being dropped; on
the evaluation set that recovered four early detections `INTER_LINEAR` missed.

A mast-mounted camera moves a few pixels in wind, and left uncorrected every ridge edge
in the scene appears in the difference image. `cv2.phaseCorrelate` against a
`cv2.createHanningWindow` recovers the global translation in about 3 ms, and
`cv2.warpAffine` undoes it. Rotation is not corrected: a pole that twists needs a
service visit, and the correlation response drops far enough to say so.

### 5.2 Sky and horizon

The horizon is where a new column first appears and also where every registration
false positive lives, so it is found rather than assumed. A gradient threshold is swept
over `cv2.Sobel` magnitudes at 256 px wide; for each candidate the first strong gradient
down each column gives a boundary, scored by how well the two regions separate in
intensity divided by how ragged the boundary is. The winner is then **refined at full
resolution**, by moving each column to the strongest vertical gradient within 14 px.

That refinement is not a detail. Without it the boundary came out a consistent 8.5
pixels high — a bias rather than noise, because the search runs downscaled and
pre-blurred — and eight pixels matters when a plume's base is often within twenty of
the skyline. On rendered scenes with a known ridge, the refinement takes the median
error from 8.5 px with a −8.5 px bias to **0.3 px with no bias**.

### 5.3 Background modelling

`cv2.createBackgroundSubtractorMOG2` with `detectShadows=True`, and the shadow label
discarded, because a shadow is the sun going behind a cloud — exactly what we must not
alarm on.

MOG2 alone is not enough at one frame a minute, because illumination changes measurably
between frames and dramatically over an hour. So the model also keeps a **time-of-day
reference** per camera in two-hour buckets, and before differencing fits a **global gain
and offset** onto it by least squares, re-fitted twice while dropping the worst 20% of
residuals. The dropped pixels are the ones that actually changed, which is the point:
the exposure model is fitted to the part of the scene that stayed the same, so a plume
covering a few percent of the frame cannot drag it. The smooth residual left by the sun
swinging across a hillside is then removed by blurring it at 1/16 scale and subtracting
it, clamped so a large event cannot subtract itself away.

When a detection is live the model is **frozen**. A plume left to soak into the
background disappears in about ten frames, after which the camera reports all clear
while the hillside burns.

### 5.4 Candidate extraction

The change mask is the union of MOG2's foreground and a **hysteresis threshold** on the
photometric difference: seeds at four times the frame's own robust noise level, grown
into connected regions above 1.6 times it. This is the single most important change in
the pipeline. A column is dense at its base and fades to two or three counts by the
time it clears the ridge, so a threshold high enough to ignore sensor noise cut the
plume off at the shoulders — the tracked box stopped growing upward and **rise**, the
most diagnostic measurement in the product, read zero on real smoke. With hysteresis,
the measured rise on a rendered plume matches the rendered rate to within 0.2 px a
minute.

Then `cv2.morphologyEx` CLOSE and OPEN, and `cv2.connectedComponentsWithStats`.
Components are ranked by area and only the largest twelve are measured; measuring first
and ranking afterwards threw away 83% of the most expensive work in the pipeline and
cost 1.1 of every 1.3 seconds a frame.

### 5.5 What each region is measured on

Inside its own bounding box, not the whole frame:

- **Texture ratio.** `cv2.Laplacian` variance inside the region against both the
  background reference and the ring just outside it, taking the lower. Smoke veils
  rather than replaces, so the detail behind it attenuates.
- **Saturation ratio and greyness**, from `cv2.cvtColor` to HSV and the maximum channel
  spread. Dropped on monochrome cameras and the remaining weights renormalised, rather
  than scored zero: a camera that cannot measure colour has not measured "no colour
  change".
- **Edge softness**, from `cv2.Sobel` on the *difference* image after a Gaussian,
  normalised by the difference's own peak. How many pixels the change takes to go from
  nothing to full: one or two for a droplet on the glass, twenty for a plume.
- **Attachment**, the signed distance from the region's base to the local skyline.
- **The base point**, the centroid of the lowest tenth of the mask — not the
  bounding-box centre, because a plume that shears downwind widens at the top, which
  drags the box centre sideways even though the fire has not moved, and that alone was
  enough to make the cloud rejector fire on real smoke.

### 5.6 Tracking and growth

**OpenCV 5 removed `TrackerCSRT`, `TrackerKCF` and the whole `legacy` namespace from
the main wheel**, so there is no built-in tracker to call. That turned out to be a good
thing: appearance trackers lock onto texture, and a plume's defining property is that
it has almost none. The association here is greedy overlap on the segmentation masks,
rescued by centroid proximity when a fast-growing plume's boxes barely intersect.

Over a track we measure area growth and monotonicity, top rise rate, base drift and
jitter, horizontal drift, width growth, and an anchor score `rise / (rise + base
drift)`. These are the real discriminators:

| | Base | Top | Area |
|---|---|---|---|
| Smoke column | anchored | rises | grows |
| Cloud | moves at wind speed | moves with it | constant |
| Dust | moves along the ground | barely rises | grows |
| Lens droplet | perfectly still | still | constant |

Base drift is a **robust net displacement** — the median position over a track's first
third against its last — not a summed path length. Path length cost real detections: a
growing plume's mask breathes five to ten pixels a frame at its lower edge as the thin
skirt crosses the threshold, so it accumulated 20 px a minute on a base that had not
moved and the erratic-motion rule threw away the smoke. The breathing is now measured
separately as jitter, where it is diagnostic: a mask that does not breathe at all is
something on the glass.

### 5.7 The learned confirmation head

`cv2.dnn` in OpenCV 5 is **ONNX-only**: `readNetFromCaffe` and `readNetFromDarknet`
are gone. Of the models that are both ONNX and safely licensed, none we could verify is
a smoke classifier — YOLOX is Apache-2.0 and loads cleanly but COCO has no smoke class,
and Ultralytics' YOLO family, which does have community smoke weights, is AGPL-3.0,
whose section 13 makes a hosted demo API a source-disclosure event.

So the model is ours: three convolutions and two dense layers, 53,978 parameters,
trained in plain numpy on `pyronear/pyro-sdis` (Apache-2.0) and exported to ONNX by
hand using only operators OpenCV 5's importer supports. It runs through
`cv2.dnn.readNetFromONNX` at 0.54 ms a 64x64 crop, and the training script verifies its
own export by re-running the validation split through `cv2.dnn`. Accuracy 0.885,
precision 0.948, recall 0.766 on that split. It is out of distribution on FIgLib — a
European network evaluated on a Californian one — contributes modest precision and no
recall, and the pipeline runs without it and says so when the file is absent.

### 5.8 Two OpenCV 5 migration traps worth recording

`cv2.FontFace` is new and useful, but its `putText` overload is
`putText(img, text, org, colour, face, size, weight)`: colour and font are **swapped**
relative to the legacy Hershey overload. Passing the old order raises "Argument
'fontFace' is required to be an integer", which is a confusing way to be told the
arguments are the wrong way round. It cost a container rebuild.

Also confirmed against the installed wheel: `VideoCapture::get()` returns −1 for
unsupported properties rather than 0, so the frame-rate fallback tests `<= 0`.

## 6. The agent loop

Image results change what the system does next. Here is precisely where.

When a camera produces a detection between 0.45 and 0.68, the **pixel column of that
detection** is converted to a bearing through the camera's azimuth and field of view,
using the pinhole relation `tan θ = (2x/w − 1)·tan(hfov/2)` — or an equiangular one
above 120°, because seven cameras in the published metadata declare a 180° field of
view, where the pinhole relation does not merely lose accuracy, it diverges.

The bearing is projected onto the map, and `Network.consultable` returns the cameras
that overlook that point, ranked by how squarely their line of sight cuts the first.
Cameras on the same summit are excluded no matter what they can see: two cameras a
metre apart give two parallel rays and no fix, and an agent that confirmed from its own
mast would be fooling itself. Each selected camera is told **where in its frame to
look**, and a detection more than 6° from that prediction is recorded as a different
thing rather than as corroboration. Different pixels select different cameras —
nothing in the loop is a fixed list, and `tests/test_agent.py` asserts that a bearing
pointing into the network finds neighbours while one pointing away finds none.

The system has exactly three actions:

1. **Read a camera it was not reading**, chosen by geometry.
2. **Re-read a camera it already read**, replaying its recent frames with the
   background model frozen, so the growth analysis gets its full span.
3. **Ask a human**, when the evidence is real but will not resolve.

It cannot pan a camera, dispatch anything, or contact anyone outside the page. That
ceiling is deliberate: the consequence of a false alarm here is a person looking at a
picture, and the consequence of a missed fire is not.

### 6.1 Corroboration ranks the queue; it does not gate it

The loop was described as a second camera *gating* the alarm: nothing raised until
another summit agrees. The held-out set does not support that description, and
`evaluation.md` §1 measures why. On the 67 test fires that two or more summits
recorded, requiring agreement finds 29.9% at 19.6 false-positive frames a camera-day;
raising one camera's own bar to 0.55 instead finds 43.3% at 16.2 — more fires for fewer
false alarms, from one camera. A gate dominated on both axes is not worth having.

So the agent's product is an **ordered queue with evidence attached**, which is what
the code has always built:

- `_crossed_bearings` → a position and an uncertainty ellipse, first in the queue;
- `_resolve_without_neighbours` → an alert from one confident camera, marked
  `NO_SECOND_VIEW`, carrying no position;
- `cross_rays` refusals → `RAYS_TOO_PARALLEL`, `BEYOND_RANGE` or `BEHIND_CAMERA`, each
  with the nearest approach of the rays, rather than a confident wrong point;
- `blind_reason` → the specific fault of a camera that cannot see, so its silence is
  never read as an empty hillside;
- `_stand_down_or_escalate` → `NEEDS_HUMAN` for anything confident and unresolved, and
  a stand-down only for a weak detection a neighbour actively contradicted.

Two design points worth noting. **Standing down is not terminal**: a lookout that saw a
cloud, checked and was satisfied goes back to watching, because treating a stand-down
as the end of the run meant one warm-up artefact could close the watch before the real
ignition. And **blind neighbours produce `NEEDS_HUMAN`, never a stand-down**: if every
camera overlooking a bearing is fogged or dark, that is a gap in cover, not an empty
hillside.

## 7. Deployment

Region **eu-west-1**, because the account is limited to two App Runner services per
region and us-east-1's two slots hold unrelated services.

- **ECR** `opencv26/firstsmoke`, a 653 MB `linux/amd64` image built in two stages. The
  builder renders the four bundled incidents, about 100 seconds of OpenCV work, so a
  cold container does not have to.
- **App Runner** `opencv26-firstsmoke`, 2 vCPU / 4 GB, left running rather than scaled
  to zero, so the first click does not pay a cold start.
- **S3** `s3://opencv26-artifacts-<aws-account-id>/firstsmoke/` holds the evaluation output
  and the trained model. Not on the request path.

The image is self-contained: a cold container serves a full incident, ending in a
crossed fix with an uncertainty ellipse, in about 32 seconds.

## 8. Evaluation

Summarised here; the full document with every failure case is
**[evaluation.md](evaluation.md)**.

The shipped configuration is a per-camera nuisance map learnt from each camera's clear
frames on other dates, and a suspicion threshold of 0.45. Both were chosen on 64
development sequences (4,959 frames, 44 cameras) and committed before the 130-sequence
test set was downloaded; all 130 are scored, over the 10,107 frames that arrived.

Section 1 carries the held-out figures. The development set, where the threshold was
chosen, gave 79.4% at 150.3 false positives per camera-day with a +438 s median, so
one-camera behaviour barely moved between the two.

What the test set settled:

- **One camera held up.** Detection and false alarms barely moved from development to
  test, which is the main thing a held-out set is for.
- **The second camera does not earn a gate** — 29.9% at 19.6 against one camera's 43.3%
  at 16.2 on the same 67 fires — which is why corroboration ranks the queue rather than
  filtering it (§6.1).
- **Calibration made no measurable difference on test**: 154.2 false positives per
  camera-day with or without it, and the same 100 fires found.
- **It never alerted ahead of the human mark.** The +240 s is against an annotator who
  reviewed each sequence in hindsight. Published detectors on the same library report
  means of 2.3 to 4.7 minutes; ours is 8.8.
- **The geometry is right and the input bearings are not.** On rendered incidents where
  we placed the fire the fix lands **13 m** from the truth inside a reported 161 m
  ellipse, while on real cameras pair fixes land a median 6.3 km apart.

On footage from outside the network the Waldo Canyon time-lapse raised 27 flags with
the highest score on the smoke column, a Grand Canyon cloud time-lapse raised 5 flags
with no fire present, and a faint prescribed-burn wisp was missed.

## 9. Limitations

1. **The false-alarm rate is high**: about one clear frame in nine, 154.2
   false-positive frames per camera-day at the shipped threshold. The persistent false
   positives on particular views are not caught by the learnt nuisance map, which made
   no measurable difference on test, and we did not identify what they physically are.
2. **We never beat the human annotator**, on development or test. A structural lag is
   built into requiring growth before speaking, and this must not be sold as early
   detection.
3. **Triangulation is not useful on real data yet**: 5 fixes on 17 test dates and a
   median 6.3 km between pair fixes. It refuses rather than guessing, which is the
   correct failure, but it is a failure.
4. **A re-aimed camera silently invalidates its bearings**, and stays wrong until the
   metadata is refreshed. The only mitigation today is that it stops agreeing with its
   neighbours. A per-camera azimuth correction fitted from long-run disagreement is the
   fix, and it is not built.
5. **Small, single-region evidence.** 63 development and 130 test fires from one network
   in southern California. Every sequence contains a fire, so the clear period is only
   forty minutes per camera and always immediately precedes an ignition; a true
   false-positive rate needs quiet days, which FIgLib does not contain.
6. **The bundled demonstration incidents are rendered**, for the licence reason in §10.
   Every measured result is from real imagery, and the upload path runs real footage.
7. **PTZ cameras are excluded**, removing 109 of the 495 cameras in the published
   metadata, because their published azimuth is a home position rather than where they
   are pointed now.

## 10. Responsible use

**Data licensing.** HPWREN's imagery and camera metadata are CC BY-NC-ND 4.0
(https://hpwren.ucsd.edu/cc.html), credited as `http://hpwren.ucsd.edu` in the
interface and on every response from the live path. The NoDerivatives term is why the
**deployed** demonstration ships rendered scenes rather than recorded frames: the
interface's whole point is to show an annotated frame with the detected region drawn on
it, and an annotated frame is a derivative work. Recorded frames are cached locally by
whoever runs the evaluation and never redistributed. The confirmation model is trained
only on `pyronear/pyro-sdis`, which is Apache-2.0, precisely so the weights carry no
licence question, and **no AGPL appears anywhere** — nothing imports `ultralytics`.

**People.** These cameras look at hillsides from tens of kilometres away and nothing in
this system detects, tracks or identifies a person. No sample media contains an
identifiable person. The one place a person could appear, a live frame from a public
camera, is behind a switch that is off by default, and those frames expire with the job.

**Autonomy.** The system stops at the alert. It cannot dispatch, pan a camera or contact
anyone, and the interface's escalation buttons are deliberately inert and say so on
hover.

**Being a good neighbour to the data source.** The live path is off unless
`FIRSTSMOKE_LIVE=1`, and the download script uses four workers with a pause between
requests, because HPWREN's usage conditions ask users to be careful about the load they
impose on a small research team's infrastructure.

**Honesty as a design property, not a disclaimer.** A camera that cannot see reports
why rather than reporting clear; a crossing that cannot be trusted is refused with a
named reason and a nearest-approach measurement; a detection carries the eight named
terms that produced its score. The single most likely way this product causes harm is
by being believed when it should not be, and every one of those choices exists to make
that harder.
