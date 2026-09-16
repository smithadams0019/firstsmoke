#!/usr/bin/env python3
"""Fetch a subset of HPWREN's FIgLib into a local cache.

FIgLib is the Fire Ignition images Library: 525 recorded sequences from HPWREN's
mountain-top cameras, each one a run of stills around a real ignition. Filenames
carry the ground truth:

    <unix_epoch>_<offset_seconds>.jpg

where the offset is signed seconds from the moment a human marked the plume as
first visible. Negative frames are labelled clear, zero and positive frames are
labelled smoke. That gives us a detection rate, a time-to-alert measured against
a human's own first sighting, and a supply of genuine negatives from the same
cameras, weather and time of day — which is the only kind of negative worth
counting.

Licence and manners
-------------------
HPWREN data is CC BY-NC-ND 4.0 (https://hpwren.ucsd.edu/cc.html) and FIgLib's
own page asks for "a credit reference to https://www.hpwren.ucsd.edu/ in
derivative work". This script therefore:

* caches **outside** the repository, so no HPWREN image is redistributed by us;
* downloads with only a handful of parallel readers and a pause, because this is
  a research network run by a small team whose usage conditions ask users to be
  careful about the load they impose;
* skips anything already on disk.

Run it with no arguments for the evaluation set used in docs/evaluation.md.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

BASE = "https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data"
CACHE = Path.home() / ".cache" / "firstsmoke" / "figlib"
USER_AGENT = "firstsmoke/0.1 (OpenCV AI Competition 2026 entry; research use)"
FILE_RE = re.compile(r"(\d{10})_([+-]?\d+)\.jpg")
"""Offsets carry an explicit sign: ``_-02400`` before ignition, ``_+02400`` after.
A pattern that forgets the plus downloads only the negatives, and leaves you with
an evaluation set containing no fires that still looks like it worked."""

# The incidents the evaluation uses. The first group are dates where two or more
# cameras on *different* summits recorded the same fire, which is what makes a
# crossed-bearing test possible at all. The second group are single-camera
# sequences, included so the detection numbers are not dominated by a handful of
# large, well-seen fires.
CROSS_CAMERA = [
    "20191001_FIRE_bh-w-mobo-c", "20191001_FIRE_lp-s-mobo-c",
    "20191001_FIRE_om-e-mobo-c", "20191001_FIRE_om-s-mobo-c",
    "20191001_FIRE_rm-w-mobo-c",
    "20200911_FIRE_lp-e-mobo-c", "20200911_FIRE_mlo-s-mobo-c",
    "20200911_FIRE_pi-s-mobo-c",
    "20180727_FIRE_bh-n-mobo-c", "20180727_FIRE_bh-s-mobo-c",
    "20180727_FIRE_bl-e-mobo-c", "20180727_FIRE_mg-w-mobo-c",
    "20180727_FIRE_wc-n-mobo-c",
    "20190829_FIRE_bl-n-mobo-c", "20190829_FIRE_pi-e-mobo-c",
    "20190829_FIRE_rm-w-mobo-c", "20190829_FIRE_smer-tcs8-mobo-c",
    "20171010_FIRE_hp-n-mobo-c", "20171010_FIRE_hp-w-mobo-c",
    "20171010_FIRE_rm-e-mobo-c",
    "20191005_FIRE_bm-e-mobo-c", "20191005_FIRE_hp-s-mobo-c",
    "20191005_FIRE_vo-n-mobo-c", "20191005_FIRE_wc-e-mobo-c",
    "20191005_FIRE_wc-n-mobo-c",
    "20160604_FIRE_rm-n-mobo-c", "20160604_FIRE_smer-tcs3-mobo-c",
    "20190717_FIRE_lp-n-mobo-c", "20190717_FIRE_pi-w-mobo-c",
    "20200705_FIRE_bm-w-mobo-c", "20200705_FIRE_wc-n-mobo-c",
    "20180806_FIRE_mg-s-mobo-c", "20180806_FIRE_vo-w-mobo-c",
]
SINGLE_CAMERA = [
    "20170519_FIRE_rm-w-mobo-c", "20170609_FIRE_sm-n-mobo-c",
    "20170807_FIRE_bh-n-mobo-c", "20170821_FIRE_lo-s-mobo-c",
    "20170826_FIRE_tp-s-mobo-c", "20170901_FIRE_om-s-mobo-c",
    "20171016_FIRE_sdsc-e-mobo-c", "20171021_FIRE_pi-e-mobo-c",
    "20180517_FIRE_rm-n-mobo-c", "20180522_FIRE_rm-e-mobo-c",
    "20180614_FIRE_hp-s-mobo-c", "20180718_FIRE_syp-w-mobo-c",
    "20180723_FIRE_tp-e-mobo-c", "20180919_FIRE_rm-e-mobo-c",
    "20190610_FIRE_bh-w-mobo-c", "20190629_FIRE_hp-n-mobo-c",
    "20190712_FIRE_om-e-mobo-c", "20190805_FIRE_sp-e-mobo-c",
    "20190813_FIRE_69bravo-e-mobo-c", "20190825_FIRE_sm-w-mobo-c",
    "20190913_FIRE_lp-n-mobo-c", "20190922_FIRE_ml-w-mobo-c",
    "20200202_FIRE_hp-w-mobo-c", "20200226_FIRE_rm-e-mobo-c",
    "20200521_FIRE_om-n-mobo-c", "20200618_FIRE_om-w-mobo-c",
    "20200831_FIRE_wc-n-mobo-c", "20201013_FIRE_cp-s-mobo-c",
    "20210204_FIRE_tp-s-mobo-c", "20210319_FIRE_om-n-mobo-c",
    "20210711_FIRE_wc-e-mobo-c", "20240907_FIRE_ws-s-mobo-c",
]
DEFAULT_SET = CROSS_CAMERA + SINGLE_CAMERA


def frame_url(sequence: str, name: str) -> str:
    """Build a frame URL, percent-encoding the plus in a post-ignition offset.

    Post-ignition frames are named ``<epoch>_+00300.jpg``. A bare ``+`` in the
    path is read as a space by the CDN in front of FIgLib and answered with a
    403, so every positive frame silently disappears from the set. Quoting it as
    ``%2B`` is the difference between an evaluation set with fires in it and one
    without.
    """
    return f"{BASE}/{sequence}/{quote(name, safe='')}"


def get(url: str, timeout: float = 60.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def list_sequence(sequence: str) -> list[tuple[str, int, int]]:
    """Return ``(filename, epoch, offset_seconds)`` for every frame in a sequence."""
    html = get(f"{BASE}/{sequence}/index.html").decode("utf-8", "replace")
    out = []
    for match in FILE_RE.finditer(html):
        out.append((match.group(0), int(match.group(1)), int(match.group(2))))
    return sorted(set(out), key=lambda item: item[2])


def downscale(data: bytes, width: int) -> bytes:
    """Resize on the way in, so the cache is a tenth of the size.

    FIgLib frames are 2048x1536 at about 700 kB. The pipeline works at 1024 px,
    so storing the originals would cost 20 GB to hold pixels we immediately
    throw away. If OpenCV is not importable the bytes pass through untouched.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return data
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape[1] <= width:
        return data
    scale = width / float(image.shape[1])
    small = cv2.resize(
        image, (width, max(1, round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA
    )
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes() if ok else data


def fetch_sequence(
    sequence: str,
    *,
    cache: Path,
    width: int,
    stride: int,
    window_s: int,
    pause: float,
    workers: int,
) -> tuple[int, int]:
    target = cache / sequence
    target.mkdir(parents=True, exist_ok=True)
    try:
        listing = list_sequence(sequence)
    except urllib.error.HTTPError as exc:
        print(f"  {sequence}: index unavailable ({exc.code})", file=sys.stderr)
        return 0, 0
    wanted = [
        item for i, item in enumerate(listing)
        if abs(item[2]) <= window_s and i % stride == 0
    ]
    missing = [
        (name, target / name)
        for name, _epoch, _offset in wanted
        if not ((target / name).exists() and (target / name).stat().st_size > 1024)
    ]
    skipped = len(wanted) - len(missing)

    def one(item: tuple[str, Path]) -> bool:
        name, path = item
        try:
            data = get(frame_url(sequence, name))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"  {sequence}/{name}: {exc}", file=sys.stderr)
            return False
        path.write_bytes(downscale(data, width))
        time.sleep(pause)
        return True

    # A handful of parallel readers, not a swarm. Four keeps a 5,000-frame pull
    # inside an hour while staying well under what a shared research CDN would
    # notice; the HPWREN usage conditions ask users to be careful with the load
    # they impose, and that is a reasonable reading of careful.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        fetched = sum(1 for ok in pool.map(one, missing) if ok)
    return fetched, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("sequences", nargs="*", default=None, help="sequence directory names")
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=1, help="keep every Nth frame")
    parser.add_argument("--window", type=int, default=2400, help="seconds either side of ignition")
    parser.add_argument(
        "--pause", type=float, default=0.1, help="seconds a worker waits after each file"
    )
    parser.add_argument("--workers", type=int, default=4, help="parallel downloads")
    parser.add_argument("--list", action="store_true", help="print the default set and exit")
    args = parser.parse_args(argv)

    if args.list:
        for name in DEFAULT_SET:
            print(name)
        return 0

    sequences = args.sequences or DEFAULT_SET
    args.cache.mkdir(parents=True, exist_ok=True)
    total_fetched = total_skipped = 0
    for i, sequence in enumerate(sequences, 1):
        fetched, skipped = fetch_sequence(
            sequence,
            cache=args.cache,
            width=args.width,
            stride=args.stride,
            window_s=args.window,
            pause=args.pause,
            workers=args.workers,
        )
        total_fetched += fetched
        total_skipped += skipped
        print(f"[{i}/{len(sequences)}] {sequence}: +{fetched} new, {skipped} cached", flush=True)
    print(f"\n{total_fetched} frames downloaded, {total_skipped} already present, in {args.cache}")
    print("Imagery: HPWREN, University of California San Diego. http://hpwren.ucsd.edu")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
