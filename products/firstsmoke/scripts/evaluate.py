#!/usr/bin/env python3
"""Run the detector over the cached FIgLib sequences and write the numbers.

    python scripts/evaluate.py                       # the whole cache
    python scripts/evaluate.py --limit 6             # a quick pass
    python scripts/evaluate.py --threshold 0.5       # a different operating point
    python scripts/evaluate.py --sweep               # detection rate against
                                                     # false positives per camera-day

Fetch the imagery first with `scripts/fetch_figlib.py`. Nothing here downloads
anything; if the cache is empty it says so rather than quietly reporting a
detection rate over zero fires.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from firstsmoke.confirm import load_confirmer
from firstsmoke.evaluate import FIGLIB_CACHE, evaluate_cache

OUT = Path(__file__).resolve().parents[1] / "docs" / "evaluation.json"


def show(index: int, total: int, name: str, result) -> None:
    mark = "hit " if result.detected else "MISS"
    offset = result.first_alert_offset_s
    when = f"{offset:+5d}s" if offset is not None else "   -  "
    print(
        f"[{index:2d}/{total}] {mark} {when}  fp {result.false_positive_frames:2d}"
        f"  peak {result.peak_confidence:.2f}  {name}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="evaluate the detector on cached FIgLib sequences")
    parser.add_argument("--cache", type=Path, default=FIGLIB_CACHE)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--stride", type=int, default=1, help="use every Nth frame")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-model", action="store_true", help="skip the ONNX confirmer")
    args = parser.parse_args(argv)

    if not args.cache.is_dir() or not any(args.cache.glob("*_FIRE_*")):
        print(
            f"no cached sequences in {args.cache}.\n"
            "Run scripts/fetch_figlib.py first; it downloads HPWREN's FIgLib into a local cache.",
            file=sys.stderr,
        )
        return 2

    confirmer = None if args.no_model else load_confirmer()
    print("confirmer:", confirmer.path.name if confirmer else "not loaded, classical evidence only")

    started = time.perf_counter()
    evaluation = evaluate_cache(
        args.cache,
        threshold=args.threshold,
        stride=args.stride,
        limit=args.limit,
        confirmer=confirmer,
        progress=show,
    )
    elapsed = time.perf_counter() - started

    payload = evaluation.to_dict()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))

    print(f"\n{len(evaluation.results)} sequences in {elapsed:.0f} s\n")
    print(f"{'threshold':>9}  {'detection':>9}  {'median':>7}  {'fp':>9}  {'fp/cam-day':>10}")
    for row in evaluation.operating_curve():
        at = row["median_time_to_alert_s"]
        median = "-" if at is None else f"{at}s"
        print(
            f"{row['threshold']:>9.2f}  {row['detection_rate'] * 100:>8.1f}%  {median:>7}  "
            f"{row['false_positive_frames']:>9d}  {row['false_positives_per_camera_day']:>10.1f}"
        )

    corroboration = payload.get("corroboration_curve") or []
    if corroboration:
        print(
            "\nWhat the second camera buys, on the dates where two summits saw the same fire:"
        )
        print(f"{'threshold':>9}  {'one camera':>22}  {'corroborated':>22}")
        print(f"{'':>9}  {'detect   fp/cam-day':>22}  {'detect   fp/cam-day':>22}")
        for row in corroboration:
            one, both = row["one_camera"], row["corroborated"]
            one_rate = one["detection_rate"] * 100
            both_rate = both["detection_rate"] * 100
            print(
                f"{row['threshold']:>9.2f}  "
                f"{one_rate:>6.1f}%  {one['false_positives_per_camera_day']:>11.1f}  "
                f"{both_rate:>6.1f}%  {both['false_positives_per_camera_day']:>11.1f}"
            )
    print("\n" + json.dumps(evaluation.summary(), indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
