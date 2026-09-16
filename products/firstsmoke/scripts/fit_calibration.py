#!/usr/bin/env python3
"""Fit the per-camera calibrations the deployed service uses.

    python scripts/fit_calibration.py --variant map

This is deliberately a different script from scripts/evaluate.py. The
evaluation fits each camera from its *other* dates so the calibration never sees
the day it is scored on, which means none of the calibrations it produces used
all of that camera's history. A deployed service has no such constraint and
should learn from everything it has seen, so this fits each camera from all of
its clear frames.

The numbers in docs/evaluation.md come from the held-out fits, never from this.
Only clear frames (FIgLib offset below zero) are read; no frame containing a fire
contributes to any calibration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from firstsmoke.calibration import CalibrationSet, ClearFrameEvidence, combine
from firstsmoke.evaluate import (
    FIGLIB_CACHE,
    VARIANTS,
    default_network,
    load_figlib_sequence,
    run_sequence,
)

OUT = Path(__file__).resolve().parents[1] / "docs" / "calibration.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="fit deployment calibrations")
    parser.add_argument("--cache", type=Path, default=FIGLIB_CACHE)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="map")
    args = parser.parse_args(argv)

    network = default_network()
    evidence: list[ClearFrameEvidence] = []
    for directory in sorted(d for d in args.cache.glob("*_FIRE_*") if d.is_dir()):
        loaded = load_figlib_sequence(directory, network)
        if loaded is None:
            continue
        sequence, labelled = loaded
        clear = [item for item in labelled if not item.is_smoke]
        result = run_sequence(sequence, clear)
        if result.evidence is not None:
            result.evidence.sequence = directory.name
            evidence.append(result.evidence)
        print(f"{directory.name}: {result.evidence.frames if result.evidence else 0} clear frames",
              flush=True)

    use_nuisance, use_ceiling = VARIANTS[args.variant]
    calibrations = CalibrationSet()
    for camera_id in sorted({e.camera_id for e in evidence}):
        fitted = combine(evidence, camera_id, use_nuisance=use_nuisance, use_ceiling=use_ceiling)
        if fitted is not None:
            calibrations.add(fitted)
    calibrations.save(args.out)
    print(json.dumps({"variant": args.variant, **calibrations.summary()}, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
