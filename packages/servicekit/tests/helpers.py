"""Analyzers and app builders for the servicekit suite.

Kept out of conftest.py so the module basename is unique workspace-wide.
"""

from __future__ import annotations

import time

import cv2
from servicekit import JobContext, ProductInfo, ServiceConfig, create_app
from visioncore import encode_jpeg


def demo_analyzer(ctx: JobContext):
    """A small but real analyzer: decodes, counts contours, attaches evidence."""
    from visioncore import decode_image, stage

    ctx.progress(10, "decoding")
    with stage("decode"):
        image = decode_image(ctx.input_path.read_bytes())

    ctx.progress(50, "thresholding")
    with stage("threshold"):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    with stage("contours"):
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ctx.note("found some shapes")
    ctx.record.metrics["contour_count"] = len(contours)
    ctx.record.metrics["mean_intensity"] = round(float(gray.mean()), 2)
    ctx.add_evidence("threshold", encode_jpeg(binary), caption="binary mask")
    if int(ctx.params.get("refuse", 0)):
        ctx.record.refuse("TOO_OBLIQUE", "synthetic refusal", obliquity_deg=61.0)
    ctx.progress(100, "done")
    return ctx.record


def exploding_analyzer(ctx: JobContext):
    raise RuntimeError("analyzer blew up")


def slow_analyzer(ctx: JobContext):
    for i in range(5):
        ctx.progress(i * 20, f"step {i}")
        time.sleep(0.02)
    ctx.record.metrics["done"] = True
    return ctx.record


def build_app(analyzer=demo_analyzer, tmp_path=None, **config_kwargs):
    product = ProductInfo(
        slug="demo", title="Demo Product", tagline="a foundation smoke test",
        accent="#ff6b35",
    )
    config = ServiceConfig(
        product=product,
        upload_dir=tmp_path / "uploads" if tmp_path else "/tmp/opencv26-tests",
        params_schema=[
            {"name": "threshold", "type": "number", "default": 127, "label": "Threshold"},
            {"name": "refuse", "type": "number", "default": 0},
        ],
        **config_kwargs,
    )
    return create_app(config, analyzer)


