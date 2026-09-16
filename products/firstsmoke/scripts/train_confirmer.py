#!/usr/bin/env python3
"""Train the smoke-confirmation classifier and export it to ONNX.

Why we train our own
--------------------
`cv2.dnn` in OpenCV 5 is ONNX-only: `readNetFromCaffe` and `readNetFromDarknet`
are gone. Of the smoke models published as ONNX, the ones with usable weights
come from the Ultralytics YOLO family, which is AGPL-3.0, and section 13 of the
AGPL makes a hosted demo API a source-disclosure event. YOLOX is Apache-2.0 and
loads cleanly in `cv2.dnn`, but its classes are COCO's and COCO has no smoke.

So we train a small convolutional classifier ourselves, on `pyronear/pyro-sdis`,
which is Apache-2.0 — a licence that permits derivative works, which matters,
because model weights trained on a no-derivatives dataset are at best a legal
argument nobody wants to have. HPWREN's FIgLib is CC BY-NC-ND and is therefore
used only to *evaluate*, never to train.

Positives are the annotated smoke boxes. Negatives are crops from the same
images, far from any box, so the classifier learns smoke against exactly the
backgrounds it will meet rather than against a different dataset's photographs.

Why the network is tiny
-----------------------
It is a confirmation step on a crop the classical pipeline has already
shortlisted, not a detector. Three convolutions and two dense layers is enough
to separate "grey translucent plume" from "ridge, cloud, shadow, road", and it
runs in about half a millisecond per crop through `cv2.dnn`. Training is plain
numpy — no torch, no CUDA, a clean checkout and two minutes.

Usage
-----
    python scripts/train_confirmer.py                 # train, evaluate, export
    python scripts/train_confirmer.py --epochs 40
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

CROP = 64
CACHE = Path.home() / ".cache" / "firstsmoke" / "pyro-sdis"
OUT = Path(__file__).resolve().parents[1] / "models"
MODEL_NAME = "smoke-confirm-v1.onnx"


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def read_labels(path: Path) -> list[tuple[float, float, float, float]]:
    """YOLO boxes as ``(cx, cy, w, h)``, normalised. Class id is ignored.

    pyro-sdis declares one class and then writes class id 1 for every box, which
    disagrees with its own data.yaml. Since there is exactly one class either
    way, the id carries no information and reading it would only propagate the
    upstream inconsistency.
    """
    if not path.is_file():
        return []
    boxes = []
    for line in path.read_text().split("\n"):
        parts = line.split()
        if len(parts) >= 5:
            boxes.append(tuple(float(p) for p in parts[1:5]))
    return boxes


def crops_from(image_path: Path, label_path: Path, rng: np.random.Generator):
    """Yield ``(crop, label)`` pairs: smoke boxes, then background patches."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return
    h, w = image.shape[:2]
    boxes = read_labels(label_path)

    taken: list[tuple[int, int, int, int]] = []
    for cx, cy, bw, bh in boxes:
        # Widen a little. The classical stage hands over a region grown by
        # morphology, so a crop cut exactly to the annotation is not what the
        # model will actually see at inference time.
        side = int(max(bw * w, bh * h) * 1.35)
        side = int(np.clip(side, 24, min(h, w)))
        x = int(np.clip(cx * w - side / 2, 0, w - side))
        y = int(np.clip(cy * h - side / 2, 0, h - side))
        taken.append((x, y, side, side))
        yield image[y : y + side, x : x + side], 1

    # Negatives from the same photograph, away from every box.
    wanted = max(1, len(boxes)) if boxes else 2
    tries = 0
    made = 0
    while made < wanted and tries < 40:
        tries += 1
        side = int(rng.integers(40, max(41, min(h, w) // 3)))
        x = int(rng.integers(0, max(1, w - side)))
        y = int(rng.integers(0, max(1, h - side)))
        if any(_overlaps((x, y, side, side), box) for box in taken):
            continue
        yield image[y : y + side, x : x + side], 0
        made += 1


def _overlaps(a, b, pad: int = 16) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (
        ax + aw + pad < bx or bx + bw + pad < ax or ay + ah + pad < by or by + bh + pad < ay
    )


def normalise(crop: np.ndarray) -> np.ndarray:
    """Exactly what `SmokeConfirmer.preprocess` does at inference time."""
    resized = cv2.resize(crop, (CROP, CROP), interpolation=cv2.INTER_AREA)
    values = resized.astype(np.float32)
    return (values - values.mean()) / (values.std() + 1e-3)


def build_set(images: Path, labels: Path, limit: int, seed: int):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for image_path in sorted(images.glob("*.jpg"))[:limit]:
        label_path = labels / f"{image_path.stem}.txt"
        for crop, label in crops_from(image_path, label_path, rng):
            if crop.size < 100:
                continue
            xs.append(normalise(crop))
            ys.append(label)
    if not xs:
        raise SystemExit(f"no crops built from {images}; run the pyro-sdis fetch first")
    return np.stack(xs)[:, None, :, :].astype(np.float32), np.array(ys, np.int64)


# --------------------------------------------------------------------------- #
# a very small convolutional network, in numpy
# --------------------------------------------------------------------------- #
def im2col(x: np.ndarray, k: int, pad: int, stride: int) -> tuple[np.ndarray, int, int]:
    n, c, h, w = x.shape
    if pad:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    oh = (h + 2 * pad - k) // stride + 1
    ow = (w + 2 * pad - k) // stride + 1
    strides = x.strides
    view = np.lib.stride_tricks.as_strided(
        x,
        shape=(n, c, oh, ow, k, k),
        strides=(strides[0], strides[1], strides[2] * stride, strides[3] * stride,
                 strides[2], strides[3]),
    )
    return view.transpose(0, 2, 3, 1, 4, 5).reshape(n * oh * ow, c * k * k), oh, ow


class Conv:
    def __init__(self, cin, cout, k, rng, pad=1, stride=1):
        fan_in = cin * k * k
        self.w = (rng.standard_normal((cout, cin, k, k)) * np.sqrt(2.0 / fan_in)).astype(np.float32)
        self.b = np.zeros(cout, np.float32)
        self.k, self.pad, self.stride = k, pad, stride
        self.cin, self.cout = cin, cout

    def forward(self, x):
        self.x_shape = x.shape
        cols, oh, ow = im2col(x, self.k, self.pad, self.stride)
        self.cols = cols
        self.oh, self.ow = oh, ow
        flat = self.w.reshape(self.cout, -1).T
        out = cols @ flat + self.b
        return out.reshape(x.shape[0], oh, ow, self.cout).transpose(0, 3, 1, 2)

    def backward(self, grad, lr):
        n = self.x_shape[0]
        g = grad.transpose(0, 2, 3, 1).reshape(-1, self.cout)
        dw = (self.cols.T @ g).T.reshape(self.w.shape)
        db = g.sum(axis=0)
        dcols = g @ self.w.reshape(self.cout, -1)
        self.w -= lr * dw / n
        self.b -= lr * db / n
        return col2im(dcols, self.x_shape, self.k, self.pad, self.stride, self.oh, self.ow)


def col2im(cols, x_shape, k, pad, stride, oh, ow):
    n, c, h, w = x_shape
    padded = np.zeros((n, c, h + 2 * pad, w + 2 * pad), np.float32)
    reshaped = cols.reshape(n, oh, ow, c, k, k).transpose(0, 3, 1, 2, 4, 5)
    for i in range(k):
        for j in range(k):
            padded[:, :, i : i + oh * stride : stride, j : j + ow * stride : stride] += (
                reshaped[:, :, :, :, i, j]
            )
    return padded[:, :, pad : pad + h, pad : pad + w] if pad else padded


class MaxPool:
    def forward(self, x):
        n, c, h, w = x.shape
        h2, w2 = h // 2, w // 2
        view = x[:, :, : h2 * 2, : w2 * 2].reshape(n, c, h2, 2, w2, 2)
        self.x_shape = x.shape
        flat = view.transpose(0, 1, 2, 4, 3, 5).reshape(n, c, h2, w2, 4)
        self.argmax = flat.argmax(axis=-1)
        return flat.max(axis=-1)

    def backward(self, grad, lr=0.0):
        n, c, h, w = self.x_shape
        h2, w2 = h // 2, w // 2
        out = np.zeros((n, c, h2, w2, 4), np.float32)
        idx = np.indices((n, c, h2, w2))
        out[idx[0], idx[1], idx[2], idx[3], self.argmax] = grad
        out = out.reshape(n, c, h2, w2, 2, 2).transpose(0, 1, 2, 4, 3, 5)
        full = np.zeros(self.x_shape, np.float32)
        full[:, :, : h2 * 2, : w2 * 2] = out.reshape(n, c, h2 * 2, w2 * 2)
        return full


class Dense:
    def __init__(self, cin, cout, rng):
        self.w = (rng.standard_normal((cin, cout)) * np.sqrt(2.0 / cin)).astype(np.float32)
        self.b = np.zeros(cout, np.float32)

    def forward(self, x):
        self.x = x
        return x @ self.w + self.b

    def backward(self, grad, lr):
        n = self.x.shape[0]
        dw = self.x.T @ grad
        db = grad.sum(axis=0)
        dx = grad @ self.w.T
        self.w -= lr * dw / n
        self.b -= lr * db / n
        return dx


class Relu:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad, lr=0.0):
        return grad * self.mask


class Net:
    """conv8 -> pool -> conv16 -> pool -> conv24 -> pool -> dense32 -> dense2."""

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.c1, self.r1, self.p1 = Conv(1, 8, 3, rng), Relu(), MaxPool()
        self.c2, self.r2, self.p2 = Conv(8, 16, 3, rng), Relu(), MaxPool()
        self.c3, self.r3, self.p3 = Conv(16, 24, 3, rng), Relu(), MaxPool()
        self.d1, self.r4 = Dense(24 * 8 * 8, 32, rng), Relu()
        self.d2 = Dense(32, 2, rng)
        self.layers = [self.c1, self.r1, self.p1, self.c2, self.r2, self.p2,
                       self.c3, self.r3, self.p3]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        self.shape = x.shape
        x = x.reshape(x.shape[0], -1)
        return self.d2.forward(self.r4.forward(self.d1.forward(x)))

    def backward(self, grad, lr):
        grad = self.d1.backward(self.r4.backward(self.d2.backward(grad, lr)), lr)
        grad = grad.reshape(self.shape)
        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def train(net: Net, x, y, *, epochs: int, batch: int, lr: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(x)
    for epoch in range(epochs):
        order = rng.permutation(n)
        losses = []
        for start in range(0, n, batch):
            idx = order[start : start + batch]
            logits = net.forward(x[idx])
            probs = softmax(logits)
            losses.append(float(-np.log(probs[np.arange(len(idx)), y[idx]] + 1e-9).mean()))
            grad = probs
            grad[np.arange(len(idx)), y[idx]] -= 1.0
            net.backward(grad, lr)
        yield epoch, float(np.mean(losses))


def evaluate(net: Net, x, y, batch: int = 256):
    probs = np.concatenate(
        [softmax(net.forward(x[i : i + batch])) for i in range(0, len(x), batch)]
    )
    p = probs[:, 1]
    predicted = (p >= 0.5).astype(np.int64)
    tp = int(((predicted == 1) & (y == 1)).sum())
    fp = int(((predicted == 1) & (y == 0)).sum())
    fn = int(((predicted == 0) & (y == 1)).sum())
    tn = int(((predicted == 0) & (y == 0)).sum())
    return {
        "accuracy": (tp + tn) / max(len(y), 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "positives": int((y == 1).sum()), "negatives": int((y == 0).sum()),
    }


# --------------------------------------------------------------------------- #
# ONNX export
# --------------------------------------------------------------------------- #
def export_onnx(net: Net, path: Path) -> Path:
    """Write the trained weights out as an ONNX graph `cv2.dnn` will load.

    The graph is built by hand rather than traced, because the training code is
    numpy and there is nothing to trace. Only operators OpenCV 5's ONNX importer
    supports are used: Conv, Relu, MaxPool, Reshape, Gemm, Softmax.
    """
    from onnx import TensorProto, helper, numpy_helper, save

    initialisers = []
    nodes = []

    def add_weight(name: str, array: np.ndarray) -> str:
        initialisers.append(numpy_helper.from_array(array.astype(np.float32), name))
        return name

    current = "input"
    for i, conv in enumerate((net.c1, net.c2, net.c3), start=1):
        w = add_weight(f"conv{i}_w", conv.w)
        b = add_weight(f"conv{i}_b", conv.b)
        nodes.append(helper.make_node(
            "Conv", [current, w, b], [f"conv{i}"], kernel_shape=[3, 3], pads=[1, 1, 1, 1],
            strides=[1, 1], name=f"conv{i}",
        ))
        nodes.append(helper.make_node("Relu", [f"conv{i}"], [f"relu{i}"], name=f"relu{i}"))
        nodes.append(helper.make_node(
            "MaxPool", [f"relu{i}"], [f"pool{i}"], kernel_shape=[2, 2], strides=[2, 2],
            name=f"pool{i}",
        ))
        current = f"pool{i}"

    shape_name = add_weight("flat_shape", np.array([-1, 24 * 8 * 8], np.float32))
    initialisers[-1] = numpy_helper.from_array(np.array([-1, 24 * 8 * 8], np.int64), "flat_shape")
    nodes.append(helper.make_node("Reshape", [current, shape_name], ["flat"], name="flatten"))

    d1w = add_weight("d1_w", net.d1.w)
    d1b = add_weight("d1_b", net.d1.b)
    nodes.append(helper.make_node("Gemm", ["flat", d1w, d1b], ["dense1"], name="dense1"))
    nodes.append(helper.make_node("Relu", ["dense1"], ["dense1_relu"], name="dense1_relu"))

    d2w = add_weight("d2_w", net.d2.w)
    d2b = add_weight("d2_b", net.d2.b)
    nodes.append(helper.make_node("Gemm", ["dense1_relu", d2w, d2b], ["logits"], name="dense2"))
    nodes.append(helper.make_node("Softmax", ["logits"], ["output"], axis=1, name="softmax"))

    graph = helper.make_graph(
        nodes, "firstsmoke-confirm",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, CROP, CROP])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])],
        initialisers,
    )
    model = helper.make_model(
        graph, producer_name="firstsmoke",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 9
    path.parent.mkdir(parents=True, exist_ok=True)
    save(model, str(path))
    return path


def verify_through_opencv(path: Path, x, y) -> dict:
    """Load the exported file the way the product will, and check it agrees."""
    net = cv2.dnn.readNetFromONNX(str(path))
    probs = []
    start = time.perf_counter()
    for sample in x:
        net.setInput(sample[None, :, :, :].astype(np.float32))
        probs.append(float(np.asarray(net.forward()).reshape(-1)[-1]))
    elapsed = (time.perf_counter() - start) / max(len(x), 1) * 1000.0
    predicted = (np.array(probs) >= 0.5).astype(np.int64)
    return {
        "accuracy_through_cv2_dnn": float((predicted == y).mean()),
        "ms_per_crop": round(elapsed, 3),
        "samples": len(x),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__[:200])
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--out", type=Path, default=OUT / MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.06)
    parser.add_argument("--train-images", type=int, default=2600)
    parser.add_argument("--val-images", type=int, default=400)
    args = parser.parse_args(argv)

    print("building crops from pyro-sdis (Apache-2.0)")
    x_train, y_train = build_set(
        args.cache / "images" / "train", args.cache / "labels" / "train", args.train_images, seed=1
    )
    x_val, y_val = build_set(
        args.cache / "val_images", args.cache / "val_labels", args.val_images, seed=2
    )
    print(f"  train {len(x_train)} crops ({int(y_train.sum())} smoke)")
    print(f"  val   {len(x_val)} crops ({int(y_val.sum())} smoke)")

    net = Net(seed=0)
    started = time.perf_counter()
    schedule = train(
        net, x_train, y_train, epochs=args.epochs, batch=args.batch, lr=args.lr
    )
    for epoch, loss in schedule:
        if epoch % 4 == 0 or epoch == args.epochs - 1:
            scores = evaluate(net, x_val, y_val)
            print(
                f"  epoch {epoch:2d}  loss {loss:.4f}  val acc {scores['accuracy']:.3f}  "
                f"precision {scores['precision']:.3f}  recall {scores['recall']:.3f}"
            )
    minutes = (time.perf_counter() - started) / 60.0

    scores = evaluate(net, x_val, y_val)
    path = export_onnx(net, args.out)
    checked = verify_through_opencv(path, x_val[:200], y_val[:200])

    report = {
        "model": path.name,
        "trained_on": "pyronear/pyro-sdis (Apache-2.0)",
        "evaluated_on": "pyronear/pyro-sdis validation split",
        "train_crops": len(x_train),
        "val_crops": len(x_val),
        "epochs": args.epochs,
        "training_minutes": round(minutes, 2),
        "parameters": sum(
            layer.w.size + layer.b.size for layer in (net.c1, net.c2, net.c3, net.d1, net.d2)
        ),
        "validation": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in scores.items()},
        "through_cv2_dnn": checked,
        "opencv_version": cv2.__version__,
    }
    (path.parent / "smoke-confirm-v1.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nwrote {path} ({path.stat().st_size / 1024:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
