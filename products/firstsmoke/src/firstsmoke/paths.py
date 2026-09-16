"""Where the runtime finds its files, in a checkout and in a container.

In a checkout everything sits two directories above the package. In the image
the package is installed into site-packages and the data is copied to /app, so
every path is also settable from the environment. The container sets all three;
a developer setting none gets the checkout layout.
"""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def _resolve(env: str, *relative: str) -> Path:
    override = os.environ.get(env)
    return Path(override) if override else PACKAGE_ROOT.joinpath(*relative)


def static_dir() -> Path:
    return _resolve("FIRSTSMOKE_STATIC", "static")


def models_dir() -> Path:
    return _resolve("FIRSTSMOKE_MODELS", "models")


def scenario_dir() -> Path:
    return _resolve("FIRSTSMOKE_SCENARIOS", "data", "scenarios")


def network_file() -> Path:
    return _resolve("FIRSTSMOKE_NETWORK", "data", "network", "hpwren_sites.js")
