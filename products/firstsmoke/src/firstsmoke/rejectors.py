"""The things that are not smoke, and how each one gives itself away.

Every wildfire camera detector that has ever been fielded fails in the same
place: it works, and then it cries wolf four times a night until the people who
were supposed to act on it stop reading the alerts. So the rejectors are not a
post-filter bolted onto a classifier. They are the product.

Each rule below names one impostor and one measurable way it differs from a
column of smoke. Each returns a :class:`Rejection` carrying the numbers that
fired it, so an operator sees *why* the system stood down, and so a false
negative can be traced to the exact rule that caused it.

A rule fires on the evidence, not on a hunch. Where a rule cannot be certain, it
returns ``None`` and lets the score decide — a rejector that guesses is a false
negative generator, and in this domain a false negative is a fire.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .cameras import Camera
from .candidates import Region
from .scene import Alignment, SceneState, Usability
from .tracks import Growth, Track


@dataclass(frozen=True)
class Rejection:
    """A named reason this candidate is not a smoke column."""

    code: str
    impostor: str
    """Plain-language name of what we think it actually is."""
    message: str
    details: dict[str, Any]
    confidence: float = 0.8
    """How sure the rule is. Below 0.5 the detector downweights rather than drops."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "impostor": self.impostor,
            "message": self.message,
            "confidence": round(self.confidence, 3),
            "details": {
                k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.details.items()
            },
        }


# --------------------------------------------------------------------------- #
# thresholds, in one place
# --------------------------------------------------------------------------- #
CLOUD_ANCHOR_MAX = 0.28
CLOUD_DRIFT_MIN_PX_MIN = 6.0
CLOUD_AREA_RATIO = (0.55, 1.8)
AIRBORNE_MARGIN_FRACTION = 0.05
"""A base this far above the ridge, as a fraction of frame height, means the
thing is not attached to the ground."""
DUST_RISE_MAX_PX_MIN = 2.0
DUST_DRIFT_MIN_PX_MIN = 5.0
DUST_GREYNESS_MAX = 0.42
STATIC_DRIFT_MAX_PX_MIN = 0.6
STATIC_AREA_RATIO = (0.85, 1.18)
STATIC_EDGE_SOFTNESS_MAX = 0.25
STATIC_JITTER_MAX_PX = 1.5
"""A plume's mask breathes by several pixels a frame as its thin skirt crosses
the detection threshold. A droplet on the glass does not breathe at all."""
FLARE_LUMA_MIN = 238.0
FLARE_SATURATED_FRACTION = 0.25
ERRATIC_BASE_DRIFT_PX_MIN = 12.0
"""A fire does not move. Neither, in the image, does the bottom of its plume:
across the whole FIgLib evaluation set the base of a real column drifts under
5 px a minute at the working resolution, and most of that is the mask breathing
rather than the base going anywhere. Anything whose base is travelling faster
than this is a registration artefact, a shadow edge sweeping across a slope, or
the tracker having jumped between two different things."""
ERRATIC_ANCHOR_MAX = 0.55

SHAKE_UNCORRECTED_PX = 3.0
RIDGE_BAND_PX = 14
GLOBAL_TEXTURE_COLLAPSE = 0.55

HABITUAL_REJECT_AT = 0.35
"""A candidate sitting where this camera raises something on at least this
fraction of its clear frames is habitual rather than new."""
BASELINE_MARGIN = 1.0
"""How far above a camera's own clear-frame 95th percentile a detection must sit
before it counts as unusual for that camera. A multiplier of 1.0 means "beat
what this camera reaches on a clear day"."""


def reject_cloud(growth: Growth, region: Region, frame_height: int) -> Rejection | None:
    """Cloud translates bodily; a column does not.

    The discriminator is the anchor score: a plume's base is nailed to the
    ignition point while its top climbs, so rise dominates base drift. A cloud
    moves all of itself at the wind speed, so rise and base drift are equal and
    the score sits near a half; a low cloud with a flat top scores lower still.
    Area is the corroborating test — cloud that is merely passing keeps its size.
    """
    airborne = region.base_below_horizon < -AIRBORNE_MARGIN_FRACTION * frame_height
    if (
        growth.frames >= 3
        and growth.anchor_score < CLOUD_ANCHOR_MAX
        and growth.centroid_drift_px_per_min > CLOUD_DRIFT_MIN_PX_MIN
        and CLOUD_AREA_RATIO[0] <= growth.area_ratio <= CLOUD_AREA_RATIO[1]
    ):
        return Rejection(
            "CLOUD_TRANSLATION",
            "cloud",
            f"the whole shape moved {growth.centroid_drift_px_per_min:.1f} px a minute while its "
            f"area changed by only {abs(1 - growth.area_ratio) * 100:.0f}%, and its base moved as "
            f"fast as its top (anchor {growth.anchor_score:.2f}); that is something passing "
            "through, not something growing out of the ground",
            {
                "anchor_score": growth.anchor_score,
                "centroid_drift_px_per_min": growth.centroid_drift_px_per_min,
                "area_ratio": growth.area_ratio,
                "airborne": airborne,
            },
            confidence=0.85 if airborne else 0.7,
        )
    return None


def reject_airborne(region: Region, frame_height: int, growth: Growth) -> Rejection | None:
    """Nothing that starts in mid-air is a new ignition.

    A plume has to touch the ground somewhere. We allow a generous margin,
    because the ridge estimate has its own error and a plume behind a ridge is
    genuinely detached in the image — but a base a twentieth of the frame above
    the skyline, with no growth downward toward it, is a cloud.
    """
    margin = AIRBORNE_MARGIN_FRACTION * frame_height
    if region.base_below_horizon < -margin and growth.top_rise_px_per_min < 3.0:
        return Rejection(
            "AIRBORNE",
            "cloud or contrail",
            f"the bottom of this shape sits {-region.base_below_horizon:.0f} px above the skyline "
            "with nothing joining it to the ground",
            {
                "base_below_horizon": region.base_below_horizon,
                "margin_px": margin,
                "top_rise_px_per_min": growth.top_rise_px_per_min,
            },
            confidence=0.75,
        )
    return None


def reject_dust(growth: Growth, region: Region) -> Rejection | None:
    """Dust runs along the ground; smoke leaves it.

    A vehicle on a fire road throws a plume that is the right colour, the right
    size and in the right place. What it does not do is rise. It also carries the
    colour of the soil, so it is measurably less neutral than wood smoke.
    """
    if (
        growth.frames >= 3
        and growth.top_rise_px_per_min < DUST_RISE_MAX_PX_MIN
        and growth.horizontal_drift_px_per_min > DUST_DRIFT_MIN_PX_MIN
        and region.base_below_horizon > 0
        and region.greyness < DUST_GREYNESS_MAX
    ):
        return Rejection(
            "GROUND_DRIFT",
            "dust or vehicle plume",
            f"it tracked sideways at {growth.horizontal_drift_px_per_min:.1f} px a minute below "
            f"the skyline without rising ({growth.top_rise_px_per_min:.1f} px a minute), and its "
            f"colour is not neutral (greyness {region.greyness:.2f})",
            {
                "top_rise_px_per_min": growth.top_rise_px_per_min,
                "horizontal_drift_px_per_min": growth.horizontal_drift_px_per_min,
                "greyness": region.greyness,
            },
            confidence=0.7,
        )
    return None


def reject_erratic(growth: Growth) -> Rejection | None:
    """The base is running around. Whatever it is, it is not rooted to the ground.

    This rule catches the residue that the named impostors miss: warm-up
    artefacts, a shadow edge crossing a slope, and the tracker stitching two
    unrelated blobs into one track. Without it, those scored high enough to send
    the agent off to consult neighbours about a bearing with nothing on it, which
    is worse than a plain false positive because it also wastes the consultation
    budget the real detection needed.
    """
    if (
        growth.frames >= 3
        and growth.base_drift_px_per_min > ERRATIC_BASE_DRIFT_PX_MIN
        and growth.anchor_score < ERRATIC_ANCHOR_MAX
    ):
        return Rejection(
            "ERRATIC_BASE",
            "a moving shadow or a tracker mistake",
            f"the bottom of this shape travelled {growth.base_drift_px_per_min:.0f} px a minute; "
            "the base of a real column stays where the fire is",
            {
                "base_drift_px_per_min": growth.base_drift_px_per_min,
                "anchor_score": growth.anchor_score,
                "frames": growth.frames,
            },
            confidence=0.8,
        )
    return None


def reject_lens_artefact(growth: Growth, region: Region) -> Rejection | None:
    """A water droplet or a dirt speck is perfectly still and has hard edges.

    This is the rule that saves a camera after rain. A droplet on the glass
    registers as a permanent local change against a background learned when the
    glass was clean, it never moves by even a pixel because it is attached to the
    camera rather than the world, and it has a sharp refractive edge that no
    plume has.
    """
    if (
        growth.frames >= 4
        and growth.centroid_drift_px_per_min < STATIC_DRIFT_MAX_PX_MIN
        and STATIC_AREA_RATIO[0] <= growth.area_ratio <= STATIC_AREA_RATIO[1]
        and growth.base_jitter_px < STATIC_JITTER_MAX_PX
        and region.edge_softness < STATIC_EDGE_SOFTNESS_MAX
    ):
        return Rejection(
            "STATIC_ARTEFACT",
            "something on the lens",
            f"unchanged for {growth.frames} frames — it moved "
            f"{growth.centroid_drift_px_per_min:.2f} px a minute, changed area by "
            f"{abs(1 - growth.area_ratio) * 100:.0f}%, and has a hard "
            f"edge (softness {region.edge_softness:.2f}); that is on the glass, not on the hill",
            {
                "centroid_drift_px_per_min": growth.centroid_drift_px_per_min,
                "area_ratio": growth.area_ratio,
                "edge_softness": region.edge_softness,
                "frames": growth.frames,
            },
            confidence=0.8,
        )
    return None


def reject_flare(region: Region, frame: np.ndarray, scene: SceneState) -> Rejection | None:
    """Sun flare is blown out; smoke never is.

    The brightness statistics come from the region's own measurements rather
    than being recomputed here. Recomputing them meant a full-frame channel max
    per region per frame, which on real imagery was two seconds in every twelve.
    """
    mean_luma = region.metrics.get("mean_luma")
    blown = region.metrics.get("blown_fraction")
    if mean_luma is None or blown is None:
        inside = region.mask > 0
        if not inside.any():
            return None
        gray = frame if frame.ndim == 2 else frame[:, :, :3].max(axis=2)
        values = gray[inside]
        mean_luma = float(values.mean())
        blown = float(np.mean(values > 250))
    if mean_luma > FLARE_LUMA_MIN and blown > FLARE_SATURATED_FRACTION:
        return Rejection(
            "FLARE",
            "lens flare or direct sun",
            f"mean brightness {mean_luma:.0f} of 255 with {blown * 100:.0f}% of it clipped; smoke "
            "scatters light, it does not saturate a sensor",
            {"mean_luma": mean_luma, "blown_fraction": blown, "scene": scene.usability.value},
            confidence=0.9,
        )
    return None


def reject_ridge_registration(
    region: Region, alignment: Alignment, frame_height: int
) -> Rejection | None:
    """A camera nudge lights up the ridge line and nothing else.

    The tell is shape and place together: a thin horizontal sliver lying exactly
    on the skyline, in a frame where stabilisation failed to lock. Either alone
    is innocent; together they are a shaken mast.
    """
    thin = region.h <= RIDGE_BAND_PX
    on_ridge = abs(region.base_below_horizon) <= RIDGE_BAND_PX
    wide = region.w > region.h * 3
    shaken = not alignment.corrected and alignment.shift_px > SHAKE_UNCORRECTED_PX
    if thin and on_ridge and wide and shaken:
        return Rejection(
            "RIDGE_REGISTRATION",
            "camera movement",
            f"a {region.w}x{region.h} px sliver sitting on the skyline in a frame that moved "
            f"{alignment.shift_px:.1f} px and could not be re-registered",
            {
                "shift_px": alignment.shift_px,
                "response": alignment.response,
                "height_px": region.h,
                "width_px": region.w,
                "frame_height": frame_height,
            },
            confidence=0.75,
        )
    return None


def reject_weather_front(
    region: Region, scene: SceneState, baseline_contrast: float | None
) -> Rejection | None:
    """When the whole frame goes milky, a milky patch is not news.

    Fog and rain arrive across the scene. If global contrast has collapsed
    against this camera's own recent baseline, a low-texture region is the
    weather, and the honest answer is that the camera has stopped being useful
    rather than that something was detected.
    """
    if baseline_contrast is None or baseline_contrast <= 1e-6:
        return None
    ratio = scene.contrast / baseline_contrast
    if ratio < GLOBAL_TEXTURE_COLLAPSE and region.texture_ratio > ratio * 0.75:
        return Rejection(
            "WEATHER_FRONT",
            "fog or rain across the whole scene",
            f"detail contrast across the entire frame fell to {ratio * 100:.0f}% of this camera's "
            "recent normal, and this patch is no softer than the rest of it",
            {
                "contrast": scene.contrast,
                "baseline_contrast": baseline_contrast,
                "contrast_ratio": ratio,
                "region_texture_ratio": region.texture_ratio,
            },
            confidence=0.8,
        )
    return None


def reject_habitual(
    region: Region, calibration, confidence: float
) -> Rejection | None:
    """This camera raises something here when nothing is happening.

    The rule that the first evaluation asked for. On eight of forty-four cameras
    the detector locked onto a persistent feature that satisfied every other
    test, because the feature genuinely grows, stays anchored and veils the
    hillside. What it does not do is be new: it was already there on frames
    labelled clear, on other days.

    Two separate conditions, because a camera can fail either way:

    * the candidate sits on cells this camera habitually lights up, or
    * the confidence does not exceed what this camera reaches on a clear day.

    Both are measured from that camera's own clear frames on *other dates*, so
    neither has seen the day being judged.
    """
    if calibration is None or not calibration.trustworthy:
        return None

    habituation = calibration.habituation(region.mask) if calibration.use_nuisance else 0.0
    if habituation >= HABITUAL_REJECT_AT:
        return Rejection(
            "HABITUAL_REGION",
            "something this camera always sees here",
            f"this camera raises a candidate in this part of its view on "
            f"{habituation * 100:.0f}% of its clear frames, learned from "
            f"{calibration.clear_frames} frames on other days; whatever it is, it was "
            "already there",
            {
                "habituation": habituation,
                "threshold": HABITUAL_REJECT_AT,
                "calibration_frames": calibration.clear_frames,
                "calibrated_from": calibration.sources,
            },
            # Scales with how habitual the location is: a cell at 0.35 is
            # suggestive, one at 0.9 is conclusive.
            confidence=float(min(0.92, 0.45 + 0.6 * (habituation - HABITUAL_REJECT_AT))),
        )

    ceiling = calibration.clear_p95 * BASELINE_MARGIN - calibration.ceiling_slack
    if calibration.use_ceiling and confidence <= ceiling and ceiling > 0.0:
        return Rejection(
            "BELOW_CAMERA_BASELINE",
            "an ordinary day on this camera",
            f"scored {confidence:.2f}, and this camera reaches {calibration.clear_p95:.2f} on "
            f"its clear frames; that is not unusual for this view",
            {
                "confidence": confidence,
                "clear_p95": calibration.clear_p95,
                "clear_median": calibration.clear_p50,
                "calibration_frames": calibration.clear_frames,
            },
            confidence=0.7,
        )
    return None


def reject_too_brief(growth: Growth, minimum_frames: int) -> Rejection | None:
    """One frame is not a column. Said out loud, because it is the whole thesis."""
    if growth.frames < minimum_frames:
        return Rejection(
            "TOO_BRIEF",
            "not yet decidable",
            f"seen in {growth.frames} frame{'s' if growth.frames != 1 else ''}; a column is "
            f"identified by how it grows, which needs at least {minimum_frames}",
            {"frames": growth.frames, "minimum": minimum_frames},
            confidence=0.4,
        )
    return None


def apply_all(
    track: Track,
    growth: Growth,
    region: Region,
    frame: np.ndarray,
    scene: SceneState,
    camera: Camera,
    alignment: Alignment,
    *,
    minimum_frames: int = 3,
    baseline_contrast: float | None = None,
    calibration=None,
    confidence: float = 0.0,
) -> list[Rejection]:
    """Run every rule. Returns all that fired, strongest first."""
    height = frame.shape[0]
    found = [
        reject_too_brief(growth, minimum_frames),
        reject_habitual(region, calibration, confidence),
        reject_flare(region, frame, scene),
        reject_lens_artefact(growth, region),
        reject_ridge_registration(region, alignment, height),
        reject_weather_front(region, scene, baseline_contrast),
        reject_airborne(region, height, growth),
        reject_cloud(growth, region, height),
        reject_erratic(growth),
    ]
    if camera.has_colour:
        found.append(reject_dust(growth, region))
    rejections = [r for r in found if r is not None]
    rejections.sort(key=lambda r: r.confidence, reverse=True)
    return rejections


def blind_reason(scene: SceneState) -> Rejection | None:
    """Turn an unusable camera into a rejection, so it reads the same as the rest."""
    if not scene.blind:
        return None
    impostor = {
        Usability.NIGHT: "darkness",
        Usability.FOG: "fog",
        Usability.LENS_OBSCURED: "an obscured lens",
        Usability.GLARE: "direct sun",
        Usability.FROZEN: "a dead feed",
    }[scene.usability]
    return Rejection(
        f"CAMERA_{scene.usability.value.upper()}",
        impostor,
        scene.reason,
        scene.to_dict(),
        confidence=1.0,
    )
