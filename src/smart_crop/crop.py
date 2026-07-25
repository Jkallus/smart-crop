"""Deterministic crop-box math and execution. See agent_spec.md for the model."""

import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from smart_crop.ratios import Target


class ResolutionFloorError(Exception):
    """Raised when a computed crop would fall below the target's minimum resolution."""


@dataclass(frozen=True)
class CropDecision:
    target: str
    worthwhile: bool
    scale: float = 1.0
    cx: float = 0.5
    cy: float = 0.5
    reason: str = ""


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def crop_box(source_w: int, source_h: int, target: Target, decision: CropDecision) -> tuple[int, int, int, int]:
    """Compute the (left, top, right, bottom) pixel box for a crop decision."""
    source_ratio = source_w / source_h

    if target.ratio > source_ratio:
        max_w, max_h = source_w, source_w / target.ratio
    elif target.ratio < source_ratio:
        max_w, max_h = source_h * target.ratio, source_h
    else:
        max_w, max_h = source_w, source_h

    box_w = max_w * decision.scale
    box_h = max_h * decision.scale

    left = _clamp(decision.cx * source_w - box_w / 2, 0, source_w - box_w)
    top = _clamp(decision.cy * source_h - box_h / 2, 0, source_h - box_h)

    return (round(left), round(top), round(left + box_w), round(top + box_h))


def apply_crop(
    source_path: Path, target: Target, decision: CropDecision, output_dir: Path, suffix: str = ""
) -> Path | None:
    """Crop source_path per decision and save into output_dir/target.folder. Returns output path, or None if skipped.

    Raises ResolutionFloorError if the model's decision would produce a crop below the target's
    minimum resolution -- a safety net independent of the model's own worthwhile/scale judgment.
    """
    if not decision.worthwhile:
        return None

    with Image.open(source_path) as img:
        source_w, source_h = img.width, img.height
        box = crop_box(source_w, source_h, target, decision)
        left, top, right, bottom = box
        box_w, box_h = right - left, bottom - top

        if box_w < target.min_w or box_h < target.min_h:
            raise ResolutionFloorError(
                f"{target.name} crop is {box_w}x{box_h}, below floor {target.min_w}x{target.min_h}"
            )

        dest_dir = output_dir / target.folder
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_name = f"{source_path.stem}{suffix}{source_path.suffix}"
        dest_path = dest_dir / dest_name

        if box == (0, 0, source_w, source_h):
            # Full-frame passthrough -- e.g. a 4:3 drone source against the ipad target at
            # scale=1.0, where source and target ratios already match and there's nothing to trim.
            # Copy the original bytes directly rather than a lossy decode/crop/re-encode round
            # trip that would produce a pixel-identical result anyway.
            shutil.copy2(source_path, dest_path)
            return dest_path

        cropped = img.crop(box)
        # quality=100 + subsampling=0 (4:4:4, no chroma downsampling) minimizes re-encode loss from
        # the unavoidable decode/crop/encode cycle; icc_profile/exif preserve color space + metadata
        # that PIL otherwise silently drops on save.
        cropped.save(
            dest_path,
            quality=100,
            subsampling=0,
            optimize=True,
            icc_profile=img.info.get("icc_profile"),
            exif=img.info.get("exif"),
        )

    return dest_path
