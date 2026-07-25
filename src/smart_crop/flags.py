"""Heuristic flags for spotting crop decisions worth a human/vision-model look.

Cheap, geometry-based checks only -- no image content inspection. Meant to cut a batch of
hundreds of decisions down to the handful actually worth eyeballing.
"""

from smart_crop.crop import CropDecision, max_crop_box
from smart_crop.ratios import Target

LOW_SCALE_THRESHOLD = 0.5
EDGE_ANCHOR_THRESHOLD = 0.05
DISAGREEMENT_CX_CY = 0.15
DISAGREEMENT_SCALE = 0.2


def decision_flags(decision: CropDecision, target: Target, source_w: int, source_h: int) -> list[str]:
    """Flags derived from a single model's decision, independent of any other model's."""
    if not decision.worthwhile:
        return ["skipped"]

    flags = []
    max_w, max_h = max_crop_box(source_w, source_h, target)

    # At scale 1.0, an axis with zero slack means the corresponding coordinate has no effect.
    # A model setting it away from 0.5 anyway is a sign it doesn't understand its own output.
    if decision.scale >= 0.99:
        if max_w >= source_w - 0.5 and abs(decision.cx - 0.5) > 0.02:
            flags.append("unused_cx")
        if max_h >= source_h - 0.5 and abs(decision.cy - 0.5) > 0.02:
            flags.append("unused_cy")

    if decision.scale < LOW_SCALE_THRESHOLD:
        flags.append("low_scale")

    if decision.cx < EDGE_ANCHOR_THRESHOLD or decision.cx > 1 - EDGE_ANCHOR_THRESHOLD:
        flags.append("edge_cx")
    if decision.cy < EDGE_ANCHOR_THRESHOLD or decision.cy > 1 - EDGE_ANCHOR_THRESHOLD:
        flags.append("edge_cy")

    return flags


def disagreement_flags(
    decisions_by_backend: dict[str, list[CropDecision]], target: Target, source_w: int, source_h: int
) -> list[str]:
    """Flags from comparing multiple backends' decisions for the same image/target.

    Only compares fine-grained cx/cy/scale when every backend proposed exactly one candidate --
    matching up N candidates against M isn't attempted, "multiple_candidates" is flagged instead.
    """
    if len(decisions_by_backend) < 2:
        return []

    if any(len(ds) > 1 for ds in decisions_by_backend.values()):
        return []  # each side already gets a "multiple_candidates" flag on its own entries

    single = {name: ds[0] for name, ds in decisions_by_backend.items()}

    flags = []
    worthwhile_values = {d.worthwhile for d in single.values()}
    if len(worthwhile_values) > 1:
        flags.append("disagreement:worthwhile")
        return flags  # cx/cy/scale aren't comparable when one side skipped

    worthwhile_decisions = [d for d in single.values() if d.worthwhile]
    if len(worthwhile_decisions) < 2:
        return flags

    cx_spread = max(d.cx for d in worthwhile_decisions) - min(d.cx for d in worthwhile_decisions)
    cy_spread = max(d.cy for d in worthwhile_decisions) - min(d.cy for d in worthwhile_decisions)
    scale_spread = max(d.scale for d in worthwhile_decisions) - min(d.scale for d in worthwhile_decisions)

    # A cx/cy spread only reflects a real difference in output pixels if at least one backend's box
    # actually has slack on that axis -- otherwise both backends' boxes span the full source
    # dimension regardless of cx/cy, and "disagreeing" on an ignored coordinate is noise, not signal.
    max_w, max_h = max_crop_box(source_w, source_h, target)
    cx_matters = any(d.scale * max_w < source_w - 0.5 for d in worthwhile_decisions)
    cy_matters = any(d.scale * max_h < source_h - 0.5 for d in worthwhile_decisions)

    if cx_spread > DISAGREEMENT_CX_CY and cx_matters:
        flags.append("disagreement:cx")
    if cy_spread > DISAGREEMENT_CX_CY and cy_matters:
        flags.append("disagreement:cy")
    if scale_spread > DISAGREEMENT_SCALE:
        flags.append("disagreement:scale")

    return flags
