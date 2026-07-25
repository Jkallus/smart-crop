from smart_crop.crop import CropDecision
from smart_crop.flags import disagreement_flags
from smart_crop.ratios import TARGETS


def test_cx_disagreement_suppressed_when_axis_has_no_slack():
    # Portrait (2:3) source -> tv (16:9): tv crops height, keeps full width, so cx has zero effect
    # on the output regardless of value. Found in a real batch (_DSC3559.jpg): qwen set cx=0 while
    # gemma left cx=0.5, but both crops spanned the full source width -- flagging that as a
    # disagreement was pure noise.
    decisions = {
        "gemma": [CropDecision(target="tv", worthwhile=True, scale=1.0, cx=0.5, cy=0.5)],
        "qwen": [CropDecision(target="tv", worthwhile=True, scale=1.0, cx=0.0, cy=0.5)],
    }
    flags = disagreement_flags(decisions, TARGETS["tv"], source_w=4016, source_h=6016)
    assert "disagreement:cx" not in flags


def test_cy_disagreement_still_flagged_when_axis_has_slack():
    # 3:2 source -> tv (16:9): tv crops height with real slack, so a cy spread reflects an actual
    # difference in output pixels and should still be flagged.
    decisions = {
        "gemma": [CropDecision(target="tv", worthwhile=True, scale=1.0, cx=0.5, cy=0.5)],
        "qwen": [CropDecision(target="tv", worthwhile=True, scale=1.0, cx=0.5, cy=0.8)],
    }
    flags = disagreement_flags(decisions, TARGETS["tv"], source_w=6000, source_h=4000)
    assert "disagreement:cy" in flags
