import filecmp

from PIL import Image

from smart_crop.crop import CropDecision, apply_crop, crop_box
from smart_crop.ratios import TARGETS


def test_wider_target_crops_height_keeps_full_width():
    # 3:2 source -> 16:9 target: target is wider, so full width kept, height trimmed.
    box = crop_box(6000, 4000, TARGETS["tv"], CropDecision(target="tv", worthwhile=True, cy=1.0))
    left, top, right, bottom = box
    assert (left, right) == (0, 6000)
    assert bottom == 4000  # cy=1.0 anchors to the bottom edge
    assert round((right - left) / (bottom - top), 3) == round(16 / 9, 3)


def test_narrower_target_crops_width_keeps_full_height():
    # 3:2 source -> 4:3 target: target is narrower, so full height kept, width trimmed.
    box = crop_box(6000, 4000, TARGETS["ipad"], CropDecision(target="ipad", worthwhile=True, cx=0.5))
    left, top, right, bottom = box
    assert (top, bottom) == (0, 4000)
    assert round((right - left) / (bottom - top), 3) == round(4 / 3, 3)


def test_matching_ratio_no_crop():
    # 4:3 drone source -> 4:3 target: exact match, full frame.
    box = crop_box(4000, 3000, TARGETS["ipad"], CropDecision(target="ipad", worthwhile=True))
    assert box == (0, 0, 4000, 3000)


def test_scale_below_one_shrinks_box_and_cx_cy_move_it():
    box = crop_box(6000, 4000, TARGETS["iphone"], CropDecision(target="iphone", worthwhile=True, scale=0.5, cx=0.2, cy=0.5))
    left, top, right, bottom = box
    max_w, max_h = 4000 * (9 / 19.5), 4000
    assert abs((right - left) - max_w * 0.5) <= 1
    assert abs((bottom - top) - max_h * 0.5) <= 1


def test_box_never_exceeds_source_bounds():
    box = crop_box(6000, 4000, TARGETS["iphone"], CropDecision(target="iphone", worthwhile=True, cx=0.0))
    left, top, right, bottom = box
    assert left >= 0 and right <= 6000
    assert top >= 0 and bottom <= 4000


def test_matching_ratio_apply_crop_copies_bytes_unchanged(tmp_path):
    # A full-frame passthrough (source ratio == target ratio, scale 1.0) should be a byte-for-byte
    # copy of the original file, not a decode/crop/re-encode round trip -- see crop.py.
    source_path = tmp_path / "source.jpg"
    Image.new("RGB", (4000, 3000), color=(120, 60, 200)).save(source_path, quality=100)

    dest_path = apply_crop(
        source_path, TARGETS["ipad"], CropDecision(target="ipad", worthwhile=True), tmp_path / "out"
    )

    assert filecmp.cmp(source_path, dest_path, shallow=False)
