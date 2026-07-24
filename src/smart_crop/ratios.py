"""Target export ratios, as W/H fractions. See agent_spec.md for the full table."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Target:
    name: str
    ratio: float  # width / height
    folder: str
    min_w: int  # resolution floor -- safety net, see agent_spec.md
    min_h: int


TARGETS: dict[str, Target] = {
    "tv": Target("tv", 16 / 9, "tv", 1920, 1080),
    "macbook": Target("macbook", 16 / 10, "macbook", 2560, 1600),
    "ultrawide": Target("ultrawide", 5120 / 2160, "ultrawide", 5120, 2160),
    "ipad": Target("ipad", 4 / 3, "ipad", 2732, 2048),
    "iphone": Target("iphone", 9 / 19.5, "iphone", 1290, 2796),
}
