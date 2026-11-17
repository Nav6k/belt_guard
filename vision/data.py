"""Turns a Roboflow YOLOv8-format export into a patch classification dataset.

Why patches instead of training a detector:
  - a fixed inspection camera over a moving belt sees a continuous strip; classifying fixed-size
    patches of that strip is a natural fit, and the damaged-patch count gives a crude size measure
  - fine-tuning a small pretrained CNN on patches trains in minutes on a CPU, a YOLO detector doesn't

Positive patches are cropped around each labelled box (label = its class). Negative patches
("intact") are cropped from regions that don't overlap any box. Boxes come from YOLO bbox lines
(cls cx cy w h) or segmentation polygons (cls x1 y1 x2 y2 ...); both are normalised to a bbox.
"""
import os
import random
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image

DATASET = Path(os.environ.get("BELT_DATASET", "vision/dataset"))
INTACT = "intact"


@dataclass
class Box:
    cls: str
    x0: float
    y0: float
    x1: float
    y1: float   # pixel coordinates


def class_names(root: Path = DATASET) -> list[str]:
    meta = yaml.safe_load((root / "data.yaml").read_text())
    names = meta["names"]
    names = list(names.values()) if isinstance(names, dict) else list(names)
    # the Roboflow export names its classes "15" and "16" (leftover annotation ids, one box of "16" in total),
    # so numeric names are merged into one "damage" class rather than training a class from a single example
    return ["damage" if str(n).isdigit() else str(n) for n in names]


def read_boxes(label_path: Path, w: int, h: int, names: list[str]) -> list[Box]:
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue                       # malformed label line: skip it, don't crash the whole export
        cls, nums = int(parts[0]), list(map(float, parts[1:]))
        if len(nums) == 4:
            cx, cy, bw, bh = nums
            x0, y0, x1, y1 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
        else:
            xs, ys = nums[0::2], nums[1::2]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        boxes.append(Box(names[cls], x0 * w, y0 * h, x1 * w, y1 * h))
    return boxes


def split_items(split: str, root: Path = DATASET) -> list[tuple[Path, list[Box]]]:
    names = class_names(root)
    items = []
    for img_path in sorted((root / split / "images").glob("*")):
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except OSError:
            continue                       # unreadable image file
        items.append((img_path, read_boxes(root / split / "labels" / (img_path.stem + ".txt"), w, h, names)))
    return items


def _overlaps(x0, y0, x1, y1, boxes: list[Box]) -> bool:
    return any(not (x1 <= b.x0 or x0 >= b.x1 or y1 <= b.y0 or y0 >= b.y1) for b in boxes)


def make_patches(items, negatives_per_image: int = 2, seed: int = 0) -> list[tuple[Image.Image, str]]:
    rng = random.Random(seed)
    out = []
    for img_path, boxes in items:
        im = Image.open(img_path).convert("RGB")
        w, h = im.size
        for b in boxes:
            # pad the box a little so the patch includes the transition from intact to damaged rubber
            pw, ph = (b.x1 - b.x0) * 0.15 + 8, (b.y1 - b.y0) * 0.15 + 8
            crop = im.crop((max(0, b.x0 - pw), max(0, b.y0 - ph), min(w, b.x1 + pw), min(h, b.y1 + ph)))
            if crop.size[0] >= 16 and crop.size[1] >= 16:
                out.append((crop, b.cls))
        side, added = int(min(w, h) * 0.3), 0
        for _ in range(negatives_per_image * 10):      # bounded attempts: busy images may have no free area
            if added >= negatives_per_image:
                break
            x0, y0 = rng.randint(0, w - side), rng.randint(0, h - side)
            if not _overlaps(x0, y0, x0 + side, y0 + side, boxes):
                out.append((im.crop((x0, y0, x0 + side, y0 + side)), INTACT))
                added += 1
    return out
