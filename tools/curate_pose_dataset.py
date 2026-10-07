"""Curate a reproducible YOLO pose subset from an already-built dataset export.

This script deliberately leaves the original test PNG frames out. The new JPG
photos in the source test directory are reassigned only because they were
explicitly supplied for this training round.
"""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import cv2
import numpy as np


OLD_TRAIN_COUNT = 1380
OLD_VAL_COUNT = 195
NEW_VAL_STEMS = {"image_42047", "image_42051", "image_42055", "image_42062", "image_42075"}
BLOCK_SIZE = 32


def image_number(path: Path) -> int:
    return int(path.stem.removeprefix("image_"))


def label_for(source: Path, image: Path) -> Path:
    return source / "labels" / image.parent.name / f"{image.stem}.txt"


def parsed_labels(label: Path) -> list[np.ndarray]:
    rows = [np.fromstring(line, sep=" ") for line in label.read_text().splitlines() if line.strip()]
    if not rows or any(len(row) != 56 or row[0] != 0 for row in rows):
        raise ValueError(f"Invalid 17-keypoint YOLO pose label: {label}")
    for row in rows:
        x, y, width, height = row[1:5]
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"Invalid bounding box: {label}")
        if not np.all(np.isin(row[7::3], [0, 1, 2])):
            raise ValueError(f"Invalid keypoint visibility: {label}")
    return rows


def feature(source: Path, image: Path) -> np.ndarray:
    pixels = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if pixels is None:
        raise ValueError(f"Cannot decode image: {image}")
    thumbnail = cv2.resize(cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY), (16, 12), interpolation=cv2.INTER_AREA)
    rows = sorted(parsed_labels(label_for(source, image)), key=lambda row: row[3] * row[4], reverse=True)
    pose_features: list[float] = [min(len(rows), 3) * 1.5]
    for row in rows[:2]:
        x, y, width, height = row[1:5]
        pose_features.extend([x * 3, y * 3, width * 3, height * 3])
        for px, py, visibility in row[5:].reshape(17, 3):
            pose_features.extend([
                (px - x) / max(width, 0.01) * 0.55 if visibility else 0,
                (py - y) / max(height, 0.01) * 0.55 if visibility else 0,
                visibility * 0.12,
            ])
    pose_features.extend([0.0] * (2 * 55 - (len(pose_features) - 1)))
    return np.concatenate((thumbnail.flatten().astype(np.float32) / 255 * 0.45,
                           np.asarray(pose_features, dtype=np.float32)))


def pick_spread(source: Path, pool: list[Path], count: int) -> list[Path]:
    """Cover the full timeline, then favor different appearance and poses per block."""
    if not 0 <= count <= len(pool):
        raise ValueError(f"Cannot select {count} from {len(pool)} images")
    pool = sorted(pool, key=image_number)
    selected: list[Path] = []
    for start in range(0, len(pool), BLOCK_SIZE):
        block = pool[start:start + BLOCK_SIZE]
        quota = round((start + len(block)) * count / len(pool)) - round(start * count / len(pool))
        if quota == 0:
            continue
        vectors = np.stack([feature(source, image) for image in block])
        timeline = np.arange(len(block), dtype=np.float32)[:, None] / max(len(block) - 1, 1)
        vectors = np.concatenate((vectors, timeline * 0.7), axis=1)
        chosen: list[int] = []
        nearest = np.full(len(block), np.inf)
        seed = int(np.argmax(np.sum((vectors - vectors.mean(axis=0)) ** 2, axis=1)))
        for _ in range(quota):
            index = seed if not chosen else int(np.argmax(nearest))
            chosen.append(index)
            nearest = np.minimum(nearest, np.sum((vectors - vectors[index]) ** 2, axis=1))
            nearest[chosen] = -1
        selected.extend(block[index] for index in sorted(chosen))
    if len(selected) != count:
        raise AssertionError((len(selected), count))
    return selected


def build(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing file: {output}")
    old_train = sorted((source / "images" / "train").glob("*.png"), key=image_number)
    old_val = sorted((source / "images" / "val").glob("*.png"), key=image_number)
    old_test = sorted((source / "images" / "test").glob("*.png"), key=image_number)
    new = sorted((source / "images" / "test").glob("*.jpg"), key=image_number)
    if (len(old_train), len(old_val), len(old_test), len(new)) != (2322, 290, 265, 25):
        raise ValueError("Source does not match the inspected dataset-v5 split")
    new_train = [path for path in new if path.stem not in NEW_VAL_STEMS]
    new_val = [path for path in new if path.stem in NEW_VAL_STEMS]
    if len(new_train) != 20 or len(new_val) != 5:
        raise AssertionError("Unexpected new-photo split")

    # Leave one 32-frame gap between the old train and validation segments.
    val_pool = old_val[32:]
    chosen = {
        "train": pick_spread(source, old_train, OLD_TRAIN_COUNT) + new_train,
        "val": pick_spread(source, val_pool, OLD_VAL_COUNT) + new_val,
    }
    if len(chosen["train"]) != 1400 or len(chosen["val"]) != 200:
        raise AssertionError("Wrong output split sizes")
    source_paths = {path for images in chosen.values() for path in images}
    if len(source_paths) != 1600:
        raise AssertionError("An image is present in more than one split")

    manifest = io.StringIO()
    writer = csv.writer(manifest)
    writer.writerow(["output_split", "filename", "source_split", "source_kind", "annotation_count"])
    label_count = 0
    for split, paths in chosen.items():
        for image in sorted(paths, key=image_number):
            rows = parsed_labels(label_for(source, image))
            label_count += len(rows)
            writer.writerow([split, image.name, image.parent.name,
                             "added_photo" if image.suffix.lower() == ".jpg" else "old_video_frame", len(rows)])

    report = f"""# Curated pose dataset from dataset-v5

- Train: 1,400 images = {OLD_TRAIN_COUNT} old video frames + {len(new_train)} added photos.
- Validation: 200 images = {OLD_VAL_COUNT} old video frames + {len(new_val)} added photos.
- Total annotations: {label_count}.
- Original test PNG frames: {len(old_test)} excluded from this archive.
- All 25 added JPG photos are included. image_42074.jpg has two person annotations; kept after the user's confirmation that two people are present. Its exact keypoints should still be reviewed at full resolution.
- Old-frame selection: 32-frame windows with visual/pose farthest-point selection, preserving temporal coverage.
- A 32-frame gap separates old train and validation pools. New photos in validation: {', '.join(sorted(NEW_VAL_STEMS))}.
- The remaining ~500-image external test set was not provided and is not in this archive.
- Only 25 photos were new; this archive cannot reproduce a 600-old/800-new mixture or guarantee prevention of forgetting. Review labels and test against the untouched external set before using the model.
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "x", compression=ZIP_STORED, allowZip64=True) as archive:
        archive.writestr("data.yaml", "train: images/train\nval: images/val\nnc: 1\nnames: [person]\nkpt_shape: [17, 3]\n")
        archive.writestr("CURATION_REPORT.md", report)
        archive.writestr("selection_manifest.csv", manifest.getvalue())
        for split, paths in chosen.items():
            for image in sorted(paths, key=image_number):
                label = label_for(source, image)
                archive.write(image, f"images/{split}/{image.name}")
                archive.write(label, f"labels/{split}/{label.name}")
    print(report)
    print(f"Created: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Extracted dataset-v5 directory")
    parser.add_argument("output", type=Path, help="New ZIP path; must not exist")
    arguments = parser.parse_args()
    build(arguments.source.resolve(), arguments.output.resolve())
