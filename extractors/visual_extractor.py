"""
extractors/visual_extractor.py
──────────────────────────────
Extracts metadata and annotations from RGB image datasets.

Handles the source-specific quirks of each agreed dataset:
  - D-Fire     : YOLO .txt labels in parallel /labels/ tree
  - Roboflow   : YOLO .txt labels, pre-split folder structure
  - Kaggle     : Folder-based labels (fire/ vs no_fire/ subdirectories)
  - Personal   : No labels (all negatives — used for false positive reduction)

All results use the shared ImageMetadata + AnnotationData dataclasses
and feed into the common Transform → Load pipeline.
"""

import logging
from pathlib import Path
from typing import List, Optional

import cv2

from extractors.base_extractor import (
    BaseExtractor,
    ExtractorResult,
    ImageMetadata,
    AnnotationData,
    CLASS_REGISTRY,
    CLASS_ID_TO_NAME,
)

log = logging.getLogger(__name__)

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class VisualExtractor(BaseExtractor):
    """
    Extracts RGB image metadata and YOLO-format annotations.

    Usage:
        extractor = VisualExtractor.for_dfire("data/raw/dfire")
        results = extractor.extract_batch(image_paths)
    """

    def __init__(
        self,
        dataset_id: str,
        dataset_name: str,
        label_strategy: str = "yolo_parallel",
        class_map: Optional[dict] = None,
    ):
        """
        Args:
            dataset_id:      Unique ID string for the dataset row in SQLite.
            dataset_name:    Human-readable name.
            label_strategy:  One of:
                               'yolo_parallel' — labels/ dir mirrors images/ dir
                               'yolo_same_dir' — .txt beside each .jpg
                               'folder_class'  — parent folder name is the class
                               'negative_only' — no labels, all images are negatives
            class_map:       Optional override for class name normalisation.
        """
        super().__init__(dataset_id, dataset_name)
        self.label_strategy = label_strategy
        self.class_map = class_map or {}

    # ── Factory constructors — one per agreed dataset ────────────────────────

    @classmethod
    def for_dfire(cls, raw_dir: str) -> "VisualExtractor":
        """
        D-Fire dataset layout:
            DFireDataset/
              train/
                images/  *.jpg
                labels/  *.txt   (YOLO format, class 0=fire, 1=smoke)
              test/
                images/  *.jpg
                labels/  *.txt
        Labels use numeric classes: 0=fire(→flame), 1=smoke.
        D-Fire has no val split — we create one during the split phase.
        """
        return cls(
            dataset_id="dfire",
            dataset_name="D-Fire Dataset",
            label_strategy="yolo_parallel",
        )

    @classmethod
    def for_roboflow(cls, raw_dir: str) -> "VisualExtractor":
        """
        Roboflow export layout (YOLOv8 format):
            fire-and-smoke-detection/
              train/
                images/  *.jpg
                labels/  *.txt
              valid/
                images/  *.jpg
                labels/  *.txt
              test/
                images/  *.jpg
                labels/  *.txt
        Note: Roboflow names the val split 'valid', not 'val'.
        Class names come from data.yaml in the root.
        """
        return cls(
            dataset_id="roboflow_fire",
            dataset_name="Roboflow Fire & Smoke",
            label_strategy="yolo_parallel",
        )

    @classmethod
    def for_kaggle(cls, raw_dir: str) -> "VisualExtractor":
        """
        Kaggle fire detection layout:
            firedetection/
              fire/      *.jpg   → class: flame
              no_fire/   *.jpg   → no annotation (negative)
        This dataset has no bounding boxes — only image-level labels.
        We treat 'fire/' images as whole-image detections (xc=0.5, yc=0.5, w=1, h=1)
        and 'no_fire/' as negatives. This is a weak label — use only for augmentation.
        """
        return cls(
            dataset_id="kaggle_fire",
            dataset_name="Kaggle Fire Detection",
            label_strategy="folder_class",
            class_map={"fire": "flame", "no_fire": None},
        )

    @classmethod
    def for_personal(cls, raw_dir: str) -> "VisualExtractor":
        """
        Personal negative captures — no labels, all images are negatives.
        These are the server room photos captured by capture_negatives.py.
        """
        return cls(
            dataset_id="personal_negatives",
            dataset_name="Personal Negatives",
            label_strategy="negative_only",
        )

    # ── Main extract method ──────────────────────────────────────────────────

    def extract(self, file_path: Path) -> ExtractorResult:
        """
        Extract metadata and annotations from a single visual image file.
        This is the method called by extract_batch() in the base class.
        """
        image_id = self.generate_id()
        errors = []

        # ── 1. Read and validate image ───────────────────────────────────────
        img = self.read_image(file_path)
        if img is None:
            return ExtractorResult(
                metadata=self._stub_metadata(image_id, file_path),
                errors=[f"Cannot decode image: {file_path.name}"],
            )

        h, w, c = self.get_image_shape(img)

        # Reject images too small to be useful (min 50×50)
        if w < 50 or h < 50:
            return ExtractorResult(
                metadata=self._stub_metadata(image_id, file_path),
                errors=[f"Image too small ({w}×{h}): {file_path.name}"],
            )

        # ── 2. Build metadata ────────────────────────────────────────────────
        metadata = ImageMetadata(
            image_id=image_id,
            dataset_id=self.dataset_id,
            file_path=str(file_path.resolve()),
            original_filename=file_path.name,
            modality="visual",
            width=w,
            height=h,
            channels=c,
            file_size_mb=self.file_size_mb(file_path),
            checksum=self.compute_checksum(file_path),
            captured_at=self.read_exif_datetime(file_path),
        )

        # ── 3. Extract annotations based on strategy ─────────────────────────
        annotations = self._extract_annotations(file_path, image_id, w, h)

        return ExtractorResult(metadata=metadata, annotations=annotations, errors=errors)

    # ── Annotation strategies ────────────────────────────────────────────────

    def _extract_annotations(
        self, file_path: Path, image_id: str, w: int, h: int
    ) -> List[AnnotationData]:

        if self.label_strategy == "yolo_parallel":
            return self._from_yolo_parallel(file_path, image_id, w, h)

        elif self.label_strategy == "yolo_same_dir":
            return self._from_yolo_same_dir(file_path, image_id, w, h)

        elif self.label_strategy == "folder_class":
            return self._from_folder_class(file_path, image_id, w, h)

        elif self.label_strategy == "negative_only":
            return []   # All personal captures are negatives

        else:
            self.log.warning("Unknown label strategy: %s", self.label_strategy)
            return []

    def _from_yolo_parallel(
        self, file_path: Path, image_id: str, w: int, h: int
    ) -> List[AnnotationData]:
        """
        Resolve label file from a parallel labels/ directory.

        Layout:
            .../images/train/img001.jpg  →  .../labels/train/img001.txt
            .../images/img001.jpg        →  .../labels/img001.txt

        The resolution replaces the first occurrence of 'images' in the
        path with 'labels', which covers both flat and split structures.
        """
        parts = file_path.parts
        try:
            # Find rightmost 'images' segment (handles nested paths safely)
            img_idx = len(parts) - 1 - parts[::-1].index("images")
        except ValueError:
            # No 'images' folder in path — try same-dir fallback
            return self._from_yolo_same_dir(file_path, image_id, w, h)

        label_parts = list(parts[:img_idx]) + ["labels"] + list(parts[img_idx + 1:])
        label_path = Path(*label_parts).with_suffix(".txt")

        return self.parse_yolo_label_file(label_path, image_id, w, h)

    def _from_yolo_same_dir(
        self, file_path: Path, image_id: str, w: int, h: int
    ) -> List[AnnotationData]:
        """Label file sits beside the image with the same stem."""
        label_path = file_path.with_suffix(".txt")
        return self.parse_yolo_label_file(label_path, image_id, w, h)

    def _from_folder_class(
        self, file_path: Path, image_id: str, w: int, h: int
    ) -> List[AnnotationData]:
        """
        Parent folder name determines class (used by Kaggle dataset).
        Images in 'fire/' → whole-image flame annotation.
        Images in 'no_fire/' → negative, no annotation.

        Weak label: bounding box covers the full image (xc=0.5, yc=0.5, w=1, h=1).
        """
        folder_name = file_path.parent.name.lower()
        raw_class = self.class_map.get(folder_name, folder_name)

        if raw_class is None:
            return []   # Negative sample

        class_name, class_id = self._resolve_class(raw_class)
        if class_name is None:
            self.log.debug("  Unresolved folder class '%s' for %s", folder_name, file_path.name)
            return []

        return [AnnotationData(
            annotation_id=self.generate_id(),
            image_id=image_id,
            class_name=class_name,
            class_id=class_id,
            confidence=0.7,         # Reduced confidence for weak/image-level labels
            x_center_norm=0.5,
            y_center_norm=0.5,
            width_norm=1.0,
            height_norm=1.0,
            x_min_px=0,
            y_min_px=0,
            x_max_px=w,
            y_max_px=h,
        )]

    # ── Helper ───────────────────────────────────────────────────────────────

    def _stub_metadata(self, image_id: str, file_path: Path) -> ImageMetadata:
        """Minimal metadata for failed extractions (for error logging)."""
        return ImageMetadata(
            image_id=image_id,
            dataset_id=self.dataset_id,
            file_path=str(file_path),
            original_filename=file_path.name,
            modality="visual",
            width=0, height=0, channels=0,
            file_size_mb=0.0,
            checksum="",
        )


# ── Dataset-specific scanner functions ────────────────────────────────────────
# These return (extractor, list_of_image_paths) ready for extract_batch().

def scan_dfire(raw_dir: str):
    """
    Scan D-Fire dataset directory and return (extractor, image_paths).

    Expected structure:
        data/raw/dfire/
          train/images/*.jpg   train/labels/*.txt
          test/images/*.jpg    test/labels/*.txt
    """
    root = Path(raw_dir)
    extractor = VisualExtractor.for_dfire(raw_dir)
    images = sorted([
        p for p in root.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS and "images" in p.parts
    ])
    log.info("D-Fire: found %d images in %s", len(images), root)
    return extractor, images


def scan_roboflow(raw_dir: str):
    """
    Scan Roboflow export directory.

    Expected structure:
        data/raw/roboflow/
          train/images/*.jpg   train/labels/*.txt
          valid/images/*.jpg   valid/labels/*.txt
          test/images/*.jpg    test/labels/*.txt
    """
    root = Path(raw_dir)
    extractor = VisualExtractor.for_roboflow(raw_dir)

    # Roboflow uses 'valid' not 'val' — include it
    images = sorted([
        p for p in root.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS and "images" in p.parts
    ])
    log.info("Roboflow: found %d images in %s", len(images), root)
    return extractor, images


def scan_kaggle(raw_dir: str):
    """
    Scan Kaggle fire detection dataset.

    Expected structure:
        data/raw/kaggle_fire/
          fire/*.jpg
          no_fire/*.jpg
    """
    root = Path(raw_dir)
    extractor = VisualExtractor.for_kaggle(raw_dir)
    images = sorted([
        p for p in root.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
        and p.parent.name.lower() in {"fire", "no_fire"}
    ])
    log.info("Kaggle: found %d images in %s", len(images), root)
    return extractor, images


def scan_personal(raw_dir: str):
    """
    Scan personal negative captures.

    Expected structure:
        data/raw/personal_negatives/*.jpg
    """
    root = Path(raw_dir)
    extractor = VisualExtractor.for_personal(raw_dir)
    images = sorted([
        p for p in root.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ])
    log.info("Personal negatives: found %d images in %s", len(images), root)
    return extractor, images