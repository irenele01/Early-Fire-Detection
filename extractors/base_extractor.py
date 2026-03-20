"""
extractors/base_extractor.py
────────────────────────────
Abstract base class shared by VisualExtractor and ThermalExtractor.

Responsibilities:
  - UUID generation for image_id / annotation_id
  - SHA-256 checksum computation
  - Image integrity validation via OpenCV decode
  - YOLO label file parsing + coordinate validation
  - Shared dataclasses used across both extractors
"""

import hashlib
import uuid
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ─── Shared dataclasses ───────────────────────────────────────────────────────

@dataclass
class AnnotationData:
    annotation_id: str
    image_id: str
    class_name: str
    class_id: int
    confidence: float                  # 1.0 for ground-truth labels
    x_center_norm: float
    y_center_norm: float
    width_norm: float
    height_norm: float
    x_min_px: Optional[int] = None
    y_min_px: Optional[int] = None
    x_max_px: Optional[int] = None
    y_max_px: Optional[int] = None


@dataclass
class ImageMetadata:
    image_id: str
    dataset_id: str
    file_path: str
    original_filename: str
    modality: str                      # 'visual' | 'thermal'
    width: int
    height: int
    channels: int
    file_size_mb: float
    checksum: str
    captured_at: Optional[str] = None  # ISO8601 string or None
    split_id: Optional[str] = None


@dataclass
class ExtractorResult:
    """Single return unit from any extractor — one image + its annotations."""
    metadata: ImageMetadata
    annotations: List[AnnotationData] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


# ─── Class name registry ─────────────────────────────────────────────────────

# Canonical class map used across all datasets.
# All dataset-specific names are normalised to these.
CLASS_REGISTRY = {
    "smoke":    0,
    "fire":     1,   # Mapped to 'flame' on export
    "flame":    1,
    "overheat": 2,
    # D-Fire uses integer-only labels (0=fire, 1=smoke)
    "0":        1,
    "1":        0,
}

CLASS_ID_TO_NAME = {0: "smoke", 1: "flame", 2: "overheat"}


# ─── Base extractor ───────────────────────────────────────────────────────────

class BaseExtractor(ABC):
    """
    Abstract base for VisualExtractor and ThermalExtractor.

    Subclasses must implement:
      - extract(file_path) -> ExtractorResult
    """

    def __init__(self, dataset_id: str, dataset_name: str):
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.log = logging.getLogger(self.__class__.__name__)

    # ── Public interface ─────────────────────────────────────────────────────

    @abstractmethod
    def extract(self, file_path: Path) -> ExtractorResult:
        """
        Extract metadata and annotations from a single image file.
        Must be implemented by subclass.
        """

    def extract_batch(self, file_paths: List[Path]) -> List[ExtractorResult]:
        """Process a list of image files, collecting results and skipping failures."""
        results = []
        for i, path in enumerate(file_paths):
            try:
                result = self.extract(path)
                results.append(result)
                if (i + 1) % 500 == 0:
                    self.log.info("  Extracted %d / %d", i + 1, len(file_paths))
            except Exception as e:
                self.log.warning("  Skipping %s: %s", path.name, e)
        return results

    # ── Shared utilities ─────────────────────────────────────────────────────

    @staticmethod
    def generate_id() -> str:
        """Generate a UUID4 string used as image_id or annotation_id."""
        return str(uuid.uuid4())

    @staticmethod
    def compute_checksum(file_path: Path) -> str:
        """SHA-256 checksum of raw file bytes. Used for deduplication."""
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def read_image(file_path: Path) -> Optional[np.ndarray]:
        """
        Decode image via OpenCV. Returns None if the file is corrupt or
        cannot be decoded — this is the primary integrity check.
        """
        img = cv2.imread(str(file_path))
        if img is None:
            return None
        return img

    @staticmethod
    def get_image_shape(img: np.ndarray) -> Tuple[int, int, int]:
        """Return (height, width, channels) from a decoded OpenCV image."""
        if img.ndim == 2:
            return img.shape[0], img.shape[1], 1
        return img.shape[0], img.shape[1], img.shape[2]

    @staticmethod
    def file_size_mb(file_path: Path) -> float:
        return file_path.stat().st_size / (1024 * 1024)

    # ── YOLO label parsing ───────────────────────────────────────────────────

    def parse_yolo_label_file(
        self,
        label_path: Path,
        image_id: str,
        img_width: int,
        img_height: int,
    ) -> List[AnnotationData]:
        """
        Parse a YOLO-format .txt label file.

        YOLO format (one detection per line):
            <class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>

        All coordinates are normalised to [0, 1].
        Returns an empty list if the file doesn't exist (negative sample).
        """
        if not label_path.exists():
            return []   # Legitimate negative sample — no annotations

        annotations = []
        with open(label_path) as f:
            lines = [l.strip() for l in f if l.strip()]

        for line_no, line in enumerate(lines):
            parts = line.split()
            if len(parts) < 5:
                self.log.debug(
                    "  Skipping malformed label line %d in %s", line_no, label_path.name
                )
                continue

            raw_class = parts[0]
            try:
                xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                self.log.debug("  Non-numeric coords in %s line %d", label_path.name, line_no)
                continue

            # Validate all values are in [0, 1]
            if not all(0.0 <= v <= 1.0 for v in [xc, yc, w, h]):
                self.log.debug(
                    "  Out-of-range coords in %s line %d: %s", label_path.name, line_no, parts
                )
                continue

            # Normalise class name using registry
            class_name, class_id = self._resolve_class(raw_class)
            if class_name is None:
                self.log.debug(
                    "  Unknown class '%s' in %s line %d — skipping", raw_class, label_path.name, line_no
                )
                continue

            # Derive pixel bounding box for convenience
            x_min = int((xc - w / 2) * img_width)
            y_min = int((yc - h / 2) * img_height)
            x_max = int((xc + w / 2) * img_width)
            y_max = int((yc + h / 2) * img_height)

            annotations.append(AnnotationData(
                annotation_id=BaseExtractor.generate_id(),
                image_id=image_id,
                class_name=class_name,
                class_id=class_id,
                confidence=1.0,         # Ground truth
                x_center_norm=xc,
                y_center_norm=yc,
                width_norm=w,
                height_norm=h,
                x_min_px=max(0, x_min),
                y_min_px=max(0, y_min),
                x_max_px=min(img_width, x_max),
                y_max_px=min(img_height, y_max),
            ))

        return annotations

    @staticmethod
    def _resolve_class(raw: str) -> Tuple[Optional[str], Optional[int]]:
        """Normalise raw class string to canonical (name, id) pair."""
        key = raw.strip().lower()
        if key in CLASS_REGISTRY:
            class_id = CLASS_REGISTRY[key]
            class_name = CLASS_ID_TO_NAME[class_id]
            return class_name, class_id
        # Try integer lookup
        try:
            cid = int(key)
            if cid in CLASS_ID_TO_NAME:
                return CLASS_ID_TO_NAME[cid], cid
        except ValueError:
            pass
        return None, None

    # ── EXIF timestamp helper ────────────────────────────────────────────────

    @staticmethod
    def read_exif_datetime(file_path: Path) -> Optional[str]:
        """
        Attempt to read capture date from EXIF.
        Returns ISO8601 string or None if unavailable.
        """
        try:
            import piexif
            exif = piexif.load(str(file_path))
            dt_bytes = exif.get("Exif", {}).get(piexif.ExifIFD.DateTimeOriginal)
            if dt_bytes:
                dt_str = dt_bytes.decode("ascii")    # "2024:01:15 10:30:00"
                return dt_str.replace(":", "-", 2)   # "2024-01-15 10:30:00"
        except Exception:
            pass
        return None