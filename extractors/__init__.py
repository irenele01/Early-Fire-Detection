"""
extractors/
──────────
ETL extraction layer for DC-EFDS-Lite.

  BaseExtractor      — shared utilities, YOLO parser, dataclasses
  VisualExtractor    — RGB image extraction (D-Fire, Roboflow, Kaggle, Personal)
  ThermalExtractor   — FLIR radiometric extraction with Planck decode
"""

from extractors.base_extractor import (
    BaseExtractor,
    ExtractorResult,
    ImageMetadata,
    AnnotationData,
    CLASS_REGISTRY,
    CLASS_ID_TO_NAME,
)

from extractors.visual_extractor import (
    VisualExtractor,
    scan_dfire,
    scan_roboflow,
    scan_kaggle,
    scan_personal,
)

from extractors.thermal_extractor import (
    ThermalExtractor,
    ThermalExtractorResult,
    ThermalMetadata,
    scan_flir,
    scan_kaist,
)

__all__ = [
    "BaseExtractor", "ExtractorResult", "ImageMetadata", "AnnotationData",
    "CLASS_REGISTRY", "CLASS_ID_TO_NAME",
    "VisualExtractor", "scan_dfire", "scan_roboflow", "scan_kaggle", "scan_personal",
    "ThermalExtractor", "ThermalExtractorResult", "ThermalMetadata",
    "scan_flir", "scan_kaist",
]