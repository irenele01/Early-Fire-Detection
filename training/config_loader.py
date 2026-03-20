"""
training/config_loader.py
──────────────────────────
Loads, validates, and resolves config/training.yaml.

Responsibilities:
  - Parse YAML with safe_load
  - Validate all required keys exist
  - Auto-detect available device (MPS → CUDA → CPU)
  - Write dataset.yaml for Ultralytics
  - Expose a single TrainingConfig dataclass to the rest of the pipeline
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


# ── Training config dataclass ─────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    # Model
    model_name:   str   = "yolov8n"
    pretrained:   bool  = True
    input_size:   int   = 320

    # Data paths
    data_path:    str   = "data/processed"
    train_dir:    str   = "images/train"
    val_dir:      str   = "images/val"
    test_dir:     str   = "images/test"
    class_names:  Dict  = field(default_factory=lambda: {0: "smoke", 1: "flame", 2: "overheat"})

    # Training hyperparameters
    epochs:       int   = 50
    batch_size:   int   = 16
    optimizer:    str   = "SGD"
    lr0:          float = 0.01
    lrf:          float = 0.01
    momentum:     float = 0.937
    weight_decay: float = 0.0005
    device:       str   = "cpu"
    patience:     int   = 20         # Early stopping patience (epochs without improvement)

    # Augmentation
    hsv_h:   float = 0.015
    hsv_s:   float = 0.7
    hsv_v:   float = 0.4
    flipud:  float = 0.0
    fliplr:  float = 0.5
    mosaic:  float = 1.0
    mixup:   float = 0.0

    # Export
    export_formats: List[str] = field(default_factory=lambda: ["onnx"])
    simplify:       bool      = True
    opset:          int       = 12

    # Output paths
    project_dir:   str = "models"
    run_name:      str = "dc_efds_run"
    weights_dir:   str = "models/weights"
    onnx_path:     str = "models/best.onnx"
    dataset_yaml:  str = "data/processed/dataset.yaml"

    @property
    def best_pt_path(self) -> Path:
        return Path(self.weights_dir) / "best.pt"

    @property
    def last_pt_path(self) -> Path:
        return Path(self.weights_dir) / "last.pt"

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


# ── YAML loader ───────────────────────────────────────────────────────────────

REQUIRED_KEYS = {
    "model":      ["name", "pretrained", "input_size"],
    "data":       ["path", "train", "val", "test", "names"],
    "training":   ["epochs", "batch_size", "optimizer", "lr0", "lrf",
                   "momentum", "weight_decay", "device"],
    "augmentation": ["hsv_h", "hsv_s", "hsv_v", "flipud", "fliplr", "mosaic"],
    "export":     ["formats", "simplify", "opset"],
}


def load_config(yaml_path: str = "config/training.yaml") -> TrainingConfig:
    """
    Load config/training.yaml and return a populated TrainingConfig.
    Exits with a clear message if the file is missing or malformed.
    """
    path = Path(yaml_path)
    if not path.exists():
        print(f"  ❌  Config not found: {path}")
        print("      Expected: config/training.yaml")
        _write_default_config(path)
        print(f"      Default config written to {path} — edit and re-run.")
        sys.exit(1)

    with open(path) as f:
        raw = yaml.safe_load(f)

    _validate_keys(raw, yaml_path)

    m  = raw["model"]
    d  = raw["data"]
    t  = raw["training"]
    a  = raw["augmentation"]
    ex = raw["export"]

    device = _resolve_device(t.get("device", "cpu"))

    cfg = TrainingConfig(
        # Model
        model_name   = m["name"],
        pretrained   = m.get("pretrained", True),
        input_size   = m["input_size"],
        # Data
        data_path    = d["path"],
        train_dir    = d["train"],
        val_dir      = d["val"],
        test_dir     = d.get("test", "images/test"),
        class_names  = d["names"],
        # Training
        epochs       = t["epochs"],
        batch_size   = t["batch_size"],
        optimizer    = t["optimizer"],
        lr0          = t["lr0"],
        lrf          = t["lrf"],
        momentum     = t["momentum"],
        weight_decay = t["weight_decay"],
        device       = device,
        patience     = t.get("patience", 20),
        # Augmentation
        hsv_h   = a["hsv_h"],
        hsv_s   = a["hsv_s"],
        hsv_v   = a["hsv_v"],
        flipud  = a["flipud"],
        fliplr  = a["fliplr"],
        mosaic  = a["mosaic"],
        mixup   = a.get("mixup", 0.0),
        # Export
        export_formats = ex["formats"],
        simplify       = ex.get("simplify", True),
        opset          = ex.get("opset", 12),
    )
    return cfg


def write_dataset_yaml(cfg: TrainingConfig) -> Path:
    """
    Write/overwrite data/processed/dataset.yaml — the file Ultralytics
    reads for data paths and class names during training.
    """
    out = Path(cfg.dataset_yaml)
    out.parent.mkdir(parents=True, exist_ok=True)

    content = {
        "path":  str(Path(cfg.data_path).resolve()),
        "train": cfg.train_dir,
        "val":   cfg.val_dir,
        "test":  cfg.test_dir,
        "nc":    cfg.num_classes,
        "names": cfg.class_names,
    }
    with open(out, "w") as f:
        yaml.dump(content, f, default_flow_style=False, sort_keys=False)
    return out


# ── Device resolution ─────────────────────────────────────────────────────────

def _resolve_device(requested: str) -> str:
    """
    Auto-detect and fall back in order: requested → MPS → CUDA → CPU.
    Returns the best available device string.
    """
    try:
        import torch
    except ImportError:
        return "cpu"

    if requested == "mps":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
        print("  ⚠️   MPS not available — falling back to CPU.")
        print("       Ensure you are on Apple Silicon with PyTorch ≥ 2.0.")
        return "cpu"

    if requested in ("cuda", "0", "1"):
        if torch.cuda.is_available():
            return requested
        print(f"  ⚠️   CUDA not available — falling back to CPU.")
        return "cpu"

    return "cpu"


# ── Config validation ─────────────────────────────────────────────────────────

def _validate_keys(raw: dict, path: str):
    missing = []
    for section, keys in REQUIRED_KEYS.items():
        if section not in raw:
            missing.append(f"  Missing section: [{section}]")
            continue
        for k in keys:
            if k not in raw[section]:
                missing.append(f"  [{section}].{k}")

    if missing:
        print(f"  ❌  Config validation failed: {path}")
        for m in missing:
            print(m)
        sys.exit(1)


# ── Default config writer ─────────────────────────────────────────────────────

DEFAULT_CONFIG = """\
model:
  name: yolov8n        # Nano variant — fits Pi 3 RAM (<400 MB)
  pretrained: true     # Start from COCO weights for faster convergence
  input_size: 320      # Critical: must match inference config on Pi 3

data:
  path: data/processed
  train: images/train
  val: images/val
  test: images/test
  names:
    0: smoke
    1: flame
    2: overheat        # Thermal class

training:
  epochs: 50
  batch_size: 16       # Reduce to 8 if MPS runs out of memory
  optimizer: SGD
  lr0: 0.01            # Initial learning rate
  lrf: 0.01            # Final LR = lr0 * lrf  (cosine schedule)
  momentum: 0.937
  weight_decay: 0.0005
  device: mps          # mps = Apple Silicon, cuda = NVIDIA, cpu = fallback
  patience: 20         # Early stopping: stop if no improvement for N epochs

augmentation:
  hsv_h: 0.015         # Minimal hue — fire has a narrow orange/red range
  hsv_s: 0.7           # Saturation: smoke (low) to flame (high)
  hsv_v: 0.4           # Value: different lighting conditions
  flipud: 0.0          # Never flip vertically — fire rises, not falls
  fliplr: 0.5          # Horizontal flip is safe
  mosaic: 1.0          # Mosaic helps with small smoke detection
  mixup: 0.0           # MixUp off — fire colours must stay distinct

export:
  formats: [onnx]
  simplify: true       # Fuse BN layers, remove identity nodes (~30% size reduction)
  opset: 12            # Last opset with full ONNX Runtime ARM support
"""


def _write_default_config(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(DEFAULT_CONFIG)