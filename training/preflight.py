"""
training/preflight.py
──────────────────────
Stage 1 — Pre-flight checks before training starts.

Validates the processed dataset from the ETL pipeline:
  - Directory structure and non-empty splits
  - Image/label pairing (spot check)
  - Class distribution (warn if severely imbalanced)
  - YAML integrity
  - Hardware readiness (MPS / CUDA / RAM)
  - Disk space for model artefacts

All checks print a clear pass/warn/fail status.
Fails fast on hard blockers. Warns on recoverable issues.
"""

import shutil
import sqlite3
import sys
from pathlib import Path
from typing import List, Tuple

IMG_EXTS = {".jpg", ".jpeg", ".png"}


# ── Check functions ───────────────────────────────────────────────────────────

def check_directories(cfg) -> Tuple[bool, List[str]]:
    """Verify that all required processed directories exist and are non-empty."""
    required = {
        "images/train":  "training images",
        "images/val":    "validation images",
        "labels/train":  "training labels",
        "labels/val":    "validation labels",
    }
    base   = Path(cfg.data_path)
    errors = []
    for subdir, desc in required.items():
        full = base / subdir
        if not full.exists():
            errors.append(f"Directory missing: {full}")
            continue
        files = list(full.iterdir())
        if not files:
            errors.append(f"Empty directory ({desc}): {full}")
    return len(errors) == 0, errors


def count_split(cfg, split: str) -> Tuple[int, int]:
    """Return (n_images, n_labeled) for a given split."""
    base     = Path(cfg.data_path)
    img_dir  = base / f"images/{split}"
    lbl_dir  = base / f"labels/{split}"

    if not img_dir.exists():
        return 0, 0

    images  = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    labeled = 0
    for img in images:
        lbl = lbl_dir / (img.stem + ".txt")
        if lbl.exists() and lbl.stat().st_size > 0:
            labeled += 1
    return len(images), labeled


def check_image_label_pairing(cfg) -> Tuple[bool, List[str]]:
    """
    Spot-check that label files exist for images and have valid YOLO content.
    Checks all val images (small set) and up to 500 training images.
    """
    base   = Path(cfg.data_path)
    errors = []
    warns  = []

    for split in ["train", "val"]:
        img_dir = base / f"images/{split}"
        lbl_dir = base / f"labels/{split}"

        if not img_dir.exists():
            continue

        images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        sample = images if split == "val" else images[:500]

        orphaned     = 0
        bad_format   = 0
        out_of_range = 0

        for img_path in sample:
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                orphaned += 1
                continue
            # Validate YOLO format: each line must have 5 floats, all in [0,1]
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 0:
                        continue
                    if len(parts) != 5:
                        bad_format += 1
                        break
                    try:
                        vals = [float(x) for x in parts[1:]]
                        if any(v < 0 or v > 1 for v in vals):
                            out_of_range += 1
                    except ValueError:
                        bad_format += 1

        if orphaned > len(sample) * 0.10:
            errors.append(
                f"[{split}] {orphaned}/{len(sample)} images have no label file "
                f"(>10% threshold — likely ETL issue)"
            )
        elif orphaned > 0:
            warns.append(
                f"[{split}] {orphaned}/{len(sample)} images without labels "
                f"(negatives — expected)"
            )

        if bad_format > 0:
            errors.append(f"[{split}] {bad_format} label files have invalid YOLO format")
        if out_of_range > 0:
            errors.append(f"[{split}] {out_of_range} labels have coordinates outside [0,1]")

    return len(errors) == 0, errors + [f"⚠️  {w}" for w in warns]


def check_class_distribution(cfg) -> Tuple[bool, List[str]]:
    """
    Check class balance in training labels.
    Warns if any class has fewer than 100 instances (likely underrepresented).
    Warns if the imbalance ratio exceeds 20:1.
    Does NOT fail — class imbalance is common in fire datasets.
    """
    lbl_dir = Path(cfg.data_path) / "labels/train"
    if not lbl_dir.exists():
        return True, []

    counts = {name: 0 for name in cfg.class_names.values()}
    total_annotations = 0

    for lbl_path in lbl_dir.glob("*.txt"):
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    try:
                        cid = int(parts[0])
                        name = cfg.class_names.get(cid, f"class_{cid}")
                        counts[name] = counts.get(name, 0) + 1
                        total_annotations += 1
                    except ValueError:
                        pass

    msgs = []
    if total_annotations == 0:
        msgs.append("⚠️  No annotations found in training labels")
        return True, msgs

    max_count = max(counts.values()) if counts.values() else 1
    min_count = min(counts.values()) if counts.values() else 0

    for cls, cnt in counts.items():
        pct = cnt / total_annotations * 100
        msgs.append(f"  {cls:<12}  {cnt:>8,}  ({pct:.1f}%)")

    msgs.append(f"  {'total':<12}  {total_annotations:>8,}")

    if min_count < 100:
        msgs.append(
            f"  ⚠️  '{min(counts, key=counts.get)}' has only {min_count} instances — "
            f"consider collecting more data for this class"
        )

    if max_count > 0 and min_count > 0 and (max_count / min_count) > 20:
        msgs.append(
            f"  ⚠️  Imbalance ratio {max_count/min_count:.0f}:1 — "
            f"training may be biased toward '{max(counts, key=counts.get)}'"
        )

    return True, msgs


def check_hardware(cfg) -> Tuple[bool, List[str]]:
    """Verify the requested compute device is available and has sufficient RAM."""
    msgs   = []
    errors = []

    try:
        import torch
    except ImportError:
        errors.append("PyTorch not installed: pip install torch")
        return False, errors

    msgs.append(f"PyTorch version : {torch.__version__}")

    if cfg.device == "mps":
        if torch.backends.mps.is_available():
            msgs.append("MPS (Apple Silicon) : ✅ available")
            # Check memory — MPS shares system RAM; warn if <8 GB total
            try:
                import psutil
                total_gb = psutil.virtual_memory().total / (1024 ** 3)
                msgs.append(f"System RAM        : {total_gb:.1f} GB")
                if total_gb < 8:
                    msgs.append(
                        f"  ⚠️  Only {total_gb:.1f} GB RAM — reduce batch_size to 8 "
                        f"in config/training.yaml if training crashes"
                    )
            except ImportError:
                pass
        else:
            errors.append(
                "MPS requested but not available. "
                "Ensure you are on Apple Silicon with PyTorch ≥ 2.0. "
                "Edit config/training.yaml: device: cpu"
            )

    elif cfg.device in ("cuda", "0", "1"):
        if torch.cuda.is_available():
            name  = torch.cuda.get_device_name(0)
            vram  = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            msgs.append(f"CUDA GPU     : ✅ {name}  ({vram:.1f} GB VRAM)")
        else:
            errors.append("CUDA requested but no GPU found. Edit device: cpu")

    else:
        msgs.append("Device       : CPU (training will be slow — ~4–8× longer than MPS)")

    return len(errors) == 0, errors + msgs


def check_disk_space(min_gb: float = 5.0) -> Tuple[bool, List[str]]:
    """Warn if less than min_gb free disk space (training artefacts can be large)."""
    usage  = shutil.disk_usage(".")
    free   = usage.free / (1024 ** 3)
    total  = usage.total / (1024 ** 3)

    if free < min_gb:
        return False, [
            f"Low disk space: {free:.1f} GB free (recommend {min_gb} GB+). "
            f"Training artefacts for YOLOv8n are ~200 MB."
        ]
    return True, [f"Disk space   : {free:.1f} GB free / {total:.1f} GB total  ✅"]


def check_dataset_yaml(cfg) -> Tuple[bool, List[str]]:
    """Verify dataset.yaml exists and matches the training config."""
    import yaml as _yaml
    path = Path(cfg.dataset_yaml)
    if not path.exists():
        return False, [f"dataset.yaml not found: {path}  — run the ETL pipeline first"]

    with open(path) as f:
        ds = _yaml.safe_load(f)

    errors = []
    for key in ("path", "train", "val", "names"):
        if key not in ds:
            errors.append(f"dataset.yaml missing key: {key}")

    if "names" in ds and ds["names"] != cfg.class_names:
        errors.append(
            f"dataset.yaml class names mismatch:\n"
            f"  yaml:   {ds['names']}\n"
            f"  config: {cfg.class_names}"
        )

    return len(errors) == 0, errors if errors else [f"dataset.yaml   : ✅ {path}"]


# ── Public entry point ────────────────────────────────────────────────────────

def run_preflight(cfg) -> bool:
    """
    Run all pre-flight checks. Returns True if training can proceed.
    Prints a formatted report. Hard failures (False) should abort training.
    """
    print()
    print("  ━━━  Stage 1: Pre-flight Checks  ━━━━━━━━━━━━━━━━━━━━")

    all_pass = True

    # ── Directories ───────────────────────────────────────────────────────────
    print("\n  [1/6] Dataset directories")
    ok, msgs = check_directories(cfg)
    _print_results(ok, msgs)
    if not ok:
        all_pass = False

    # ── Image counts ──────────────────────────────────────────────────────────
    print("\n  [2/6] Split image counts")
    for split in ["train", "val", "test"]:
        n_img, n_lbl = count_split(cfg, split)
        neg = n_img - n_lbl
        status = "✅" if n_img > 0 else "⚠️ "
        print(f"       {status}  {split:<6}  {n_img:>7,} images  "
              f"{n_lbl:>7,} labeled  {neg:>6,} negatives")

    # ── Label pairing ─────────────────────────────────────────────────────────
    print("\n  [3/6] Image-label pairing")
    ok, msgs = check_image_label_pairing(cfg)
    _print_results(ok, msgs)
    if not ok:
        all_pass = False

    # ── Class distribution ────────────────────────────────────────────────────
    print("\n  [4/6] Class distribution (training set)")
    _, msgs = check_class_distribution(cfg)
    for m in msgs:
        print(f"       {m}")

    # ── Hardware ──────────────────────────────────────────────────────────────
    print("\n  [5/6] Hardware and compute")
    ok, msgs = check_hardware(cfg)
    _print_results(ok, msgs)
    if not ok:
        all_pass = False

    # ── Disk space ────────────────────────────────────────────────────────────
    print("\n  [6/6] Disk space and dataset.yaml")
    ok_disk, msgs_disk = check_disk_space()
    ok_yaml, msgs_yaml = check_dataset_yaml(cfg)
    _print_results(ok_disk and ok_yaml, msgs_disk + msgs_yaml)
    if not (ok_disk and ok_yaml):
        all_pass = False

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    if all_pass:
        print("  ✅  Pre-flight passed — starting training.")
    else:
        print("  ❌  Pre-flight failed — fix issues above before training.")

    return all_pass


def _print_results(ok: bool, msgs: List[str]):
    for msg in msgs:
        prefix = "  ❌ " if (not ok and not msg.startswith("⚠️")) else "       "
        print(f"{prefix}{msg}")
    if not msgs:
        status = "✅" if ok else "❌"
        print(f"       {status}")