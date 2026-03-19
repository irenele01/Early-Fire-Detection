"""
Step 2 of 5 — Download all training datasets.

Datasets:
  Visual  (fire/smoke):
    1. D-Fire        — git clone, automated          (~1.2 GB, 21,000 images)
    2. Roboflow      — API key or manual download    (~0.5 GB,  6,400 images)
    3. Kaggle Fire   — kaggle CLI or manual download (~2.0 GB, 17,000 images)

  Thermal:
    4. FLIR Thermal  — git clone, automated          (~3.0 GB, 14,000 images)
    5. KAIST         — manual request (optional)     (~65 GB,  95,000 images)

  Personal:
    6. Personal      — captured in Step 3 (skip here)

Usage:
    python scripts/download_all_datasets.py                         # all automated
    python scripts/download_all_datasets.py --datasets dfire flir   # specific
    python scripts/download_all_datasets.py --skip-large            # skip KAIST
    python scripts/download_all_datasets.py --check                 # status only
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ── Utilities ─────────────────────────────────────────────────────────────────

def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.suffix.lower() in IMG_EXTS)


def run(cmd: list, timeout: int = 1800) -> bool:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    try:
        result = subprocess.run(cmd, timeout=timeout)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  ❌  Timed out after {timeout}s")
        return False
    except FileNotFoundError:
        print(f"  ❌  Command not found: {cmd[0]}")
        return False


def git_ok() -> bool:
    return subprocess.run(["git", "--version"], capture_output=True).returncode == 0


def _section(title: str):
    print(f"\n  {'─'*52}")
    print(f"  {title}")
    print(f"  {'─'*52}")


# ── Dataset 1: D-Fire ─────────────────────────────────────────────────────────

def download_dfire() -> bool:
    """
    D-Fire — 21,000 images, YOLO bboxes, research license.
    Classes: 0 = flame, 1 = smoke (+ 9,838 true negatives).
    Cite: de Venâncio et al., Engineering Applications of AI, 2022.
    """
    _section("D-Fire Dataset  (automated git clone)")
    dest = RAW / "dfire"

    existing = count_images(dest)
    if existing > 1000:
        print(f"  ✅  Already downloaded: {existing:,} images")
        return True

    if not git_ok():
        print("  ❌  git not found — install git and retry.")
        return False

    print("  Cloning (~1.2 GB, a few minutes)...")
    ok = run(["git", "clone", "--depth=1",
              "https://github.com/gaiasd/DFireDataset.git", str(dest)])

    if ok:
        n = count_images(dest)
        print(f"  ✅  D-Fire: {n:,} images → {dest}")
    else:
        print("  ❌  Clone failed.")
        print("      Manual: https://github.com/gaiasd/DFireDataset")
    return ok


# ── Dataset 2: Roboflow ───────────────────────────────────────────────────────

def download_roboflow() -> bool:
    """
    Roboflow Fire & Smoke — 6,400 images, MIT license, high-quality bboxes.
    URL: https://universe.roboflow.com/middle-east-tech-university/fire-and-smoke-detection-hiwia
    Requires: ROBOFLOW_API_KEY env var  OR  manual download.
    """
    _section("Roboflow Fire & Smoke  (API key required)")
    dest = RAW / "roboflow"

    existing = count_images(dest)
    if existing > 100:
        print(f"  ✅  Already downloaded: {existing:,} images")
        return True

    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()

    if not api_key:
        print("  ⚠️   ROBOFLOW_API_KEY not set.")
        print()
        print("  Option A — API download:")
        print("    1. Sign up free at https://roboflow.com")
        print("    2. Visit: https://universe.roboflow.com/middle-east-tech-university")
        print("              /fire-and-smoke-detection-hiwia")
        print("    3. Click 'Download' → YOLOv8 → 'show download code'")
        print("    4. Copy your API key then run:")
        print("         export ROBOFLOW_API_KEY=<your_key>")
        print("         python scripts/download_all_datasets.py --datasets roboflow")
        print()
        print("  Option B — manual download:")
        print("    1. Same URL above → Export → YOLOv8 → ZIP")
        print(f"   2. Unzip to: {dest.resolve()}/")
        print("       Expected: roboflow/train/images/*.jpg")
        print("                 roboflow/train/labels/*.txt")
        print("                 roboflow/valid/images/*.jpg  ← 'valid' not 'val'")
        return False

    try:
        from roboflow import Roboflow
    except ImportError:
        print("  ❌  roboflow package missing: pip install roboflow")
        return False

    print("  Downloading via Roboflow API...")
    try:
        rf      = Roboflow(api_key=api_key)
        project = rf.workspace("middle-east-tech-university").project(
            "fire-and-smoke-detection-hiwia"
        )
        project.version(1).download("yolov8", location=str(dest))
        n = count_images(dest)
        print(f"  ✅  Roboflow: {n:,} images → {dest}")
        return True
    except Exception as e:
        print(f"  ❌  Roboflow API error: {e}")
        return False


# ── Dataset 3: Kaggle ─────────────────────────────────────────────────────────

def download_kaggle() -> bool:
    """
    Kaggle Fire Detection — 17,000 images, folder-based labels.
    Classes: fire/ (positive), no_fire/ (negative).
    Requires: kaggle CLI + ~/.kaggle/kaggle.json API credentials.
    """
    _section("Kaggle Fire Detection  (CLI required)")
    dest = RAW / "kaggle_fire"

    existing = count_images(dest)
    if existing > 1000:
        print(f"  ✅  Already downloaded: {existing:,} images")
        return True

    kaggle_ok = subprocess.run(["kaggle", "--version"], capture_output=True).returncode == 0
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"

    if not kaggle_ok or not kaggle_json.exists():
        print("  ⚠️   Kaggle CLI not configured.")
        print()
        print("  Setup:")
        print("    1. pip install kaggle")
        print("    2. https://www.kaggle.com → Account → API → Create New Token")
        print("    3. mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json")
        print("    4. chmod 600 ~/.kaggle/kaggle.json")
        print("    5. Re-run this script")
        print()
        print("  Manual download:")
        print("    https://www.kaggle.com/datasets/phasukkk/firedetection")
        print(f"   Unzip to: {dest.resolve()}/")
        print("       Expected: kaggle_fire/fire/*.jpg")
        print("                 kaggle_fire/no_fire/*.jpg")
        return False

    dest.mkdir(parents=True, exist_ok=True)
    print("  Downloading via Kaggle CLI (~2 GB)...")
    ok = run(["kaggle", "datasets", "download",
               "-d", "phasukkk/firedetection",
               "-p", str(dest), "--unzip"], timeout=600)

    if ok:
        n = count_images(dest)
        print(f"  ✅  Kaggle Fire: {n:,} images → {dest}")
    else:
        print("  ❌  Kaggle download failed — see manual instructions above.")
    return ok


# ── Dataset 4: FLIR Thermal ───────────────────────────────────────────────────

def download_flir() -> bool:
    """
    FLIR Thermal Dataset — 14,000 radiometric JPEG images.
    Contains embedded Planck constants for full temperature extraction.
    License: Free for research and commercial use.
    """
    _section("FLIR Thermal Dataset  (automated git clone)")
    dest = RAW / "flir_thermal"

    existing = count_images(dest)
    if existing > 1000:
        print(f"  ✅  Already downloaded: {existing:,} images")
        return True

    if not git_ok():
        print("  ❌  git not found.")
        return False

    print("  Cloning (~3 GB, may take 10-15 minutes)...")
    ok = run(["git", "clone", "--depth=1",
              "https://github.com/flir/thermal-dataset.git", str(dest)], timeout=1800)

    if ok:
        n = count_images(dest)
        print(f"  ✅  FLIR Thermal: {n:,} images → {dest}")
    else:
        print("  ❌  Clone failed.")
        print("      Manual: https://www.flir.com/oem/adas/adas-dataset-form/")
        print("      Alt:    https://github.com/flir/thermal-dataset")
    return ok


# ── Dataset 5: KAIST (optional) ───────────────────────────────────────────────

def download_kaist() -> bool:
    """
    KAIST Multi-Spectral — 95,000 paired visible+thermal frames. (~65 GB)
    Requires manual access request. Optional for initial training.
    """
    _section("KAIST Multi-Spectral  (manual request — optional)")
    dest = RAW / "kaist"

    existing = count_images(dest)
    if existing > 1000:
        print(f"  ✅  Already downloaded: {existing:,} images")
        return True

    print("  ⚠️   KAIST requires a manual access request — cannot be automated.")
    print()
    print("  Steps:")
    print("    1. Visit: http://multispectral.ece.kaist.ac.kr/")
    print("    2. Submit the access request form (approved within 1–2 days)")
    print("    3. Download thermal (lwir) subset only to save space:")
    print("         set00.tar – set05.tar  (~30 GB instead of 65 GB)")
    print(f"   4. Extract to: {dest.resolve()}/")
    print("       Expected: kaist/sets/set00/V000/lwir/*.png")
    print()
    print("  ℹ️   KAIST is optional for initial training.")
    print("      D-Fire + FLIR provide sufficient coverage to start.")
    print("      Re-run with --datasets kaist after downloading.")
    return True  # Non-blocking


# ── Status check ──────────────────────────────────────────────────────────────

DATASET_META = {
    "dfire":    ("D-Fire",          RAW / "dfire",            21000),
    "roboflow": ("Roboflow",        RAW / "roboflow",          6400),
    "kaggle":   ("Kaggle Fire",     RAW / "kaggle_fire",      17000),
    "flir":     ("FLIR Thermal",    RAW / "flir_thermal",     14000),
    "kaist":    ("KAIST",           RAW / "kaist",                0),  # optional
    "personal": ("Personal (Step 3)", RAW / "personal_negatives", 500),
}


def print_status():
    print()
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │  Dataset status                                         │")
    print("  ├──────────────────────┬──────────┬──────────┬───────────┤")
    print("  │  Dataset             │  Found   │  Target  │  Status   │")
    print("  ├──────────────────────┼──────────┼──────────┼───────────┤")

    total = 0
    for key, (name, path, target) in DATASET_META.items():
        n = count_images(path)
        total += n
        if target == 0:
            status = "optional"
        elif n >= target * 0.8:
            status = "✅ ready"
        elif n > 0:
            status = "⚠️  partial"
        else:
            status = "❌ missing"
        print(f"  │  {name:<20}│ {n:>8,} │ {target:>8,} │  {status:<8} │")

    print("  ├──────────────────────┴──────────┴──────────┴───────────┤")
    print(f"  │  Total raw images: {total:,}{' '*(37-len(str(total)))}│")
    print("  └─────────────────────────────────────────────────────────┘")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

DOWNLOADERS = {
    "dfire":    download_dfire,
    "roboflow": download_roboflow,
    "kaggle":   download_kaggle,
    "flir":     download_flir,
    "kaist":    download_kaist,
}

DEFAULT_DATASETS = ["dfire", "roboflow", "kaggle", "flir"]


def main():
    parser = argparse.ArgumentParser(
        description="DC-EFDS-Lite — Step 2: Download all training datasets"
    )
    parser.add_argument(
        "--datasets", nargs="+",
        choices=list(DOWNLOADERS.keys()),
        default=DEFAULT_DATASETS,
        help="Datasets to download (default: dfire roboflow kaggle flir)",
    )
    parser.add_argument("--skip-large", action="store_true", help="Skip KAIST (65 GB)")
    parser.add_argument("--check",      action="store_true", help="Status check only, no download")
    args = parser.parse_args()

    print()
    print("╔════════════════════════════════════════════════════════╗")
    print("║  DC-EFDS-Lite  ·  Step 2 of 5: Download Datasets      ║")
    print("╚════════════════════════════════════════════════════════╝")

    if args.check:
        print_status()
        return

    datasets = args.datasets
    if args.skip_large:
        datasets = [d for d in datasets if d != "kaist"]
        print("\n  ℹ️   --skip-large: KAIST excluded.")

    results = {ds: DOWNLOADERS[ds]() for ds in datasets}

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("  ══════════════════════  Summary  ══════════════════════")
    succeeded = [k for k, v in results.items() if v]
    failed    = [k for k, v in results.items() if not v]
    for ds in succeeded:
        n = count_images(DATASET_META[ds][1])
        print(f"  ✅  {ds:<15}  {n:>7,} images")
    for ds in failed:
        print(f"  ❌  {ds:<15}  — manual action needed (see above)")

    print_status()

    if failed:
        print("  ⚠️   Some datasets need manual download — training can proceed without them.")
    else:
        print("  ✅  All automated datasets ready.")

    print()
    print("  ─────────────────────────────────────────────────────")
    print("  Next:  python scripts/capture_personal_data.py")
    print()


if __name__ == "__main__":
    main()