#!/usr/bin/env python3
"""
Step 4 of 5 — Run the full ETL pipeline.

Follows the ETL diagram exactly:
  EXTRACT   → scan all raw datasets, read metadata + annotations
  TRANSFORM → resize 320×320, privacy blur, label validate, quality check
  LOAD      → write to SQLite + processed filesystem + dataset.yaml

Usage:
    python scripts/run_etl.py
    python scripts/run_etl.py --datasets dfire roboflow personal
    python scripts/run_etl.py --dry-run          # count only, no writes
    python scripts/run_etl.py --skip-transform   # load raw paths into DB
    python scripts/run_etl.py --resume           # skip already-loaded images
"""

import argparse
import random
import sqlite3
import sys
import time
import uuid
import yaml
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# ── Path setup (run from project root) ───────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from extractors.base_extractor import ExtractorResult, ImageMetadata, AnnotationData
from extractors.thermal_extractor import ThermalExtractorResult, ThermalMetadata
from extractors.visual_extractor  import scan_dfire, scan_roboflow, scan_kaggle, scan_personal
from extractors.thermal_extractor import scan_flir, scan_kaist

# ── Constants ─────────────────────────────────────────────────────────────────

RAW_DIR   = Path("data/raw")
PROCESSED = Path("data/processed")
DB_PATH   = Path("database/dc_efds.db")
LOG_PATH  = Path("logs/etl.log")

SPLIT_IDS = {"train": "split-train", "val": "split-val", "test": "split-test"}

ETL_CONFIG = {
    "splits":       {"train": 0.70, "val": 0.20, "test": 0.10},
    "stratify_by":  ["modality", "class"],
    "target_size":  (320, 320),
    "jpeg_quality": 85,
    "min_dim":      50,
    "max_file_mb":  50.0,
    "privacy_blur": True,
    "blur_kernel":  (51, 51),
}

# Map dataset key → (raw subdirectory, modality)
DATASET_MAP = {
    "dfire":    ("dfire",             "visual"),
    "roboflow": ("roboflow",          "visual"),
    "kaggle":   ("kaggle_fire",       "visual"),
    "personal": ("personal_negatives","visual"),
    "flir":     ("flir_thermal",      "thermal"),
    "kaist":    ("kaist",             "thermal"),
}

SCANNERS = {
    "dfire":    lambda: scan_dfire(str(RAW_DIR / "dfire")),
    "roboflow": lambda: scan_roboflow(str(RAW_DIR / "roboflow")),
    "kaggle":   lambda: scan_kaggle(str(RAW_DIR / "kaggle_fire")),
    "personal": lambda: scan_personal(str(RAW_DIR / "personal_negatives")),
    "flir":     lambda: scan_flir(str(RAW_DIR / "flir_thermal")),
    "kaist":    lambda: scan_kaist(str(RAW_DIR / "kaist")),
}


# ── Logging ───────────────────────────────────────────────────────────────────

import logging

LOG_PATH.parent.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_PATH), mode="a"),
    ],
)
log = logging.getLogger("etl")


# ── Database helpers ──────────────────────────────────────────────────────────

def get_conn(db_path: Path = DB_PATH) -> sqlite3.Connection:
    if not db_path.exists():
        log.error("Database not found: %s", db_path)
        log.error("Run: python scripts/init_database.py first.")
        sys.exit(1)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous  = NORMAL")
    conn.execute("PRAGMA cache_size   = -131072")  # 128 MB cache for ETL speed
    return conn


def already_loaded(conn: sqlite3.Connection, dataset_id: str) -> set:
    """Return set of original_filename values already in DB for this dataset."""
    rows = conn.execute(
        "SELECT original_filename FROM images WHERE dataset_id = ?", (dataset_id,)
    ).fetchall()
    return {r[0] for r in rows}


def upsert_dataset_meta(conn: sqlite3.Connection, dataset_id: str, modality: str):
    conn.execute(
        "INSERT OR IGNORE INTO datasets (dataset_id, name, modality) VALUES (?,?,?)",
        (dataset_id, dataset_id, modality),
    )
    conn.commit()


# ── TRANSFORM stage ───────────────────────────────────────────────────────────

class Transformer:
    """
    Applies the transformation chain from the ETL diagram:
      1. Resize to 320×320
      2. Privacy blur  (Haar face detector — GDPR compliance)
      3. Label conversion / coordinate adjustment
      4. Quality validation
    """

    W, H = ETL_CONFIG["target_size"]
    Q    = ETL_CONFIG["jpeg_quality"]

    def __init__(self):
        self._face_cascade: Optional[cv2.CascadeClassifier] = None
        if ETL_CONFIG["privacy_blur"]:
            path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cc   = cv2.CascadeClassifier(path)
            if not cc.empty():
                self._face_cascade = cc

    def run(
        self,
        result:     ExtractorResult,
        split_name: str,
        skip_copy:  bool = False,
    ) -> Tuple[Optional[Path], List[AnnotationData], List[str]]:
        """
        Returns (processed_path, adjusted_annotations, errors).
        processed_path is None if quality validation fails.
        """
        meta = result.metadata
        src  = Path(meta.file_path)
        errs = self._validate(meta, src)
        if errs:
            return None, [], errs

        if skip_copy:
            return src, result.annotations, []

        # ── Read source ──────────────────────────────────────────────────────
        img = cv2.imread(str(src))
        if img is None:
            return None, [], [f"cv2.imread failed: {src.name}"]

        # ── Resize ───────────────────────────────────────────────────────────
        img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)

        # ── Privacy blur ──────────────────────────────────────────────────────
        if self._face_cascade is not None:
            img = self._blur_faces(img)

        # ── Output paths ─────────────────────────────────────────────────────
        sub  = "images" if meta.modality == "visual" else "images_thermal"
        idir = PROCESSED / sub / split_name
        ldir = PROCESSED / "labels" / split_name
        idir.mkdir(parents=True, exist_ok=True)
        ldir.mkdir(parents=True, exist_ok=True)

        out_img = idir / f"{meta.image_id}.jpg"
        out_lbl = ldir / f"{meta.image_id}.txt"

        cv2.imwrite(str(out_img), img, [cv2.IMWRITE_JPEG_QUALITY, self.Q])

        # ── Adjust pixel bbox coords to new size ──────────────────────────────
        adjusted = [
            AnnotationData(
                annotation_id = ann.annotation_id,
                image_id      = ann.image_id,
                class_name    = ann.class_name,
                class_id      = ann.class_id,
                confidence    = ann.confidence,
                x_center_norm = ann.x_center_norm,
                y_center_norm = ann.y_center_norm,
                width_norm    = ann.width_norm,
                height_norm   = ann.height_norm,
                x_min_px = int((ann.x_center_norm - ann.width_norm  / 2) * self.W),
                y_min_px = int((ann.y_center_norm - ann.height_norm / 2) * self.H),
                x_max_px = int((ann.x_center_norm + ann.width_norm  / 2) * self.W),
                y_max_px = int((ann.y_center_norm + ann.height_norm / 2) * self.H),
            )
            for ann in result.annotations
        ]

        # ── Write YOLO label file ─────────────────────────────────────────────
        with open(out_lbl, "w") as f:
            for ann in adjusted:
                f.write(
                    f"{ann.class_id} {ann.x_center_norm:.6f} {ann.y_center_norm:.6f} "
                    f"{ann.width_norm:.6f} {ann.height_norm:.6f}\n"
                )

        return out_img, adjusted, []

    @staticmethod
    def _validate(meta: ImageMetadata, src: Path) -> List[str]:
        errs = []
        if not src.exists():
            errs.append(f"Missing: {src}")
        if meta.width < ETL_CONFIG["min_dim"] or meta.height < ETL_CONFIG["min_dim"]:
            errs.append(f"Too small: {meta.width}×{meta.height}")
        if meta.file_size_mb > ETL_CONFIG["max_file_mb"]:
            errs.append(f"Too large: {meta.file_size_mb:.1f} MB")
        return errs

    def _blur_faces(self, img: np.ndarray) -> np.ndarray:
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
        )
        for (x, y, w, h) in faces:
            img[y:y+h, x:x+w] = cv2.GaussianBlur(
                img[y:y+h, x:x+w], ETL_CONFIG["blur_kernel"], 0
            )
        return img


# ── SPLIT ASSIGNMENT ──────────────────────────────────────────────────────────

def assign_splits(
    results: List[ExtractorResult],
    cfg:     dict,
) -> List[Tuple[ExtractorResult, str]]:
    """
    Stratified split by (modality × primary_class).

    Ensures visual/thermal balance and smoke/flame/negative balance
    are maintained in each of the three splits — not just globally.
    """
    random.seed(42)

    strata: dict = defaultdict(list)
    for r in results:
        modality = r.metadata.modality
        if r.annotations:
            cls = max(r.annotations, key=lambda a: a.confidence).class_name
        else:
            cls = "negative"
        strata[f"{modality}_{cls}"].append(r)

    train_r = cfg["splits"]["train"]
    val_r   = cfg["splits"]["val"]

    assigned = []
    for stratum, items in strata.items():
        random.shuffle(items)
        n       = len(items)
        n_train = int(n * train_r)
        n_val   = int(n * val_r)
        for i, r in enumerate(items):
            split = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
            assigned.append((r, split))

    counts = defaultdict(int)
    for _, s in assigned:
        counts[s] += 1
    log.info("  Split assignment: train=%d  val=%d  test=%d",
             counts["train"], counts["val"], counts["test"])
    return assigned


# ── LOAD helpers ──────────────────────────────────────────────────────────────

def load_image(
    conn:       sqlite3.Connection,
    meta:       ImageMetadata,
    split_name: str,
    proc_path:  Path,
) -> bool:
    try:
        conn.execute("""
            INSERT OR REPLACE INTO images
              (image_id, dataset_id, split_id, file_path, original_filename,
               modality, width, height, channels, file_size_mb,
               captured_at, checksum, validated, privacy_cleared)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                meta.image_id, meta.dataset_id, SPLIT_IDS[split_name],
                str(proc_path.resolve()),
                meta.original_filename,
                meta.modality,
                ETL_CONFIG["target_size"][0],  # always 320 post-resize
                ETL_CONFIG["target_size"][1],
                meta.channels,
                proc_path.stat().st_size / (1024 * 1024),
                meta.captured_at,
                meta.checksum,
                True,   # validated = True (passed quality check)
                ETL_CONFIG["privacy_blur"],
            ),
        )
        return True
    except sqlite3.Error as e:
        log.debug("  load_image error %s: %s", meta.image_id, e)
        return False


def load_annotations(
    conn:        sqlite3.Connection,
    annotations: List[AnnotationData],
) -> int:
    loaded = 0
    for ann in annotations:
        try:
            conn.execute("""
                INSERT OR REPLACE INTO annotations
                  (annotation_id, image_id, class_name, class_id, confidence,
                   x_center_norm, y_center_norm, width_norm, height_norm,
                   x_min_px, y_min_px, x_max_px, y_max_px, annotation_source)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    ann.annotation_id, ann.image_id,
                    ann.class_name, ann.class_id, ann.confidence,
                    ann.x_center_norm, ann.y_center_norm,
                    ann.width_norm, ann.height_norm,
                    ann.x_min_px, ann.y_min_px, ann.x_max_px, ann.y_max_px,
                    "auto",
                ),
            )
            loaded += 1
        except sqlite3.Error:
            pass
    return loaded


def load_thermal(
    conn:    sqlite3.Connection,
    thermal: ThermalMetadata,
):
    try:
        conn.execute("""
            INSERT OR REPLACE INTO thermal_metadata
              (image_id, min_temperature_c, max_temperature_c, avg_temperature_c,
               median_temperature_c, emissivity, reflected_temp_c, atmospheric_temp_c,
               relative_humidity, distance_to_object_m, temperature_unit,
               radiometric_data_path, thermal_palette,
               planck_r1, planck_b, planck_f, planck_o, planck_k)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                thermal.image_id,
                thermal.min_temperature_c, thermal.max_temperature_c,
                thermal.avg_temperature_c, thermal.median_temperature_c,
                thermal.emissivity, thermal.reflected_temp_c, thermal.atmospheric_temp_c,
                thermal.relative_humidity, thermal.distance_to_object_m,
                thermal.temperature_unit, thermal.radiometric_data_path,
                thermal.thermal_palette,
                thermal.planck_r1, thermal.planck_b, thermal.planck_f,
                thermal.planck_o, thermal.planck_k,
            ),
        )
    except sqlite3.Error:
        pass


def update_dataset_counts(conn: sqlite3.Connection, dataset_id: str):
    conn.execute("""
        UPDATE datasets SET
          total_images = (
              SELECT COUNT(*) FROM images WHERE dataset_id = ?),
          total_annotations = (
              SELECT COUNT(a.annotation_id)
              FROM annotations a JOIN images i ON a.image_id = i.image_id
              WHERE i.dataset_id = ?),
          updated_at = datetime('now')
        WHERE dataset_id = ?
    """, (dataset_id, dataset_id, dataset_id))


def update_split_counts(conn: sqlite3.Connection):
    for split_id in SPLIT_IDS.values():
        conn.execute(
            "UPDATE data_splits SET image_count = "
            "(SELECT COUNT(*) FROM images WHERE split_id = ?) WHERE split_id = ?",
            (split_id, split_id),
        )


def write_dataset_yaml() -> Path:
    out = PROCESSED / "dataset.yaml"
    PROCESSED.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.dump({
            "path":  str(PROCESSED.resolve()),
            "train": "images/train",
            "val":   "images/val",
            "test":  "images/test",
            "names": {0: "smoke", 1: "flame", 2: "overheat"},
        }, f, default_flow_style=False, sort_keys=False)
    return out


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────

def run_etl(
    datasets:       List[str],
    dry_run:        bool = False,
    skip_transform: bool = False,
    resume:         bool = False,
):
    t_start = time.time()

    log.info("")
    log.info("╔════════════════════════════════════════════════════════╗")
    log.info("║  DC-EFDS-Lite  ·  Step 4 of 5: Run ETL Pipeline       ║")
    log.info("╚════════════════════════════════════════════════════════╝")
    log.info("  Datasets : %s", datasets)
    log.info("  dry_run  : %s  |  skip_transform : %s  |  resume : %s",
             dry_run, skip_transform, resume)
    log.info("")

    conn        = get_conn()
    transformer = Transformer()

    all_results:    List[ExtractorResult]   = []
    dataset_ids:    List[str]               = []

    # ══════════════════════════════════════════════════════════════════════════
    #  EXTRACT
    # ══════════════════════════════════════════════════════════════════════════
    log.info("━━━  EXTRACT  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    for ds_name in datasets:
        raw_sub, modality = DATASET_MAP[ds_name]
        raw_path = RAW_DIR / raw_sub

        if not raw_path.exists() or not any(raw_path.rglob("*.jpg")):
            log.warning("  [SKIP] %s — raw directory empty or missing: %s", ds_name, raw_path)
            continue

        log.info("  Scanning: %-15s  (%s)", ds_name, raw_path)
        extractor, image_paths = SCANNERS[ds_name]()

        if dry_run:
            log.info("  [DRY RUN] %d images found in %s", len(image_paths), ds_name)
            continue

        if resume:
            loaded_names = already_loaded(conn, extractor.dataset_id)
            image_paths  = [p for p in image_paths if p.name not in loaded_names]
            log.info("  Resume: %d new images after skipping %d already loaded",
                     len(image_paths), len(loaded_names))

        upsert_dataset_meta(conn, extractor.dataset_id, modality)
        results = extractor.extract_batch(image_paths)

        valid  = [r for r in results if r.is_valid]
        errors = [r for r in results if not r.is_valid]
        log.info("  %-15s  %5d valid   %3d errors", ds_name, len(valid), len(errors))

        all_results.extend(valid)
        dataset_ids.append(extractor.dataset_id)

    if dry_run:
        log.info("")
        log.info("  Dry run complete — no writes performed.")
        conn.close()
        return

    if not all_results:
        log.error("  No valid images found. Check that datasets are downloaded.")
        conn.close()
        sys.exit(1)

    log.info("")
    log.info("  Total extracted: %d valid images", len(all_results))

    # ══════════════════════════════════════════════════════════════════════════
    #  TRANSFORM + SPLIT ASSIGNMENT
    # ══════════════════════════════════════════════════════════════════════════
    log.info("")
    log.info("━━━  TRANSFORM  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    assigned = assign_splits(all_results, ETL_CONFIG)

    # ══════════════════════════════════════════════════════════════════════════
    #  LOAD
    # ══════════════════════════════════════════════════════════════════════════
    log.info("")
    log.info("━━━  LOAD  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    total_images      = 0
    total_annotations = 0
    total_thermal     = 0
    total_errors      = 0
    BATCH             = 500

    for i, (result, split_name) in enumerate(tqdm(assigned, desc="  ETL", unit="img")):
        proc_path, adj_anns, t_errs = transformer.run(
            result, split_name, skip_copy=skip_transform
        )

        if t_errs:
            total_errors += 1
            continue

        ok_img = load_image(conn, result.metadata, split_name, proc_path)
        if not ok_img:
            total_errors += 1
            continue

        n_ann = load_annotations(conn, adj_anns)

        thermal = getattr(result, "thermal", None)
        if thermal is not None:
            load_thermal(conn, thermal)
            total_thermal += 1

        total_images      += 1
        total_annotations += n_ann

        if (i + 1) % BATCH == 0:
            conn.commit()

    conn.commit()

    # ── Update counters ───────────────────────────────────────────────────────
    for ds_id in dataset_ids:
        update_dataset_counts(conn, ds_id)
    update_split_counts(conn)
    conn.commit()

    # ── Write dataset.yaml ────────────────────────────────────────────────────
    yaml_path = write_dataset_yaml()
    log.info("  dataset.yaml written: %s", yaml_path)

    conn.close()

    # ══════════════════════════════════════════════════════════════════════════
    #  SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t_start

    log.info("")
    log.info("━━━  ETL COMPLETE  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info("")

    conn2 = get_conn()

    visual_ct  = conn2.execute(
        "SELECT COUNT(*) FROM images WHERE modality='visual'").fetchone()[0]
    thermal_ct = conn2.execute(
        "SELECT COUNT(*) FROM images WHERE modality='thermal'").fetchone()[0]
    ann_ct     = conn2.execute(
        "SELECT COUNT(*) FROM annotations").fetchone()[0]
    db_mb      = DB_PATH.stat().st_size / (1024 * 1024)

    v_ok  = "✅" if visual_ct  >= 25000 else "⚠️ "
    t_ok  = "✅" if thermal_ct >= 14000 else "⚠️ "
    a_ok  = "✅" if ann_ct     >= 50000 else "⚠️ "
    db_ok = "✅" if db_mb      >= 100   else "⚠️ "

    print()
    print("  ┌──────────────────────────────────────────────────────┐")
    print("  │  ETL Results                                         │")
    print("  ├──────────────────────────────────────────────────────┤")
    print(f"  │  {v_ok}  Visual images   :  {visual_ct:>7,}  (target: 25,000+)  │")
    print(f"  │  {t_ok}  Thermal images  :  {thermal_ct:>7,}  (target: 14,000+)  │")
    print(f"  │  {a_ok}  Annotations     :  {ann_ct:>7,}  (target: 50,000+)  │")
    print(f"  │  {db_ok}  Database size   :  {db_mb:>7.0f} MB  (target: 500+ MB) │")
    print(f"  │      Errors         :  {total_errors:>7,}                      │")
    print(f"  │      Elapsed        :  {elapsed:>7.0f}s                      │")
    print("  ├──────────────────────────────────────────────────────┤")

    for split in ["train", "val", "test"]:
        n = conn2.execute(
            "SELECT image_count FROM data_splits WHERE split_name=?", (split,)
        ).fetchone()[0]
        print(f"  │      {split:<6}  :  {n:>7,} images                       │")

    print("  └──────────────────────────────────────────────────────┘")

    conn2.close()

    print()
    print("  ─────────────────────────────────────────────────────────")
    print("  Next:  python scripts/validate_db.py")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DC-EFDS-Lite — Step 4: Run ETL Pipeline")
    parser.add_argument(
        "--datasets", nargs="+",
        choices=list(SCANNERS.keys()),
        default=list(SCANNERS.keys()),
        help="Datasets to process (default: all)",
    )
    parser.add_argument("--dry-run",        action="store_true", help="Count only, no writes")
    parser.add_argument("--skip-transform", action="store_true", help="Skip resize/copy, DB only")
    parser.add_argument("--resume",         action="store_true", help="Skip already-loaded images")
    args = parser.parse_args()

    run_etl(
        datasets       = args.datasets,
        dry_run        = args.dry_run,
        skip_transform = args.skip_transform,
        resume         = args.resume,
    )