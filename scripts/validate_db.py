#!/usr/bin/env python3
"""
scripts/validate_db.py
─────────────────────────────────────────────────────────────────
Step 5 of 5 — Validate the ETL output database.

Checks every aspect of the pipeline output before training:

  1.  Database connectivity and integrity check
  2.  Target image counts  (✅ 25,000+ visual | ✅ 14,000+ thermal)
  3.  Target annotations   (✅ 50,000+)
  4.  Database size        (✅ ~500 MB – 1 GB)
  5.  Split distribution   (70 / 20 / 10 with ±5% tolerance)
  6.  Class balance across splits
  7.  Per-dataset record counts
  8.  Annotation coordinate bounds  (all values in [0,1])
  9.  File existence check  (sampled)
  10. Label file pairing
  11. Thermal metadata coverage
  12. Duplicate image detection (checksum)
  13. dataset.yaml validity

Expected terminal output on success:
  ✅ Visual images: 25,000+
  ✅ Thermal images: 14,000+
  ✅ Annotations: 50,000+
  ✅ Database size: ~500 MB - 1 GB

Usage:
    python scripts/validate_db.py
    python scripts/validate_db.py --db database/dc_efds.db
    python scripts/validate_db.py --sample 0.10   # check 10% of files
    python scripts/validate_db.py --fix-counts     # recalculate dataset counters
"""

import argparse
import random
import sqlite3
import sys
from pathlib import Path

import yaml

DEFAULT_DB        = Path("database/dc_efds.db")
DEFAULT_PROCESSED = Path("data/processed")
DEFAULT_SAMPLE    = 0.05   # 5% file existence sample

# ── Targets ───────────────────────────────────────────────────────────────────

TARGETS = {
    "visual_images":  25_000,
    "thermal_images": 14_000,
    "annotations":    50_000,
    "db_size_mb_min": 100,     # warn below this
    "db_size_mb_ok":  500,     # ✅ above this
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _conn(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        print(f"\n  ❌  Database not found: {db_path}")
        print("      Run: python scripts/init_database.py  then  python scripts/run_etl.py")
        sys.exit(1)
    c = sqlite3.connect(str(db_path))
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    return c


def _section(n: int, title: str):
    print(f"\n  ── Check {n:02d}: {title}")


def _pass(msg: str):  print(f"  ✅  {msg}")
def _warn(msg: str):  print(f"  ⚠️   {msg}")
def _fail(msg: str):  print(f"  ❌  {msg}"); return 1


# ══════════════════════════════════════════════════════════════════════════════
#  Individual checks
# ══════════════════════════════════════════════════════════════════════════════

def check_connectivity(conn: sqlite3.Connection, db_path: Path) -> int:
    _section(1, "Database connectivity and integrity")
    result = conn.execute("PRAGMA integrity_check").fetchone()[0]
    if result != "ok":
        return _fail(f"Integrity check failed: {result}")
    size_mb = db_path.stat().st_size / (1024 * 1024)
    _pass(f"Integrity: ok  |  Size: {size_mb:.1f} MB  |  Path: {db_path}")
    return 0


def check_image_counts(conn: sqlite3.Connection) -> int:
    _section(2, "Image count targets")
    failures = 0

    visual  = conn.execute("SELECT COUNT(*) FROM images WHERE modality='visual'").fetchone()[0]
    thermal = conn.execute("SELECT COUNT(*) FROM images WHERE modality='thermal'").fetchone()[0]
    total   = visual + thermal
    db_mb   = 0
    try:
        db_mb = DEFAULT_DB.stat().st_size / (1024 * 1024)
    except Exception:
        pass

    if visual >= TARGETS["visual_images"]:
        _pass(f"Visual images   : {visual:>8,}  (target: {TARGETS['visual_images']:,}+)  ✅")
    else:
        _warn(f"Visual images   : {visual:>8,}  (target: {TARGETS['visual_images']:,}+)  — below target")
        failures += 1

    if thermal >= TARGETS["thermal_images"]:
        _pass(f"Thermal images  : {thermal:>8,}  (target: {TARGETS['thermal_images']:,}+)  ✅")
    else:
        _warn(f"Thermal images  : {thermal:>8,}  (target: {TARGETS['thermal_images']:,}+)  — below target")

    if db_mb >= TARGETS["db_size_mb_ok"]:
        _pass(f"Database size   : {db_mb:>7.0f} MB  (target: {TARGETS['db_size_mb_ok']}+ MB)  ✅")
    elif db_mb >= TARGETS["db_size_mb_min"]:
        _warn(f"Database size   : {db_mb:>7.0f} MB  (expected ~500 MB after full ETL)")
    else:
        _warn(f"Database size   : {db_mb:>7.0f} MB  — very small, ETL may be incomplete")

    _pass(f"Total images    : {total:>8,}")
    return failures


def check_annotation_counts(conn: sqlite3.Connection) -> int:
    _section(3, "Annotation count targets")
    ann_total = conn.execute("SELECT COUNT(*) FROM annotations").fetchone()[0]

    if ann_total >= TARGETS["annotations"]:
        _pass(f"Annotations     : {ann_total:>8,}  (target: {TARGETS['annotations']:,}+)  ✅")
        return 0
    else:
        _warn(f"Annotations     : {ann_total:>8,}  (target: {TARGETS['annotations']:,}+)  — below target")
        return 1


def check_split_distribution(conn: sqlite3.Connection) -> int:
    _section(4, "Split distribution  (70 / 20 / 10 ±5%)")
    rows = conn.execute("""
        SELECT ds.split_name, COUNT(*) AS cnt
        FROM images i
        JOIN data_splits ds ON i.split_id = ds.split_id
        GROUP BY ds.split_name
        ORDER BY ds.split_name
    """).fetchall()

    total    = sum(r["cnt"] for r in rows)
    failures = 0
    expected = {"train": 70, "val": 20, "test": 10}

    for row in rows:
        name = row["split_name"]
        cnt  = row["cnt"]
        pct  = cnt / total * 100 if total else 0
        exp  = expected.get(name, 0)
        ok   = abs(pct - exp) <= 5

        if ok:
            _pass(f"{name:<6}  {cnt:>8,} images  ({pct:.1f}%,  expected ~{exp}%)")
        else:
            _warn(f"{name:<6}  {cnt:>8,} images  ({pct:.1f}%,  expected ~{exp}%)")
            failures += 1

    if total == 0:
        _fail("No images in any split — ETL has not run yet.")
        failures += 1

    return failures


def check_class_balance(conn: sqlite3.Connection) -> int:
    _section(5, "Class distribution across splits")

    rows = conn.execute("""
        SELECT ds.split_name, a.class_name, COUNT(*) AS cnt
        FROM annotations a
        JOIN images      i  ON a.image_id = i.image_id
        JOIN data_splits ds ON i.split_id = ds.split_id
        GROUP BY ds.split_name, a.class_name
        ORDER BY ds.split_name, a.class_name
    """).fetchall()

    neg = conn.execute(
        "SELECT COUNT(*) FROM images WHERE image_id NOT IN "
        "(SELECT DISTINCT image_id FROM annotations)"
    ).fetchone()[0]

    if not rows:
        _warn("No annotations found — check extractor output.")
        return 1

    current_split = None
    for row in rows:
        if row["split_name"] != current_split:
            current_split = row["split_name"]
            print(f"       {current_split}:")
        print(f"         {row['class_name']:<12}  {row['cnt']:>7,}")

    _pass(f"Negative images (no annotation): {neg:,}")
    return 0


def check_per_dataset(conn: sqlite3.Connection) -> int:
    _section(6, "Per-dataset record counts")

    rows = conn.execute("""
        SELECT d.name, d.modality, d.total_images, d.total_annotations
        FROM datasets d
        ORDER BY d.modality, d.total_images DESC
    """).fetchall()

    icon = {"visual": "🔥", "thermal": "🌡️", "multi-modal": "🔀"}
    for row in rows:
        i   = icon.get(row["modality"], "·")
        img = row["total_images"]
        ann = row["total_annotations"]
        status = "✅" if img > 0 else "⚠️ "
        print(f"       {status}  {i} {row['name']:<38}  {img:>7,} imgs  {ann:>8,} anns")

    return 0


def check_coordinate_bounds(conn: sqlite3.Connection) -> int:
    _section(7, "Annotation coordinate bounds  [0, 1]")

    bad = conn.execute("""
        SELECT COUNT(*) FROM annotations WHERE
            x_center_norm < 0 OR x_center_norm > 1 OR
            y_center_norm < 0 OR y_center_norm > 1 OR
            width_norm    < 0 OR width_norm    > 1 OR
            height_norm   < 0 OR height_norm   > 1
    """).fetchone()[0]

    if bad == 0:
        _pass("All annotation coordinates are within [0, 1]")
        return 0
    else:
        return _fail(f"{bad:,} annotations have out-of-range coordinates")


def check_file_existence(conn: sqlite3.Connection, sample_rate: float) -> int:
    _section(8, f"File existence  ({int(sample_rate*100)}% random sample)")

    all_paths = conn.execute("SELECT file_path FROM images").fetchall()
    n_sample  = max(1, int(len(all_paths) * sample_rate))
    sample    = random.sample(all_paths, min(n_sample, len(all_paths)))
    missing   = sum(1 for r in sample if not Path(r["file_path"]).exists())

    if missing == 0:
        _pass(f"All {len(sample):,} sampled files exist on disk")
        return 0
    elif missing / len(sample) < 0.05:
        _warn(f"{missing}/{len(sample)} sampled files missing  (<5% threshold, acceptable)")
        return 0
    else:
        return _fail(f"{missing}/{len(sample)} sampled files missing  (>5% — check paths)")


def check_label_pairing(conn: sqlite3.Connection, sample_rate: float) -> int:
    _section(9, f"Label file pairing  ({int(sample_rate*100)}% sample)")

    all_paths = conn.execute(
        "SELECT file_path FROM images WHERE modality='visual'"
    ).fetchall()
    if not all_paths:
        _warn("No visual images to check label pairing.")
        return 0

    n_sample = max(1, int(len(all_paths) * sample_rate))
    sample   = random.sample(all_paths, min(n_sample, len(all_paths)))
    missing  = 0
    checked  = 0

    for row in sample:
        img_p = Path(row["file_path"])
        parts = img_p.parts
        try:
            idx   = len(parts) - 1 - parts[::-1].index("images")
            lbl_p = Path(*list(parts[:idx]) + ["labels"] + list(parts[idx+1:])).with_suffix(".txt")
            if not lbl_p.exists():
                missing += 1
            checked += 1
        except ValueError:
            pass

    if checked == 0:
        _warn("Could not resolve label paths — check processed directory structure.")
        return 0

    if missing == 0:
        _pass(f"All {checked:,} sampled label files present")
        return 0
    elif missing / checked < 0.05:
        _warn(f"{missing}/{checked} label files missing  (<5% threshold)")
        return 0
    else:
        return _fail(f"{missing}/{checked} label files missing")


def check_thermal_metadata(conn: sqlite3.Connection) -> int:
    _section(10, "Thermal metadata coverage")

    thermal_imgs = conn.execute(
        "SELECT COUNT(*) FROM images WHERE modality='thermal'"
    ).fetchone()[0]
    thermal_meta = conn.execute(
        "SELECT COUNT(*) FROM thermal_metadata"
    ).fetchone()[0]
    with_temp = conn.execute(
        "SELECT COUNT(*) FROM thermal_metadata WHERE max_temperature_c IS NOT NULL"
    ).fetchone()[0]

    if thermal_imgs == 0:
        _warn("No thermal images loaded yet  (FLIR dataset not processed)")
        return 0

    coverage = thermal_meta / thermal_imgs * 100 if thermal_imgs else 0
    temp_cov = with_temp / thermal_imgs * 100 if thermal_imgs else 0

    _pass(f"Thermal images          : {thermal_imgs:,}")
    _pass(f"Metadata rows           : {thermal_meta:,}  ({coverage:.0f}% coverage)")
    if with_temp > 0:
        _pass(f"With Planck temp decode : {with_temp:,}  ({temp_cov:.0f}% of thermal images)")
    else:
        _warn("No Planck temperature data — exiftool may not be installed  (brew install exiftool)")
    return 0


def check_duplicates(conn: sqlite3.Connection) -> int:
    _section(11, "Duplicate image detection  (checksum)")

    dup_groups = conn.execute("""
        SELECT checksum, COUNT(*) AS cnt
        FROM images
        WHERE checksum IS NOT NULL AND checksum != ''
        GROUP BY checksum HAVING cnt > 1
    """).fetchall()

    if not dup_groups:
        _pass("No duplicate images detected")
        return 0

    total_dups = sum(r["cnt"] - 1 for r in dup_groups)
    _warn(f"{len(dup_groups)} duplicate groups  ({total_dups} redundant images)")
    print("       To remove duplicates:")
    print("         sqlite3 database/dc_efds.db \\")
    print('           "DELETE FROM images WHERE rowid NOT IN '
          '(SELECT MIN(rowid) FROM images GROUP BY checksum);"')
    return 0   # Warning only, not a blocking failure


def check_dataset_yaml(processed_dir: Path) -> int:
    _section(12, "dataset.yaml")

    yaml_path = processed_dir / "dataset.yaml"
    if not yaml_path.exists():
        return _fail(f"dataset.yaml not found at {yaml_path}")

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    required = {"path", "train", "val", "test", "names"}
    missing  = required - set(cfg.keys())
    if missing:
        return _fail(f"dataset.yaml missing keys: {missing}")

    classes = list(cfg["names"].values())
    _pass(f"dataset.yaml valid  |  {len(classes)} classes: {classes}")
    _pass(f"Training path: {cfg['path']}")
    return 0


# ── Fix counts utility ────────────────────────────────────────────────────────

def fix_counts(conn: sqlite3.Connection):
    print("\n  Recalculating dataset counters...")
    datasets = conn.execute("SELECT dataset_id FROM datasets").fetchall()
    for row in datasets:
        ds_id = row["dataset_id"]
        conn.execute("""
            UPDATE datasets SET
              total_images = (SELECT COUNT(*) FROM images WHERE dataset_id = ?),
              total_annotations = (
                  SELECT COUNT(a.annotation_id)
                  FROM annotations a JOIN images i ON a.image_id = i.image_id
                  WHERE i.dataset_id = ?),
              updated_at = datetime('now')
            WHERE dataset_id = ?
        """, (ds_id, ds_id, ds_id))

    for split_name in ["train", "val", "test"]:
        split_id = f"split-{split_name}"
        conn.execute(
            "UPDATE data_splits SET image_count = "
            "(SELECT COUNT(*) FROM images WHERE split_id = ?) WHERE split_id = ?",
            (split_id, split_id),
        )
    conn.commit()
    print("  ✅  Counters updated.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DC-EFDS-Lite — Step 5: Validate database"
    )
    parser.add_argument("--db",        default=str(DEFAULT_DB))
    parser.add_argument("--processed", default=str(DEFAULT_PROCESSED))
    parser.add_argument("--sample",    type=float, default=DEFAULT_SAMPLE,
                        help="File existence sample rate 0–1 (default: 0.05)")
    parser.add_argument("--fix-counts", action="store_true",
                        help="Recalculate dataset/split counters before checking")
    args = parser.parse_args()

    random.seed(0)
    db_path       = Path(args.db)
    processed_dir = Path(args.processed)

    print()
    print("╔════════════════════════════════════════════════════════╗")
    print("║  DC-EFDS-Lite  ·  Step 5 of 5: Validate Database      ║")
    print("╚════════════════════════════════════════════════════════╝")

    conn = _conn(db_path)

    if args.fix_counts:
        fix_counts(conn)

    failures = 0
    failures += check_connectivity(conn, db_path)
    failures += check_image_counts(conn)
    failures += check_annotation_counts(conn)
    failures += check_split_distribution(conn)
    failures += check_class_balance(conn)
    failures += check_per_dataset(conn)
    failures += check_coordinate_bounds(conn)
    failures += check_file_existence(conn, args.sample)
    failures += check_label_pairing(conn, args.sample)
    failures += check_thermal_metadata(conn)
    failures += check_duplicates(conn)
    failures += check_dataset_yaml(processed_dir)
    conn.close()

    # ── Final verdict ─────────────────────────────────────────────────────────
    print()
    print("  ══════════════════════════════════════════════════════")

    # Pull actual numbers for the summary line
    conn2 = _conn(db_path)
    vis   = conn2.execute("SELECT COUNT(*) FROM images WHERE modality='visual'").fetchone()[0]
    thm   = conn2.execute("SELECT COUNT(*) FROM images WHERE modality='thermal'").fetchone()[0]
    ann   = conn2.execute("SELECT COUNT(*) FROM annotations").fetchone()[0]
    mb    = db_path.stat().st_size / (1024 * 1024)
    conn2.close()

    print(f"  {'✅' if vis >= 25000 else '⚠️ '}  Visual images  : {vis:,}")
    print(f"  {'✅' if thm >= 14000 else '⚠️ '}  Thermal images : {thm:,}")
    print(f"  {'✅' if ann >= 50000 else '⚠️ '}  Annotations    : {ann:,}")
    print(f"  {'✅' if mb  >= 500   else '⚠️ '}  Database size  : {mb:.0f} MB")
    print()

    if failures == 0:
        print("  ✅  All checks passed — database is ready for training.")
        print()
        print("  ─────────────────────────────────────────────────────")
        print("  Next:  python scripts/train.py")
        print()
    else:
        print(f"  ⚠️   {failures} check(s) need attention — review output above.")
        print("      Training can still proceed if only ⚠️  warnings (not ❌ failures).")
        print()
        print("  ─────────────────────────────────────────────────────")
        print("  Next:  python scripts/train.py  (or fix issues first)")
        print()
        if failures >= 3:
            sys.exit(1)


if __name__ == "__main__":
    main()