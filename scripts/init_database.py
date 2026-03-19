"""
Step 1 of 5 — Initialise the SQLite database and project structure.

Creates:
  - All tables, indexes, and views from the agreed schema
  - data_splits rows  (70 / 20 / 10)
  - Dataset registry rows for all 6 agreed datasets
  - Full directory tree  (data/, models/, logs/, etc.)

Safe to re-run — all statements use IF NOT EXISTS / INSERT OR IGNORE.

Usage:
    python scripts/init_database.py
    python scripts/init_database.py --db database/dc_efds.db
    python scripts/init_database.py --reset    # back up + recreate (destructive)
"""

import argparse
import sqlite3
import sys
from pathlib import Path

DEFAULT_DB = Path("database/dc_efds.db")

DIRECTORY_TREE = [
    "data/raw/dfire",
    "data/raw/roboflow",
    "data/raw/kaggle_fire",
    "data/raw/personal_negatives",
    "data/raw/flir_thermal",
    "data/raw/kaist",
    "data/processed/images/train",
    "data/processed/images/val",
    "data/processed/images/test",
    "data/processed/images_thermal/train",
    "data/processed/images_thermal/val",
    "data/processed/images_thermal/test",
    "data/processed/labels/train",
    "data/processed/labels/val",
    "data/processed/labels/test",
    "data/alerts",
    "database/backups",
    "models/weights",
    "logs",
    "src/inference",
    "src/alerts",
    "src/dashboard",
    "extractors",
    "config",
    "tests",
]

DATA_SPLITS = [
    ("split-train", "train", 0.70),
    ("split-val",   "val",   0.20),
    ("split-test",  "test",  0.10),
]

DATASETS = [
    (
        "dfire", "D-Fire Dataset", "visual",
        "21,000+ fire/smoke images with YOLO bboxes. "
        "Classes: flame (1,164), smoke (5,867), both (4,658), negative (9,838). "
        "License: Research — cite de Venâncio et al. 2022.",
    ),
    (
        "roboflow_fire", "Roboflow Fire & Smoke", "visual",
        "6,400+ diverse environment images with high-quality bboxes. "
        "License: MIT (commercial use allowed).",
    ),
    (
        "kaggle_fire", "Kaggle Fire Detection", "visual",
        "17,000+ images with folder-based labels (fire/ vs no_fire/). "
        "Weak labels — used for augmentation and negative samples. License: Research.",
    ),
    (
        "personal_negatives", "Personal Server Room Negatives", "visual",
        "500+ frames captured from the target deployment environment. "
        "All negatives — critical for false positive reduction.",
    ),
    (
        "flir_thermal", "FLIR Thermal Dataset", "thermal",
        "14,000+ radiometric JPEG images with embedded Planck calibration constants. "
        "Full per-pixel temperature extraction via Planck equation. "
        "License: Free for research and commercial use.",
    ),
    (
        "kaist", "KAIST Multi-Spectral Dataset", "multi-modal",
        "95,000+ paired visible + thermal (lwir) frames. Optional — 65GB full download. "
        "License: Research — access request at http://multispectral.ece.kaist.ac.kr/",
    ),
]

# ── Full schema — exact match to the agreed schema.sql document ───────────────

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS datasets (
    dataset_id          TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    description         TEXT,
    modality            TEXT CHECK(modality IN ('visual', 'thermal', 'multi-modal')),
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_images        INTEGER   DEFAULT 0,
    total_annotations   INTEGER   DEFAULT 0
);

CREATE TABLE IF NOT EXISTS data_splits (
    split_id    TEXT PRIMARY KEY,
    split_name  TEXT  CHECK(split_name IN ('train', 'val', 'test')),
    split_ratio REAL  CHECK(split_ratio > 0 AND split_ratio <= 1),
    image_count INTEGER DEFAULT 0,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS images (
    image_id          TEXT PRIMARY KEY,
    dataset_id        TEXT NOT NULL,
    split_id          TEXT,
    file_path         TEXT NOT NULL,
    original_filename TEXT,
    modality          TEXT CHECK(modality IN ('visual', 'thermal')) NOT NULL,
    width             INTEGER NOT NULL,
    height            INTEGER NOT NULL,
    channels          INTEGER DEFAULT 3,
    file_size_mb      REAL,
    captured_at       TIMESTAMP,
    checksum          TEXT,
    validated         BOOLEAN DEFAULT FALSE,
    privacy_cleared   BOOLEAN DEFAULT FALSE,
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)  ON DELETE CASCADE,
    FOREIGN KEY (split_id)   REFERENCES data_splits(split_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS annotations (
    annotation_id     TEXT PRIMARY KEY,
    image_id          TEXT NOT NULL,
    class_name        TEXT NOT NULL,
    class_id          INTEGER,
    confidence        REAL CHECK(confidence    >= 0 AND confidence    <= 1),
    x_center_norm     REAL CHECK(x_center_norm >= 0 AND x_center_norm <= 1),
    y_center_norm     REAL CHECK(y_center_norm >= 0 AND y_center_norm <= 1),
    width_norm        REAL CHECK(width_norm    >= 0 AND width_norm    <= 1),
    height_norm       REAL CHECK(height_norm   >= 0 AND height_norm   <= 1),
    x_min_px          INTEGER,
    y_min_px          INTEGER,
    x_max_px          INTEGER,
    y_max_px          INTEGER,
    annotation_source TEXT CHECK(annotation_source IN ('manual','auto','hybrid')) DEFAULT 'auto',
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (image_id) REFERENCES images(image_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS thermal_metadata (
    image_id              TEXT PRIMARY KEY,
    min_temperature_c     REAL,
    max_temperature_c     REAL,
    avg_temperature_c     REAL,
    median_temperature_c  REAL,
    emissivity            REAL    DEFAULT 0.95,
    reflected_temp_c      REAL,
    atmospheric_temp_c    REAL,
    relative_humidity     REAL,
    distance_to_object_m  REAL,
    temperature_unit      TEXT    DEFAULT 'Celsius',
    radiometric_data_path TEXT,
    thermal_palette       TEXT,
    planck_r1             REAL,
    planck_b              REAL,
    planck_f              REAL,
    planck_o              REAL,
    planck_k              REAL,
    created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (image_id) REFERENCES images(image_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS alerts (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp         TEXT    NOT NULL DEFAULT (datetime('now')),
    detection_type    TEXT    NOT NULL CHECK(detection_type IN ('heat_warning','smoke','flame','overheat')),
    detection_source  TEXT    CHECK(detection_source IN ('visual','thermal','fusion')),
    max_temperature_c REAL,
    confidence        REAL    NOT NULL,
    bbox_coords       TEXT,
    frame_path        TEXT,
    resolved          BOOLEAN DEFAULT FALSE,
    resolved_at       TEXT,
    notes             TEXT
);

CREATE TABLE IF NOT EXISTS system_metrics (
    timestamp             TEXT    PRIMARY KEY DEFAULT (datetime('now')),
    cpu_percent           REAL,
    memory_mb             REAL,
    inference_time_ms     REAL,
    fps_achieved          REAL,
    camera_connected      BOOLEAN,
    wifi_signal_strength  INTEGER,
    temperature_c         REAL
);

CREATE INDEX IF NOT EXISTS idx_images_dataset    ON images(dataset_id);
CREATE INDEX IF NOT EXISTS idx_images_split      ON images(split_id);
CREATE INDEX IF NOT EXISTS idx_images_modality   ON images(modality);
CREATE INDEX IF NOT EXISTS idx_images_checksum   ON images(checksum);
CREATE INDEX IF NOT EXISTS idx_annotations_image ON annotations(image_id);
CREATE INDEX IF NOT EXISTS idx_annotations_class ON annotations(class_name);
CREATE INDEX IF NOT EXISTS idx_thermal_max_temp  ON thermal_metadata(max_temperature_c);
CREATE INDEX IF NOT EXISTS idx_alerts_timestamp  ON alerts(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_unresolved ON alerts(resolved) WHERE resolved = FALSE;

CREATE VIEW IF NOT EXISTS v_dataset_stats AS
SELECT
    d.dataset_id, d.name, d.modality,
    COUNT(DISTINCT i.image_id)                                            AS total_images,
    COUNT(DISTINCT a.annotation_id)                                       AS total_annotations,
    COUNT(DISTINCT CASE WHEN i.modality='visual'  THEN i.image_id END)   AS visual_images,
    COUNT(DISTINCT CASE WHEN i.modality='thermal' THEN i.image_id END)   AS thermal_images,
    AVG(CASE WHEN tm.max_temperature_c IS NOT NULL THEN tm.max_temperature_c END) AS avg_max_temp
FROM datasets d
LEFT JOIN images           i  ON d.dataset_id = i.dataset_id
LEFT JOIN annotations      a  ON i.image_id   = a.image_id
LEFT JOIN thermal_metadata tm ON i.image_id   = tm.image_id
GROUP BY d.dataset_id;

CREATE VIEW IF NOT EXISTS v_alert_summary AS
SELECT
    detection_type,
    COUNT(*)                                      AS total_alerts,
    COUNT(CASE WHEN resolved=FALSE THEN 1 END)    AS unresolved,
    AVG(confidence)                               AS avg_confidence,
    MAX(timestamp)                                AS last_alert
FROM alerts
GROUP BY detection_type;

CREATE VIEW IF NOT EXISTS v_split_class_balance AS
SELECT ds.split_name, a.class_name, COUNT(*) AS annotation_count
FROM annotations a
JOIN images      i  ON a.image_id = i.image_id
JOIN data_splits ds ON i.split_id = ds.split_id
GROUP BY ds.split_name, a.class_name
ORDER BY ds.split_name, a.class_name;

CREATE VIEW IF NOT EXISTS v_etl_progress AS
SELECT
    d.name AS dataset, d.modality,
    d.total_images, d.total_annotations,
    COUNT(DISTINCT i.split_id) AS splits_populated
FROM datasets d
LEFT JOIN images i ON d.dataset_id = i.dataset_id
GROUP BY d.dataset_id;
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_banner():
    print()
    print("╔════════════════════════════════════════════════════════╗")
    print("║  DC-EFDS-Lite  ·  Step 1 of 5: Initialize Database    ║")
    print("╚════════════════════════════════════════════════════════╝")
    print()


def _create_dirs():
    print("  [1/4] Creating directory tree...")
    new = sum(1 for d in DIRECTORY_TREE if not Path(d).exists())
    for d in DIRECTORY_TREE:
        Path(d).mkdir(parents=True, exist_ok=True)
    print(f"        {len(DIRECTORY_TREE)} directories ready  ({new} created)")


def _apply_schema(db_path: Path) -> sqlite3.Connection:
    print(f"\n  [2/4] Applying schema → {db_path}")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous  = NORMAL")
    conn.execute("PRAGMA cache_size   = -65536")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    t = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
    v = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='view'").fetchone()[0]
    ix = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='index'").fetchone()[0]
    print(f"        {t} tables  ·  {v} views  ·  {ix} indexes applied")
    return conn


def _seed_splits(conn: sqlite3.Connection):
    print("\n  [3/4] Seeding data_splits...")
    for sid, sname, ratio in DATA_SPLITS:
        conn.execute(
            "INSERT OR IGNORE INTO data_splits (split_id, split_name, split_ratio) VALUES (?,?,?)",
            (sid, sname, ratio),
        )
    conn.commit()
    for _, sname, ratio in DATA_SPLITS:
        print(f"        {sname:<6}  {ratio*100:.0f}%")


def _seed_datasets(conn: sqlite3.Connection):
    print("\n  [4/4] Registering datasets...")
    for ds_id, name, modality, desc in DATASETS:
        conn.execute(
            "INSERT OR IGNORE INTO datasets (dataset_id, name, modality, description) VALUES (?,?,?,?)",
            (ds_id, name, modality, desc),
        )
    conn.commit()
    icon = {"visual": "🔥", "thermal": "🌡️", "multi-modal": "🔀"}
    for ds_id, name, modality, _ in DATASETS:
        print(f"        {icon.get(modality,'·')}  {name}")


def _verify(conn: sqlite3.Connection, db_path: Path) -> bool:
    missing_t = {"datasets","data_splits","images","annotations",
                 "thermal_metadata","alerts","system_metrics"} - {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    missing_v = {"v_dataset_stats","v_alert_summary",
                 "v_split_class_balance","v_etl_progress"} - {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='view'").fetchall()
    }
    if missing_t or missing_v:
        print(f"  ❌  Missing tables: {missing_t}  Missing views: {missing_v}")
        return False

    chk = conn.execute("PRAGMA integrity_check").fetchone()[0]
    sz  = db_path.stat().st_size / (1024 * 1024)
    ds  = conn.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
    sp  = conn.execute("SELECT COUNT(*) FROM data_splits").fetchone()[0]

    print()
    print("  ┌──────────────────────────────────────────┐")
    print("  │  Verification result                     │")
    print("  ├──────────────────────────────────────────┤")
    print(f"  │  Database size   :  {sz:.3f} MB               │")
    print(f"  │  Datasets seeded :  {ds}                        │")
    print(f"  │  Splits seeded   :  {sp}  (train / val / test)  │")
    print(f"  │  Integrity check :  {chk}                     │")
    print("  └──────────────────────────────────────────┘")
    return chk == "ok"


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DC-EFDS-Lite — Step 1: Initialize Database")
    parser.add_argument("--db",    default=str(DEFAULT_DB), help="Path to SQLite file")
    parser.add_argument("--reset", action="store_true",     help="Back up + recreate (destructive)")
    args   = parser.parse_args()
    db_path = Path(args.db)

    _print_banner()

    if args.reset:
        answer = input("  ⚠️   --reset will wipe all data. Type YES to confirm: ")
        if answer.strip().upper() != "YES":
            print("  Aborted.")
            sys.exit(0)
        if db_path.exists():
            bak = db_path.with_suffix(".backup.db")
            db_path.rename(bak)
            print(f"  ⚠️   Backed up to: {bak}")

    _create_dirs()
    conn = _apply_schema(db_path)
    _seed_splits(conn)
    _seed_datasets(conn)
    ok   = _verify(conn, db_path)
    conn.close()

    print()
    if ok:
        print("  ✅  Step 1 complete.")
        print(f"      Database: {db_path.resolve()}")
        print()
        print("  ─────────────────────────────────────────────────────")
        print("  Next:  python scripts/download_all_datasets.py")
        print()
    else:
        print("  ❌  Verification failed — check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()