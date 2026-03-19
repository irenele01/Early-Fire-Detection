PRAGMA foreign_keys = ON;

--Core tables
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    modality TEXT CHECK (modality IN('visual','thermal','multi-modal')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_images INTEGER DEFAULT 0,
    total_annotations INTEGER DEFAULT 0 );

CREATE TABLE IF NOT EXISTS data_splits (
    split_id TEXT PRIMARY KEY,
    split_name TEXT CHECK(split_name IN ('train', 'val', 'test')),
    split_ratio REAL CHECK(split_ratio > 0 AND split_ratio <= 1),
    image_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS images (
    image_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    split_id TEXT,
    file_path TEXT NOT NULL,
    original_filename TEXT,
    modality TEXT CHECK(modality IN ('visual', 'thermal')) NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    channels INTEGER DEFAULT 3,
    file_size_mb REAL,
    captured_at TIMESTAMP,
    checksum TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    FOREIGN KEY (split_id) REFERENCES data_splits(split_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS annotations (
    annotation_id TEXT PRIMARY KEY,
    image_id TEXT NOT NULL,
    class_name TEXT NOT NULL,
    class_id INTEGER,
    confidence REAL CHECK(confidence >= 0 AND confidence <= 1),
    x_center_norm REAL CHECK(x_center_norm >= 0 AND x_center_norm <= 1),
    y_center_norm REAL CHECK(y_center_norm >= 0 AND y_center_norm <= 1),
    width_norm REAL CHECK(width_norm >= 0 AND width_norm <= 1),
    height_norm REAL CHECK(height_norm >= 0 AND height_norm <= 1),
    x_min_px INTEGER,
    y_min_px INTEGER,
    x_max_px INTEGER,
    y_max_px INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (image_id) REFERENCES images(image_id) ON DELETE CASCADE
);

-- Thermal-Specific Table
CREATE TABLE IF NOT EXISTS thermal_metadata (
    image_id TEXT PRIMARY KEY,
    min_temperature_c REAL,
    max_temperature_c REAL,
    avg_temperature_c REAL,
    median_temperature_c REAL,
    emissivity REAL DEFAULT 0.95,
    reflected_temp_c REAL,
    atmospheric_temp_c REAL,
    relative_humidity REAL,
    distance_to_object_m REAL,
    temperature_unit TEXT DEFAULT 'Celsius',
    radiometric_data_path TEXT,
    thermal_palette TEXT,
    planck_r1 REAL,
    planck_b REAL,
    planck_f REAL,
    planck_o REAL,
    planck_k REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (image_id) REFERENCES images(image_id) ON DELETE CASCADE
);

-- Production Alert Tables (Runtime)
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    detection_type TEXT NOT NULL CHECK(detection_type IN ('heat_warning', 'smoke', 'flame', 'overheat')),
    detection_source TEXT CHECK(detection_source IN ('visual', 'thermal', 'fusion')),
    max_temperature_c REAL,
    confidence REAL NOT NULL,
    bbox_coords TEXT,
    frame_path TEXT,
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS system_metrics (
    timestamp TEXT PRIMARY KEY DEFAULT (datetime('now')),
    cpu_percent REAL,
    memory_mb REAL,
    inference_time_ms REAL,
    fps_achieved REAL,
    camera_connected BOOLEAN,
    wifi_signal_strength INTEGER,
    temperature_c REAL
);

CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_annotations_class ON annotations(class_name);
CREATE INDEX IF NOT EXISTS idx_thermal_max_temp ON thermal_metadata(max_temperature_c);
CREATE INDEX IF NOT EXISTS idx_thermal_max_temp ON thermal_metadata(max_temperature_c);

-- Views
CREATE VIEW IF NOT EXISTS v_dataset_stats AS
SELECT 
    d.dataset_id, d.name, d.modality,
    COUNT(DISTINCT i.image_id) as total_images,
    COUNT(DISTINCT a.annotation_id) as total_annotations,
    COUNT(DISTINCT CASE WHEN i.modality = 'visual' THEN i.image_id END) as visual_images,
    COUNT(DISTINCT CASE WHEN i.modality = 'thermal' THEN i.image_id END) as thermal_images,
    AVG(CASE WHEN tm.max_temperature_c IS NOT NULL THEN tm.max_temperature_c END) as avg_max_temp
FROM datasets d
LEFT JOIN images i ON d.dataset_id = i.dataset_id
LEFT JOIN annotations a ON i.image_id = a.image_id
LEFT JOIN thermal_metadata tm ON i.image_id = tm.image_id
GROUP BY d.dataset_id;

CREATE VIEW IF NOT EXISTS v_alert_summary AS
SELECT 
    detection_type,
    COUNT(*) as total_alerts,
    COUNT(CASE WHEN resolved = FALSE THEN 1 END) as unresolved,
    AVG(confidence) as avg_confidence,
    MAX(timestamp) as last_alert
FROM alerts
GROUP BY detection_type;
