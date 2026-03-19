# DC-EFDS-Lite — System Design Document

**Version:** 1.0  
**Status:** Production  
**Author:** Irene Le 
**Last Updated:** March 2026

---

## Table of contents

1. [System overview](#1-system-overview)
2. [Architecture decisions](#2-architecture-decisions)
3. [Component design](#3-component-design)
4. [Data pipeline](#4-data-pipeline)
5. [Inference engine](#5-inference-engine)
6. [Database schema](#6-database-schema)
7. [Alert system](#7-alert-system)
8. [Model training pipeline](#8-model-training-pipeline)
9. [Security model](#9-security-model)
10. [Performance characteristics](#10-performance-characteristics)
11. [Failure modes and mitigations](#11-failure-modes-and-mitigations)
12. [Deployment architecture](#12-deployment-architecture)

---

## 1. System overview

EFDS-Lite is a two-environment system: a **development environment** (MacBook M3) responsible for data engineering and model training, and a **production environment** (Raspberry Pi 3) responsible for real-time inference and alerting.

The two environments communicate only once — at deploy time — via a single ONNX model file transfer over SSH. After deployment, the Pi runs fully air-gapped from the training environment.

### System context

```
┌─────────────────────────────────────────────────────────────┐
│  Local Network (192.168.x.x)                                │
│                                                             │
│   iPhone 6          Raspberry Pi 3         User Device     │
│  ┌────────┐         ┌───────────┐          ┌────────────┐  │
│  │IP Webcam│──HTTP──►│ Inference │──Telegram►│   Phone    │  │
│  │ :8080  │  MJPEG  │ Pipeline  │   (TLS)  │ Dashboard  │  │
│  └────────┘         └─────┬─────┘          └────────────┘  │
│                           │                                 │
│                    ┌──────▼──────┐                          │
│                    │  SQLite DB  │                          │
│                    │  + Streamlit│                          │
│                    └─────────────┘                          │
└─────────────────────────────────────────────────────────────┘

  MacBook M3  (development only — not connected at runtime)
  ┌──────────────────────────────────────────────┐
  │  Datasets → ETL → SQLite → Training → ONNX  │
  │                               ▼              │
  │                     scp best.onnx → Pi 3    │
  └──────────────────────────────────────────────┘
```

### Key design constraints

| Constraint | Implication |
|-----------|-------------|
| Pi 3 has 1GB RAM | Model must be <10MB ONNX; frame buffer capped at 4 frames |
| Pi 3 is quad-core ARM Cortex-A53 | CPU-only inference; ONNX Runtime, not PyTorch |
| No cloud dependency | All inference, storage, and logic runs on-device |
| Privacy requirement | Video frames never written to disk; only alert snapshots saved |
| False positive tolerance < 1%/day | Temporal validation + personal negative training required |

---

## 2. Architecture decisions

### 2.1 Producer-consumer frame pipeline

The video capture thread and the inference thread are explicitly decoupled via a bounded queue. This is the most critical architectural decision in the system.

**Why:** OpenCV's `VideoCapture.read()` blocks until a frame arrives from the network. If inference runs in the same thread, any network hiccup (common on Wi-Fi) stalls the entire pipeline. With separate threads, the capture thread can reconnect without affecting inference, and inference can take however long it needs without dropping incoming frames.

**The queue is bounded and lossy by design.** When the inference thread is slower than the capture thread, the oldest frame is dropped rather than buffered. This prioritises freshness over completeness — in a fire detection system, a 3-second-old frame is worse than no frame at all.

```python
# Non-blocking drop: always keep the latest frame
if self._frame_queue.full():
    self._frame_queue.get_nowait()   # discard oldest
self._frame_queue.put_nowait(frame)  # enqueue latest
```

### 2.2 ONNX Runtime over PyTorch on the Pi

PyTorch on ARM is slow for inference and consumes significantly more memory. ONNX Runtime with `CPUExecutionProvider` and `ORT_ENABLE_ALL` graph optimisation delivers 2–3× faster inference on the Pi 3 with lower RAM usage. The model is exported from Ultralytics with `opset=12` and `simplify=True` to produce a clean single-graph ONNX file.

### 2.3 Pre-allocated input tensor

A common source of latency jitter on embedded systems is heap allocation. The inference engine allocates the input blob once at `__init__` time and reuses it every frame via `np.copyto()`. On a Pi 3 this reduces per-frame allocation time from ~3ms to ~0.1ms and reduces GC pressure.

```python
# Allocated once
self._input_blob = np.zeros((1, 3, 320, 320), dtype=np.float32)

# Per-frame: writes into the buffer, no new allocation
np.copyto(self._input_blob[0], rgb.transpose(2, 0, 1) / 255.0)
```

### 2.4 YOLOv8n at 320×320

YOLOv8 comes in five sizes (n/s/m/l/x). The nano variant at 320×320 input is the only configuration that achieves <500ms inference on a Pi 3 CPU while staying under 400MB RAM. Larger models or higher resolution inputs would require a Pi 4 or better.

### 2.5 Temporal validation with consecutive-frame gating

A single positive detection does not trigger an alert. The system requires `persistence_frames` (default: 5) **consecutive** positive frames before confirming an event. This is distinct from a sliding window approach — a single clean frame resets the streak to zero.

This design choice reduces false positives from transient lighting changes (a camera flash, a monitor turning on) to near zero, at the cost of ~1 second of additional detection latency (5 frames × ~200ms inference).

```
Frame sequence:  + + + - + + + + +
Streak counter:  1 2 3 0 1 2 3 4 5 → ALERT at frame 9
```

A sliding window (`all(buffer)`) would fire at frame 8 (4 positives out of 5), which misses the reset caused by the false negative at frame 4.

### 2.6 SQLite over PostgreSQL

The system runs on a device with no guaranteed network uptime. SQLite with WAL journal mode provides:
- Zero configuration and zero daemon processes
- Safe concurrent reads from the dashboard while the pipeline writes
- Adequate throughput for the write pattern (~1 alert row + 1 metrics row per minute)

Write-Ahead Logging (`PRAGMA journal_mode=WAL`) allows the Streamlit dashboard to read while the pipeline is writing, eliminating locking contention without requiring a separate database server.

---

## 3. Component design

### 3.1 Component map

```
┌──────────────────────────────────────────────────────────────────┐
│  Pipeline (main loop)                                            │
│                                                                  │
│  ┌─────────────┐    ┌──────────────────────────────────────┐    │
│  │SourceAdapter│    │          InferenceEngine              │    │
│  │             │    │  ┌────────────┐  ┌─────────────────┐ │    │
│  │ CaptureThread────►│  │ preprocess │──►   ONNX session  │ │    │
│  │ frame queue  │    │  └────────────┘  └────────┬────────┘ │    │
│  │ reconnect    │    │                           │          │    │
│  └─────────────┘    │                    ┌───────▼──────┐  │    │
│                     │                    │ parse_output  │  │    │
│                     │                    │ NMS filter    │  │    │
│                     └────────────────────┴───────┬───────┘    │
│                                                  │             │
│                              ┌───────────────────▼──────────┐ │
│                              │     TemporalValidator         │ │
│                              │     consecutive streak        │ │
│                              └───────────────┬──────────────┘ │
│                                              │ confirmed?      │
│                         ┌────────────────────▼──────────────┐ │
│                         │         FusionEngine               │ │
│                         │  visual + thermal → Alert level    │ │
│                         └──────────────┬─────────────────────┘ │
│                                        │                        │
│                         ┌──────────────▼──────────────────────┐│
│                         │       AlertDispatcher               ││
│                         │  Telegram · SQLite · GPIO · file    ││
│                         └─────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 SourceAdapter

Responsibility: abstract all video sources behind a single `read()` interface, handle reconnect logic, and manage the frame drop queue.

| Method | Behaviour |
|--------|-----------|
| `start()` | Opens VideoCapture, starts daemon capture thread |
| `stop()` | Sets stop event, releases capture |
| `read()` | Non-blocking; returns `None` if queue empty |
| `_capture_loop()` | Reads frames, applies `frame_skip`, non-blocking enqueue |
| `_connect()` | Called on init and on any read failure; resets VideoCapture |

The capture thread is a Python daemon thread — it is automatically killed when the main process exits, which prevents zombie processes on Pi restart.

### 3.3 InferenceEngine

Responsibility: load ONNX model, run inference, return structured detections.

The inference session is configured with:
- `inter_op_num_threads=1` — Pi 3 benefits from fewer threads due to memory bandwidth limits
- `intra_op_num_threads=2` — parallelism within operators (two physical cores available after OS overhead)
- `ORT_ENABLE_ALL` — full graph fusion, constant folding, memory pattern optimisation

The warmup pass on `__init__` runs one dummy inference. The first inference call always triggers JIT kernel compilation in ONNX Runtime, causing a 200–800ms spike. Running it at startup prevents this spike from appearing mid-detection.

### 3.4 TemporalValidator

Responsibility: gate alerts behind a consecutive-frame persistence requirement and a cooldown window.

State:
- `_streak: int` — number of consecutive positive frames
- `_last_alert_time: float` — Unix timestamp of last confirmed alert

The cooldown prevents alert spam during a sustained fire event. The default 30-second cooldown means repeated Telegram messages are suppressed, but the system continues logging every detection to the database.

### 3.5 FusionEngine

Responsibility: combine visual detections and thermal temperature data into a tiered alert decision.

Alert levels and their trigger logic:

| Level | Condition | Action |
|-------|-----------|--------|
| 1 — Critical | Thermal > 80°C OR (flame detected AND thermal > 60°C) | Telegram + GPIO + snapshot |
| 2 — High | Flame detected (no thermal) OR (smoke + thermal > 60°C) | Telegram + GPIO |
| 3 — Warning | Smoke detected OR thermal > 60°C (standalone) | Telegram only |

Thermal input is `Optional[float]` — if no thermal camera is connected, the engine falls back to visual-only logic without any code changes.

### 3.6 AlertDispatcher

Responsibility: fan out a confirmed alert to all configured sinks.

Thread safety: all SQLite writes are protected by `threading.Lock`. The Streamlit dashboard reads from the same database concurrently, which is safe under WAL mode.

The Telegram dispatcher sends a `sendPhoto` request when a snapshot is available, falling back to `sendMessage` on any I/O error. Network failures are caught and logged without crashing the pipeline.

---

## 4. Data pipeline

### 4.1 ETL overview

```
Raw sources                 Transform                    Load
───────────                 ─────────                    ────
D-Fire (21K)   ─┐
Roboflow (6K)  ─┤──► VisualExtractor ──► Resize 320px ──► data/processed/
Kaggle (17K)   ─┤                        BGR→RGB           images/{train,val,test}
Personal (500) ─┘                         Validate         labels/{train,val,test}
                                          Split 70/20/10   dataset.yaml
FLIR (14K)     ──── ThermalExtractor ──► Planck decode ──► thermal_metadata table
KAIST (95K)         (optional)            Temp extract      images/thermal/
```

### 4.2 Data split strategy

The 70/20/10 split is stratified by both modality (visual/thermal) and class (smoke/flame/overheat/negative) to ensure the validation and test sets reflect the training distribution. A random split without stratification risks placing all difficult edge cases (e.g. steam that looks like smoke) into the training set, which inflates validation mAP.

### 4.3 Negative sample importance

The personal negative capture is architecturally critical, not optional. A model trained only on public fire datasets will never see:
- Your specific server rack LEDs (which can resemble flame at certain angles)
- Your UPS unit battery indicator lights
- Reflections off metallic rack surfaces
- Dust particles in air cooling vents

Without these negatives, the model optimises for the public dataset distribution and produces false positives in the real deployment environment. The target is 500+ images captured across multiple lighting conditions (day, night, LED-lit).

### 4.4 Visual extractor

For each image the extractor computes:
- SHA-256 checksum (deduplication)
- EXIF capture timestamp (provenance)
- Image integrity check via OpenCV decode
- YOLO label file parsing and normalisation validation (all coordinates must be in [0,1])

### 4.5 Thermal extractor

FLIR radiometric JPEG files embed temperature data in proprietary EXIF tags. The extractor reads:
- Planck constants (R1, B, F, O, K) from EXIF
- Per-pixel raw counts → temperature via `T = B / ln(R1 / (raw + O) + F) - 273.15`
- Summary statistics (min, max, avg, median) stored in `thermal_metadata` table

---

## 5. Inference engine

### 5.1 Input preprocessing pipeline

```
Raw frame (640×480 BGR)
        │
        ▼
cv2.resize(320, 320)          # Bilinear interpolation
        │
        ▼
cv2.cvtColor(BGR → RGB)       # YOLO expects RGB
        │
        ▼
transpose (HWC → CHW)         # PyTorch/ONNX channel-first format
        │
        ▼
/ 255.0  →  float32           # Normalise to [0, 1]
        │
        ▼
np.copyto(pre_allocated_blob)  # Write into [1,3,320,320] buffer
```

### 5.2 Output parsing

YOLOv8 ONNX output shape: `[1, 4+num_classes, num_anchors]`

The output is transposed to `[num_anchors, 4+num_classes]` where the first 4 columns are `[cx, cy, w, h]` in normalised coordinates and the remaining columns are per-class scores (not sigmoid-activated — raw logits).

Post-processing steps:
1. `argmax` across class columns to get predicted class and raw score
2. Confidence threshold filter (default 0.50)
3. Convert `cx,cy,w,h` → `x1,y1,x2,y2` in original pixel coordinates
4. `cv2.dnn.NMSBoxes` for non-maximum suppression (IOU threshold 0.45)

### 5.3 Inference latency profile (Pi 3 benchmarks)

| Stage | Time (ms) |
|-------|-----------|
| Frame read from queue | ~1 |
| Resize + colour convert | ~8 |
| `np.copyto` normalise | ~2 |
| ONNX session run | ~220–320 |
| Output parse + NMS | ~5 |
| **Total per-frame** | **~240–340ms** |

With `frame_skip=3`, the effective processing rate is one frame every ~280ms, or approximately 3.5 effective FPS. The capture thread continues buffering at the source's native 15 FPS; only every 3rd captured frame enters the inference queue.

---

## 6. Database schema

### 6.1 Training database (`dc_efds.db`)

Used during ETL and training on the Mac. Not deployed to Pi.

```sql
datasets          -- dataset provenance and metadata
images            -- per-image record with checksum and split assignment
annotations       -- YOLO bounding boxes, normalised coordinates
thermal_metadata  -- temperature stats and Planck parameters per image
data_splits       -- train/val/test split definitions
```

Key design decisions:
- `image_id` is a UUID string, not an integer — avoids collision when merging multiple datasets
- `privacy_cleared` boolean on images table — supports GDPR compliance workflow
- All foreign keys enforced via `PRAGMA foreign_keys = ON`
- Indexes on `(dataset_id, modality)` and `(image_id)` for ETL query performance

### 6.2 Production database (`alerts.db`)

Deployed to Pi. Minimal schema — only what's needed for alerting and the dashboard.

```sql
alerts            -- one row per confirmed alert event
system_metrics    -- one row per minute: CPU, RAM, FPS, inference time
```

```sql
CREATE TABLE alerts (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp         TEXT    DEFAULT (datetime('now')),
    detection_type    TEXT    NOT NULL,   -- 'smoke' | 'flame' | 'overheat' | 'heat_warning'
    detection_source  TEXT,               -- 'visual' | 'thermal' | 'fusion'
    max_temperature_c REAL,
    confidence        REAL    NOT NULL,
    bbox_coords       TEXT,               -- "x1,y1,x2,y2" pixel coords
    frame_path        TEXT,               -- path to saved snapshot JPG
    resolved          INTEGER DEFAULT 0
);

CREATE TABLE system_metrics (
    timestamp         TEXT    PRIMARY KEY DEFAULT (datetime('now')),
    cpu_percent       REAL,
    memory_mb         REAL,
    inference_time_ms REAL,
    fps_achieved      REAL,
    camera_connected  INTEGER
);
```

Indexes:
- `idx_alerts_timestamp DESC` — dashboard query for recent alerts
- `idx_alerts_unresolved` partial index — active alert monitoring
- Metrics table uses `timestamp` as primary key, preventing duplicate rows per minute

Retention: a maintenance cron job purges `alerts` rows older than 30 days and `system_metrics` rows older than 7 days.

---

## 7. Alert system

### 7.1 Alert decision tree

```
Detections from InferenceEngine
         │
         ▼
  TemporalValidator
  (5 consecutive frames)
         │
    confirmed?
    /         \
  No           Yes
  │             │
  │         FusionEngine
  │       (visual + thermal)
  │             │
  │      ┌──────┴───────┐
  │    Alert?          None
  │      │               │
  │   Dispatch         Log only
  │   ├── SQLite
  │   ├── Telegram
  │   ├── GPIO
  │   └── Snapshot
  │
  Continue loop
```

### 7.2 Telegram message format

```
[CRITICAL] DC-EFDS Detection
Type: FLAME
Source: FUSION
Confidence: 87%
Temp: 73.2°C
[attached: alert_snapshot.jpg]
```

The bot sends a photo message when a snapshot is available. Photo messages arrive with the caption inline in Telegram, which is important for rapid triage — the user sees the frame and the metadata simultaneously without having to open a separate message.

### 7.3 Alert rate limiting

Beyond the `cooldown_seconds` in the temporal validator, the system caps Telegram messages at 20 per hour to prevent flooding during a sustained event. This is implemented as a token bucket in `AlertDispatcher`.

---

## 8. Model training pipeline

### 8.1 Training environment

Training runs on MacBook M3 using Apple Metal Performance Shaders (MPS) via PyTorch's `torch.backends.mps` backend. MPS provides GPU acceleration without requiring CUDA or a separate GPU.

For users without Apple Silicon, the training config supports `device: cuda` (NVIDIA) or falls back to `device: cpu`.

### 8.2 Model selection rationale

YOLOv8 nano (`yolov8n`) was selected over competing architectures based on:
- Sub-500ms inference on ARM Cortex-A53 (Pi 3)
- 3.2M parameters — fits comfortably in Pi 3 RAM alongside OS and dashboard
- Ultralytics ONNX export is stable and well-tested on ARM
- Strong community fire/smoke dataset availability in YOLO label format

Alternatives considered:
- MobileNetV3 SSD — faster but worse mAP on small smoke detections
- YOLOv8s — better mAP but 450ms+ inference on Pi 3, too close to the 500ms limit
- TensorFlow Lite — rejected due to ONNX Runtime's better ARM optimisation

### 8.3 Augmentation strategy

The augmentation config deliberately constrains fire-specific colour augmentation:

```yaml
hsv_h: 0.015    # Minimal hue shift — fire has a narrow orange/red hue range
hsv_s: 0.7      # Saturation variation covers smoke (low sat) to flame (high sat)
hsv_v: 0.4      # Value variation covers different lighting conditions
flipud: 0.0     # Fire never appears upside down — disable vertical flip
fliplr: 0.5     # Horizontal flip is safe and doubles effective dataset size
mosaic: 1.0     # Mosaic augmentation improves small object detection
```

The `flipud: 0.0` setting is often overlooked in generic YOLO training but is important here — training on inverted fire images would teach the model a spatial prior that doesn't exist in deployment.

### 8.4 Export to ONNX

```python
model.export(
    format='onnx',
    imgsz=320,      # Must match inference config
    simplify=True,  # Removes redundant nodes via onnx-simplifier
    opset=12,       # opset 12 is the last version with full ONNX Runtime ARM support
)
```

The export produces a ~6MB ONNX file. `simplify=True` reduces this from ~9MB by fusing batch normalisation layers and removing identity nodes. The resulting model graph has ~85 operators versus ~140 in the unsimplified version.

---

## 9. Security model

### 9.1 Threat model

| Threat | Likelihood | Mitigations |
|--------|-----------|-------------|
| Unauthorised LAN access to Pi | Medium | UFW firewall; SSH key-only; local network only |
| Alert spam / denial of service | Medium | Rate limiting (20 alerts/hour); cooldown window |
| Video interception | Low | No external streams; Telegram uses TLS |
| Model tampering | Low | SHA-256 checksum on model file at startup |
| Physical Pi access | Low | Locked enclosure; systemd auto-restart |

### 9.2 Network configuration

```
Inbound allowed (local network only):
  - TCP 22  (SSH) — from 192.168.x.x/24 only
  - TCP 8501 (Streamlit) — from 192.168.x.x/24 only

Outbound allowed:
  - TCP 443 (Telegram API) — outbound only, TLS
  - TCP 80/443 (system updates)

All other inbound: DENY
```

### 9.3 Data privacy

| Data type | Storage | Retention |
|-----------|---------|-----------|
| Live video frames | RAM only, never written | 0 days |
| Alert snapshots | Local file (chmod 600) | 30 days |
| Detection metadata | SQLite (chmod 600) | 30 days |
| System metrics | SQLite | 7 days |
| Training images | Mac only, never on Pi | Indefinite |

The Pi never stores raw video. Only the single frame at the moment of a confirmed alert is saved as a JPEG, and only when `save_snapshots: true` is configured.

---

## 10. Performance characteristics

### 10.1 Latency breakdown — alert end-to-end

```
Frame captured by iPhone          t=0
Frame arrives at Pi (MJPEG)       t=0 + ~50ms  (network + decode)
Frame enters queue                t=0 + ~55ms
Inference engine picks up frame   t=0 + ~55ms  (immediately, no wait)
ONNX inference completes          t=0 + ~330ms
Temporal validator: frame N of 5  t=0 + ~330ms
...4 more consecutive frames...
5th frame confirmed               t=0 + ~1650ms (~1.6s total streak)
Alert dispatched                  t=0 + ~1660ms
Telegram HTTP request             t=0 + ~1660ms
Telegram message delivered        t=0 + ~2000–2500ms (network dependent)
```

End-to-end target of 2 seconds is achievable with 5-frame persistence and typical home network latency.

### 10.2 Resource usage (Pi 3, steady state)

| Resource | Typical | Peak |
|----------|---------|------|
| RAM | 320–380MB | 420MB |
| CPU | 55–70% | 85% (during inference) |
| CPU temperature | 55–62°C | 72°C |
| Disk writes | ~2KB/min (metrics) | ~200KB (alert snapshot) |
| Network receive | ~800KB/s (MJPEG stream) | 1.2MB/s |

Heatsinks are required on the Pi 3 CPU. Without them, the BCM2837 throttles at 80°C, dropping inference to unpredictable speeds.

### 10.3 Model performance targets

| Metric | Target | Notes |
|--------|--------|-------|
| mAP@50 | > 0.85 | Measured on held-out test set |
| Precision | > 0.80 | Low false positive rate is prioritised over recall |
| Recall | > 0.75 | Some missed detections acceptable; temporal validator compensates |
| Inference (Pi 3) | < 500ms | Hard limit for real-time monitoring |
| Model size (ONNX) | < 10MB | Fits in SD card and RAM with headroom |

---

## 11. Failure modes and mitigations

### 11.1 Camera stream loss

**Detection:** `VideoCapture.read()` returns `ok=False`

**Behaviour:** Capture thread logs a warning, sleeps 5 seconds, calls `_connect()`. The inference loop continues draining the queue until empty, then idles at `time.sleep(0.01)` until new frames arrive. No crash, no alert storm.

**Dashboard indicator:** `camera_connected = 0` in `system_metrics`

### 11.2 Pi 3 thermal throttling

**Detection:** Inference time exceeds 800ms (2× normal)

**Mitigation:** `CPUQuota=80%` in systemd service limits sustained CPU usage to prevent thermal runaway. Heatsinks and optional USB fan reduce steady-state temperature by 10–15°C.

**Operational guidance:** If temperatures consistently exceed 70°C, reduce `fps_target` in config or increase `frame_skip` to 4.

### 11.3 SQLite lock contention

**Risk:** Streamlit dashboard reading while pipeline is writing.

**Mitigation:** WAL journal mode allows concurrent readers. The pipeline uses a `threading.Lock` to serialise writes from the main inference thread. Streamlit is a separate process; it gets a shared read lock from SQLite automatically.

### 11.4 Telegram API unavailability

**Behaviour:** `AlertDispatcher._send_telegram()` wraps all requests in `try/except`. A failed Telegram dispatch logs the error and continues — the alert is still written to SQLite and the GPIO still triggers. The system does not retry failed Telegram messages (to avoid alert storms after reconnection).

### 11.5 Model file corruption

**Detection:** `onnxruntime.InferenceSession()` raises on load.

**Mitigation:** `validate_install.py` checks model integrity before every deployment. The systemd service has `Restart=on-failure` but a corrupted model will cause the service to restart and immediately fail again — the on-call engineer must re-deploy the model from the Mac.

---

## 12. Deployment architecture

### 12.1 Service management

The pipeline runs as a systemd unit with:
- `After=network.target` — waits for network before starting (required for camera stream connection)
- `Restart=on-failure` with `RestartSec=15` — 15-second delay prevents restart storms
- `MemoryMax=450M` — OOM-kills the service before the kernel OOM killer acts
- `CPUQuota=80%` — leaves CPU headroom for systemd, logging, and SSH

### 12.2 Active learning feedback loop

False positives captured in the dashboard are tagged and saved to `data/personal/false_positives/`. A quarterly retraining run on the Mac includes these images in the negative sample set, reducing false positive rate over time.

```
Production alert (false positive)
    │
    ▼ (user tags in dashboard)
data/personal/false_positives/YYYY-MM/
    │
    ▼ (quarterly)
ETL pipeline re-runs with updated personal dataset
    │
    ▼
Retrain → new best.onnx → deploy to Pi
```

### 12.3 Maintenance schedule

| Task | Frequency | Command |
|------|-----------|---------|
| Database backup | Weekly | `cp database/alerts.db database/backups/alerts_$(date +%Y%m%d).db` |
| Old alert cleanup | Monthly | `sqlite3 alerts.db "DELETE FROM alerts WHERE timestamp < datetime('now', '-30 days')"` |
| Log rotation | Weekly | `logrotate config/logrotate.conf` |
| Model retraining | Quarterly | Re-run training pipeline with accumulated false positives |
| OS updates | Monthly | `sudo apt update && sudo apt upgrade` |
| Hardware inspection | Quarterly | Check heatsink contact, SD card health (`sudo smartctl`) |

---

## Appendix — Key configuration reference

```yaml
# config/pipeline.yaml — all tunable parameters

camera:
  source: "http://192.168.1.XX:8080/video"
  fps_target: 15
  resolution: [640, 480]

model:
  path: "models/best.onnx"
  input_size: 320                # Must match training config
  confidence_threshold: 0.50     # Lower: more sensitive. Higher: fewer false alarms.
  iou_threshold: 0.45            # NMS overlap threshold

inference:
  frame_skip: 3                  # 1 = every frame, 3 = every 3rd frame
  persistence_frames: 5          # Consecutive positives to confirm. Range: 3–10.
  cooldown_seconds: 30           # Min gap between alerts. Range: 10–120.
  queue_maxsize: 4               # Frame buffer depth. Keep at 4 for Pi 3.

thermal:
  warning_temp_c: 60.0           # First threshold — Level 3 alert
  critical_temp_c: 80.0          # Second threshold — Level 1 alert

alerts:
  save_snapshots: true
  snapshot_dir: "data/alerts"

database:
  path: "database/alerts.db"
  retention_days: 30
```