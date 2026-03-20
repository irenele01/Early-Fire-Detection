"""
src/inference/alerts.py
────────────────────────
Stage 6 — Alert Dispatcher → Telegram / GPIO LED / SQLite Log

Implements the right side of the inference architecture diagram:
  Alert Dispatcher → Telegram
                   → GPIO LED
                   → SQLite Log

Each sink is independent — a failure in Telegram does NOT block GPIO or SQLite.
All SQLite writes are protected by a threading.Lock for safe concurrent access
alongside the Streamlit dashboard reader.

Alert levels:
  Level 1 — CRITICAL : thermal >80°C OR visual flame + thermal >60°C
  Level 2 — HIGH     : visual flame (no thermal) OR smoke + thermal >60°C
  Level 3 — WARNING  : visual smoke OR thermal >60°C standalone
"""

import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.inference.engine import Detection

log = logging.getLogger("alerts")


# ── Alert dataclass ───────────────────────────────────────────────────────────

@dataclass
class Alert:
    level:          int               # 1=critical, 2=high, 3=warning
    detection_type: str               # 'smoke' | 'flame' | 'overheat' | 'heat_warning'
    source:         str               # 'visual' | 'thermal' | 'fusion'
    confidence:     float
    detection:      Optional[Detection] = None
    max_temp_c:     Optional[float]     = None

    LEVEL_LABEL = {1: "CRITICAL", 2: "HIGH", 3: "WARNING"}
    LEVEL_EMOJI = {1: "🔥 [CRITICAL]", 2: "⚠️ [HIGH]", 3: "ℹ️ [WARNING]"}

    @property
    def label(self) -> str:
        return self.LEVEL_LABEL.get(self.level, "ALERT")

    @property
    def emoji_label(self) -> str:
        return self.LEVEL_EMOJI.get(self.level, "🔔 [ALERT]")


# ── Alert Dispatcher ──────────────────────────────────────────────────────────

class AlertDispatcher:
    """
    Fans out a confirmed alert to all configured output sinks:
      1. SQLite Log  (always)
      2. Telegram    (if telegram_enabled and credentials set)
      3. GPIO LED    (if gpio_enabled and running on Pi)

    Each sink runs independently — one failure never blocks another.
    Includes a rate limiter (max N Telegram messages per hour).
    """

    def __init__(self, cfg):
        self.cfg      = cfg
        self._db_lock = threading.Lock()
        self._db: Optional[sqlite3.Connection] = None

        # Rate limiter state
        self._telegram_timestamps: list = []   # Recent message timestamps

        # Initialise sinks
        self._init_db()
        self._init_gpio()

        Path(cfg.snapshot_dir).mkdir(parents=True, exist_ok=True)

    # ── Main dispatch method ──────────────────────────────────────────────────

    def dispatch(self, alert: Alert, frame: Optional[np.ndarray] = None):
        """
        Send alert to all configured sinks.
        Called from the main inference loop when TemporalValidator confirms.
        """
        log.warning(
            "ALERT L%d [%s] | type=%s | source=%s | conf=%.0f%% | temp=%s",
            alert.level, alert.label,
            alert.detection_type, alert.source,
            alert.confidence * 100,
            f"{alert.max_temp_c:.1f}°C" if alert.max_temp_c else "N/A",
        )

        # Save snapshot (used by Telegram and stored path in DB)
        snapshot_path = self._save_snapshot(alert, frame)

        # ── Sink 1: SQLite Log (always, never blocked by other failures) ──────
        try:
            self._log_to_db(alert, snapshot_path)
        except Exception as e:
            log.error("SQLite log failed: %s", e)

        # ── Sink 2: Telegram ──────────────────────────────────────────────────
        if self.cfg.telegram_enabled and self.cfg.telegram_token and self.cfg.telegram_chat_id:
            if self._rate_limit_ok():
                try:
                    self._send_telegram(alert, snapshot_path)
                except Exception as e:
                    log.error("Telegram dispatch failed: %s", e)
            else:
                log.warning(
                    "Telegram rate limit reached (%d/hour) — alert logged but not sent",
                    self.cfg.rate_limit_per_hour,
                )

        # ── Sink 3: GPIO LED / Buzzer ─────────────────────────────────────────
        if self.cfg.gpio_enabled:
            try:
                self._trigger_gpio(alert)
            except Exception as e:
                log.debug("GPIO trigger failed (expected on non-Pi hardware): %s", e)

    # ── Sink 1: SQLite Log ────────────────────────────────────────────────────

    def _init_db(self):
        """Create or connect to the alerts database. Creates tables if missing."""
        db_path = Path(self.cfg.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode = WAL")   # Safe concurrent reads
        self._db.execute("PRAGMA synchronous  = NORMAL")
        with self._db_lock:
            self._db.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp         TEXT    NOT NULL DEFAULT (datetime('now')),
                    detection_type    TEXT    NOT NULL
                        CHECK(detection_type IN ('heat_warning','smoke','flame','overheat')),
                    detection_source  TEXT
                        CHECK(detection_source IN ('visual','thermal','fusion')),
                    max_temperature_c REAL,
                    confidence        REAL    NOT NULL,
                    bbox_coords       TEXT,
                    frame_path        TEXT,
                    resolved          INTEGER DEFAULT 0,
                    resolved_at       TEXT,
                    notes             TEXT
                )
            """)
            self._db.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    timestamp             TEXT    PRIMARY KEY DEFAULT (datetime('now')),
                    cpu_percent           REAL,
                    memory_mb             REAL,
                    inference_time_ms     REAL,
                    fps_achieved          REAL,
                    camera_connected      INTEGER,
                    wifi_signal_strength  INTEGER,
                    temperature_c         REAL
                )
            """)
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_alerts_ts  ON alerts(timestamp DESC)"
            )
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_alerts_unresolved "
                "ON alerts(resolved) WHERE resolved = 0"
            )
            self._db.commit()
        log.info("Database ready: %s", self.cfg.db_path)

    def _log_to_db(self, alert: Alert, snapshot_path: Optional[str]):
        """Write one alert row to the SQLite database."""
        bbox_str = None
        if alert.detection:
            bbox_str = ",".join(str(int(v)) for v in alert.detection.bbox)

        with self._db_lock:
            self._db.execute("""
                INSERT INTO alerts
                  (detection_type, detection_source, max_temperature_c,
                   confidence, bbox_coords, frame_path)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    alert.detection_type,
                    alert.source,
                    alert.max_temp_c,
                    round(alert.confidence, 4),
                    bbox_str,
                    snapshot_path,
                ),
            )
            self._db.commit()
        log.debug("Alert logged to SQLite")

    def log_metrics(
        self,
        inference_ms:   float,
        fps:            float,
        camera_ok:      bool,
        wifi_rssi:      Optional[int]   = None,
        cpu_temp_c:     Optional[float] = None,
    ):
        """Write one system metrics row. Called every 60 seconds from the pipeline."""
        cpu_pct = 0.0
        mem_mb  = 0.0
        try:
            import psutil
            cpu_pct = psutil.cpu_percent(interval=None)
            mem_mb  = psutil.Process().memory_info().rss / (1024 * 1024)
        except ImportError:
            pass

        with self._db_lock:
            self._db.execute("""
                INSERT OR REPLACE INTO system_metrics
                  (cpu_percent, memory_mb, inference_time_ms, fps_achieved,
                   camera_connected, wifi_signal_strength, temperature_c)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (cpu_pct, mem_mb, round(inference_ms, 2), round(fps, 2),
                 int(camera_ok), wifi_rssi, cpu_temp_c),
            )
            self._db.commit()

    def purge_old_records(self):
        """Delete alerts and metrics older than retention_days. Call on startup."""
        days = self.cfg.retention_days
        with self._db_lock:
            self._db.execute(
                "DELETE FROM alerts WHERE timestamp < datetime('now', ?)",
                (f"-{days} days",),
            )
            self._db.execute(
                "DELETE FROM system_metrics WHERE timestamp < datetime('now', '-7 days')"
            )
            self._db.commit()
        log.info("Purged records older than %d days", days)

    # ── Sink 2: Telegram ──────────────────────────────────────────────────────

    def _send_telegram(self, alert: Alert, snapshot_path: Optional[str]):
        """Send alert via Telegram Bot API with optional photo attachment."""
        import requests

        temp_str = f"\nTemp: {alert.max_temp_c:.1f}°C" if alert.max_temp_c else ""
        msg = (
            f"{alert.emoji_label} DC-EFDS Alert\n"
            f"Type: {alert.detection_type.upper()}\n"
            f"Source: {alert.source}\n"
            f"Confidence: {alert.confidence:.0%}"
            f"{temp_str}"
        )
        base = f"https://api.telegram.org/bot{self.cfg.telegram_token}"

        if snapshot_path and Path(snapshot_path).exists():
            with open(snapshot_path, "rb") as f:
                resp = requests.post(
                    f"{base}/sendPhoto",
                    data={"chat_id": self.cfg.telegram_chat_id, "caption": msg},
                    files={"photo": f},
                    timeout=8,
                )
        else:
            resp = requests.post(
                f"{base}/sendMessage",
                json={"chat_id": self.cfg.telegram_chat_id, "text": msg},
                timeout=8,
            )

        if resp.ok:
            self._telegram_timestamps.append(time.time())
            log.info("Telegram alert sent (L%d %s)", alert.level, alert.detection_type)
        else:
            log.error("Telegram API error %d: %s", resp.status_code, resp.text[:200])

    def _rate_limit_ok(self) -> bool:
        """
        Token bucket rate limiter: max cfg.rate_limit_per_hour messages per hour.
        Prunes timestamps older than 1 hour before checking.
        """
        now     = time.time()
        one_hr  = 3600.0
        self._telegram_timestamps = [
            t for t in self._telegram_timestamps if (now - t) < one_hr
        ]
        return len(self._telegram_timestamps) < self.cfg.rate_limit_per_hour

    # ── Sink 3: GPIO ──────────────────────────────────────────────────────────

    def _init_gpio(self):
        """Set up GPIO pins. Silently skips if not on a Raspberry Pi."""
        if not self.cfg.gpio_enabled:
            return
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            GPIO.setup(self.cfg.gpio_led_pin, GPIO.OUT, initial=GPIO.LOW)
            if self.cfg.gpio_buzzer_pin is not None:
                GPIO.setup(self.cfg.gpio_buzzer_pin, GPIO.OUT, initial=GPIO.LOW)
            self._gpio = GPIO
            log.info(
                "GPIO ready — LED pin %d  buzzer pin %s",
                self.cfg.gpio_led_pin,
                self.cfg.gpio_buzzer_pin,
            )
        except (ImportError, RuntimeError) as e:
            log.debug("GPIO not available (expected on non-Pi): %s", e)
            self._gpio = None

    def _trigger_gpio(self, alert: Alert):
        """
        Flash LED and optionally pulse buzzer based on alert level.
        Level 1: continuous on + buzzer  |  Level 2: 3 flashes  |  Level 3: 1 flash
        """
        if not hasattr(self, "_gpio") or self._gpio is None:
            return

        GPIO    = self._gpio
        led     = self.cfg.gpio_led_pin
        buzzer  = self.cfg.gpio_buzzer_pin

        def flash(n: int, on_s: float = 0.3, off_s: float = 0.2):
            for _ in range(n):
                GPIO.output(led, GPIO.HIGH)
                if buzzer is not None:
                    GPIO.output(buzzer, GPIO.HIGH)
                time.sleep(on_s)
                GPIO.output(led, GPIO.LOW)
                if buzzer is not None:
                    GPIO.output(buzzer, GPIO.LOW)
                time.sleep(off_s)

        if alert.level == 1:
            flash(5, on_s=0.5, off_s=0.1)   # Critical: 5 rapid flashes
        elif alert.level == 2:
            flash(3, on_s=0.3, off_s=0.2)   # High: 3 flashes
        else:
            flash(1, on_s=0.2, off_s=0.0)   # Warning: 1 short flash

    # ── Snapshot helper ───────────────────────────────────────────────────────

    def _save_snapshot(
        self, alert: Alert, frame: Optional[np.ndarray]
    ) -> Optional[str]:
        """Save the alert frame as a JPEG snapshot. Returns file path or None."""
        if not self.cfg.save_snapshots or frame is None:
            return None

        ts       = int(time.time())
        filename = f"alert_{ts}_{alert.detection_type}_L{alert.level}.jpg"
        out_path = Path(self.cfg.snapshot_dir) / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        log.debug("Snapshot saved: %s", out_path)
        return str(out_path)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def close(self):
        """Clean up DB connection and GPIO on shutdown."""
        if self._db:
            self._db.close()
        if hasattr(self, "_gpio") and self._gpio is not None:
            try:
                self._gpio.cleanup()
            except Exception:
                pass
        log.info("AlertDispatcher closed.")