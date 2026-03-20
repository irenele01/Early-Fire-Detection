"""
src/inference/config.py
────────────────────────
Loads config/pipeline.yaml into a typed PipelineConfig dataclass.

All inference modules import PipelineConfig from here — no module
reads YAML directly. This gives one place to change defaults and
one place to validate.
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


# ── Config dataclass ──────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    # ── Camera ────────────────────────────────────────────────────────────────
    source:             str        = "0"
    resolution:         List[int]  = field(default_factory=lambda: [640, 480])
    fps_target:         int        = 15
    reconnect_delay_s:  float      = 5.0
    queue_maxsize:      int        = 4

    # ── Model ─────────────────────────────────────────────────────────────────
    model_path:         str        = "models/best.onnx"
    input_size:         int        = 320
    conf_threshold:     float      = 0.50
    iou_threshold:      float      = 0.45
    class_names:        Dict       = field(default_factory=lambda: {0: "smoke", 1: "flame", 2: "overheat"})

    # ── Inference ─────────────────────────────────────────────────────────────
    frame_skip:                 int   = 3
    alert_persistence_frames:   int   = 5
    cooldown_seconds:           float = 30.0
    threads:                    int   = 1

    # ── Alerts ────────────────────────────────────────────────────────────────
    telegram_enabled:     bool          = False
    telegram_token:       str           = ""
    telegram_chat_id:     str           = ""
    gpio_enabled:         bool          = False
    gpio_led_pin:         int           = 17
    gpio_buzzer_pin:      Optional[int] = 27
    save_snapshots:       bool          = True
    snapshot_dir:         str           = "data/alerts"
    rate_limit_per_hour:  int           = 20

    # ── Database ──────────────────────────────────────────────────────────────
    db_path:          str = "database/alerts.db"
    retention_days:   int = 30

    # ── Display ───────────────────────────────────────────────────────────────
    show_window:      bool = False
    show_fps_overlay: bool = True
    show_bbox:        bool = True

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def class_name_list(self) -> List[str]:
        """Return class names indexed by class ID."""
        return [self.class_names.get(i, str(i)) for i in range(len(self.class_names))]

    @property
    def model_path_exists(self) -> bool:
        return Path(self.model_path).exists()


# ── Loader ────────────────────────────────────────────────────────────────────

def load_config(yaml_path: str = "config/pipeline.yaml") -> PipelineConfig:
    """
    Load pipeline.yaml and return a populated PipelineConfig.
    Environment variables TELEGRAM_TOKEN and TELEGRAM_CHAT_ID override
    any values in the YAML file.
    """
    path = Path(yaml_path)
    if not path.exists():
        print(f"  ⚠️   Config not found: {path} — using defaults.")
        print(f"      Create config/pipeline.yaml or pass --config <path>")
        return PipelineConfig()

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    cam   = raw.get("camera",    {})
    model = raw.get("model",     {})
    inf   = raw.get("inference", {})
    alrt  = raw.get("alerts",    {})
    db    = raw.get("database",  {})
    disp  = raw.get("display",   {})

    # Telegram credentials: .env overrides YAML
    tg_token   = os.getenv("TELEGRAM_TOKEN",   alrt.get("telegram_token",   ""))
    tg_chat_id = os.getenv("TELEGRAM_CHAT_ID", alrt.get("telegram_chat_id", ""))

    cfg = PipelineConfig(
        # Camera
        source            = str(cam.get("source", "0")),
        resolution        = cam.get("resolution", [640, 480]),
        fps_target        = int(cam.get("fps_target", 15)),
        reconnect_delay_s = float(cam.get("reconnect_delay_s", 5.0)),
        queue_maxsize     = int(cam.get("queue_maxsize", 4)),
        # Model
        model_path        = str(model.get("path", "models/best.onnx")),
        input_size        = int(model.get("input_size", 320)),
        conf_threshold    = float(model.get("confidence_threshold", 0.50)),
        iou_threshold     = float(model.get("iou_threshold", 0.45)),
        class_names       = model.get("class_names", {0: "smoke", 1: "flame", 2: "overheat"}),
        # Inference
        frame_skip               = int(inf.get("frame_skip", 3)),
        alert_persistence_frames = int(inf.get("alert_persistence_frames", 5)),
        cooldown_seconds         = float(inf.get("cooldown_seconds",  30.0)),
        threads                  = int(inf.get("threads", 1)),
        # Alerts
        telegram_enabled    = bool(alrt.get("telegram_enabled", False)),
        telegram_token      = tg_token,
        telegram_chat_id    = tg_chat_id,
        gpio_enabled        = bool(alrt.get("gpio_enabled", False)),
        gpio_led_pin        = int(alrt.get("gpio_led_pin", 17)),
        gpio_buzzer_pin     = alrt.get("gpio_buzzer_pin", 27),
        save_snapshots      = bool(alrt.get("save_snapshots", True)),
        snapshot_dir        = str(alrt.get("snapshot_dir", "data/alerts")),
        rate_limit_per_hour = int(alrt.get("rate_limit_per_hour", 20)),
        # Database
        db_path         = str(db.get("path", "database/alerts.db")),
        retention_days  = int(db.get("retention_days", 30)),
        # Display
        show_window      = bool(disp.get("show_window", False)),
        show_fps_overlay = bool(disp.get("show_fps_overlay", True)),
        show_bbox        = bool(disp.get("show_bbox", True)),
    )
    return cfg