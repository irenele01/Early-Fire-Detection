"""
src/inference/pipeline.py
──────────────────────────
Main inference pipeline orchestrator.

Wires together all stages from the inference architecture diagram:

  Camera Stream
      ↓
  Frame Capture  (camera.CameraStream — producer thread)
      ↓
  Frame Skip?    (camera.CameraStream — frame_skip counter)
  ├── Skip → back to Camera Stream
  └── Process →
          ↓
      Preprocess 320×320  (engine.InferenceEngine.preprocess)
          ↓
      ONNX Inference       (engine.InferenceEngine.run_inference)
          ↓
      Post-Processing NMS  (engine.InferenceEngine.postprocess)
          ↓
      Temporal Validator   (validator.TemporalValidator)
      ├── Not Confirmed → loop back to Camera Stream
      └── Confirmed →
              ↓
          Alert Dispatcher (alerts.AlertDispatcher)
          ├── Telegram
          ├── GPIO LED
          └── SQLite Log

The pipeline runs in the main thread. The camera capture runs in a
daemon thread. All synchronisation is via the bounded drop queue.
"""

import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import cv2

from src.inference.camera    import CameraStream
from src.inference.engine    import InferenceEngine
from src.inference.validator import TemporalValidator
from src.inference.alerts    import Alert, AlertDispatcher
from src.inference.overlay   import draw_frame

log = logging.getLogger("pipeline")


class InferencePipeline:
    """
    Single-entry-point class for the real-time inference pipeline.
    Initialises all components from a PipelineConfig and runs the
    inference loop until stopped.
    """

    def __init__(self, cfg):
        self.cfg = cfg

        # Ensure required directories exist
        Path("logs").mkdir(exist_ok=True)
        Path(cfg.snapshot_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.db_path).parent.mkdir(parents=True, exist_ok=True)

        log.info("Initialising pipeline components...")

        # Stage 1 — Camera
        self.camera    = CameraStream(cfg)

        # Stages 2-4 — Preprocess + Inference + NMS
        self.engine    = InferenceEngine(cfg)

        # Stage 5 — Temporal Validator
        self.validator = TemporalValidator(
            persistence_frames = cfg.alert_persistence_frames,
            cooldown_seconds   = cfg.cooldown_seconds,
        )

        # Stage 6 — Alert Dispatcher (Telegram + GPIO + SQLite)
        self.dispatcher = AlertDispatcher(cfg)
        self.dispatcher.purge_old_records()

        # Metrics state
        self._fps             = 0.0
        self._last_inf_ms     = 0.0
        self._running         = False
        self._last_alert: Optional[Alert] = None

        # Graceful shutdown on Ctrl+C / SIGTERM
        signal.signal(signal.SIGINT,  self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self):
        """Start the pipeline and block until stopped."""
        log.info("Starting camera stream...")
        if not self.camera.start():
            log.error("Cannot start camera — check config.camera.source")
            sys.exit(1)

        self._running       = True
        frame_count         = 0
        fps_t0              = time.perf_counter()
        last_metric_log     = time.time()

        log.info("Pipeline running.  Press Ctrl+C to stop.")
        self._print_startup_summary()

        try:
            while self._running:
                # ── Read next frame from queue ────────────────────────────────
                frame = self.camera.read()
                if frame is None:
                    time.sleep(0.005)    # 5 ms idle — avoid busy-wait
                    continue

                # ── Stages 2-4: Preprocess → Inference → NMS ─────────────────
                detections, inf_ms   = self.engine.infer(frame)
                self._last_inf_ms    = inf_ms
                frame_count         += 1

                # ── Stage 5: Temporal Validator ───────────────────────────────
                confirmed, best_det = self.validator.update(detections)

                # ── Stage 6: Alert Dispatcher (Confirmed path only) ───────────
                alert = None
                if confirmed and best_det is not None:
                    alert = Alert(
                        level          = self._detection_level(best_det),
                        detection_type = best_det.class_name,
                        source         = "visual",
                        confidence     = best_det.confidence,
                        detection      = best_det,
                    )
                    self._last_alert = alert
                    self.dispatcher.dispatch(alert, frame)

                # ── FPS tracking ──────────────────────────────────────────────
                now     = time.perf_counter()
                elapsed = now - fps_t0
                if elapsed >= 1.0:
                    self._fps   = frame_count / elapsed
                    frame_count = 0
                    fps_t0      = now

                # ── System metrics log (every 60 s) ───────────────────────────
                if time.time() - last_metric_log > 60:
                    self.dispatcher.log_metrics(
                        inference_ms = inf_ms,
                        fps          = self._fps,
                        camera_ok    = self.camera.is_connected,
                    )
                    last_metric_log = time.time()

                # ── Display (optional, dev mode only) ─────────────────────────
                if self.cfg.show_window:
                    annotated = draw_frame(
                        frame        = frame,
                        detections   = detections,
                        alert        = alert or self._last_alert,
                        validator    = self.validator,
                        inference_ms = inf_ms,
                        fps          = self._fps,
                        cfg          = self.cfg,
                    )
                    cv2.imshow("DC-EFDS-Lite", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        log.info("Quit key pressed.")
                        break

        except Exception as e:
            log.exception("Pipeline error: %s", e)
        finally:
            self._shutdown()

    # ── Alert level logic ─────────────────────────────────────────────────────

    @staticmethod
    def _detection_level(detection) -> int:
        """
        Assign alert level from detection class.
        Level 1 = flame (highest urgency)
        Level 2 = overheat
        Level 3 = smoke (earliest warning, lowest urgency)
        """
        return {"flame": 1, "overheat": 2, "smoke": 3}.get(detection.class_name, 2)

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def _handle_signal(self, signum, frame):
        log.info("Shutdown signal received.")
        self._running = False

    def _shutdown(self):
        log.info("Shutting down pipeline...")
        self.camera.stop()
        self.dispatcher.close()
        if self.cfg.show_window:
            cv2.destroyAllWindows()
        log.info("Pipeline stopped cleanly.")

    # ── Startup summary ───────────────────────────────────────────────────────

    def _print_startup_summary(self):
        cfg = self.cfg
        print()
        print("  ┌──────────────────────────────────────────────────────┐")
        print("  │  DC-EFDS-Lite  ·  Inference Pipeline                 │")
        print("  ├──────────────────────────────────────────────────────┤")
        print(f"  │  Camera         :  {cfg.source[:45]:<45}  │")
        print(f"  │  Model          :  {cfg.model_path:<45}  │")
        print(f"  │  Input size     :  {cfg.input_size}×{cfg.input_size:<41}  │")
        print(f"  │  Frame skip     :  1 of every {cfg.frame_skip} frames"
              f"{'':>24}  │")
        print(f"  │  Persistence    :  {cfg.alert_persistence_frames} consecutive frames"
              f"{'':>24}  │")
        print(f"  │  Confidence     :  {cfg.conf_threshold:.0%}"
              f"{'':>39}  │")
        print(f"  │  Telegram       :  {'✅ enabled' if cfg.telegram_enabled else '❌ disabled':<45}  │")
        print(f"  │  GPIO           :  {'✅ enabled' if cfg.gpio_enabled else '❌ disabled':<45}  │")
        print(f"  │  Display window :  {'✅ on' if cfg.show_window else '❌ headless':<45}  │")
        print("  └──────────────────────────────────────────────────────┘")
        print()