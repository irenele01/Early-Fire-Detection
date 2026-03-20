"""
src/inference/camera.py
────────────────────────
Stage 1 — Camera Stream → Frame Capture → Frame Skip?

Implements the left side of the inference architecture diagram:
  Camera Stream → Frame Capture → [Frame Skip? → Skip back / Process forward]

Design decisions:
  - Dedicated daemon thread for capture (producer)
  - Bounded drop queue: oldest frame discarded when full
    → always delivers the FRESHEST frame, never stale ones
  - frame_skip counter: only every Nth frame is enqueued for inference
  - Auto-reconnect on stream loss with configurable delay
  - Thread-safe stop via Event
"""

import logging
import queue
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("camera")


class CameraStream:
    """
    Producer thread: reads from the camera and enqueues
    selected frames for the inference consumer.

    Frame Skip logic (matches the diagram diamond node):
      - Every frame is captured from the camera
      - Only every cfg.frame_skip-th frame is forwarded to the queue
      - When the queue is full, the oldest frame is dropped (not the newest)
    """

    def __init__(self, cfg):
        self.cfg             = cfg
        self._cap: Optional[cv2.VideoCapture] = None
        self._queue          = queue.Queue(maxsize=cfg.queue_maxsize)
        self._stop_event     = threading.Event()
        self._lock           = threading.Lock()
        self._connected      = threading.Event()
        self._frame_counter  = 0        # Raw frame counter for frame_skip logic
        self._thread         = threading.Thread(
            target=self._capture_loop, daemon=True, name="CameraCapture"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """
        Open the camera and start the capture thread.
        Returns True if the camera opened successfully.
        """
        try:
            self._open()
        except RuntimeError as e:
            log.error("Camera failed to open: %s", e)
            return False
        self._thread.start()
        log.info("Camera stream started: %s", self.cfg.source)
        return True

    def stop(self):
        """Signal the capture thread to stop and release the camera."""
        self._stop_event.set()
        self._thread.join(timeout=3.0)
        with self._lock:
            if self._cap and self._cap.isOpened():
                self._cap.release()
        log.info("Camera stream stopped.")

    def read(self) -> Optional[np.ndarray]:
        """
        Non-blocking read — returns the latest processed frame or None.
        Called by the inference loop (consumer).
        """
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _open(self):
        """Open VideoCapture with minimal internal buffering."""
        src = int(self.cfg.source) if str(self.cfg.source).isdigit() else self.cfg.source
        with self._lock:
            if self._cap and self._cap.isOpened():
                self._cap.release()
            cap = cv2.VideoCapture(src)
            # Minimise OpenCV's internal frame buffer — we manage freshness ourselves
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # Set requested resolution
            w, h = self.cfg.resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open source: {self.cfg.source}")
            self._cap = cap
        self._connected.set()
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info("Camera opened: %dx%d  source=%s", actual_w, actual_h, self.cfg.source)

    def _capture_loop(self):
        """
        Producer loop — runs in a daemon thread.

        Frame skip logic:
          frame_counter increments on every successful read.
          Only frames where (frame_counter % frame_skip == 0) are forwarded.
          This matches the 'Frame Skip?' diamond in the architecture diagram.
        """
        while not self._stop_event.is_set():
            # ── Read one frame ────────────────────────────────────────────────
            with self._lock:
                if self._cap is None:
                    time.sleep(0.1)
                    continue
                ok, frame = self._cap.read()

            if not ok or frame is None:
                self._connected.clear()
                log.warning(
                    "Frame read failed — reconnecting in %ss", self.cfg.reconnect_delay_s
                )
                time.sleep(self.cfg.reconnect_delay_s)
                try:
                    self._open()
                except RuntimeError as e:
                    log.error("Reconnect failed: %s", e)
                continue

            self._connected.set()
            self._frame_counter += 1

            # ── Frame Skip decision (diagram diamond) ─────────────────────────
            # Skip: go back to camera stream
            if self._frame_counter % self.cfg.frame_skip != 0:
                continue   # "Skip" path — discard, loop back

            # Process: enqueue for inference
            # Drop oldest frame if queue is full — freshness over completeness
            if self._queue.full():
                try:
                    self._queue.get_nowait()   # Discard stale
                except queue.Empty:
                    pass
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                pass   # Race-condition safety — skip silently