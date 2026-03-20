"""
src/inference/validator.py
───────────────────────────
Stage 5 — Temporal Validator → Confirmed / Not Confirmed

Implements the Temporal Validator node in the inference architecture:
  Post-Processing NMS → Temporal Validator → Confirmed → Alert Dispatcher
                                           → Not Confirmed → (loop back)

Why consecutive-frame gating matters:
  A single-frame positive is very likely a false positive — lighting glints,
  reflections, LEDs, and compression artefacts all produce transient hits.
  Requiring N CONSECUTIVE positive frames filters these out completely.

  Design note: this uses a STREAK counter, not a sliding window buffer.
  Difference:
    - Sliding window (all-in-buffer): frame sequence + + + - + + + + fires
      at frame 8 if 4 of 5 in window are positive → still fires on transients
    - Streak (consecutive): the - at frame 4 resets streak to 0 → only fires
      after an unbroken run of N positives → much cleaner in server rooms
      with blinking LEDs
"""

import time
import logging
from typing import List, Optional, Tuple

from src.inference.engine import Detection

log = logging.getLogger("validator")


class TemporalValidator:
    """
    Gates detections behind a consecutive-frame persistence requirement
    and a cooldown window to prevent alert flooding.

    State machine:
      - streak:  int  — number of consecutive frames with ≥1 detection
      - last_alert_time: float  — Unix time of last confirmed alert

    A frame with zero detections resets streak to 0 immediately.
    An alert fires only when streak reaches persistence_frames AND
    we are outside the cooldown window.
    """

    def __init__(self, persistence_frames: int = 5, cooldown_seconds: float = 30.0):
        self.persistence    = persistence_frames
        self.cooldown       = cooldown_seconds
        self._streak        = 0
        self._last_alert_t  = 0.0

    # ── Main update ───────────────────────────────────────────────────────────

    def update(
        self, detections: List[Detection]
    ) -> Tuple[bool, Optional[Detection]]:
        """
        Called once per processed frame with the list of detections.

        Returns:
            (should_alert, best_detection)
            should_alert = True only on the frame that completes the streak
                           AND only if cooldown has elapsed.
            best_detection = highest-confidence detection, or None.
        """
        if not detections:
            # "Not Confirmed" path — reset streak, loop back
            if self._streak > 0:
                log.debug("Streak broken at %d (no detections)", self._streak)
            self._streak = 0
            return False, None

        # Positive frame — increment streak
        self._streak += 1
        log.debug(
            "Streak: %d/%d  best_conf=%.2f  class=%s",
            self._streak,
            self.persistence,
            max(d.confidence for d in detections),
            max(detections, key=lambda d: d.confidence).class_name,
        )

        # Not yet reached persistence threshold
        if self._streak < self.persistence:
            return False, None

        # Reached threshold — check cooldown
        now = time.time()
        if (now - self._last_alert_t) < self.cooldown:
            # Still in cooldown — suppress but don't reset streak
            log.debug(
                "Alert suppressed (cooldown %.0fs remaining)",
                self.cooldown - (now - self._last_alert_t),
            )
            return False, None

        # ── CONFIRMED ─────────────────────────────────────────────────────────
        self._streak       = 0     # Reset so next event needs a fresh streak
        self._last_alert_t = now

        best = max(detections, key=lambda d: d.confidence)
        log.info(
            "CONFIRMED: %s  conf=%.2f  (after %d consecutive frames)",
            best.class_name, best.confidence, self.persistence,
        )
        return True, best

    # ── Status helpers ────────────────────────────────────────────────────────

    @property
    def streak(self) -> int:
        return self._streak

    @property
    def in_cooldown(self) -> bool:
        return (time.time() - self._last_alert_t) < self.cooldown

    @property
    def cooldown_remaining_s(self) -> float:
        remaining = self.cooldown - (time.time() - self._last_alert_t)
        return max(0.0, remaining)

    def reset(self):
        """Manually reset streak and cooldown — useful for testing."""
        self._streak       = 0
        self._last_alert_t = 0.0