"""
src/inference/engine.py
────────────────────────
Stages 2-4 — Preprocess 320×320 → ONNX Inference → Post-Processing NMS

Implements the centre section of the inference architecture diagram:
  Preprocess 320×320 → ONNX Inference → Post-Processing NMS

Key performance optimisations for Raspberry Pi 3:
  - Input tensor pre-allocated once at init (no per-frame heap allocation)
  - np.copyto() writes directly into the pre-allocated buffer
  - ONNX Runtime warmup pass on load (first inference triggers JIT — do it early)
  - Single-threaded ORT session (stable on Pi 3 ARM Cortex-A53)
  - ORT_ENABLE_ALL graph optimisation (BN fusion, constant folding)
  - cv2.dnn.NMSBoxes for NMS (faster than pure-Python alternatives)
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("engine")


# ── Detection dataclass ───────────────────────────────────────────────────────

@dataclass
class Detection:
    """One detected object from a single frame."""
    class_id:   int
    class_name: str
    confidence: float
    bbox:       Tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels

    @property
    def bbox_ints(self) -> Tuple[int, int, int, int]:
        return tuple(int(v) for v in self.bbox)

    def to_dict(self) -> Dict:
        return {
            "class_id":   self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox":       list(self.bbox),
        }


# ── Inference engine ──────────────────────────────────────────────────────────

class InferenceEngine:
    """
    Wraps ONNX Runtime with a pre-allocated input buffer and warmup pass.

    Stage 2 — Preprocess 320×320:
      Raw BGR frame → resize → RGB → CHW → float32 normalised → blob

    Stage 3 — ONNX Inference:
      blob → ORT session → raw output tensor

    Stage 4 — Post-Processing NMS:
      raw output → confidence filter → NMS → list[Detection]
    """

    def __init__(self, cfg):
        self.cfg        = cfg
        self.input_size = cfg.input_size
        self._session   = None

        self._load_session()
        self._warmup()

    # ── Stage 2: Preprocess ───────────────────────────────────────────────────

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        BGR frame → normalised float32 tensor [1, 3, H, W].

        Uses np.copyto() into a pre-allocated buffer to avoid allocating
        a new array on every frame — critical for low-RAM devices.
        """
        sz      = self.input_size
        resized = cv2.resize(frame, (sz, sz), interpolation=cv2.INTER_LINEAR)
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # In-place write: HWC float32 → CHW, normalise [0,1]
        np.copyto(
            self._input_blob[0],
            rgb.transpose(2, 0, 1).astype(np.float32) / 255.0,
        )
        return self._input_blob

    # ── Stage 3: ONNX Inference ───────────────────────────────────────────────

    def run_inference(self, blob: np.ndarray) -> np.ndarray:
        """Feed pre-processed blob into ONNX Runtime. Returns raw output."""
        return self._session.run(None, {self._input_name: blob})[0]

    # ── Stage 4: Post-Processing NMS ──────────────────────────────────────────

    def postprocess(
        self,
        raw_output: np.ndarray,
        orig_h: int,
        orig_w: int,
    ) -> List[Detection]:
        """
        YOLOv8 ONNX output shape: [1, 4 + num_classes, num_anchors]

        Steps:
          1. Transpose to [num_anchors, 4 + num_classes]
          2. Extract cx,cy,w,h and per-class scores
          3. Confidence threshold filter
          4. Convert to x1,y1,x2,y2 in original image pixel coordinates
          5. cv2.dnn.NMSBoxes (IOU-based non-maximum suppression)
        """
        preds = raw_output[0].T          # [num_anchors, 4 + num_classes]
        boxes  = preds[:, :4]            # cx, cy, w, h  (normalised to input_size)
        scores = preds[:, 4:]            # per-class scores

        class_ids    = np.argmax(scores, axis=1)
        confidences  = scores[np.arange(len(scores)), class_ids]

        # Confidence threshold filter
        mask = confidences >= self.cfg.conf_threshold
        if not mask.any():
            return []

        boxes        = boxes[mask]
        confidences  = confidences[mask]
        class_ids    = class_ids[mask]

        # Scale cx,cy,w,h from input_size back to original frame dimensions
        sx = orig_w / self.input_size
        sy = orig_h / self.input_size

        x1 = (boxes[:, 0] - boxes[:, 2] / 2) * sx
        y1 = (boxes[:, 1] - boxes[:, 3] / 2) * sy
        x2 = (boxes[:, 0] + boxes[:, 2] / 2) * sx
        y2 = (boxes[:, 1] + boxes[:, 3] / 2) * sy

        # NMS — cv2.dnn.NMSBoxes expects [x, y, w, h] format
        nms_boxes = [
            [float(x1[i]), float(y1[i]),
             float(x2[i] - x1[i]), float(y2[i] - y1[i])]
            for i in range(len(x1))
        ]
        indices = cv2.dnn.NMSBoxes(
            nms_boxes,
            confidences.tolist(),
            self.cfg.conf_threshold,
            self.cfg.iou_threshold,
        )

        detections = []
        if len(indices) > 0:
            flat = indices.flatten() if hasattr(indices, "flatten") else indices
            for i in flat:
                cid  = int(class_ids[i])
                name = self.cfg.class_names.get(cid, f"class_{cid}")
                detections.append(Detection(
                    class_id   = cid,
                    class_name = name,
                    confidence = float(confidences[i]),
                    bbox       = (float(x1[i]), float(y1[i]),
                                  float(x2[i]), float(y2[i])),
                ))
        return detections

    # ── Combined infer() — runs all 3 stages ─────────────────────────────────

    def infer(self, frame: np.ndarray) -> Tuple[List[Detection], float]:
        """
        Convenience method: run stages 2→3→4 on a raw BGR frame.
        Returns (detections, inference_ms).
        """
        h, w = frame.shape[:2]
        blob       = self.preprocess(frame)
        t0         = time.perf_counter()
        raw_output = self.run_inference(blob)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        detections = self.postprocess(raw_output, h, w)
        log.debug("Inference %.1f ms  →  %d detections", elapsed_ms, len(detections))
        return detections, elapsed_ms

    # ── Session loading ───────────────────────────────────────────────────────

    def _load_session(self):
        """Load ONNX model and configure the session for Pi 3 deployment."""
        try:
            import onnxruntime as ort
        except ImportError:
            raise RuntimeError("onnxruntime not installed: pip install onnxruntime")

        import os
        if not os.path.exists(self.cfg.model_path):
            raise FileNotFoundError(
                f"ONNX model not found: {self.cfg.model_path}\n"
                f"Run training first: python scripts/train.py"
            )

        opts = ort.SessionOptions()
        # Pi 3 optimisation: single intra-op thread — more stable than 2+
        opts.intra_op_num_threads        = self.cfg.threads
        opts.inter_op_num_threads        = 1
        opts.graph_optimization_level    = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level          = 3    # Suppress ORT INFO noise

        log.info("Loading ONNX model: %s", self.cfg.model_path)
        self._session    = ort.InferenceSession(
            self.cfg.model_path,
            sess_options = opts,
            providers    = ["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        in_shape         = self._session.get_inputs()[0].shape
        log.info("Model loaded  input=%s  shape=%s", self._input_name, in_shape)

        # Pre-allocate input tensor — reused every frame
        sz = self.input_size
        self._input_blob = np.zeros((1, 3, sz, sz), dtype=np.float32)

    def _warmup(self):
        """
        One dummy inference pass — triggers ONNX Runtime kernel JIT compilation.
        Without this, the FIRST real inference call takes 200-800ms extra.
        """
        log.info("Warming up ONNX model...")
        t0 = time.perf_counter()
        self._session.run(None, {self._input_name: self._input_blob})
        ms = (time.perf_counter() - t0) * 1000
        log.info("Warmup complete: %.1f ms", ms)