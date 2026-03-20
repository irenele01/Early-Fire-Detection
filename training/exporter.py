"""
training/exporter.py
─────────────────────
Stages 5 & 6 — ONNX Conversion + Pi 3 Validation.

Maps to the architecture diagram:
  Export Best Model → ONNX Conversion → Pi 3 Validation

Stage 5 — ONNX Conversion:
  - Export best.pt → best.onnx using Ultralytics export
  - Verify the ONNX graph (onnx.checker)
  - Verify with ONNX Runtime inference (dummy input)
  - Report model size and operator count

Stage 6 — Pi 3 Validation:
  - Simulate Pi 3 constraints: single thread, no SIMD tricks
  - Benchmark inference latency (target: <500 ms)
  - Benchmark RAM usage (target: <400 MB)
  - Report go/no-go for deployment
"""

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ── Export result ─────────────────────────────────────────────────────────────

@dataclass
class ExportResult:
    success:            bool
    onnx_path:          Optional[Path] = None
    model_size_mb:      float          = 0.0
    operator_count:     int            = 0
    # ONNX Runtime verification
    ort_verified:       bool           = False
    ort_warmup_ms:      float          = 0.0
    ort_inference_ms:   float          = 0.0    # Mean of 10 runs
    # Pi 3 simulation
    pi3_estimated_ms:   float          = 0.0
    pi3_ram_mb:         float          = 0.0
    pi3_passes:         bool           = False
    error_message:      str            = ""

    # Pi 3 deployment targets
    PI3_LATENCY_MS_MAX = 500.0
    PI3_RAM_MB_MAX     = 400.0

    @property
    def ready_for_pi3(self) -> bool:
        return (
            self.success and
            self.ort_verified and
            self.pi3_passes
        )


# ── Exporter ──────────────────────────────────────────────────────────────────

class Exporter:
    """
    Handles ONNX export, graph verification, and Pi 3 simulation.
    """

    # Pi 3 ARM Cortex-A53 is roughly 15-20× slower than M3 on ONNX inference.
    # This multiplier is used to estimate Pi 3 latency from the local benchmark.
    PI3_SLOWDOWN_FACTOR = 18.0

    def __init__(self, cfg):
        self.cfg = cfg

    def run(self, best_pt: Path) -> ExportResult:
        print()
        print("  ━━━  Stage 5: ONNX Conversion  ━━━━━━━━━━━━━━━━━━━")

        result = ExportResult(success=False)

        # ── 1. Export with Ultralytics ────────────────────────────────────────
        onnx_path = self._export_ultralytics(best_pt, result)
        if onnx_path is None:
            return result

        # Copy to canonical location
        target = Path(self.cfg.onnx_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(onnx_path, target)
        result.onnx_path    = target
        result.model_size_mb = target.stat().st_size / (1024 * 1024)

        print(f"\n  ONNX model : {target}")
        print(f"  Size       : {result.model_size_mb:.2f} MB")

        # ── 2. ONNX graph verification ────────────────────────────────────────
        self._verify_onnx_graph(target, result)

        # ── 3. ONNX Runtime inference check ──────────────────────────────────
        self._verify_ort_inference(target, result)

        # ── 4. Pi 3 validation (Stage 6) ─────────────────────────────────────
        print()
        print("  ━━━  Stage 6: Pi 3 Validation  ━━━━━━━━━━━━━━━━━━━")
        self._simulate_pi3(target, result)

        result.success = True
        return result

    # ── Stage 5: ONNX export ─────────────────────────────────────────────────

    def _export_ultralytics(self, best_pt: Path, result: ExportResult) -> Optional[Path]:
        """
        Use Ultralytics built-in export.
        Settings tuned for ONNX Runtime on ARM Cortex-A53 (Pi 3).
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            result.error_message = "ultralytics not installed"
            return None

        print(f"\n  Exporting {best_pt.name} → ONNX...")
        print(f"  Input size : {self.cfg.input_size}×{self.cfg.input_size}")
        print(f"  Opset      : {self.cfg.opset}  (max ARM ONNX Runtime support)")
        print(f"  Simplify   : {self.cfg.simplify}  (BN fusion + identity removal)")

        model = YOLO(str(best_pt))
        try:
            exported = model.export(
                format   = "onnx",
                imgsz    = self.cfg.input_size,
                simplify = self.cfg.simplify,
                opset    = self.cfg.opset,
                dynamic  = False,    # Static input shape — faster on Pi 3
                half     = False,    # FP16 not reliable on ONNX Runtime CPU
                device   = "cpu",    # Export always on CPU for portability
            )
        except Exception as e:
            result.error_message = f"Ultralytics export failed: {e}"
            print(f"  ❌  {result.error_message}")
            return None

        # Ultralytics returns the path as a string
        onnx_path = Path(str(exported))
        if not onnx_path.exists():
            # Try conventional location: same dir as best.pt with .onnx suffix
            onnx_path = best_pt.with_suffix(".onnx")

        if not onnx_path.exists():
            result.error_message = "ONNX file not found after export"
            print(f"  ❌  {result.error_message}")
            return None

        print(f"  ✅  Exported to: {onnx_path}")
        return onnx_path

    # ── Stage 5: Graph verification ───────────────────────────────────────────

    def _verify_onnx_graph(self, onnx_path: Path, result: ExportResult):
        """Validate the ONNX graph structure using onnx.checker."""
        print("\n  Verifying ONNX graph...")
        try:
            import onnx
            model_proto = onnx.load(str(onnx_path))
            onnx.checker.check_model(model_proto)

            # Count operators (proxy for model complexity)
            ops = {node.op_type for node in model_proto.graph.node}
            result.operator_count = len(model_proto.graph.node)

            print(f"  ✅  Graph valid")
            print(f"      Nodes     : {result.operator_count}")
            print(f"      Op types  : {len(ops)}  "
                  f"({', '.join(sorted(ops)[:8])}{'...' if len(ops) > 8 else ''})")

        except ImportError:
            print("  ⚠️   onnx package not installed (pip install onnx) — skipping graph check")
        except Exception as e:
            print(f"  ⚠️   ONNX graph check failed: {e}")

    # ── Stage 5: ORT inference verification ──────────────────────────────────

    def _verify_ort_inference(self, onnx_path: Path, result: ExportResult):
        """
        Run ONNX Runtime inference on a dummy input.
        Measures warmup time (first inference — JIT) and steady-state mean
        across 10 runs.
        """
        print("\n  Verifying ONNX Runtime inference...")
        try:
            import onnxruntime as ort
            import numpy as np

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1   # Pi 3 sim: single-threaded
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.log_severity_level = 3    # Suppress ORT INFO logs

            session   = ort.InferenceSession(str(onnx_path), opts,
                                             providers=["CPUExecutionProvider"])
            inp_name  = session.get_inputs()[0].name
            inp_shape = session.get_inputs()[0].shape
            sz        = self.cfg.input_size
            dummy     = np.zeros((1, 3, sz, sz), dtype=np.float32)

            # Warmup pass — measures JIT compilation overhead
            t_warm = time.perf_counter()
            session.run(None, {inp_name: dummy})
            result.ort_warmup_ms = (time.perf_counter() - t_warm) * 1000

            # 10 steady-state runs
            times = []
            for _ in range(10):
                t0 = time.perf_counter()
                session.run(None, {inp_name: dummy})
                times.append((time.perf_counter() - t0) * 1000)

            result.ort_inference_ms = sum(times) / len(times)
            result.ort_verified     = True

            print(f"  ✅  ORT inference verified")
            print(f"      Input shape    : {inp_shape}")
            print(f"      Warmup         : {result.ort_warmup_ms:.1f} ms")
            print(f"      Mean (10 runs) : {result.ort_inference_ms:.1f} ms  "
                  f"(on {_get_cpu_info()})")

        except ImportError:
            print("  ⚠️   onnxruntime not installed (pip install onnxruntime) — skipping ORT check")
        except Exception as e:
            print(f"  ⚠️   ORT inference check failed: {e}")

    # ── Stage 6: Pi 3 simulation ──────────────────────────────────────────────

    def _simulate_pi3(self, onnx_path: Path, result: ExportResult):
        """
        Estimate Pi 3 performance by applying a calibrated slowdown factor
        to the local ONNX Runtime benchmark.

        The Pi 3 BCM2837 ARM Cortex-A53 at 1.4 GHz is approximately
        15–20× slower than a MacBook M3 for float32 inference on YOLOv8n.
        We use the conservative 18× factor as our estimate.

        Also estimates RAM usage from model size + runtime overhead.
        """
        print()

        if result.ort_inference_ms > 0:
            # Scale from local measurement
            result.pi3_estimated_ms = result.ort_inference_ms * self.PI3_SLOWDOWN_FACTOR
        else:
            # Fallback: use model size as a heuristic (~30 ms per MB for YOLOv8n)
            result.pi3_estimated_ms = result.model_size_mb * 30.0

        # RAM estimate: model params + ORT session overhead + frame buffer
        #   model loaded  : ~model_size_mb * 4  (float32 = 4× compressed size)
        #   ORT overhead  : ~80 MB fixed
        #   2 frame buffers (640×480 BGR): ~2 × 0.9 MB ≈ 2 MB
        result.pi3_ram_mb = (result.model_size_mb * 4) + 80.0 + 2.0

        latency_ok = result.pi3_estimated_ms <= ExportResult.PI3_LATENCY_MS_MAX
        ram_ok     = result.pi3_ram_mb     <= ExportResult.PI3_RAM_MB_MAX
        result.pi3_passes = latency_ok and ram_ok

        lat_icon = "✅" if latency_ok else "❌"
        ram_icon = "✅" if ram_ok     else "❌"

        print("  ┌──────────────────────────────────────────────────────┐")
        print("  │  Pi 3 Deployment Estimate                            │")
        print("  ├──────────────────────────────────────────────────────┤")
        print(f"  │  Local inference   :  {result.ort_inference_ms:>7.1f} ms  (this machine)     │")
        print(f"  │  Slowdown factor   :  {self.PI3_SLOWDOWN_FACTOR:>7.1f}×   (ARM A53 vs M3)     │")
        print(f"  │  {lat_icon}  Pi 3 latency    :  {result.pi3_estimated_ms:>7.1f} ms  (target <500 ms)  │")
        print(f"  │  {ram_icon}  Pi 3 RAM est.   :  {result.pi3_ram_mb:>7.1f} MB  (target <400 MB)  │")
        print("  ├──────────────────────────────────────────────────────┤")

        if result.pi3_passes:
            print("  │  ✅  Model meets Pi 3 deployment requirements.       │")
        else:
            if not latency_ok:
                print(f"  │  ❌  Latency too high — try reducing imgsz to 256.  │")
            if not ram_ok:
                print(f"  │  ❌  RAM too high — use Pi 4 (2 GB) instead.        │")

        print("  └──────────────────────────────────────────────────────┘")
        print()
        print("  NOTE: This is a CPU-simulation estimate.")
        print("  Run on actual Pi 3 with:")
        print("    python scripts/validate_install.py --model models/best.onnx")


# ── Helper ────────────────────────────────────────────────────────────────────

def _get_cpu_info() -> str:
    try:
        import platform
        return platform.processor() or platform.machine()
    except Exception:
        return "unknown CPU"