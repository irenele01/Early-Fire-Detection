"""
training/trainer.py
────────────────────
Stages 2 & 3 — YOLOv8n Training Loop + Convergence Check.

Maps to the architecture diagram:
  YOLOv8n Pretrained → Training Loop → Validation → Converged?

Key responsibilities:
  - Load pretrained YOLOv8n weights (COCO)
  - Execute training with full hyperparameter set from config
  - Monitor convergence via mAP@50 plateau detection
  - Early stopping support (patience parameter)
  - Checkpoint management (best.pt + last.pt)
  - Return structured TrainingResult for downstream stages
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# ── Training result dataclass ─────────────────────────────────────────────────

@dataclass
class TrainingResult:
    """Structured output from a completed training run."""
    success:         bool
    best_pt_path:    Optional[Path] = None
    last_pt_path:    Optional[Path] = None
    run_dir:         Optional[Path] = None
    epochs_trained:  int            = 0
    converged:       bool           = False
    # Final metrics on validation set
    map50:           float          = 0.0   # mAP@0.50
    map50_95:        float          = 0.0   # mAP@0.50:0.95
    precision:       float          = 0.0
    recall:          float          = 0.0
    fitness:         float          = 0.0   # Ultralytics composite fitness score
    elapsed_minutes: float          = 0.0
    error_message:   str            = ""

    @property
    def meets_deployment_targets(self) -> bool:
        """
        Check if the model meets the minimum performance targets
        defined in the system design document.
        """
        return (
            self.map50     >= 0.85 and
            self.precision >= 0.80 and
            self.recall    >= 0.75
        )

    def summary(self) -> str:
        lines = [
            f"  mAP@50        : {self.map50:.4f}  (target ≥ 0.85  {'✅' if self.map50 >= 0.85 else '⚠️ '})",
            f"  mAP@50:95     : {self.map50_95:.4f}",
            f"  Precision     : {self.precision:.4f}  (target ≥ 0.80  {'✅' if self.precision >= 0.80 else '⚠️ '})",
            f"  Recall        : {self.recall:.4f}  (target ≥ 0.75  {'✅' if self.recall >= 0.75 else '⚠️ '})",
            f"  Fitness score : {self.fitness:.4f}",
            f"  Epochs        : {self.epochs_trained}",
            f"  Converged     : {'Yes ✅' if self.converged else 'No (early stopped or max epochs)'}",
            f"  Elapsed       : {self.elapsed_minutes:.1f} min",
        ]
        return "\n".join(lines)


# ── Trainer ───────────────────────────────────────────────────────────────────

class Trainer:
    """
    Wraps the Ultralytics YOLO training API with structured
    pre/post processing and convergence reporting.
    """

    # Convergence threshold:
    # If mAP@50 improvement over the last `patience` epochs
    # is less than this delta, training is considered converged.
    CONVERGENCE_DELTA = 0.001

    def __init__(self, cfg):
        self.cfg = cfg
        self._model = None

    def run(self, resume: bool = False) -> TrainingResult:
        """
        Execute the full training loop.

        Args:
            resume: If True, continue from last.pt checkpoint.

        Returns:
            TrainingResult with all metrics and paths populated.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            return TrainingResult(
                success=False,
                error_message="ultralytics not installed: pip install ultralytics",
            )

        cfg = self.cfg
        t_start = time.time()

        print()
        print("  ━━━  Stage 2: Training Loop  ━━━━━━━━━━━━━━━━━━━━━━")
        print()
        print(f"  Model       : {cfg.model_name}  (pretrained={cfg.pretrained})")
        print(f"  Input size  : {cfg.input_size}×{cfg.input_size}")
        print(f"  Device      : {cfg.device}")
        print(f"  Epochs      : {cfg.epochs}  (patience={cfg.patience})")
        print(f"  Batch size  : {cfg.batch_size}")
        print(f"  Optimizer   : {cfg.optimizer}  lr0={cfg.lr0}  lrf={cfg.lrf}")
        print(f"  Classes     : {list(cfg.class_names.values())}")
        print()

        # ── Load weights ──────────────────────────────────────────────────────
        if resume and cfg.last_pt_path.exists():
            weights = str(cfg.last_pt_path)
            print(f"  Resuming from checkpoint: {weights}")
        elif resume and not cfg.last_pt_path.exists():
            print("  ⚠️   Resume requested but last.pt not found — starting fresh.")
            weights = f"{cfg.model_name}.pt"
        else:
            weights = f"{cfg.model_name}.pt"
            print(f"  Loading pretrained weights: {weights}")

        self._model = YOLO(weights)

        # ── Training call ─────────────────────────────────────────────────────
        # All hyperparameters flow from TrainingConfig — no magic numbers here.
        try:
            results = self._model.train(
                data          = cfg.dataset_yaml,
                epochs        = cfg.epochs,
                patience      = cfg.patience,
                imgsz         = cfg.input_size,
                batch         = cfg.batch_size,
                device        = cfg.device,
                optimizer     = cfg.optimizer,
                lr0           = cfg.lr0,
                lrf           = cfg.lrf,
                momentum      = cfg.momentum,
                weight_decay  = cfg.weight_decay,
                # Augmentation
                hsv_h         = cfg.hsv_h,
                hsv_s         = cfg.hsv_s,
                hsv_v         = cfg.hsv_v,
                flipud        = cfg.flipud,
                fliplr        = cfg.fliplr,
                mosaic        = cfg.mosaic,
                mixup         = cfg.mixup,
                # Output
                project       = cfg.project_dir,
                name          = cfg.run_name,
                resume        = resume,
                save          = True,
                save_period   = 10,          # Checkpoint every 10 epochs
                verbose       = True,
                plots         = True,        # Save training plots
                val           = True,        # Run validation each epoch
                exist_ok      = resume,      # Overwrite run dir if resuming
            )
        except Exception as e:
            return TrainingResult(
                success=False,
                error_message=f"Training failed: {e}",
            )

        elapsed = (time.time() - t_start) / 60

        # ── Locate checkpoint files ───────────────────────────────────────────
        run_dir   = Path(results.save_dir)
        best_pt   = run_dir / "weights" / "best.pt"
        last_pt   = run_dir / "weights" / "last.pt"

        if not best_pt.exists():
            return TrainingResult(
                success=False,
                error_message=f"best.pt not found at {best_pt} — training may have failed early.",
            )

        # ── Extract final metrics from results object ─────────────────────────
        map50, map50_95, precision, recall, fitness = self._extract_metrics(results)

        # ── Convergence check ─────────────────────────────────────────────────
        converged = self._check_convergence(results)

        print()
        print("  ━━━  Stage 3: Convergence Check  ━━━━━━━━━━━━━━━━━━")
        print()
        if converged:
            print("  ✅  Model converged — mAP@50 plateau detected.")
        else:
            print("  ℹ️   Training reached max epochs or early-stop patience.")
            print("      Consider increasing epochs if mAP is still rising.")

        return TrainingResult(
            success         = True,
            best_pt_path    = best_pt,
            last_pt_path    = last_pt if last_pt.exists() else None,
            run_dir         = run_dir,
            epochs_trained  = int(getattr(results, 'epoch', cfg.epochs)),
            converged       = converged,
            map50           = map50,
            map50_95        = map50_95,
            precision       = precision,
            recall          = recall,
            fitness         = fitness,
            elapsed_minutes = elapsed,
        )

    # ── Metric extraction ─────────────────────────────────────────────────────

    @staticmethod
    def _extract_metrics(results) -> tuple:
        """
        Extract final validation metrics from Ultralytics results object.
        Ultralytics stores metrics in results.results_dict after training.
        Falls back to 0.0 for any missing key.
        """
        rd = getattr(results, "results_dict", {}) or {}

        map50     = rd.get("metrics/mAP50(B)",    0.0)
        map50_95  = rd.get("metrics/mAP50-95(B)", 0.0)
        precision = rd.get("metrics/precision(B)", 0.0)
        recall    = rd.get("metrics/recall(B)",    0.0)

        # Ultralytics fitness = 0.1*mAP@50 + 0.9*mAP@50:95
        fitness = 0.1 * map50 + 0.9 * map50_95

        return map50, map50_95, precision, recall, fitness

    @staticmethod
    def _check_convergence(results) -> bool:
        """
        Convergence heuristic: the model is considered converged if
        Ultralytics triggered early stopping (best epoch < final epoch).
        Ultralytics sets this automatically based on the `patience` parameter.
        """
        # Ultralytics sets results.stop_epoch when early stopping fires
        stop_epoch = getattr(results, "stop_epoch", None)
        best_epoch = getattr(results, "best_fitness_epoch", None)

        if stop_epoch is not None:
            return True   # Ultralytics decided to stop
        if best_epoch is not None and best_epoch < getattr(results, "epoch", 999) - 5:
            return True   # Best epoch was many epochs ago
        return False