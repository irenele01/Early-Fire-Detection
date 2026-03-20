"""
Model Training Pipeline — DC-EFDS-Lite

Implements the full training architecture:

  [1] Processed Dataset  → Pre-flight validation
  [2] YOLOv8n Pretrained → Load weights + configure hyperparameters
  [3] Training Loop      → Forward / backward / optimise per epoch
  [4] Validation         → Per-epoch mAP@50 on val split
  [5] Converged?         → Convergence + early stopping check
  [6] Export Best Model  → Copy best.pt to models/weights/
  [7] ONNX Conversion    → Export + simplify + verify
  [8] Pi 3 Validation    → Latency/RAM estimation for deployment

Activation:
    source venv/bin/activate
    python scripts/train.py

Advanced usage:
    python scripts/train.py --config config/training.yaml
    python scripts/train.py --epochs 100 --device mps
    python scripts/train.py --resume                      # continue from last.pt
    python scripts/train.py --eval-only                   # evaluate existing model
    python scripts/train.py --export-only                 # re-export to ONNX only
    python scripts/train.py --no-export                   # skip ONNX export
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
# Allow running from project root or scripts/ directory
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from training.config_loader import load_config, write_dataset_yaml
from training.preflight     import run_preflight
from training.trainer       import Trainer
from training.evaluator     import evaluate_on_test_set, save_evaluation_report
from training.exporter      import Exporter


# ── Banner ────────────────────────────────────────────────────────────────────

def _banner():
    print()
    print("╔════════════════════════════════════════════════════════╗")
    print("║   DC-EFDS-Lite  ·  Model Training Pipeline            ║")
    print("╚════════════════════════════════════════════════════════╝")
    print()
    print("  Architecture:")
    print("  Processed Dataset → YOLOv8n Pretrained → Training Loop")
    print("  → Validation → Converged? → Export Best Model")
    print("  → ONNX Conversion → Pi 3 Validation")
    print()


# ── Stage helpers ─────────────────────────────────────────────────────────────

def _copy_best_weights(src: Path, cfg) -> Path:
    """Copy best.pt from the Ultralytics run directory to models/weights/."""
    dest_dir = Path(cfg.weights_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "best.pt"
    shutil.copy2(src, dest)
    print(f"\n  Best weights saved: {dest}")
    return dest


def _print_config_summary(cfg):
    print("  ┌──────────────────────────────────────────────────────┐")
    print("  │  Training Configuration                              │")
    print("  ├──────────────────────────────────────────────────────┤")
    print(f"  │  Model       :  {cfg.model_name:<35}  │")
    print(f"  │  Input size  :  {cfg.input_size}×{cfg.input_size:<31}  │")
    print(f"  │  Device      :  {cfg.device:<35}  │")
    print(f"  │  Epochs      :  {cfg.epochs:<35}  │")
    print(f"  │  Batch size  :  {cfg.batch_size:<35}  │")
    print(f"  │  Optimizer   :  {cfg.optimizer}  lr0={cfg.lr0}  momentum={cfg.momentum:<8}  │")
    print(f"  │  Classes     :  {str(list(cfg.class_names.values())):<35}  │")
    print(f"  │  Data path   :  {cfg.data_path:<35}  │")
    print("  └──────────────────────────────────────────────────────┘")
    print()


def _print_final_summary(train_result, eval_result, export_result, elapsed_total: float):
    print()
    print("╔════════════════════════════════════════════════════════╗")
    print("║   Training Pipeline Complete                          ║")
    print("╚════════════════════════════════════════════════════════╝")
    print()

    # Training metrics
    if train_result and train_result.success:
        print("  Training metrics (validation set):")
        print(train_result.summary())
        print()

    # Test set evaluation
    if eval_result and "error" not in eval_result:
        map50     = eval_result.get("map50", 0)
        precision = eval_result.get("precision", 0)
        recall    = eval_result.get("recall", 0)
        print(f"  Test set  — mAP@50={map50:.4f}  P={precision:.4f}  R={recall:.4f}")
        print()

    # ONNX + Pi 3
    if export_result and export_result.success:
        print(f"  ONNX model  : {export_result.onnx_path}  ({export_result.model_size_mb:.1f} MB)")
        print(f"  ORT latency : {export_result.ort_inference_ms:.1f} ms  (this machine)")
        print(f"  Pi 3 est.   : {export_result.pi3_estimated_ms:.0f} ms  (target <500 ms)  "
              f"{'✅' if export_result.pi3_passes else '⚠️ '}")
        print()

    print(f"  Total elapsed : {elapsed_total/60:.1f} minutes")
    print()

    if export_result and export_result.ready_for_pi3:
        print("  ✅  Model ready for Pi 3 deployment.")
        print()
        print("  Deploy to Pi 3:")
        print("    scp models/best.onnx pi@<pi-ip>:~/dc-efds-lite/models/")
        print("    ssh pi@<pi-ip>")
        print("    python scripts/validate_install.py --model models/best.onnx")
        print("    sudo systemctl start dc-efds.service")
    else:
        print("  ⚠️   Model training complete but review warnings above before deploying.")

    print()
    print("  ─────────────────────────────────────────────────────────")
    print("  Quick reference:")
    print("    View training plots  : open models/dc_efds_run/")
    print("    Re-run evaluation    : python scripts/train.py --eval-only")
    print("    Re-export ONNX       : python scripts/train.py --export-only")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DC-EFDS-Lite — Model Training Pipeline"
    )
    parser.add_argument("--config",      default="config/training.yaml",
                        help="Path to training.yaml (default: config/training.yaml)")
    parser.add_argument("--epochs",      type=int,   default=None,
                        help="Override epochs in config")
    parser.add_argument("--batch",       type=int,   default=None,
                        help="Override batch_size in config")
    parser.add_argument("--device",      default=None,
                        choices=["mps", "cuda", "cpu", "0", "1"],
                        help="Override device in config")
    parser.add_argument("--resume",      action="store_true",
                        help="Resume from last.pt checkpoint")
    parser.add_argument("--no-preflight", action="store_true",
                        help="Skip pre-flight checks (not recommended)")
    parser.add_argument("--no-export",   action="store_true",
                        help="Skip ONNX export stage")
    parser.add_argument("--eval-only",   action="store_true",
                        help="Only run evaluation on existing models/weights/best.pt")
    parser.add_argument("--export-only", action="store_true",
                        help="Only run ONNX export on existing models/weights/best.pt")
    args = parser.parse_args()

    _banner()
    t_start = time.time()

    # ── Load config ───────────────────────────────────────────────────────────
    cfg = load_config(args.config)

    # CLI overrides
    if args.epochs:  cfg.epochs     = args.epochs
    if args.batch:   cfg.batch_size = args.batch
    if args.device:
        from training.config_loader import _resolve_device
        cfg.device = _resolve_device(args.device)

    _print_config_summary(cfg)

    # ── Write dataset.yaml ────────────────────────────────────────────────────
    ds_yaml = write_dataset_yaml(cfg)
    print(f"  dataset.yaml written: {ds_yaml}")

    # ── Eval-only mode ────────────────────────────────────────────────────────
    if args.eval_only:
        best_pt = Path(cfg.weights_dir) / "best.pt"
        if not best_pt.exists():
            print(f"  ❌  best.pt not found: {best_pt}")
            print("       Run training first: python scripts/train.py")
            sys.exit(1)
        print(f"\n  Eval-only mode — evaluating: {best_pt}")
        eval_result = evaluate_on_test_set(best_pt, cfg)
        save_evaluation_report(eval_result)
        return

    # ── Export-only mode ──────────────────────────────────────────────────────
    if args.export_only:
        best_pt = Path(cfg.weights_dir) / "best.pt"
        if not best_pt.exists():
            print(f"  ❌  best.pt not found: {best_pt}")
            sys.exit(1)
        print(f"\n  Export-only mode — exporting: {best_pt}")
        exporter = Exporter(cfg)
        result   = exporter.run(best_pt)
        if not result.success:
            print(f"  ❌  Export failed: {result.error_message}")
            sys.exit(1)
        return

    # ═════════════════════════════════════════════════════════════════════════
    #  STAGE 1 — Pre-flight
    # ═════════════════════════════════════════════════════════════════════════
    if not args.no_preflight:
        ok = run_preflight(cfg)
        if not ok:
            print()
            print("  ❌  Pre-flight failed. Fix the issues above and retry.")
            sys.exit(1)
    else:
        print("  ⚠️   Pre-flight skipped (--no-preflight).")

    # ═════════════════════════════════════════════════════════════════════════
    #  STAGES 2–5 — Training loop + convergence
    # ═════════════════════════════════════════════════════════════════════════
    trainer      = Trainer(cfg)
    train_result = trainer.run(resume=args.resume)

    if not train_result.success:
        print(f"\n  ❌  Training failed: {train_result.error_message}")
        sys.exit(1)

    # Print training summary
    print()
    print("  Training complete:")
    print(train_result.summary())

    # ═════════════════════════════════════════════════════════════════════════
    #  STAGE 6 — Export best.pt to models/weights/
    # ═════════════════════════════════════════════════════════════════════════
    print()
    print("  ━━━  Stage 6: Save Best Model  ━━━━━━━━━━━━━━━━━━━━━")
    best_pt = _copy_best_weights(train_result.best_pt_path, cfg)

    # ═════════════════════════════════════════════════════════════════════════
    #  STAGE 7 — Test set evaluation
    # ═════════════════════════════════════════════════════════════════════════
    eval_result = evaluate_on_test_set(best_pt, cfg, split="test")
    save_evaluation_report(eval_result)

    # ═════════════════════════════════════════════════════════════════════════
    #  STAGE 8+9 — ONNX export + Pi 3 validation
    # ═════════════════════════════════════════════════════════════════════════
    export_result = None
    if not args.no_export:
        exporter      = Exporter(cfg)
        export_result = exporter.run(best_pt)

        if not export_result.success:
            print(f"\n  ⚠️   ONNX export failed: {export_result.error_message}")
            print("       Training weights are still saved at models/weights/best.pt")
    else:
        print("\n  ⚠️   ONNX export skipped (--no-export).")

    # ═════════════════════════════════════════════════════════════════════════
    #  Final summary
    # ═════════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t_start
    _print_final_summary(train_result, eval_result, export_result, elapsed)


if __name__ == "__main__":
    main()