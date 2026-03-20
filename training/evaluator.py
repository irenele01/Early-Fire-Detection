"""
training/evaluator.py
──────────────────────
Stage 4 — Post-training evaluation on the held-out test set.

This runs after "Converged?" → "Export Best Model" in the diagram.
It gives the final honest performance numbers on data the model
never saw during training or validation.

Evaluates:
  - mAP@50 and mAP@50:95 (primary metrics)
  - Per-class precision, recall, F1
  - Confusion matrix (saved as PNG)
  - Speed (inference time on the training machine)

All results are written to models/evaluation_report.json and printed
in a structured table.
"""

import json
import time
from pathlib import Path
from typing import Dict, Optional

# ── Evaluation result ─────────────────────────────────────────────────────────

DEPLOYMENT_TARGETS = {
    "map50":     0.85,
    "precision": 0.80,
    "recall":    0.75,
}


def evaluate_on_test_set(
    model_path: Path,
    cfg,
    split: str = "test",
) -> Dict:
    """
    Run model.val() on the test split and return a structured metrics dict.

    Args:
        model_path : Path to best.pt
        cfg        : TrainingConfig
        split      : 'test' (default) or 'val'

    Returns:
        dict with all metrics, or error key on failure.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        return {"error": "ultralytics not installed"}

    if not model_path.exists():
        return {"error": f"Model not found: {model_path}"}

    print()
    print("  ━━━  Stage 4: Test Set Evaluation  ━━━━━━━━━━━━━━━━━")
    print(f"  Model  : {model_path}")
    print(f"  Split  : {split}")
    print(f"  Device : {cfg.device}")
    print()

    model = YOLO(str(model_path))

    t0 = time.perf_counter()
    try:
        metrics = model.val(
            data    = cfg.dataset_yaml,
            imgsz   = cfg.input_size,
            batch   = cfg.batch_size,
            device  = cfg.device,
            split   = split,
            plots   = True,
            save_json = False,
            verbose = False,
        )
    except Exception as e:
        return {"error": f"Evaluation failed: {e}"}

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # ── Extract metrics ───────────────────────────────────────────────────────
    rd = getattr(metrics, "results_dict", {}) or {}

    map50     = rd.get("metrics/mAP50(B)",    0.0)
    map50_95  = rd.get("metrics/mAP50-95(B)", 0.0)
    precision = rd.get("metrics/precision(B)", 0.0)
    recall    = rd.get("metrics/recall(B)",    0.0)
    fitness   = 0.1 * map50 + 0.9 * map50_95

    # Per-class breakdown (Ultralytics exposes via metrics.box)
    per_class = {}
    try:
        box = metrics.box
        if hasattr(box, "maps") and hasattr(box, "names"):
            for i, name in box.names.items():
                per_class[name] = {
                    "map50":     float(box.maps[i]) if i < len(box.maps) else 0.0,
                    "precision": float(box.p[i])    if hasattr(box, "p") and i < len(box.p) else 0.0,
                    "recall":    float(box.r[i])    if hasattr(box, "r") and i < len(box.r) else 0.0,
                    "f1":        float(box.f1[i])   if hasattr(box, "f1") and i < len(box.f1) else 0.0,
                }
    except Exception:
        pass

    result = {
        "split":           split,
        "map50":           round(map50, 4),
        "map50_95":        round(map50_95, 4),
        "precision":       round(precision, 4),
        "recall":          round(recall, 4),
        "fitness":         round(fitness, 4),
        "elapsed_ms":      round(elapsed_ms, 1),
        "per_class":       per_class,
        "meets_targets":   {
            k: round({"map50": map50, "precision": precision, "recall": recall}[k], 4) >= v
            for k, v in DEPLOYMENT_TARGETS.items()
        },
    }

    # ── Print report ──────────────────────────────────────────────────────────
    _print_evaluation_report(result)

    return result


def _print_evaluation_report(result: Dict):
    map50     = result["map50"]
    map50_95  = result["map50_95"]
    precision = result["precision"]
    recall    = result["recall"]
    fitness   = result["fitness"]

    print("  ┌──────────────────────────────────────────────────────┐")
    print("  │  Test Set Evaluation Results                         │")
    print("  ├──────────────────────────────────────────────────────┤")
    print(f"  │  mAP@50        :  {map50:.4f}   {'✅' if map50 >= 0.85 else '⚠️ '}  (target ≥ 0.85)        │")
    print(f"  │  mAP@50:95     :  {map50_95:.4f}                                │")
    print(f"  │  Precision     :  {precision:.4f}   {'✅' if precision >= 0.80 else '⚠️ '}  (target ≥ 0.80)        │")
    print(f"  │  Recall        :  {recall:.4f}   {'✅' if recall >= 0.75 else '⚠️ '}  (target ≥ 0.75)        │")
    print(f"  │  Fitness score :  {fitness:.4f}                                │")
    print("  ├──────────────────────────────────────────────────────┤")

    if result.get("per_class"):
        print("  │  Per-class breakdown:                                │")
        for cls, m in result["per_class"].items():
            p   = m.get("precision", 0)
            r   = m.get("recall", 0)
            m50 = m.get("map50", 0)
            f1  = m.get("f1", 0)
            print(f"  │    {cls:<10}  P={p:.3f}  R={r:.3f}  mAP50={m50:.3f}  F1={f1:.3f}  │")
        print("  ├──────────────────────────────────────────────────────┤")

    overall_ok = all(result["meets_targets"].values())
    verdict    = "✅  Model meets all deployment targets." if overall_ok else \
                 "⚠️   Some targets not met — review before deploying."
    print(f"  │  {verdict:<52}│")
    print("  └──────────────────────────────────────────────────────┘")


def save_evaluation_report(result: Dict, out_path: str = "models/evaluation_report.json"):
    """Write evaluation results to JSON for record keeping."""
    import datetime
    result["evaluated_at"] = datetime.datetime.now().isoformat()
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Evaluation report saved: {p}")