#!/usr/bin/env python3
"""
src/inference/run_pipeline.py
──────────────────────────────
CLI entry point for the DC-EFDS-Lite inference pipeline.

Loads config/pipeline.yaml, applies any CLI overrides,
and runs the full inference loop.

Usage:
    # Standard (uses config/pipeline.yaml)
    python src/inference/run_pipeline.py

    # Override source (iPhone stream)
    python src/inference/run_pipeline.py --source http://192.168.1.100:8080/video

    # Headless Pi 3 (no display window)
    python src/inference/run_pipeline.py --headless

    # Dev mode (show window, lower threshold to see more detections)
    python src/inference/run_pipeline.py --display --conf 0.35

    # Point at a video file for offline testing
    python src/inference/run_pipeline.py --source data/test_video.mp4
"""

import argparse
import logging
import sys
from pathlib import Path

# ── Path setup — run from project root ───────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.inference.config   import load_config
from src.inference.pipeline import InferencePipeline


def setup_logging(log_path: str = "logs/pipeline.log"):
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers= [
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="a"),
        ],
    )


def main():
    parser = argparse.ArgumentParser(
        description="DC-EFDS-Lite — Real-time fire detection pipeline"
    )
    parser.add_argument(
        "--config", default="config/pipeline.yaml",
        help="Path to pipeline.yaml (default: config/pipeline.yaml)",
    )
    parser.add_argument(
        "--source", default=None,
        help="Video source override: '0'=webcam, URL=stream, path=file",
    )
    parser.add_argument(
        "--model", default=None,
        help="Path to ONNX model (overrides config)",
    )
    parser.add_argument(
        "--conf", type=float, default=None,
        help="Confidence threshold override (0–1)",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Force headless mode — disable display window",
    )
    parser.add_argument(
        "--display", action="store_true",
        help="Force display window on (dev mode)",
    )
    parser.add_argument(
        "--skip", type=int, default=None,
        help="Frame skip override (default: from config)",
    )
    args = parser.parse_args()

    setup_logging()

    # Load config
    cfg = load_config(args.config)

    # Apply CLI overrides
    if args.source:  cfg.source         = args.source
    if args.model:   cfg.model_path     = args.model
    if args.conf:    cfg.conf_threshold = args.conf
    if args.skip:    cfg.frame_skip     = args.skip
    if args.headless: cfg.show_window   = False
    if args.display:  cfg.show_window   = True

    # Validate model exists before starting
    if not cfg.model_path_exists:
        print(f"\n  ❌  ONNX model not found: {cfg.model_path}")
        print("      Run training first:")
        print("        source venv/bin/activate")
        print("        python scripts/train.py")
        sys.exit(1)

    # Run
    pipeline = InferencePipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()