"""
pipeline.py
===========
Top-level pipeline for the county championship betting model.
Run individual steps or the full pipeline end-to-end.

Usage:
    # Full pipeline
    python pipeline.py

    # Single step only
    python pipeline.py --step parse

    # Override default data paths
    python pipeline.py --bronze data/bronze/cricsheet --silver data/silver/cricsheet
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path so src imports work when running from root
sys.path.insert(0, str(Path(__file__).parent))

from src.parse_cricsheet import run as parse_cricsheet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

STEPS = ["parse"]  # extend as we add feature_engineering, model, etc.


def run_step(step: str, args: argparse.Namespace):
    start = time.time()
    logger.info(f"{'─' * 50}")
    logger.info(f"Step: {step}")
    logger.info(f"{'─' * 50}")

    if step == "parse":
        parse_cricsheet(
            bronze_dir=Path(args.bronze),
            silver_dir=Path(args.silver),
        )
    else:
        raise ValueError(f"Unknown step: {step}")

    elapsed = time.time() - start
    logger.info(f"Step '{step}' completed in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="County championship model pipeline")
    parser.add_argument(
        "--step",
        choices=STEPS + ["all"],
        default="all",
        help="Which pipeline step to run (default: all)",
    )
    parser.add_argument(
        "--bronze",
        default="data/bronze/cricsheet",
        help="Bronze layer input directory",
    )
    parser.add_argument(
        "--silver",
        default="data/silver/cricsheet",
        help="Silver layer output directory",
    )
    args = parser.parse_args()

    steps_to_run = STEPS if args.step == "all" else [args.step]

    logger.info(f"Running pipeline steps: {steps_to_run}")
    pipeline_start = time.time()

    for step in steps_to_run:
        run_step(step, args)

    total = time.time() - pipeline_start
    logger.info(f"{'─' * 50}")
    logger.info(f"Pipeline complete in {total:.1f}s")


if __name__ == "__main__":
    main()
