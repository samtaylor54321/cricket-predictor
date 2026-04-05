"""
pipeline.py
===========
Top-level pipeline for the county championship betting model.
Runs all steps end-to-end or individually.

Steps:
  parse    — bronze → silver: parse Cricsheet JSON files
  features — silver → gold:   build Elo, form, and XI strength features
  model    — gold:             train model and write evaluation metrics

Usage:
    # Full pipeline
    python pipeline.py

    # Single step
    python pipeline.py --step features

    # Override default data paths
    python pipeline.py --bronze data/bronze/cricsheet \\
                       --silver data/silver/cricsheet \\
                       --gold   data/gold

    # Find value on an upcoming match (after model is trained)
    python pipeline.py --step value \\
                       --home-team Lancashire --away-team Surrey \\
                       --home-odds 2.10 --away-odds 1.80 \\
                       --elo-diff 45 --home-form 0.6 --away-form 0.4
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.feature_engineering import run as run_features
from src.model import load_model, run as run_model, value_bet
from src.parse_cricsheet import run as run_parse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TRAIN_STEPS = ["parse", "features", "model"]


# ─────────────────────────────────────────────
# STEP RUNNERS
# ─────────────────────────────────────────────

def step_parse(args: argparse.Namespace):
    run_parse(
        bronze_dir=Path(args.bronze),
        silver_dir=Path(args.silver),
    )


def step_features(args: argparse.Namespace):
    run_features(
        silver_dir=Path(args.silver),
        gold_dir=Path(args.gold),
    )


def step_model(args: argparse.Namespace):
    run_model(
        gold_dir=Path(args.gold),
        model_path=Path(args.gold) / "model.pkl",
    )


def step_value(args: argparse.Namespace):
    """
    Find value on a single upcoming match using the trained model.
    Feature values can be passed as CLI args or will fall back to
    neutral defaults (useful for quick checks).
    """
    model_path = Path(args.gold) / "model.pkl"
    if not model_path.exists():
        logger.error(f"No model found at {model_path} — run 'python pipeline.py --step model' first")
        sys.exit(1)

    model = load_model(model_path)

    features = {
        "elo_diff":              args.elo_diff,
        "home_won_toss":         args.home_won_toss,
        "home_form":             args.home_form,
        "away_form":             args.away_form,
        "venue_home_win_rate":   args.venue_home_win_rate,
        "batting_strength_diff": args.batting_strength_diff,
        "bowling_strength_diff": args.bowling_strength_diff,
    }

    result = value_bet(
        model=model,
        match_features=features,
        home_odds=args.home_odds,
        away_odds=args.away_odds,
    )

    home = args.home_team or "Home"
    away = args.away_team or "Away"

    logger.info("─" * 50)
    logger.info(f"  {home} vs {away}")
    logger.info("─" * 50)
    logger.info(f"  Model probs    {home}: {result['model_home_prob']:.1%}  |  {away}: {result['model_away_prob']:.1%}")
    logger.info(f"  Fair probs     {home}: {result['fair_home_prob']:.1%}  |  {away}: {result['fair_away_prob']:.1%}")
    logger.info(f"  Edge           {home}: {result['home_edge']:+.1%}  |  {away}: {result['away_edge']:+.1%}")
    logger.info("─" * 50)

    if result["value_side"]:
        side = home if result["value_side"] == "home" else away
        logger.info(f"  VALUE BET: {side} @ {result['odds']}")
        logger.info(f"  Kelly stake  : {result['kelly_stake']:.1%} of bankroll")
        logger.info(f"  Expected val : {result['expected_value']:+.3f} per £1 staked")
    else:
        logger.info("  No value found at current odds.")

    logger.info("─" * 50)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="County championship model pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Step selection
    parser.add_argument(
        "--step",
        choices=TRAIN_STEPS + ["all", "value"],
        default="all",
        help="Pipeline step to run",
    )

    # Data paths
    parser.add_argument("--bronze", default="data/bronze/cricsheet")
    parser.add_argument("--silver", default="data/silver/cricsheet")
    parser.add_argument("--gold",   default="data/gold")

    # Value bet args (used when --step value)
    parser.add_argument("--home-team",  default=None)
    parser.add_argument("--away-team",  default=None)
    parser.add_argument("--home-odds",  type=float, default=2.00)
    parser.add_argument("--away-odds",  type=float, default=2.00)
    parser.add_argument("--elo-diff",              type=float, default=0.0)
    parser.add_argument("--home-won-toss",         type=float, default=0.0)
    parser.add_argument("--home-form",             type=float, default=0.5)
    parser.add_argument("--away-form",             type=float, default=0.5)
    parser.add_argument("--venue-home-win-rate",   type=float, default=0.5)
    parser.add_argument("--batting-strength-diff", type=float, default=0.0)
    parser.add_argument("--bowling-strength-diff", type=float, default=0.0)

    args = parser.parse_args()

    if args.step == "value":
        step_value(args)
        return

    steps_to_run = TRAIN_STEPS if args.step == "all" else [args.step]
    step_map = {
        "parse":    step_parse,
        "features": step_features,
        "model":    step_model,
    }

    pipeline_start = time.time()
    logger.info(f"Running steps: {steps_to_run}")

    for step in steps_to_run:
        t = time.time()
        logger.info(f"{'─' * 50}")
        logger.info(f"Step: {step}")
        logger.info(f"{'─' * 50}")
        step_map[step](args)
        logger.info(f"Step '{step}' done in {time.time() - t:.1f}s")

    logger.info(f"{'─' * 50}")
    logger.info(f"Pipeline complete in {time.time() - pipeline_start:.1f}s")


if __name__ == "__main__":
    main()
