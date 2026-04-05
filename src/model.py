"""
src/model.py
============
Trains a logistic regression on gold-layer features, evaluates it,
and exposes a value_bet() function for use on upcoming fixtures.

Gold (input):
  data/gold/features.csv

Gold (output):
  data/gold/model.pkl       — fitted pipeline
  data/gold/evaluation.json — cross-val accuracy and calibration stats

Called via:
    python src/model.py

Or from pipeline.py:
    from src.model import run
    run()

Finding value on an upcoming match:
    from src.model import load_model, value_bet
    model = load_model()
    result = value_bet(model, match_features, bookmaker_odds)
"""

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

DEFAULT_GOLD      = Path("data/gold")
DEFAULT_MODEL_PATH = Path("data/gold/model.pkl")

# Features used in training — order matters for inference
FEATURE_COLS = [
    "elo_diff",
    "home_won_toss",
    "home_form",
    "away_form",
    "venue_home_win_rate",
    "batting_strength_diff",
    "bowling_strength_diff",
]

# Minimum edge (model prob - fair prob) to flag as value
MIN_EDGE = 0.03

# Fractional Kelly multiplier — quarter Kelly for safety
KELLY_FRACTION = 0.25


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def prepare_training_data(features_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Filter to rows with a valid target and no missing features.
    Returns (X, y).
    """
    df = features_df.dropna(subset=["home_win"]).copy()
    df["home_win"] = df["home_win"].astype(int)

    # Fill venue_home_win_rate NaNs with column median before dropping other NaN rows
    # (venue rate is NaN for first match at a ground — expected, not missing data)
    if df["venue_home_win_rate"].isna().any():
        median = df["venue_home_win_rate"].median()
        df["venue_home_win_rate"] = df["venue_home_win_rate"].fillna(median)
        logger.info(f"  Filled venue_home_win_rate NaNs with median {median:.3f}")

    # Now drop rows where any other feature is missing
    other_features = [c for c in FEATURE_COLS if c != "venue_home_win_rate"]
    df = df.dropna(subset=other_features)

    X = df[FEATURE_COLS]
    y = df["home_win"]
    return X, y


def build_pipeline() -> Pipeline:
    """
    Logistic regression with standard scaling and probability calibration.
    Calibration ensures predicted probabilities are reliable —
    a predicted 60% should win about 60% of the time.
    """
    base = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        C=1.0,
        random_state=42,
    )
    calibrated = CalibratedClassifierCV(base, method="isotonic", cv=5)

    return Pipeline([
        ("scaler",    StandardScaler()),
        ("classifier", calibrated),
    ])


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Cross-validate the model and return evaluation metrics.
    Uses StratifiedKFold to preserve class balance across folds.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accuracy_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    brier_scores    = cross_val_score(model, X, y, cv=cv, scoring="neg_brier_score")
    roc_scores      = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

    metrics = {
        "n_samples":        int(len(y)),
        "home_win_rate":    float(y.mean()),
        "cv_accuracy_mean": float(accuracy_scores.mean()),
        "cv_accuracy_std":  float(accuracy_scores.std()),
        "cv_brier_mean":    float(-brier_scores.mean()),   # lower is better
        "cv_roc_auc_mean":  float(roc_scores.mean()),
        "cv_roc_auc_std":   float(roc_scores.std()),
    }

    logger.info(f"  Accuracy  : {metrics['cv_accuracy_mean']:.1%} ± {metrics['cv_accuracy_std']:.1%}")
    logger.info(f"  ROC AUC   : {metrics['cv_roc_auc_mean']:.3f} ± {metrics['cv_roc_auc_std']:.3f}")
    logger.info(f"  Brier     : {metrics['cv_brier_mean']:.3f}  (baseline: {y.mean() * (1 - y.mean()):.3f})")

    return metrics


# ─────────────────────────────────────────────
# PERSISTENCE
# ─────────────────────────────────────────────

def save_model(model: Pipeline, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"  Model saved to {path}")


def load_model(path: Path = DEFAULT_MODEL_PATH) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────
# VALUE BET FINDER
# ─────────────────────────────────────────────

def remove_vig(home_odds: float, away_odds: float) -> tuple[float, float]:
    """
    Convert bookmaker decimal odds to vig-free fair probabilities.
    The overround is removed proportionally across both outcomes.
    """
    raw_home = 1 / home_odds
    raw_away = 1 / away_odds
    total    = raw_home + raw_away
    return raw_home / total, raw_away / total


def kelly_stake(prob: float, decimal_odds: float, fraction: float = KELLY_FRACTION) -> float:
    """
    Fractional Kelly criterion stake as a proportion of bankroll.
    Returns 0 if the bet has no edge.
    """
    b     = decimal_odds - 1
    kelly = (b * prob - (1 - prob)) / b
    return max(0.0, kelly * fraction)


def value_bet(
    model: Pipeline,
    match_features: dict,
    home_odds: float,
    away_odds: float,
    min_edge: float = MIN_EDGE,
) -> dict:
    """
    Given a match and bookmaker odds, return the value assessment.

    Args:
        model:          Fitted model pipeline from load_model()
        match_features: Dict with keys matching FEATURE_COLS.
                        NaN is allowed for venue_home_win_rate
                        (unknown venue — will be filled with training median).
        home_odds:      Bet365 decimal odds for home win (draw no bet)
        away_odds:      Bet365 decimal odds for away win (draw no bet)
        min_edge:       Minimum probability edge to flag as a value bet

    Returns dict with:
        model_home_prob:  model's estimated home win probability
        fair_home_prob:   vig-removed bookmaker implied probability
        home_edge:        model_home_prob - fair_home_prob
        away_edge:        equivalent for away side
        value_side:       'home', 'away', or None
        kelly_stake:      recommended fraction of bankroll (0 if no value)
        expected_value:   expected profit per £1 staked (0 if no value)
    """
    X = pd.DataFrame([{col: match_features.get(col, np.nan) for col in FEATURE_COLS}])

    # Fill venue NaN with a neutral 0.5 if not available
    X["venue_home_win_rate"] = X["venue_home_win_rate"].fillna(0.5)

    model_home_prob = float(model.predict_proba(X)[0][1])
    model_away_prob = 1 - model_home_prob

    fair_home_prob, fair_away_prob = remove_vig(home_odds, away_odds)

    home_edge = model_home_prob - fair_home_prob
    away_edge = model_away_prob - fair_away_prob

    # Determine best value side
    value_side = None
    best_edge  = min_edge  # only flag if edge exceeds threshold

    if home_edge >= best_edge:
        value_side = "home"
        best_edge  = home_edge
    if away_edge >= best_edge:
        value_side = "away"

    if value_side == "home":
        stake = kelly_stake(model_home_prob, home_odds)
        ev    = model_home_prob * (home_odds - 1) - (1 - model_home_prob)
        odds  = home_odds
    elif value_side == "away":
        stake = kelly_stake(model_away_prob, away_odds)
        ev    = model_away_prob * (away_odds - 1) - (1 - model_away_prob)
        odds  = away_odds
    else:
        stake = 0.0
        ev    = 0.0
        odds  = None

    return {
        "model_home_prob": round(model_home_prob, 3),
        "model_away_prob": round(model_away_prob, 3),
        "fair_home_prob":  round(fair_home_prob,  3),
        "fair_away_prob":  round(fair_away_prob,  3),
        "home_edge":       round(home_edge, 3),
        "away_edge":       round(away_edge, 3),
        "value_side":      value_side,
        "odds":            odds,
        "kelly_stake":     round(stake, 3),
        "expected_value":  round(ev, 3),
    }


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

def run(
    gold_dir:    Path = DEFAULT_GOLD,
    model_path:  Path = DEFAULT_MODEL_PATH,
) -> Pipeline:
    """
    Load gold features, train model, evaluate, save.
    Returns the fitted model.
    """
    gold_dir   = Path(gold_dir)
    model_path = Path(model_path)

    features_path = gold_dir / "features.csv"
    if not features_path.exists():
        raise FileNotFoundError(f"features.csv not found in {gold_dir} — run feature step first")

    logger.info(f"Loading features from {features_path}")
    features_df = pd.read_csv(features_path)

    logger.info("Preparing training data...")
    X, y = prepare_training_data(features_df)
    logger.info(f"  Training rows  : {len(X)}")
    logger.info(f"  Home win rate  : {y.mean():.1%}")

    logger.info("Evaluating model (cross-validation)...")
    model = build_pipeline()
    metrics = evaluate(model, X, y)

    logger.info("Training final model on full dataset...")
    model.fit(X, y)

    save_model(model, model_path)

    eval_path = gold_dir / "evaluation.json"
    with open(eval_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"  Evaluation saved to {eval_path}")

    return model


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Train the county championship model")
    parser.add_argument("--gold",  type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    run(gold_dir=args.gold, model_path=args.model)
