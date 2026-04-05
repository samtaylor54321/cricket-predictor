"""
tests/test_model.py
====================
Unit tests for src/model.py

Run with:
    pytest tests/test_model.py -v
"""

import json
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.model import (
    FEATURE_COLS,
    build_pipeline,
    kelly_stake,
    load_model,
    prepare_training_data,
    remove_vig,
    run,
    save_model,
    value_bet,
)


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture
def sample_features():
    """Synthetic features DataFrame with enough rows to cross-validate."""
    rng = np.random.default_rng(42)
    n   = 100

    df = pd.DataFrame({
        "match_id":              [f"m{i:03d}" for i in range(n)],
        "match_date":            pd.date_range("2019-04-01", periods=n, freq="7D"),
        "season":                ["2019"] * 50 + ["2022"] * 50,
        "division":              [1] * n,
        "home_team":             ["Surrey"] * n,
        "away_team":             ["Kent"] * n,
        "venue":                 ["The Oval"] * n,
        "elo_diff":              rng.normal(0, 80, n),
        "home_won_toss":         rng.integers(0, 2, n).astype(float),
        "home_form":             rng.uniform(0.2, 0.8, n),
        "away_form":             rng.uniform(0.2, 0.8, n),
        "venue_home_win_rate":   rng.uniform(0.4, 0.7, n),
        "batting_strength_diff": rng.normal(0, 5, n),
        "bowling_strength_diff": rng.normal(0, 5, n),
        "home_win":              rng.integers(0, 2, n).astype(float),
    })
    return df


@pytest.fixture
def features_with_nans(sample_features):
    """Features with some NaN values in venue_home_win_rate."""
    df = sample_features.copy()
    df.loc[:5, "venue_home_win_rate"] = np.nan
    return df


@pytest.fixture
def fitted_model(sample_features):
    """A model fitted on sample data."""
    X, y = prepare_training_data(sample_features)
    model = build_pipeline()
    model.fit(X, y)
    return model


# ─────────────────────────────────────────────
# remove_vig
# ─────────────────────────────────────────────

class TestRemoveVig:
    def test_probabilities_sum_to_one(self):
        home, away = remove_vig(2.10, 1.80)
        assert abs(home + away - 1.0) < 1e-9

    def test_lower_odds_higher_probability(self):
        home, away = remove_vig(3.00, 1.50)
        assert away > home

    def test_even_odds_equal_probability(self):
        home, away = remove_vig(2.00, 2.00)
        assert abs(home - away) < 1e-9

    def test_vig_is_removed(self):
        """Raw implied probs sum to > 1 due to overround; after removal they sum to 1."""
        home_raw = 1 / 2.10
        away_raw = 1 / 1.80
        assert home_raw + away_raw > 1.0  # confirms overround exists
        home, away = remove_vig(2.10, 1.80)
        assert abs(home + away - 1.0) < 1e-9


# ─────────────────────────────────────────────
# kelly_stake
# ─────────────────────────────────────────────

class TestKellyStake:
    def test_positive_edge_returns_positive_stake(self):
        stake = kelly_stake(prob=0.60, decimal_odds=2.00)
        assert stake > 0

    def test_no_edge_returns_zero(self):
        # 50% chance at evens = no edge
        stake = kelly_stake(prob=0.50, decimal_odds=2.00)
        assert stake == 0.0

    def test_negative_edge_returns_zero(self):
        stake = kelly_stake(prob=0.40, decimal_odds=2.00)
        assert stake == 0.0

    def test_fractional_kelly_applied(self):
        """Full Kelly on prob=0.6, odds=2.0 is 0.2; quarter Kelly = 0.05."""
        full_kelly = kelly_stake(prob=0.60, decimal_odds=2.00, fraction=1.0)
        quarter    = kelly_stake(prob=0.60, decimal_odds=2.00, fraction=0.25)
        assert abs(quarter - full_kelly * 0.25) < 1e-9

    def test_stake_does_not_exceed_one(self):
        # Even with massive edge, should stay well below 1
        stake = kelly_stake(prob=0.95, decimal_odds=10.00, fraction=0.25)
        assert stake < 1.0


# ─────────────────────────────────────────────
# prepare_training_data
# ─────────────────────────────────────────────

class TestPrepareTrainingData:
    def test_returns_correct_shapes(self, sample_features):
        X, y = prepare_training_data(sample_features)
        assert X.shape[1] == len(FEATURE_COLS)
        assert len(X) == len(y)

    def test_only_feature_cols_in_X(self, sample_features):
        X, _ = prepare_training_data(sample_features)
        assert list(X.columns) == FEATURE_COLS

    def test_nan_rows_dropped(self, sample_features):
        df = sample_features.copy()
        df.loc[0, "elo_diff"] = np.nan
        X, y = prepare_training_data(df)
        assert len(X) == len(sample_features) - 1

    def test_venue_nan_filled_not_dropped(self, features_with_nans):
        X, y = prepare_training_data(features_with_nans)
        assert X["venue_home_win_rate"].isna().sum() == 0
        assert len(X) == len(features_with_nans)  # no rows dropped

    def test_target_is_integer(self, sample_features):
        _, y = prepare_training_data(sample_features)
        assert y.dtype in (int, np.int64, np.int32)


# ─────────────────────────────────────────────
# build_pipeline
# ─────────────────────────────────────────────

class TestBuildPipeline:
    def test_returns_pipeline(self):
        model = build_pipeline()
        assert isinstance(model, Pipeline)

    def test_pipeline_fits_and_predicts(self, sample_features):
        X, y = prepare_training_data(sample_features)
        model = build_pipeline()
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(y)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_sums_to_one(self, sample_features):
        X, y = prepare_training_data(sample_features)
        model = build_pipeline()
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape[1] == 2
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_probabilities_between_zero_and_one(self, sample_features):
        X, y = prepare_training_data(sample_features)
        model = build_pipeline()
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert (proba >= 0).all() and (proba <= 1).all()


# ─────────────────────────────────────────────
# save_model / load_model
# ─────────────────────────────────────────────

class TestModelPersistence:
    def test_save_and_load_roundtrip(self, tmp_path, fitted_model, sample_features):
        path = tmp_path / "model.pkl"
        save_model(fitted_model, path)
        loaded = load_model(path)

        X, _ = prepare_training_data(sample_features)
        original_proba = fitted_model.predict_proba(X)
        loaded_proba   = loaded.predict_proba(X)
        np.testing.assert_allclose(original_proba, loaded_proba, atol=1e-9)

    def test_saved_file_is_pickle(self, tmp_path, fitted_model):
        path = tmp_path / "model.pkl"
        save_model(fitted_model, path)
        with open(path, "rb") as f:
            obj = pickle.load(f)
        assert isinstance(obj, Pipeline)


# ─────────────────────────────────────────────
# value_bet
# ─────────────────────────────────────────────

class TestValueBet:
    @pytest.fixture
    def neutral_features(self):
        return {
            "elo_diff":              0.0,
            "home_won_toss":         0.0,
            "home_form":             0.5,
            "away_form":             0.5,
            "venue_home_win_rate":   0.5,
            "batting_strength_diff": 0.0,
            "bowling_strength_diff": 0.0,
        }

    def test_returns_expected_keys(self, fitted_model, neutral_features):
        result = value_bet(fitted_model, neutral_features, 2.10, 1.80)
        expected_keys = {
            "model_home_prob", "model_away_prob",
            "fair_home_prob",  "fair_away_prob",
            "home_edge",       "away_edge",
            "value_side",      "odds",
            "kelly_stake",     "expected_value",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_probs_sum_to_one(self, fitted_model, neutral_features):
        result = value_bet(fitted_model, neutral_features, 2.10, 1.80)
        assert abs(result["model_home_prob"] + result["model_away_prob"] - 1.0) < 1e-3

    def test_no_value_returns_none_side(self, fitted_model, neutral_features):
        # Very tight odds with small edge should return no value
        result = value_bet(fitted_model, neutral_features, 1.95, 1.95, min_edge=0.99)
        assert result["value_side"] is None
        assert result["kelly_stake"] == 0.0

    def test_strong_home_favourite_flags_home(self, sample_features):
        """Model trained on data where high elo_diff reliably predicts home win."""
        rng = np.random.default_rng(0)
        n   = 200
        elo = rng.normal(0, 80, n)
        # home_win strongly correlated with elo_diff so model learns direction
        home_win = (elo + rng.normal(0, 20, n) > 0).astype(float)
        df = pd.DataFrame({
            "elo_diff":              elo,
            "home_won_toss":         np.zeros(n),
            "home_form":             np.full(n, 0.5),
            "away_form":             np.full(n, 0.5),
            "venue_home_win_rate":   np.full(n, 0.5),
            "batting_strength_diff": np.zeros(n),
            "bowling_strength_diff": np.zeros(n),
            "home_win":              home_win,
        })
        X, y = prepare_training_data(df)
        model = build_pipeline()
        model.fit(X, y)

        features = {
            "elo_diff":              400.0,
            "home_won_toss":         0.0,
            "home_form":             0.5,
            "away_form":             0.5,
            "venue_home_win_rate":   0.5,
            "batting_strength_diff": 0.0,
            "bowling_strength_diff": 0.0,
        }
        result = value_bet(model, features, home_odds=1.80, away_odds=2.20, min_edge=0.01)
        assert result["value_side"] == "home"
        assert result["kelly_stake"] > 0
        assert result["expected_value"] > 0

    def test_missing_venue_rate_handled(self, fitted_model, neutral_features):
        """NaN venue rate should not raise."""
        features = dict(neutral_features)
        features["venue_home_win_rate"] = np.nan
        result = value_bet(fitted_model, features, 2.10, 1.80)
        assert "model_home_prob" in result

    def test_edges_sum_approximately(self, fitted_model, neutral_features):
        """home_edge + away_edge should sum to approximately 0 (model probs sum to 1,
        fair probs sum to 1, so edges cancel)."""
        result = value_bet(fitted_model, neutral_features, 2.10, 1.80)
        assert abs(result["home_edge"] + result["away_edge"]) < 0.01


# ─────────────────────────────────────────────
# run (integration)
# ─────────────────────────────────────────────

class TestRun:
    def test_creates_model_and_evaluation(self, tmp_path, sample_features):
        gold = tmp_path / "gold"
        gold.mkdir()
        sample_features.to_csv(gold / "features.csv", index=False)

        model_path = gold / "model.pkl"
        run(gold_dir=gold, model_path=model_path)

        assert model_path.exists()
        assert (gold / "evaluation.json").exists()

    def test_evaluation_has_expected_keys(self, tmp_path, sample_features):
        gold = tmp_path / "gold"
        gold.mkdir()
        sample_features.to_csv(gold / "features.csv", index=False)

        run(gold_dir=gold, model_path=gold / "model.pkl")

        with open(gold / "evaluation.json") as f:
            metrics = json.load(f)

        assert "cv_accuracy_mean" in metrics
        assert "cv_roc_auc_mean"  in metrics
        assert "cv_brier_mean"    in metrics
        assert "n_samples"        in metrics

    def test_raises_if_features_missing(self, tmp_path):
        gold = tmp_path / "gold"
        gold.mkdir()
        with pytest.raises(FileNotFoundError, match="features.csv"):
            run(gold_dir=gold, model_path=gold / "model.pkl")

    def test_returned_model_can_predict(self, tmp_path, sample_features):
        gold = tmp_path / "gold"
        gold.mkdir()
        sample_features.to_csv(gold / "features.csv", index=False)

        model = run(gold_dir=gold, model_path=gold / "model.pkl")
        X, _  = prepare_training_data(sample_features)
        preds = model.predict(X)
        assert len(preds) == len(X)
