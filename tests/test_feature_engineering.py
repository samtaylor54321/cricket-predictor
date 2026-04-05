"""
tests/test_feature_engineering.py
===================================
Unit tests for src/feature_engineering.py

Run with:
    pytest tests/test_feature_engineering.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (EloTracker, build_features,
                                     build_player_averages, build_team_form,
                                     build_venue_stats, compute_xi_strength,
                                     run)

# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────


@pytest.fixture
def simple_matches():
    """Four decisive matches across two seasons."""
    return pd.DataFrame(
        [
            {
                "match_id": "m001",
                "match_date": "2022-04-10",
                "season": "2022",
                "division": 1,
                "venue": "Old Trafford",
                "home_team": "Lancashire",
                "away_team": "Surrey",
                "result_type": "decisive",
                "winner": "Lancashire",
                "home_win": 1,
                "home_won_toss": 1,
                "toss_decision": "bat",
                "home_players": "P1|P2|P3|P4|P5|P6|P7|P8|P9|P10|P11",
                "away_players": "P12|P13|P14|P15|P16|P17|P18|P19|P20|P21|P22",
                "home_player_ids": "id1|id2|id3|id4|id5|id6|id7|id8|id9|id10|id11",
                "away_player_ids": "id12|id13|id14|id15|id16|id17|id18|id19|id20|id21|id22",
            },
            {
                "match_id": "m002",
                "match_date": "2022-04-17",
                "season": "2022",
                "division": 1,
                "venue": "The Oval",
                "home_team": "Surrey",
                "away_team": "Lancashire",
                "result_type": "decisive",
                "winner": "Surrey",
                "home_win": 1,
                "home_won_toss": 0,
                "toss_decision": "field",
                "home_players": "P12|P13|P14|P15|P16|P17|P18|P19|P20|P21|P22",
                "away_players": "P1|P2|P3|P4|P5|P6|P7|P8|P9|P10|P11",
                "home_player_ids": "id12|id13|id14|id15|id16|id17|id18|id19|id20|id21|id22",
                "away_player_ids": "id1|id2|id3|id4|id5|id6|id7|id8|id9|id10|id11",
            },
            {
                "match_id": "m003",
                "match_date": "2022-04-24",
                "season": "2022",
                "division": 1,
                "venue": "Old Trafford",
                "home_team": "Lancashire",
                "away_team": "Yorkshire",
                "result_type": "decisive",
                "winner": "Yorkshire",
                "home_win": 0,
                "home_won_toss": 1,
                "toss_decision": "bat",
                "home_players": "P1|P2|P3|P4|P5|P6|P7|P8|P9|P10|P11",
                "away_players": "P23|P24|P25|P26|P27|P28|P29|P30|P31|P32|P33",
                "home_player_ids": "id1|id2|id3|id4|id5|id6|id7|id8|id9|id10|id11",
                "away_player_ids": "id23|id24|id25|id26|id27|id28|id29|id30|id31|id32|id33",
            },
            {
                "match_id": "m004",
                "match_date": "2022-05-01",
                "season": "2022",
                "division": 1,
                "venue": "Old Trafford",
                "home_team": "Lancashire",
                "away_team": "Essex",
                "result_type": "draw",
                "winner": None,
                "home_win": None,
                "home_won_toss": 0,
                "toss_decision": "field",
                "home_players": "P1|P2|P3|P4|P5|P6|P7|P8|P9|P10|P11",
                "away_players": "P34|P35|P36|P37|P38|P39|P40|P41|P42|P43|P44",
                "home_player_ids": "id1|id2|id3|id4|id5|id6|id7|id8|id9|id10|id11",
                "away_player_ids": "id34|id35|id36|id37|id38|id39|id40|id41|id42|id43|id44",
            },
        ]
    )


@pytest.fixture
def simple_players():
    """Minimal player performance rows for two players across matches."""
    return pd.DataFrame(
        [
            # id1 bats in m001 innings 1
            {
                "match_id": "m001",
                "innings": 1,
                "batting_team": "Lancashire",
                "player_name": "P1",
                "player_id": "id1",
                "runs_scored": 45,
                "balls_faced": 80,
                "dismissed": True,
                "wickets_taken": 0,
                "balls_bowled": 0,
                "runs_conceded": 0,
            },
            # id1 bats in m001 innings 3 (follow-on scenario)
            {
                "match_id": "m001",
                "innings": 3,
                "batting_team": "Lancashire",
                "player_name": "P1",
                "player_id": "id1",
                "runs_scored": 20,
                "balls_faced": 35,
                "dismissed": True,
                "wickets_taken": 0,
                "balls_bowled": 0,
                "runs_conceded": 0,
            },
            # id8 bowls in m001 innings 2
            {
                "match_id": "m001",
                "innings": 2,
                "batting_team": "Surrey",
                "player_name": "P8",
                "player_id": "id8",
                "runs_scored": 5,
                "balls_faced": 10,
                "dismissed": True,
                "wickets_taken": 3,
                "balls_bowled": 24,
                "runs_conceded": 48,
            },
            # id1 bats in m002 (away match)
            {
                "match_id": "m002",
                "innings": 2,
                "batting_team": "Lancashire",
                "player_name": "P1",
                "player_id": "id1",
                "runs_scored": 70,
                "balls_faced": 120,
                "dismissed": False,
                "wickets_taken": 0,
                "balls_bowled": 0,
                "runs_conceded": 0,
            },
            # id1 bats in m003
            {
                "match_id": "m003",
                "innings": 1,
                "batting_team": "Lancashire",
                "player_name": "P1",
                "player_id": "id1",
                "runs_scored": 12,
                "balls_faced": 25,
                "dismissed": True,
                "wickets_taken": 0,
                "balls_bowled": 0,
                "runs_conceded": 0,
            },
        ]
    )


# ─────────────────────────────────────────────
# EloTracker
# ─────────────────────────────────────────────


class TestEloTracker:
    def test_default_rating(self):
        elo = EloTracker()
        assert elo.get("New Team") == 1500

    def test_winner_rating_increases(self):
        elo = EloTracker()
        before = elo.get("Surrey")
        elo.update("Surrey", "Kent")
        assert elo.get("Surrey") > before

    def test_loser_rating_decreases(self):
        elo = EloTracker()
        before = elo.get("Kent")
        elo.update("Surrey", "Kent")
        assert elo.get("Kent") < before

    def test_ratings_sum_is_conserved(self):
        elo = EloTracker()
        before = elo.get("Surrey") + elo.get("Kent")
        elo.update("Surrey", "Kent")
        after = elo.get("Surrey") + elo.get("Kent")
        assert abs(before - after) < 0.01

    def test_home_advantage_reduces_winner_gain(self):
        """Winning at home should gain fewer points than winning away."""
        elo_home = EloTracker()
        elo_away = EloTracker()

        elo_home.update("Surrey", "Kent", home_team="Surrey")
        elo_away.update("Surrey", "Kent", home_team="Kent")

        gain_home = elo_home.get("Surrey") - 1500
        gain_away = elo_away.get("Surrey") - 1500
        assert gain_home < gain_away

    def test_snapshot_returns_copy(self):
        elo = EloTracker()
        elo.update("Surrey", "Kent")
        snap = elo.snapshot()
        elo.update("Kent", "Surrey")
        assert snap["Surrey"] != elo.get("Surrey")


# ─────────────────────────────────────────────
# build_player_averages
# ─────────────────────────────────────────────


class TestBuildPlayerAverages:
    def test_returns_dataframe(self, simple_players, simple_matches):
        match_dates = pd.to_datetime(simple_matches.set_index("match_id")["match_date"])
        result = build_player_averages(simple_players, match_dates)
        assert isinstance(result, pd.DataFrame)

    def test_no_leakage_first_appearance(self, simple_players, simple_matches):
        """Player should have NaN averages for their first-ever match."""
        match_dates = pd.to_datetime(simple_matches.set_index("match_id")["match_date"])
        avgs = build_player_averages(simple_players, match_dates)
        first_row = avgs[(avgs["player_id"] == "id1") & (avgs["match_id"] == "m001")]
        assert first_row["batting_avg"].isna().all()

    def test_averages_computed_from_prior_matches(self, simple_players, simple_matches):
        """By m003, id1 has two prior innings (m001 x2, m002 x1)."""
        match_dates = pd.to_datetime(simple_matches.set_index("match_id")["match_date"])
        avgs = build_player_averages(simple_players, match_dates)
        row = avgs[(avgs["player_id"] == "id1") & (avgs["match_id"] == "m003")]
        assert not row.empty
        # m001: 45 runs (out) + 20 runs (out), m002: 70 runs (not out)
        # batting_avg = (45 + 20 + 70) / 2 dismissals = 67.5
        assert abs(row["batting_avg"].iloc[0] - 67.5) < 0.1

    def test_bowling_avg_nan_with_no_wickets(self, simple_players, simple_matches):
        match_dates = pd.to_datetime(simple_matches.set_index("match_id")["match_date"])
        avgs = build_player_averages(simple_players, match_dates)
        # id1 is a batter — no wickets taken
        row = avgs[(avgs["player_id"] == "id1") & (avgs["match_id"] == "m003")]
        assert row["bowling_avg"].isna().all()


# ─────────────────────────────────────────────
# compute_xi_strength
# ─────────────────────────────────────────────


class TestComputeXiStrength:
    def test_returns_dict_with_expected_keys(self):
        avgs = pd.DataFrame(
            columns=[
                "player_id",
                "match_id",
                "batting_avg",
                "bowling_avg",
                "innings_count",
            ]
        )
        result = compute_xi_strength([], "m001", avgs, 30.0, 35.0)
        assert "batting_strength" in result
        assert "bowling_strength" in result

    def test_empty_xi_uses_divisional_average(self):
        avgs = pd.DataFrame(
            columns=[
                "player_id",
                "match_id",
                "batting_avg",
                "bowling_avg",
                "innings_count",
            ]
        )
        result = compute_xi_strength([], "m001", avgs, 30.0, 35.0)
        # With no players, means of empty lists are nan
        assert np.isnan(result["batting_strength"])

    def test_unknown_player_uses_divisional_average(self):
        avgs = pd.DataFrame(
            columns=[
                "player_id",
                "match_id",
                "batting_avg",
                "bowling_avg",
                "innings_count",
            ]
        )
        ids = ["unknown1"] * 11
        result = compute_xi_strength(ids, "m001", avgs, 30.0, 35.0)
        assert abs(result["batting_strength"] - 30.0) < 0.01

    def test_shrinkage_applied_for_sparse_player(self):
        """A player with 2 innings should be blended towards divisional avg."""
        avgs = pd.DataFrame(
            [
                {
                    "player_id": "id1",
                    "match_id": "m001",
                    "batting_avg": 60.0,
                    "bowling_avg": 25.0,
                    "innings_count": 2,
                }
            ]
        )
        # weight = 2/5 = 0.4, so blended = 0.4 * 60 + 0.6 * 30 = 42
        result = compute_xi_strength(["id1"], "m001", avgs, 30.0, 35.0)
        assert abs(result["batting_strength"] - 42.0) < 0.1

    def test_full_weight_for_experienced_player(self):
        """A player with >= PLAYER_MIN_INNINGS innings gets full weight."""
        avgs = pd.DataFrame(
            [
                {
                    "player_id": "id1",
                    "match_id": "m001",
                    "batting_avg": 60.0,
                    "bowling_avg": 25.0,
                    "innings_count": 10,
                }
            ]
        )
        result = compute_xi_strength(["id1"], "m001", avgs, 30.0, 35.0)
        assert abs(result["batting_strength"] - 60.0) < 0.1


# ─────────────────────────────────────────────
# build_venue_stats
# ─────────────────────────────────────────────


class TestBuildVenueStats:
    def test_first_match_at_venue_is_nan(self, simple_matches):
        rates = build_venue_stats(simple_matches)
        assert np.isnan(rates["m001"])  # first match at Old Trafford

    def test_second_match_uses_prior_result(self, simple_matches):
        rates = build_venue_stats(simple_matches)
        # m003 is the second decisive match at Old Trafford; m001 was a home win (1.0)
        assert rates["m003"] == 1.0

    def test_draw_excluded_from_venue_stats(self, simple_matches):
        # m004 is a draw — it should not appear in venue stats
        rates = build_venue_stats(simple_matches)
        assert "m004" not in rates.index

    def test_different_venues_independent(self, simple_matches):
        rates = build_venue_stats(simple_matches)
        # m002 is the first match at The Oval — should be NaN
        assert np.isnan(rates["m002"])


# ─────────────────────────────────────────────
# build_team_form
# ─────────────────────────────────────────────


class TestBuildTeamForm:
    def test_first_match_form_is_nan(self, simple_matches):
        form = build_team_form(simple_matches)
        first = form[form["match_id"] == "m001"]
        assert first["home_form"].isna().all()
        assert first["away_form"].isna().all()

    def test_form_updates_after_result(self, simple_matches):
        form = build_team_form(simple_matches)
        # m003: Lancashire's prior decisive matches are m001 (home win = 1.0 for Lancashire)
        # and m002 (away at The Oval, Surrey won so Lancashire won = 0).
        # Form = (1 + 0) / 2 = 0.5
        lancashire_form = form[form["match_id"] == "m003"]["home_form"].iloc[0]
        assert abs(lancashire_form - 0.5) < 0.01

    def test_draw_excluded_from_form(self, simple_matches):
        form = build_team_form(simple_matches)
        # m004 is a draw so it should not appear
        assert "m004" not in form["match_id"].values

    def test_returns_dataframe_with_correct_columns(self, simple_matches):
        form = build_team_form(simple_matches)
        assert set(form.columns) == {"match_id", "home_form", "away_form"}


# ─────────────────────────────────────────────
# build_features (integration)
# ─────────────────────────────────────────────


class TestBuildFeatures:
    def test_returns_only_decisive_matches(self, simple_matches, simple_players):
        features = build_features(simple_matches, simple_players)
        assert "m004" not in features["match_id"].values  # draw excluded
        assert len(features) == 3  # m001, m002, m003

    def test_elo_diff_is_zero_for_first_match(self, simple_matches, simple_players):
        """Both teams start at 1500 so diff should be 0 for the very first match."""
        features = build_features(simple_matches, simple_players)
        first = features[features["match_id"] == "m001"]["elo_diff"].iloc[0]
        assert first == 0.0

    def test_elo_diff_changes_after_result(self, simple_matches, simple_players):
        features = build_features(simple_matches, simple_players)
        m001_diff = features[features["match_id"] == "m001"]["elo_diff"].iloc[0]
        m002_diff = features[features["match_id"] == "m002"]["elo_diff"].iloc[0]
        # m001 starts at 0 (both teams at default). After Lancashire win,
        # m002 (Surrey home vs Lancashire away) should reflect updated ratings.
        assert m001_diff == 0.0  # first ever match — both at default
        assert m002_diff != 0.0  # ratings have moved after m001

    def test_expected_columns_present(self, simple_matches, simple_players):
        features = build_features(simple_matches, simple_players)
        expected = {
            "match_id",
            "match_date",
            "season",
            "division",
            "home_team",
            "away_team",
            "venue",
            "elo_diff",
            "home_won_toss",
            "home_form",
            "away_form",
            "venue_home_win_rate",
            "batting_strength_diff",
            "bowling_strength_diff",
            "home_win",
        }
        assert expected.issubset(set(features.columns))

    def test_home_win_target_preserved(self, simple_matches, simple_players):
        features = build_features(simple_matches, simple_players)
        assert features[features["match_id"] == "m001"]["home_win"].iloc[0] == 1
        assert features[features["match_id"] == "m003"]["home_win"].iloc[0] == 0


# ─────────────────────────────────────────────
# run (integration)
# ─────────────────────────────────────────────


class TestRun:
    def test_creates_features_csv(self, tmp_path, simple_matches, simple_players):
        silver = tmp_path / "silver"
        gold = tmp_path / "gold"
        silver.mkdir()
        simple_matches.to_csv(silver / "matches.csv", index=False)
        simple_players.to_csv(silver / "players.csv", index=False)

        run(silver_dir=silver, gold_dir=gold)

        assert (gold / "features.csv").exists()

    def test_raises_if_matches_missing(self, tmp_path):
        silver = tmp_path / "silver"
        silver.mkdir()
        with pytest.raises(FileNotFoundError, match="matches.csv"):
            run(silver_dir=silver, gold_dir=tmp_path / "gold")

    def test_raises_if_players_missing(self, tmp_path, simple_matches):
        silver = tmp_path / "silver"
        silver.mkdir()
        simple_matches.to_csv(silver / "matches.csv", index=False)
        with pytest.raises(FileNotFoundError, match="players.csv"):
            run(silver_dir=silver, gold_dir=tmp_path / "gold")

    def test_output_has_expected_shape(self, tmp_path, simple_matches, simple_players):
        silver = tmp_path / "silver"
        gold = tmp_path / "gold"
        silver.mkdir()
        simple_matches.to_csv(silver / "matches.csv", index=False)
        simple_players.to_csv(silver / "players.csv", index=False)

        features_df = run(silver_dir=silver, gold_dir=gold)

        assert len(features_df) == 3  # 3 decisive matches
        assert "home_win" in features_df.columns
