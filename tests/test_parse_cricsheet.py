"""
tests/test_parse_cricsheet.py
=============================
Unit tests for src/parse_cricsheet.py

Run with:
    pytest tests/test_parse_cricsheet.py -v
"""

import json
import pytest
from pathlib import Path

from src.parse_cricsheet import (
    parse_outcome,
    parse_match,
    parse_player_performances,
    run,
)


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture
def sample_match_data():
    """Minimal valid county championship match — home win."""
    return {
        "meta": {"data_version": "1.1.0"},
        "info": {
            "balls_per_over": 6,
            "city": "Bristol",
            "dates": ["2022-04-07", "2022-04-08", "2022-04-09", "2022-04-10"],
            "event": {"name": "LV= Insurance County Championship", "group": 1},
            "gender": "male",
            "match_type": "MDM",
            "outcome": {"winner": "Gloucestershire", "by": {"wickets": 5}},
            "players": {
                "Gloucestershire": ["GH Roderick", "CDJ Dent", "James Bracey",
                                    "GT Hankins", "DA Payne", "CN Miles",
                                    "JMR Taylor", "MD Taylor", "J Shaw",
                                    "DA Wheeldon", "BG Charlesworth"],
                "Yorkshire":      ["AD Hales", "JA Tattersall", "HC Brook",
                                   "JE Root", "GS Ballance", "TT Bresnan",
                                   "JA Leaning", "BO Coad", "MD Fisher",
                                   "JW Shutt", "B Mike"],
            },
            "registry": {
                "people": {
                    "GH Roderick": "7081848a",
                    "CDJ Dent":    "10435ca2",
                    "James Bracey":"5bfdc77a",
                    "AD Hales":    "abc12345",
                    "JE Root":     "xyz99999",
                }
            },
            "season": "2022",
            "teams": ["Gloucestershire", "Yorkshire"],
            "toss": {"winner": "Gloucestershire", "decision": "bat"},
            "venue": "County Ground, Bristol",
        },
        "innings": [
            {
                "team": "Gloucestershire",
                "overs": [
                    {
                        "over": 0,
                        "deliveries": [
                            {
                                "batter": "GH Roderick",
                                "bowler": "BO Coad",
                                "non_striker": "CDJ Dent",
                                "runs": {"batter": 4, "extras": 0, "total": 4},
                            },
                            {
                                "batter": "GH Roderick",
                                "bowler": "BO Coad",
                                "non_striker": "CDJ Dent",
                                "runs": {"batter": 0, "extras": 0, "total": 0},
                                "wickets": [
                                    {"kind": "caught", "player_out": "GH Roderick",
                                     "fielders": [{"name": "AD Hales"}]}
                                ],
                            },
                        ],
                    }
                ],
            },
            {
                "team": "Yorkshire",
                "overs": [
                    {
                        "over": 0,
                        "deliveries": [
                            {
                                "batter": "AD Hales",
                                "bowler": "DA Payne",
                                "non_striker": "JE Root",
                                "runs": {"batter": 0, "extras": 0, "total": 0},
                                "wickets": [
                                    {"kind": "lbw", "player_out": "AD Hales"}
                                ],
                            },
                            {
                                "batter": "JE Root",
                                "bowler": "DA Payne",
                                "non_striker": "JA Tattersall",
                                "extras": {"wides": 1},
                                "runs": {"batter": 0, "extras": 1, "total": 1},
                            },
                        ],
                    }
                ],
            },
        ],
    }


@pytest.fixture
def draw_match_data(sample_match_data):
    """Variant where the match ends in a draw."""
    data = json.loads(json.dumps(sample_match_data))  # deep copy
    data["info"]["outcome"] = {"result": "draw"}
    return data


@pytest.fixture
def non_county_data(sample_match_data):
    """Variant that is not a county championship match."""
    data = json.loads(json.dumps(sample_match_data))
    data["info"]["event"]["name"] = "T20 Blast"
    return data


@pytest.fixture
def match_json_file(tmp_path, sample_match_data):
    """Write sample match to a temp JSON file and return the path."""
    filepath = tmp_path / "test_match_123.json"
    filepath.write_text(json.dumps(sample_match_data))
    return filepath


@pytest.fixture
def draw_json_file(tmp_path, draw_match_data):
    filepath = tmp_path / "test_match_draw.json"
    filepath.write_text(json.dumps(draw_match_data))
    return filepath


@pytest.fixture
def non_county_json_file(tmp_path, non_county_data):
    filepath = tmp_path / "test_match_t20.json"
    filepath.write_text(json.dumps(non_county_data))
    return filepath


# ─────────────────────────────────────────────
# parse_outcome
# ─────────────────────────────────────────────

class TestParseOutcome:
    def test_winner_by_wickets(self):
        winner, result = parse_outcome({"winner": "Surrey", "by": {"wickets": 6}})
        assert winner == "Surrey"
        assert result == "decisive"

    def test_winner_by_runs(self):
        winner, result = parse_outcome({"winner": "Kent", "by": {"runs": 45}})
        assert winner == "Kent"
        assert result == "decisive"

    def test_winner_by_innings(self):
        winner, result = parse_outcome({"winner": "Essex", "by": {"innings": 1, "runs": 34}})
        assert winner == "Essex"
        assert result == "decisive"

    def test_draw(self):
        winner, result = parse_outcome({"result": "draw"})
        assert winner is None
        assert result == "draw"

    def test_tie(self):
        winner, result = parse_outcome({"result": "tie"})
        assert winner is None
        assert result == "tie"

    def test_no_result(self):
        winner, result = parse_outcome({"result": "no result"})
        assert winner is None
        assert result == "no_result"

    def test_empty_outcome(self):
        winner, result = parse_outcome({})
        assert winner is None
        assert result == "no_result"


# ─────────────────────────────────────────────
# parse_match
# ─────────────────────────────────────────────

class TestParseMatch:
    def test_returns_none_for_non_county(self, non_county_json_file):
        assert parse_match(non_county_json_file) is None

    def test_match_id_is_filename_stem(self, match_json_file):
        result = parse_match(match_json_file)
        assert result["match_id"] == "test_match_123"

    def test_home_team_is_first_team(self, match_json_file):
        result = parse_match(match_json_file)
        assert result["home_team"] == "Gloucestershire"
        assert result["away_team"] == "Yorkshire"

    def test_home_win_is_one_for_home_winner(self, match_json_file):
        result = parse_match(match_json_file)
        assert result["home_win"] == 1

    def test_home_win_is_zero_for_away_winner(self, tmp_path, sample_match_data):
        data = json.loads(json.dumps(sample_match_data))
        data["info"]["outcome"] = {"winner": "Yorkshire"}
        filepath = tmp_path / "away_win.json"
        filepath.write_text(json.dumps(data))
        result = parse_match(filepath)
        assert result["home_win"] == 0

    def test_home_win_is_none_for_draw(self, draw_json_file):
        result = parse_match(draw_json_file)
        assert result["home_win"] is None
        assert result["result_type"] == "draw"

    def test_result_type_decisive(self, match_json_file):
        result = parse_match(match_json_file)
        assert result["result_type"] == "decisive"

    def test_division_extracted(self, match_json_file):
        result = parse_match(match_json_file)
        assert result["division"] == 1

    def test_season_extracted(self, match_json_file):
        result = parse_match(match_json_file)
        assert result["season"] == "2022"

    def test_toss_fields(self, match_json_file):
        result = parse_match(match_json_file)
        assert result["toss_winner"] == "Gloucestershire"
        assert result["toss_decision"] == "bat"
        assert result["home_won_toss"] == 1
        assert result["toss_uncontested"] is False

    def test_uncontested_toss(self, tmp_path, sample_match_data):
        data = json.loads(json.dumps(sample_match_data))
        data["info"]["toss"]["uncontested"] = True
        filepath = tmp_path / "uncontested.json"
        filepath.write_text(json.dumps(data))
        result = parse_match(filepath)
        assert result["toss_uncontested"] is True

    def test_players_stored_as_pipe_separated(self, match_json_file):
        result = parse_match(match_json_file)
        home_players = result["home_players"].split("|")
        assert "GH Roderick" in home_players
        assert "CDJ Dent" in home_players
        assert len(home_players) == 11

    def test_player_ids_populated_for_known_players(self, match_json_file):
        result = parse_match(match_json_file)
        ids = result["home_player_ids"].split("|")
        # First player GH Roderick has a registry entry
        assert ids[0] == "7081848a"

    def test_player_ids_empty_string_for_unknown_players(self, match_json_file):
        result = parse_match(match_json_file)
        # Not all Yorkshire players are in the registry fixture
        away_ids = result["away_player_ids"].split("|")
        assert "" in away_ids  # some should be missing

    def test_venue_extracted(self, match_json_file):
        result = parse_match(match_json_file)
        assert result["venue"] == "County Ground, Bristol"


# ─────────────────────────────────────────────
# parse_player_performances
# ─────────────────────────────────────────────

class TestParsePlayerPerformances:
    def test_returns_list(self, sample_match_data):
        rows = parse_player_performances(sample_match_data, "test_123")
        assert isinstance(rows, list)
        assert len(rows) > 0

    def test_runs_scored_correctly(self, sample_match_data):
        rows = parse_player_performances(sample_match_data, "test_123")
        roderick = next(r for r in rows if r["player_name"] == "GH Roderick")
        assert roderick["runs_scored"] == 4

    def test_batter_marked_dismissed(self, sample_match_data):
        rows = parse_player_performances(sample_match_data, "test_123")
        roderick = next(r for r in rows if r["player_name"] == "GH Roderick")
        assert roderick["dismissed"] is True

    def test_wicket_credited_to_bowler(self, sample_match_data):
        rows = parse_player_performances(sample_match_data, "test_123")
        # BO Coad took the wicket of GH Roderick in innings 1
        coad = next(
            r for r in rows
            if r["player_name"] == "BO Coad" and r["innings"] == 1
        )
        assert coad["wickets_taken"] == 1

    def test_wicket_credited_to_bowler_innings_2(self, sample_match_data):
        rows = parse_player_performances(sample_match_data, "test_123")
        payne = next(
            r for r in rows
            if r["player_name"] == "DA Payne" and r["innings"] == 2
        )
        assert payne["wickets_taken"] == 1

    def test_wide_not_counted_as_legal_ball(self, sample_match_data):
        rows = parse_player_performances(sample_match_data, "test_123")
        # DA Payne bowled 1 legal ball and 1 wide in innings 2
        payne = next(
            r for r in rows
            if r["player_name"] == "DA Payne" and r["innings"] == 2
        )
        assert payne["balls_bowled"] == 1

    def test_wide_runs_counted_in_runs_conceded(self, sample_match_data):
        rows = parse_player_performances(sample_match_data, "test_123")
        payne = next(
            r for r in rows
            if r["player_name"] == "DA Payne" and r["innings"] == 2
        )
        assert payne["runs_conceded"] == 1

    def test_run_out_not_credited_to_bowler(self, tmp_path, sample_match_data):
        data = json.loads(json.dumps(sample_match_data))
        data["innings"][0]["overs"][0]["deliveries"][1]["wickets"] = [
            {"kind": "run out", "player_out": "GH Roderick"}
        ]
        rows = parse_player_performances(data, "test_123")
        coad = next(
            r for r in rows
            if r["player_name"] == "BO Coad" and r["innings"] == 1
        )
        assert coad["wickets_taken"] == 0

    def test_innings_number_set_correctly(self, sample_match_data):
        rows = parse_player_performances(sample_match_data, "test_123")
        innings_nums = {r["innings"] for r in rows}
        assert innings_nums == {1, 2}

    def test_player_id_populated_from_registry(self, sample_match_data):
        rows = parse_player_performances(sample_match_data, "test_123")
        roderick = next(r for r in rows if r["player_name"] == "GH Roderick")
        assert roderick["player_id"] == "7081848a"

    def test_match_id_set_on_all_rows(self, sample_match_data):
        rows = parse_player_performances(sample_match_data, "test_abc")
        assert all(r["match_id"] == "test_abc" for r in rows)


# ─────────────────────────────────────────────
# run (integration)
# ─────────────────────────────────────────────

class TestRun:
    def test_creates_output_files(self, tmp_path, sample_match_data):
        bronze = tmp_path / "bronze"
        silver = tmp_path / "silver"
        bronze.mkdir()
        (bronze / "match_001.json").write_text(json.dumps(sample_match_data))

        run(bronze_dir=bronze, silver_dir=silver)

        assert (silver / "matches.csv").exists()
        assert (silver / "players.csv").exists()

    def test_skips_non_county_files(self, tmp_path, sample_match_data, non_county_data):
        bronze = tmp_path / "bronze"
        bronze.mkdir()
        (bronze / "county.json").write_text(json.dumps(sample_match_data))
        (bronze / "t20.json").write_text(json.dumps(non_county_data))

        matches_df, _ = run(
            bronze_dir=bronze,
            silver_dir=tmp_path / "silver",
        )
        assert len(matches_df) == 1

    def test_matches_df_has_expected_columns(self, tmp_path, sample_match_data):
        bronze = tmp_path / "bronze"
        bronze.mkdir()
        (bronze / "m.json").write_text(json.dumps(sample_match_data))

        matches_df, _ = run(bronze_dir=bronze, silver_dir=tmp_path / "silver")

        expected_cols = {
            "match_id", "match_date", "season", "division",
            "home_team", "away_team", "result_type", "home_win",
            "toss_winner", "toss_decision", "home_won_toss",
        }
        assert expected_cols.issubset(set(matches_df.columns))

    def test_players_df_has_expected_columns(self, tmp_path, sample_match_data):
        bronze = tmp_path / "bronze"
        bronze.mkdir()
        (bronze / "m.json").write_text(json.dumps(sample_match_data))

        _, players_df = run(bronze_dir=bronze, silver_dir=tmp_path / "silver")

        expected_cols = {
            "match_id", "innings", "batting_team", "player_name",
            "player_id", "runs_scored", "balls_faced", "dismissed",
            "wickets_taken", "balls_bowled", "runs_conceded",
        }
        assert expected_cols.issubset(set(players_df.columns))

    def test_raises_if_no_json_files(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            run(bronze_dir=empty, silver_dir=tmp_path / "silver")

    def test_decisive_draw_split(self, tmp_path, sample_match_data, draw_match_data):
        bronze = tmp_path / "bronze"
        bronze.mkdir()
        (bronze / "decisive.json").write_text(json.dumps(sample_match_data))
        (bronze / "draw.json").write_text(json.dumps(draw_match_data))

        matches_df, _ = run(bronze_dir=bronze, silver_dir=tmp_path / "silver")

        assert (matches_df["result_type"] == "decisive").sum() == 1
        assert (matches_df["result_type"] == "draw").sum() == 1
        assert matches_df.loc[matches_df["result_type"] == "decisive", "home_win"].iloc[0] == 1
        assert matches_df.loc[matches_df["result_type"] == "draw", "home_win"].isna().all()
