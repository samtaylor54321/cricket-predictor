"""
src/parse_cricsheet.py
======================
Parses raw Cricsheet county championship JSON files (bronze layer)
into two clean CSVs (silver layer):

  data/silver/cricsheet/matches.csv  — one row per match
  data/silver/cricsheet/players.csv  — one row per player per innings

Called directly via:
    python src/parse_cricsheet.py

Or imported and called from pipeline.py:
    from src.parse_cricsheet import run
    run()
"""

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Default paths relative to project root
DEFAULT_BRONZE = Path("data/bronze/cricsheet")
DEFAULT_SILVER = Path("data/silver/cricsheet")


# ─────────────────────────────────────────────
# OUTCOME PARSING
# ─────────────────────────────────────────────

def parse_outcome(outcome: dict) -> tuple[str | None, str]:
    """
    Returns (winner, result_type).

    result_type: 'decisive' | 'draw' | 'tie' | 'no_result'
    winner:      winning team name, or None
    """
    if "winner" in outcome:
        return outcome["winner"], "decisive"

    result = outcome.get("result", "").lower()
    if result == "draw":
        return None, "draw"
    if result == "tie":
        return None, "tie"
    return None, "no_result"


# ─────────────────────────────────────────────
# SINGLE MATCH PARSER
# ─────────────────────────────────────────────

def parse_match(filepath: Path) -> dict | None:
    """
    Parse one Cricsheet JSON file into a flat match record.
    Returns None if the file is not a county championship match.
    """
    with open(filepath) as f:
        data = json.load(f)

    info = data.get("info", {})

    event_name = info.get("event", {}).get("name", "").lower()
    if "county championship" not in event_name:
        return None

    teams = info.get("teams", [])
    if len(teams) != 2:
        return None

    dates      = info.get("dates", [])
    match_date = dates[0] if dates else None
    division   = info.get("event", {}).get("group", None)
    season     = info.get("season", None)
    venue      = info.get("venue", None)
    city       = info.get("city", None)

    toss             = info.get("toss", {})
    toss_winner      = toss.get("winner", None)
    toss_decision    = toss.get("decision", None)
    toss_uncontested = toss.get("uncontested", False)

    outcome          = info.get("outcome", {})
    winner, result_type = parse_outcome(outcome)

    # teams[0] is home by convention in Cricsheet county data
    home_team = teams[0]
    away_team = teams[1]

    players  = info.get("players", {})
    registry = info.get("registry", {}).get("people", {})

    home_players = players.get(home_team, [])
    away_players = players.get(away_team, [])

    home_win = None
    if result_type == "decisive":
        home_win = 1 if winner == home_team else 0

    return {
        "match_id":           filepath.stem,
        "match_date":         match_date,
        "season":             season,
        "division":           division,
        "venue":              venue,
        "city":               city,
        "home_team":          home_team,
        "away_team":          away_team,
        "toss_winner":        toss_winner,
        "toss_decision":      toss_decision,
        "toss_uncontested":   toss_uncontested,
        "home_won_toss":      1 if toss_winner == home_team else 0,
        "result_type":        result_type,
        "winner":             winner,
        "home_win":           home_win,
        "home_players":       "|".join(home_players),
        "away_players":       "|".join(away_players),
        "home_player_ids":    "|".join(registry.get(p, "") for p in home_players),
        "away_player_ids":    "|".join(registry.get(p, "") for p in away_players),
    }


# ─────────────────────────────────────────────
# PLAYER PERFORMANCE PARSER
# ─────────────────────────────────────────────

def parse_player_performances(data: dict, match_id: str) -> list[dict]:
    """
    Walk all deliveries in a match and compute per-player batting
    and bowling stats per innings.

    Returns a list of dicts, one per (player, innings) combination.
    """
    registry = data.get("info", {}).get("registry", {}).get("people", {})
    rows = []

    for innings_idx, innings in enumerate(data.get("innings", [])):
        batting_team = innings.get("team")

        batting_runs:    dict[str, int]  = {}
        batting_balls:   dict[str, int]  = {}
        batting_out:     dict[str, bool] = {}
        bowling_runs:    dict[str, int]  = {}
        bowling_balls:   dict[str, int]  = {}
        bowling_wickets: dict[str, int]  = {}

        for over in innings.get("overs", []):
            for delivery in over.get("deliveries", []):
                batter = delivery.get("batter")
                bowler = delivery.get("bowler")
                runs   = delivery.get("runs", {})
                extras = delivery.get("extras", {})

                if batter:
                    batting_runs[batter]  = batting_runs.get(batter, 0)  + runs.get("batter", 0)
                    batting_balls[batter] = batting_balls.get(batter, 0) + 1

                if bowler:
                    is_wide  = "wides"   in extras
                    is_noball = "noballs" in extras
                    bowling_runs[bowler] = bowling_runs.get(bowler, 0) + runs.get("total", 0)
                    if not (is_wide or is_noball):
                        bowling_balls[bowler] = bowling_balls.get(bowler, 0) + 1

                for wicket in delivery.get("wickets", []):
                    kind      = wicket.get("kind", "")
                    dismissed = wicket.get("player_out", batter)
                    batting_out[dismissed] = True
                    if kind not in ("run out", "obstructing the field", "retired hurt"):
                        bowling_wickets[bowler] = bowling_wickets.get(bowler, 0) + 1

        all_players = set(batting_runs) | set(bowling_balls)
        for player in all_players:
            rows.append({
                "match_id":      match_id,
                "innings":       innings_idx + 1,
                "batting_team":  batting_team,
                "player_name":   player,
                "player_id":     registry.get(player, ""),
                "runs_scored":   batting_runs.get(player, 0),
                "balls_faced":   batting_balls.get(player, 0),
                "dismissed":     batting_out.get(player, False),
                "wickets_taken": bowling_wickets.get(player, 0),
                "balls_bowled":  bowling_balls.get(player, 0),
                "runs_conceded": bowling_runs.get(player, 0),
            })

    return rows


# ─────────────────────────────────────────────
# BATCH PIPELINE
# ─────────────────────────────────────────────

def run(
    bronze_dir: Path = DEFAULT_BRONZE,
    silver_dir: Path = DEFAULT_SILVER,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse all JSON files in bronze_dir and write CSVs to silver_dir.
    Returns (matches_df, players_df).
    """
    bronze_dir = Path(bronze_dir)
    silver_dir = Path(silver_dir)
    silver_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(bronze_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {bronze_dir}")

    logger.info(f"Found {len(json_files)} JSON files in {bronze_dir}")

    match_rows  = []
    player_rows = []
    skipped     = 0
    not_county  = 0

    for i, filepath in enumerate(json_files):
        if i % 100 == 0:
            logger.info(f"  Processing file {i + 1}/{len(json_files)}")

        try:
            with open(filepath) as f:
                data = json.load(f)

            match = parse_match(filepath)
            if match is None:
                not_county += 1
                continue

            match_rows.append(match)
            player_rows.extend(
                parse_player_performances(data, match["match_id"])
            )

        except Exception as e:
            logger.warning(f"Could not parse {filepath.name}: {e}")
            skipped += 1

    matches_df = pd.DataFrame(match_rows).sort_values("match_date").reset_index(drop=True)
    players_df = pd.DataFrame(player_rows)

    matches_path = silver_dir / "matches.csv"
    players_path = silver_dir / "players.csv"
    matches_df.to_csv(matches_path, index=False)
    players_df.to_csv(players_path, index=False)

    decisive = (matches_df["result_type"] == "decisive").sum()
    draws    = (matches_df["result_type"] == "draw").sum()
    other    = len(matches_df) - decisive - draws

    logger.info(f"Parse complete.")
    logger.info(f"  Files processed : {len(json_files)}")
    logger.info(f"  Not county      : {not_county}")
    logger.info(f"  Errors skipped  : {skipped}")
    logger.info(f"  Matches parsed  : {len(matches_df)}")
    logger.info(f"    Decisive      : {decisive}")
    logger.info(f"    Draws         : {draws}")
    logger.info(f"    Other         : {other}")
    logger.info(f"  Player rows     : {len(players_df)}")
    logger.info(f"  Written to      : {silver_dir}")

    return matches_df, players_df


# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Parse Cricsheet county championship JSON files")
    parser.add_argument("--bronze", type=Path, default=DEFAULT_BRONZE)
    parser.add_argument("--silver", type=Path, default=DEFAULT_SILVER)
    args = parser.parse_args()

    run(bronze_dir=args.bronze, silver_dir=args.silver)
