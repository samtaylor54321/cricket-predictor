"""
src/feature_engineering.py
===========================
Reads silver-layer CSVs and produces a gold-layer features CSV
ready for model training and inference.

Silver (input):
  data/silver/cricsheet/matches.csv
  data/silver/cricsheet/players.csv

Gold (output):
  data/gold/features.csv

Each row in features.csv corresponds to one match and contains:
  - match metadata (id, date, season, division, venue)
  - Elo rating difference at the time of the match (no leakage)
  - Home advantage flag
  - Toss features
  - XI strength scores (batting and bowling) for each side
  - Rolling form in decisive matches only
  - Venue home win rate
  - target: home_win (1/0, blank for draws — excluded from model training)

Called via:
    python src/feature_engineering.py

Or from pipeline.py:
    from src.feature_engineering import run
    run()
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_SILVER = Path("data/silver/cricsheet")
DEFAULT_GOLD = Path("data/gold")

# How many recent decisive matches to use for rolling form
FORM_WINDOW = 5

# How many recent innings to use for player form averages
PLAYER_FORM_INNINGS = 10

# Minimum innings before we trust a player's own average
# (below this we blend with the divisional average)
PLAYER_MIN_INNINGS = 5

# Elo hyperparameters
ELO_DEFAULT = 1500
ELO_K = 32
ELO_HOME_BONUS = 50


# ─────────────────────────────────────────────
# ELO ENGINE
# ─────────────────────────────────────────────


class EloTracker:
    """
    Tracks team Elo ratings across matches.
    Ratings are updated after each decisive match only —
    draws don't update ratings since they're not in our target.
    """

    def __init__(self):
        self.ratings: dict[str, float] = {}

    def get(self, team: str) -> float:
        return self.ratings.get(team, ELO_DEFAULT)

    def expected(self, rating_a: float, rating_b: float) -> float:
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update(self, winner: str, loser: str, home_team: str | None = None):
        """Update ratings after a decisive result."""
        ra = self.get(winner)
        rb = self.get(loser)

        home_bonus = 0
        if home_team == winner:
            home_bonus = ELO_HOME_BONUS
        elif home_team == loser:
            home_bonus = -ELO_HOME_BONUS

        ea = self.expected(ra + home_bonus, rb)
        self.ratings[winner] = ra + ELO_K * (1.0 - ea)
        self.ratings[loser] = rb + ELO_K * (0.0 - (1 - ea))

    def snapshot(self) -> dict[str, float]:
        return dict(self.ratings)


# ─────────────────────────────────────────────
# PLAYER FORM
# ─────────────────────────────────────────────


def build_player_averages(
    players_df: pd.DataFrame,
    match_dates: pd.Series,
) -> pd.DataFrame:
    """
    For each (player_id, match_id) pair, compute the player's
    rolling batting and bowling averages over their last
    PLAYER_FORM_INNINGS innings *before* that match.

    Returns a DataFrame indexed by (player_id, match_id) with:
      - batting_avg:  runs per dismissal over recent innings
      - bowling_avg:  runs conceded per wicket over recent innings
      - innings_count: number of innings in the rolling window
    """
    # Attach match dates to player rows
    df = players_df.merge(
        match_dates.rename("match_date"),
        on="match_id",
        how="left",
    ).sort_values(["player_id", "match_date", "innings"])

    records = []

    for player_id, group in df.groupby("player_id"):
        group = group.reset_index(drop=True)

        for i, row in group.iterrows():
            # Only use innings strictly before this match
            prior = group[group["match_date"] < row["match_date"]].tail(
                PLAYER_FORM_INNINGS
            )

            if len(prior) == 0:
                batting_avg = np.nan
                bowling_avg = np.nan
                innings_count = 0
            else:
                # Batting average: total runs / total dismissals (min 1 to avoid div/0)
                total_runs = prior["runs_scored"].sum()
                total_dismissals = prior["dismissed"].sum()
                batting_avg = total_runs / max(total_dismissals, 1)

                # Bowling average: runs conceded / wickets taken (nan if no wickets)
                total_wkts = prior["wickets_taken"].sum()
                total_conceded = prior["runs_conceded"].sum()
                bowling_avg = total_conceded / total_wkts if total_wkts > 0 else np.nan

                innings_count = len(prior)

            records.append(
                {
                    "player_id": player_id,
                    "match_id": row["match_id"],
                    "batting_avg": batting_avg,
                    "bowling_avg": bowling_avg,
                    "innings_count": innings_count,
                }
            )

    return pd.DataFrame(records)


def compute_xi_strength(
    player_ids: list[str],
    match_id: str,
    player_avgs: pd.DataFrame,
    division_batting_avg: float,
    division_bowling_avg: float,
) -> dict:
    """
    Given an XI (list of player IDs) and a match, compute aggregate
    batting and bowling strength scores.

    Players with fewer than PLAYER_MIN_INNINGS are shrunk towards
    the divisional average proportionally to how many innings they have.
    """
    batting_scores = []
    bowling_scores = []

    match_avgs = player_avgs[player_avgs["match_id"] == match_id].set_index("player_id")

    for pid in player_ids:
        if not pid:
            # Unknown player — use divisional average
            batting_scores.append(division_batting_avg)
            bowling_scores.append(division_bowling_avg)
            continue

        if pid not in match_avgs.index:
            batting_scores.append(division_batting_avg)
            bowling_scores.append(division_bowling_avg)
            continue

        row = match_avgs.loc[pid]
        # loc can return a Series (multiple rows) if player appears twice — take first
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        n = int(row["innings_count"])
        weight = min(n / PLAYER_MIN_INNINGS, 1.0)  # 0→1 as innings accumulate

        bat_raw = (
            row["batting_avg"]
            if not np.isnan(row["batting_avg"])
            else division_batting_avg
        )
        bowl_raw = (
            row["bowling_avg"]
            if not np.isnan(row["bowling_avg"])
            else division_bowling_avg
        )

        # Shrink towards divisional average when sample is thin
        batting_scores.append(weight * bat_raw + (1 - weight) * division_batting_avg)
        bowling_scores.append(weight * bowl_raw + (1 - weight) * division_bowling_avg)

    return {
        "batting_strength": np.mean(batting_scores[:6]),  # top 6 batters
        "bowling_strength": np.mean(bowling_scores[-5:]),  # last 5 (bowlers bat lower)
    }


# ─────────────────────────────────────────────
# VENUE STATS
# ─────────────────────────────────────────────


def build_venue_stats(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each match, compute the historical home win rate at that venue
    using only matches played *before* this one.

    Returns a Series indexed by match_id.
    """
    decisive = matches_df[matches_df["result_type"] == "decisive"].copy()
    decisive = decisive.sort_values("match_date").reset_index(drop=True)

    venue_win_rate = {}

    for i, row in decisive.iterrows():
        venue = row["venue"]
        prior = decisive[
            (decisive["venue"] == venue) & (decisive["match_date"] < row["match_date"])
        ]
        if len(prior) == 0:
            venue_win_rate[row["match_id"]] = np.nan
        else:
            venue_win_rate[row["match_id"]] = prior["home_win"].mean()

    return pd.Series(venue_win_rate, name="venue_home_win_rate")


# ─────────────────────────────────────────────
# ROLLING FORM
# ─────────────────────────────────────────────


def build_team_form(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each team in each match, compute their win rate over their
    last FORM_WINDOW decisive matches before this match.

    Returns a DataFrame with columns: match_id, home_form, away_form.
    """
    decisive = matches_df[matches_df["result_type"] == "decisive"].copy()
    decisive = decisive.sort_values("match_date").reset_index(drop=True)

    # Build a flat record of (team, date, won) for every decisive match
    team_results: list[dict] = []
    for _, row in decisive.iterrows():
        team_results.append(
            {
                "team": row["home_team"],
                "match_date": row["match_date"],
                "won": float(row["home_win"]),
            }
        )
        team_results.append(
            {
                "team": row["away_team"],
                "match_date": row["match_date"],
                "won": 1.0 - float(row["home_win"]),
            }
        )
    tr = pd.DataFrame(team_results).sort_values(["team", "match_date"])

    form_records = []
    for _, row in decisive.iterrows():
        home_prior = tr[
            (tr["team"] == row["home_team"]) & (tr["match_date"] < row["match_date"])
        ].tail(FORM_WINDOW)

        away_prior = tr[
            (tr["team"] == row["away_team"]) & (tr["match_date"] < row["match_date"])
        ].tail(FORM_WINDOW)

        form_records.append(
            {
                "match_id": row["match_id"],
                "home_form": home_prior["won"].mean()
                if len(home_prior) > 0
                else np.nan,
                "away_form": away_prior["won"].mean()
                if len(away_prior) > 0
                else np.nan,
            }
        )

    return pd.DataFrame(form_records)


# ─────────────────────────────────────────────
# MAIN FEATURE BUILD
# ─────────────────────────────────────────────


def build_features(
    matches_df: pd.DataFrame,
    players_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine all features into a single gold-layer DataFrame.
    Only decisive matches are included (draws have no target variable).
    """
    matches_df = matches_df.copy()
    matches_df["match_date"] = pd.to_datetime(matches_df["match_date"])
    matches_df = matches_df.sort_values("match_date").reset_index(drop=True)

    decisive = matches_df[matches_df["result_type"] == "decisive"].copy()

    # ── Elo ratings ──────────────────────────────────────────────────
    elo = EloTracker()
    elo_diffs = {}

    for _, row in matches_df.iterrows():
        mid = row["match_id"]

        # Snapshot ratings BEFORE this match
        if row["result_type"] == "decisive":
            elo_diffs[mid] = elo.get(row["home_team"]) - elo.get(row["away_team"])
            elo.update(
                winner=row["winner"],
                loser=row["away_team"]
                if row["winner"] == row["home_team"]
                else row["home_team"],
                home_team=row["home_team"],
            )

    decisive["elo_diff"] = decisive["match_id"].map(elo_diffs)

    # ── Venue stats ───────────────────────────────────────────────────
    venue_rates = build_venue_stats(matches_df)
    decisive = decisive.join(venue_rates, on="match_id")

    # ── Rolling form ──────────────────────────────────────────────────
    form_df = build_team_form(matches_df)
    decisive = decisive.merge(form_df, on="match_id", how="left")

    # ── Player form & XI strength ─────────────────────────────────────
    match_dates = matches_df.set_index("match_id")["match_date"]
    player_avgs = build_player_averages(players_df, match_dates)

    # Divisional averages as fallback for sparse players
    div_batting_avg = players_df["runs_scored"].sum() / max(
        players_df["dismissed"].sum(), 1
    )
    div_bowling_avg = players_df["runs_conceded"].sum() / max(
        players_df["wickets_taken"].sum(), 1
    )

    xi_rows = []
    for _, row in decisive.iterrows():
        home_ids = [p for p in row["home_player_ids"].split("|") if p]
        away_ids = [p for p in row["away_player_ids"].split("|") if p]

        home_strength = compute_xi_strength(
            home_ids,
            row["match_id"],
            player_avgs,
            div_batting_avg,
            div_bowling_avg,
        )
        away_strength = compute_xi_strength(
            away_ids,
            row["match_id"],
            player_avgs,
            div_batting_avg,
            div_bowling_avg,
        )

        xi_rows.append(
            {
                "match_id": row["match_id"],
                "home_batting_strength": home_strength["batting_strength"],
                "home_bowling_strength": home_strength["bowling_strength"],
                "away_batting_strength": away_strength["batting_strength"],
                "away_bowling_strength": away_strength["bowling_strength"],
                "batting_strength_diff": home_strength["batting_strength"]
                - away_strength["batting_strength"],
                "bowling_strength_diff": home_strength["bowling_strength"]
                - away_strength["bowling_strength"],
            }
        )

    xi_df = pd.DataFrame(xi_rows)
    decisive = decisive.merge(xi_df, on="match_id", how="left")

    # ── Assemble final feature set ────────────────────────────────────
    feature_cols = [
        "match_id",
        "match_date",
        "season",
        "division",
        "home_team",
        "away_team",
        "venue",
        # Features
        "elo_diff",
        "home_won_toss",
        "home_form",
        "away_form",
        "venue_home_win_rate",
        "batting_strength_diff",
        "bowling_strength_diff",
        # Target
        "home_win",
    ]

    return decisive[feature_cols].reset_index(drop=True)


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────


def run(
    silver_dir: Path = DEFAULT_SILVER,
    gold_dir: Path = DEFAULT_GOLD,
) -> pd.DataFrame:
    """
    Load silver CSVs, build features, write gold CSV.
    Returns the features DataFrame.
    """
    silver_dir = Path(silver_dir)
    gold_dir = Path(gold_dir)
    gold_dir.mkdir(parents=True, exist_ok=True)

    matches_path = silver_dir / "matches.csv"
    players_path = silver_dir / "players.csv"

    if not matches_path.exists():
        raise FileNotFoundError(
            f"matches.csv not found in {silver_dir} — run parse step first"
        )
    if not players_path.exists():
        raise FileNotFoundError(
            f"players.csv not found in {silver_dir} — run parse step first"
        )

    logger.info(f"Loading silver data from {silver_dir}")
    matches_df = pd.read_csv(matches_path)
    players_df = pd.read_csv(players_path)

    logger.info(f"  Matches loaded : {len(matches_df)}")
    logger.info(f"  Player rows    : {len(players_df)}")

    logger.info("Building features...")
    features_df = build_features(matches_df, players_df)

    out_path = gold_dir / "features.csv"
    features_df.to_csv(out_path, index=False)

    complete = features_df["home_win"].notna().sum()
    logger.info(f"Feature build complete.")
    logger.info(f"  Feature rows   : {len(features_df)}")
    logger.info(f"  With target    : {complete}")
    logger.info(f"  Written to     : {out_path}")

    return features_df


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Build model features from silver layer"
    )
    parser.add_argument("--silver", type=Path, default=DEFAULT_SILVER)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    args = parser.parse_args()

    run(silver_dir=args.silver, gold_dir=args.gold)
