# County Championship Betting Model

A value bet finder for the County Championship, targeting Bet365's draw-no-bet markets. The model estimates the probability that the home team wins a decisive match, then compares that against the bookmaker's implied probability to identify edges.

## Background

The core hypothesis is that County Championship markets are less efficiently priced than international cricket. Thinner betting volume, less bookmaker analytical resource, and 18 teams producing a high volume of matches all contribute to a market where a well-constructed model can find consistent edges — particularly in mid-table Division 1 fixtures and across Division 2.

Because Bet365 offer draw-no-bet only, draws are excluded from the model entirely. This simplifies the problem to a binary classification: given that this match produces a result, does the home team win?

---

## Project structure

```
cricket-model/
│
├── main.py                  # Top-level pipeline runner
│
├── src/
│   ├── __init__.py
│   ├── parse_cricsheet.py       # Bronze → silver: parse raw JSON files
│   ├── feature_engineering.py  # Silver → gold: build model features  [upcoming]
│   └── model.py                # Train model and find value bets       [upcoming]
│
├── data/
│   ├── bronze/
│   │   └── cricsheet/          # Raw Cricsheet JSON files (one per match)
│   └── silver/
│       └── cricsheet/
│           ├── matches.csv     # One row per match
│           └── players.csv     # One row per player per innings
│
└── tests/
    ├── __init__.py
    └── test_parse_cricsheet.py
```

---

## Data

Match data comes from [Cricsheet](https://cricsheet.org/downloads/), which provides free ball-by-ball JSON files for every County Championship match since 2016. Download the County Championship zip and unzip the contents into `data/bronze/cricsheet/`.

The pipeline produces two silver-layer CSVs:

**matches.csv** — one row per match with the following key fields:

| column                                | description                                                    |
| ------------------------------------- | -------------------------------------------------------------- |
| `match_id`                            | Cricsheet file stem (e.g. `947089`)                            |
| `match_date`                          | First day of the match                                         |
| `season`                              | Season year                                                    |
| `division`                            | 1 or 2                                                         |
| `home_team` / `away_team`             | Team names                                                     |
| `result_type`                         | `decisive`, `draw`, `tie`, or `no_result`                      |
| `home_win`                            | **Target variable** — 1 (home win), 0 (away win), blank (draw) |
| `toss_winner` / `toss_decision`       | Toss result                                                    |
| `toss_uncontested`                    | True if away team chose to field without a toss                |
| `home_players` / `away_players`       | Pipe-separated XI in batting order                             |
| `home_player_ids` / `away_player_ids` | Pipe-separated Cricsheet registry IDs                          |

**players.csv** — one row per player per innings:

| column          | description                                         |
| --------------- | --------------------------------------------------- |
| `match_id`      | Links to matches.csv                                |
| `innings`       | 1–4                                                 |
| `player_id`     | Stable Cricsheet registry ID                        |
| `runs_scored`   | Batter runs (excluding extras)                      |
| `balls_faced`   | Balls faced by batter                               |
| `dismissed`     | True if the batter was out                          |
| `wickets_taken` | Wickets credited to bowler (excludes run outs)      |
| `balls_bowled`  | Legal deliveries only (wides and no-balls excluded) |
| `runs_conceded` | Total runs conceded (including extras)              |

---

## Setup

```bash
# Clone and install dependencies
pip install pandas scikit-learn xgboost pytest

# Place Cricsheet JSON files in the bronze layer
unzip county_championship.zip -d data/bronze/cricsheet/
```

---

## Running the pipeline

```bash
# Full pipeline (all steps)
python pipeline.py

# Parse step only
python pipeline.py --step parse

# Override default data paths
python pipeline.py --bronze path/to/bronze --silver path/to/silver
```

---

## Running the tests

```bash
pytest tests/ -v
```

Tests use temporary directories and do not touch the `data/` folder.

---

## Modelling approach

### Target variable

Binary classification on decisive matches only. Draws are excluded from the training set — Bet365's draw-no-bet structure means they are irrelevant to the P&L.

### Features (planned)

- Elo rating difference between home and away team, updated after every decisive result
- Home advantage factor
- Aggregate batting and bowling form of the announced XI
- Venue-specific home win rate
- Recent form in decisive matches only (rolling window, draws excluded)

### Player form

When XIs are announced, player averages over recent first-class appearances are aggregated into a single team strength score per side. Players with fewer than a threshold number of appearances fall back to the divisional average for their role, keeping the model scoreable for debuts and call-ups.

### Value identification

A bet has value when:

```
model_probability > 1 / bookmaker_decimal_odds
```

Stake sizing uses fractional Kelly criterion (quarter Kelly by default) to account for model uncertainty.

---

## Design decisions

**Draw-no-bet simplifies the problem.** Rather than modelling a three-way W/D/L distribution, the model only needs to estimate p(home win | result). This is a cleaner binary problem that trains well on the available data volume.

**Elo as the foundation.** Team-level Elo ratings are simple, interpretable, and self-correcting. They handle the fact that county squads evolve across a season better than static reputation-based priors.

**Coach selection as a signal.** Rather than independently modelling player roles, the announced XI is treated as the coach's forward-looking estimate of the best available combination. Player form is aggregated from that XI rather than trying to predict selection.

**Venue draw rates are noted but not modelled directly.** In a draw-no-bet market, the draw rate at a given venue affects bet volume (more draws = fewer settled bets) but not the direction of the model prediction. It is tracked for informational purposes.
