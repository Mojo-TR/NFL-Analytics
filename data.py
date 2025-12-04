import nflreadpy as nfl
import polars as pl
import pandas as pd

pbp = nfl.load_pbp(seasons=2025)

off_pass_eff = (
        pbp.filter(pl.col("play_type") == "pass").group_by("posteam")
        .agg([
            pl.sum("epa").alias("total_passing_epa"),
            pl.count("play_id").alias("total_pass_attempts")
        ])
        .with_columns(
            (pl.col("total_passing_epa") / pl.col("total_pass_attempts")).alias("epa_per_pass")
        )
    )

off_rush_eff  = (
    pbp.filter(pl.col("play_type") == "run").group_by("posteam")
    .agg([
        pl.sum("epa").alias("total_rush_epa"),
        pl.count("play_id").alias("total_rushes")
    ])
    .with_columns([
        (pl.col("total_rush_epa") / pl.col("total_rushes")).alias("epa_per_rush")
    ])
)

off_pass_sr = (
    pbp.filter(pl.col("play_type") == "pass")
       .group_by("posteam")
       .agg(
           pl.mean("success").alias("pass_success_rate")
       )
)

off_run_sr = (
    pbp.filter(pl.col("play_type") == "run")
       .group_by("posteam")
       .agg(
           pl.mean("success").alias("run_success_rate")
       )
)

total_offense = (
    pbp
    .filter(pl.col("yards_gained").is_not_null())
    .group_by("posteam")
    .agg([
        pl.sum("yards_gained").alias("total_off_yards"),
        pl.n_unique("game_id").alias("games_played")
    ])
    .with_columns([
        (pl.col("total_off_yards") / pl.col("games_played")).alias("off_yards_per_game")
    ])
)

scoring_offense = (
    pbp
    .group_by(["posteam", "game_id"])
    .agg([
        pl.max("total_home_score").alias("home_points"),
        pl.max("total_away_score").alias("away_points"),
        pl.first("home_team").alias("home_team"),
        pl.first("away_team").alias("away_team"), 
    ])
    .with_columns([
        pl.when(pl.col("posteam") == pl.col("home_team")).then(pl.col("home_points"))
        .otherwise(pl.col("away_points"))
        .alias("points_scored")
    ])
    .group_by("posteam")
    .agg([
        pl.sum("points_scored").alias("total_points_scored"),
        pl.n_unique("game_id").alias("games_played")
    ])
    .with_columns([
        (pl.col("total_points_scored") / pl.col("games_played")).round(1).alias("scoring_offense_per_game")
    ])
)

scoring_pct = (
    pbp
    .group_by(["posteam", "drive", "game_id"])
    .agg([
        pl.max("drive_ended_with_score").alias("scored")
    ])
    .group_by("posteam")
    .agg([
        (pl.mean("scored") * 100).alias("scoring_pct")
    ])
)

def_passing = (
        pbp.filter(pl.col("play_type") == "pass").group_by("defteam")
        .agg([
            pl.sum("epa").alias("total_passing_epa"),
            pl.count("play_id").alias("total_pass_attempts")
        ])
        .with_columns(
            (pl.col("total_passing_epa") / pl.col("total_pass_attempts")).alias("epa_per_pass")
        )
    )

def_rushing = (
    pbp.filter(pl.col("play_type") == "run").group_by("defteam")
    .agg([
        pl.sum("epa").alias("total_rush_epa"),
        pl.count("play_id").alias("total_rushes")
    ])
    .with_columns([
        (pl.col("total_rush_epa") / pl.col("total_rushes")).alias("epa_per_rush")
    ])
)

total_defense = (
    pbp
    .filter(pl.col("yards_gained").is_not_null())
    .group_by("defteam")
    .agg([
        pl.sum("yards_gained").alias("total_yards_allowed"),
        pl.n_unique("game_id").alias("games_played_def")
    ])
    .with_columns([
        (pl.col("total_yards_allowed") / pl.col("games_played_def")).alias("yards_allowed_per_game")
    ])
)

scoring_defense = (
    pbp
    .group_by(["defteam", "game_id"])
    .agg([
        pl.max("total_home_score").alias("home_points"),
        pl.max("total_away_score").alias("away_points"),
        pl.first("home_team").alias("home_team"),
        pl.first("away_team").alias("away_team"), 
    ])
    .with_columns([
        pl.when(pl.col("defteam") == pl.col("home_team")).then(pl.col("away_points"))
        .otherwise(pl.col("home_points"))
        .alias("points_allowed")
    ])
    .group_by("defteam")
    .agg([
        pl.sum("points_allowed").alias("total_points_allowed"),
        pl.n_unique("game_id").alias("games_played_def")
    ])
    .with_columns([
        (pl.col("total_points_allowed") / pl.col("games_played_def")).round(1).alias("scoring_defense_per_game")
    ])
)

stop_rate = (
    pbp
    .group_by(["defteam", "drive", "game_id"])
    .agg([
        pl.max("drive_ended_with_score").alias("scored")
    ])
    .group_by("defteam")
    .agg([
        ((1 - pl.mean("scored")) * 100).alias("stop_rate")
    ])
)

def_pass_sr = (
    pbp.filter(pl.col("play_type") == "pass")
       .group_by("posteam")
       .agg(
           pl.mean("success").alias("pass_success_rate")
       )
)