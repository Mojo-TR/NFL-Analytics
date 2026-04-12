"""This script calculates offensive and defensive metrics for NFL teams based on play-by-play data, 
grades the teams on a standardized scale, ranks them, and generates visualizations of the rankings. 
It also saves the rankings to CSV files and sends an email with the generated figures 
as attachments."""

import base64
import math
import os
import smtplib
from email.message import EmailMessage
from io import BytesIO
from pathlib import Path
import nflreadpy as nfl
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc
import polars as pl
import pandas as pd
import requests
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

pbp = nfl.load_pbp()
pbp = pbp.filter(pl.col("season") == pbp["season"].unique()[0]).to_pandas()
pbp = pbp[(pbp["play_type"] != "no_play") & (pbp["season_type"] == "REG")]
teams = nfl.load_teams().to_pandas()

SEASON = pbp["season"].unique()[0]
WEEK = max(pbp["week"])
OUTPUT_DIR = Path(f"figures/images/{SEASON}_season_metrics")
OUTPUT_DIR_CSV = Path(f"csv/{SEASON}_season_metrics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_CSV.mkdir(parents=True, exist_ok=True)

EMAIL_ADDRESS = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASS")
TO_EMAIL = EMAIL_ADDRESS

WEIGHTS = {
    "yards": 0.2,
    "scoring": 0.2,
    "pct": 0.1,
    "epa_rush": 0.25,
    "epa_pass": 0.25,
}

def grade_metric(df, col, high_is_good=True):
    """Function to grade a metric on a 0-100 scale based on its z-score percentile."""

    mean = df[col].mean()
    std = df[col].std()

    z = (df[col] - mean) / std

    if not high_is_good:
        z = -z

    z = z.clip(-3, 3)

    grade = 0.5 * (1 + z.map(lambda v: math.erf(v / math.sqrt(2)))) * 100

    return grade

home_scoring = (
    pbp.groupby(["home_team", "game_id"])
    .agg(
        scored=("home_score", "max"),
        allowed=("away_score", "max")
    )
    .reset_index()
    .rename(columns={"home_team": "team"})
)

away_scoring = (
    pbp.groupby(["away_team", "game_id"])
    .agg(
        scored=("away_score", "max"),
        allowed=("home_score", "max")
    )
    .reset_index()
    .rename(columns={"away_team": "team"})
)

team_scoring = pd.concat([home_scoring, away_scoring])

def get_metrics(pbp, side):
    
    if side == "offense":
        group_col = "posteam"
        score_col = "scored"
    else:
        group_col = "defteam"
        score_col = "allowed"

    stats = (
        pbp.groupby(group_col)
        .agg(
            epa_pass = ("epa", lambda x: x[pbp.loc[x.index, "play_type"] == "pass"].mean()),
            epa_rush = ("epa", lambda x: x[pbp.loc[x.index, "play_type"] == "run"].mean()),
        )
        .reset_index()
        .rename(columns={group_col: "team"})
    )
    
    team_yards = (
        pbp.groupby([group_col, "game_id"])
        .agg(
            yards=("yards_gained", "sum")
        )
        .reset_index()
    )

    team_yards_per_game = (
        team_yards.groupby(group_col)
        .agg(
            yards_per_game=("yards", "mean")
        )
        .reset_index()
        .rename(columns={group_col: "team"})
    )

    scoring_per_game = (
        team_scoring.groupby("team")
        .agg(
            scoring_per_game=(score_col, "mean")
        )
        .reset_index()
    )

    scoring_pct = (
        pbp.groupby([group_col, "game_id", "drive"])
        .agg(scoring_drive=("drive_ended_with_score", "max"))
        .reset_index()
        .groupby(group_col)
        .agg(scoring_pct=("scoring_drive", "mean"))
        .reset_index()
        .rename(columns={group_col: "team"})
    )

    if side == "defense":
        scoring_pct["scoring_pct"] = 1 - scoring_pct["scoring_pct"]
    
    stats = (
        stats.merge(team_yards_per_game, on="team")
        .merge(scoring_per_game, on="team")
        .merge(scoring_pct, on="team")
    )
    
    high_is_good = {
        "epa_pass": False,
        "epa_rush": False,
        "yards_per_game": False,
        "scoring_per_game": False,
        "scoring_pct": True
    }   

    for col in ["epa_pass", "epa_rush", "yards_per_game", "scoring_per_game", "scoring_pct"]:
        if side == "defense":
            stats[f"{col}_grade"] = grade_metric(stats, col) if high_is_good[col] else grade_metric(stats, col, high_is_good=False)
        else:
            stats[f"{col}_grade"] = grade_metric(stats, col)

    stats["overall_grade"] = (
        stats["epa_pass_grade"] * WEIGHTS["epa_pass"]
        + stats["epa_rush_grade"] * WEIGHTS["epa_rush"]
        + stats["yards_per_game_grade"] * WEIGHTS["yards"]
        + stats["scoring_per_game_grade"] * WEIGHTS["scoring"]
        + stats["scoring_pct_grade"] * WEIGHTS["pct"]
    )
    
    filename = OUTPUT_DIR_CSV / f"{side}_rankings.csv"
    stats.to_csv(filename, index=False)
    print(f"Saved {side} rankings to {filename}")
    
    return stats

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def grade_to_color(value, min_val=0, max_val=100):
    """Maps a 0-100 grade to a red -> white -> green gradient."""
    t = (value - min_val) / (max_val - min_val)
    t = max(0, min(1, t))

    red   = (220, 38, 38)    # low end
    white = (255, 255, 255)  # midpoint
    green = (76, 187, 23)    # high end

    if t < 0.5:
        s = t / 0.5
        r = int(red[0] + s * (white[0] - red[0]))
        g = int(red[1] + s * (white[1] - red[1]))
        b = int(red[2] + s * (white[2] - red[2]))
    else:
        s = (t - 0.5) / 0.5
        r = int(white[0] + s * (green[0] - white[0]))
        g = int(white[1] + s * (green[1] - white[1]))
        b = int(white[2] + s * (green[2] - white[2]))

    return f"rgb({r},{g},{b})"

def grade_to_blue(value, min_val=0, max_val=100):
        t = max(0, min(1, (value - min_val) / (max_val - min_val)))
        return pc.sample_colorscale("Blues", t)[0]

def make_rank_table(df, side):

    df = (
        df.merge(teams[["team_abbr", "team_color", "team_color2"]], left_on="team", right_on="team_abbr")
        .drop(columns=["team_abbr"])
        .sort_values("overall_grade", ascending=False)
    )

    if side == "offense":
        title = f"Offensive Rankings - Week {WEEK}, {SEASON} Season" if WEEK < 18 else f"Offensive Rankings - {SEASON} Season"
        display_names = {
            "team": "TEAM",
            "epa_rush_grade": "EPA/RUSH",
            "epa_pass_grade": "EPA/PASS",
            "scoring_per_game_grade": "PTS/GM",
            "yards_per_game_grade": "YDS/GM",
            "scoring_pct_grade": "SC%",
            "overall_grade": "RATING",
        }
        # raw stat to display instead of grade
        raw_stats = {
            "epa_rush_grade": "epa_rush",
            "epa_pass_grade": "epa_pass",
            "scoring_per_game_grade": "scoring_per_game",
            "yards_per_game_grade": "yards_per_game",
            "scoring_pct_grade": "scoring_pct",
        }
        column_order = ["", "TEAM", "RATING", "EPA/RUSH", "EPA/PASS", "PTS/GM", "YDS/GM", "SC%"]
    else:
        title = f"Defensive Rankings - Week {WEEK}, {SEASON} Season" if WEEK < 18 else f"Defensive Rankings - {SEASON} Season"
        display_names = {
            "team": "TEAM",
            "epa_rush_grade": "EPA/RUSH",
            "epa_pass_grade": "EPA/PASS",
            "scoring_per_game_grade": "PTS/GM",
            "yards_per_game_grade": "YDS/GM",
            "scoring_pct_grade": "STOP%",
            "overall_grade": "RATING",
        }
        raw_stats = {
            "epa_rush_grade": "epa_rush",
            "epa_pass_grade": "epa_pass",
            "scoring_per_game_grade": "scoring_per_game",
            "yards_per_game_grade": "yards_per_game",
            "scoring_pct_grade": "scoring_pct",
        }
        column_order = ["", "TEAM", "RATING", "EPA/RUSH", "EPA/PASS", "PTS/GM", "YDS/GM", "STOP%"]
        
    formats = {
        "RATING":   lambda v: f"{v:.1f}",
        "EPA/RUSH": lambda v: f"{v:.2f}",
        "EPA/PASS": lambda v: f"{v:.2f}",
        "PTS/GM":   lambda v: f"{v:.1f}",
        "YDS/GM":   lambda v: f"{v:.1f}",
        "SC%":      lambda v: f"{v:.1%}",
        "STOP%":    lambda v: f"{v:.1%}",
    }
    
    # rank each stat column (1 = best)
    rank_cols = {
        "epa_rush_grade":         "epa_rush",
        "epa_pass_grade":         "epa_pass",
        "scoring_per_game_grade": "scoring_per_game",
        "yards_per_game_grade":   "yards_per_game",
        "scoring_pct_grade":      "scoring_pct",
        "overall_grade":          "overall_grade",
    }
    
    for grade_col, stat_col in rank_cols.items():
        df[f"{grade_col}_rank"] = df[grade_col].rank(ascending=False).astype(int)

    rating_note = "Rating = Average of standardized 0–100 grades (z-score → percentile)"
    inv_names = {v: k for k, v in display_names.items()}

    table_values = []
    colors = []
    text_colors = []
    
    for col in column_order:
        raw_col = inv_names.get(col)
        fmt = formats.get(col, lambda v: str(v))  # fallback to plain string
        
        if col == "":
            table_values.append(list(range(1, len(df) + 1)))
            colors.append(["lightgray"] * len(df))
            text_colors.append(["black"] * len(df))
        elif col == "TEAM":
            table_values.append(df["team"].tolist())
            colors.append(df["team_color"].tolist())
            text_colors.append(df["team_color2"].tolist())
        elif col == "RATING":
            table_values.append([fmt(v) for v in df[raw_col]])
            colors.append([grade_to_color(v) for v in df[raw_col]])
            text_colors.append(["black"] * len(df))
        else:
            stat_col = raw_stats[raw_col]
            rank_col = f"{raw_col}_rank"
            table_values.append([
                f"{fmt(v)} ({r})" for v, r in zip(df[stat_col], df[rank_col])
            ])
            colors.append([grade_to_blue(v) for v in df[raw_col]])
            text_colors.append(["white" if v > 60 else "black" for v in df[raw_col]])

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=column_order,
            fill_color="lightgrey",
            align="center",
            font=dict(size=12, color="black"),
        ),
        cells=dict(
            values=table_values,
            fill_color=colors,
            font=dict(color=text_colors),
            align="center"
        )
    )])

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        annotations=[dict(
            text=rating_note,
            x=0.5, y=1.02,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=12, color="gray"),
            align="center"
        )],
        width=650,
        height=730,
        margin=dict(l=0, r=0, t=60, b=0),
        paper_bgcolor="white"
    )
    
    fig.data[0].columnwidth = [10] + [40] * 2 + [50] * 5

    return fig

def plot_ratings(odf, ddf, teams):
    
    """Function to create a scatter plot of offensive vs defensive ratings for each team, with team logos as markers and diagonal lines indicating tiers of performance."""
    
    off_rating = odf[['team', 'overall_grade']].rename(columns={'overall_grade': 'offense_score'}).copy()
    def_rating = ddf[['team', 'overall_grade']].rename(columns={'overall_grade': 'defense_score'}).copy()
    merged = off_rating.merge(def_rating, on='team', how='inner')

    team_meta = teams[["team_abbr", "team_logo_espn"]].copy()

    # join metadata
    merged = merged.merge(team_meta, left_on="team", right_on="team_abbr", how="inner")    

    def url_to_base64(url):
        try:
            r = requests.get(url)
            r.status_code
            img = Image.open(BytesIO(r.content)).convert("RGBA")

            # resize for consistency
            img.thumbnail((80, 80), Image.Resampling.LANCZOS)

            buf = BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except :
            return None

    merged["logo_b64"] = merged["team_logo_espn"].apply(url_to_base64)

    fig = go.Figure()

    # Invisible scatter for coordinate mapping
    fig.add_trace(go.Scatter(
        x=merged["offense_score"],
        y=merged["defense_score"],
        mode="markers",
        marker=dict(size=0, opacity=0),
        showlegend=False
    ))

    # Add team logos as images
    for _, row in merged.iterrows():
        if row["logo_b64"] is None:
            continue

        fig.add_layout_image(
            dict(
                source="data:image/png;base64," + row["logo_b64"],
                x=row["offense_score"],
                y=row["defense_score"],
                xref="x",
                yref="y",
                sizex=5, 
                sizey=5,
                xanchor="center",
                yanchor="middle",
                layer="above"
            )
        )

    # Add diagonal lines (y = -x + c)
    intercepts = np.arange(25, 250, 25)
    for c in intercepts:
        fig.add_trace(go.Scatter(
            x=[0, 100],
            y=[0 + c, -100 + c],
            mode="lines",
            line=dict(color="lightgray", width=1, dash="solid"),
            showlegend=False
        ))

    # Axis + layout
    fig.update_layout(
        title=f"Team Ratings & Tiers — Week {WEEK}, {SEASON} Season" if WEEK < 18 else f"Team Ratings & Tiers — {SEASON} Season",
        xaxis=dict(
            title="Offensive Rating",
            linecolor="black",
            gridcolor="lightgray",
            griddash="dash",
            range=[0, 100],
            dtick=10,
            zeroline=False
        ),
        yaxis=dict(
            title="Defensive Rating",
            linecolor="black",
            gridcolor="lightgray",
            griddash="dash",
            range=[0, 100],
            dtick=10,
            zeroline=False
        ),
        plot_bgcolor="#ffffff",
        paper_bgcolor="white",
        width=900,
        height=700
    )
    
    # Quadrant labels
    P = 10  # padding from edges

    top_y    = 0 + P
    bottom_y = 100 - P
    left_x   = 100 - P
    right_x  = 0 + P

    # Top-right
    fig.add_annotation(
        x=right_x, y=top_y,
        text="Strong Offense<br>Strong Defense",
        showarrow=False,
        font=dict(size=12),
        xref="x", yref="y"
    )

    # Top-left
    fig.add_annotation(
        x=left_x, y=top_y,
        text="Weak Offense<br>Strong Defense",
        showarrow=False,
        font=dict(size=12),
        xref="x", yref="y"
    )

    # Bottom-right
    fig.add_annotation(
        x=right_x, y=bottom_y,
        text="Strong Offense<br>Weak Defense",
        showarrow=False,
        font=dict(size=12),
        xref="x", yref="y"
    )

    # Bottom-left
    fig.add_annotation(
        x=left_x, y=bottom_y,
        text="Weak Offense<br>Weak Defense",
        showarrow=False,
        font=dict(size=12),
        xref="x", yref="y"
    )
    
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    return fig

def send_email(subject, body, to_email, attachments):
    
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = to_email    
    msg.set_content(body)

    for file_path in attachments:
        with open(file_path, "rb") as f:
            file_data = f.read()
            file_name = os.path.basename(f.name)
        msg.add_attachment(file_data, maintype="image", subtype="jpeg", filename=file_name)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        smtp.send_message(msg)


# Metric Calculation
off_stats = get_metrics(pbp, "offense")
def_stats = get_metrics(pbp, "defense")

off_fig = make_rank_table(off_stats, "offense")
def_fig = make_rank_table(def_stats, "defense")
rating_fig = plot_ratings(off_stats, def_stats, teams)

def_fig.write_image(f"{OUTPUT_DIR}/defense_rankings_week_{WEEK}.png" if WEEK < 18 else f"{OUTPUT_DIR}/defense_rankings_{SEASON}_season.png")
off_fig.write_image(f"{OUTPUT_DIR}/offense_rankings_week_{WEEK}.png" if WEEK < 18 else f"{OUTPUT_DIR}/offense_rankings_{SEASON}_season.png")
rating_fig.write_image(f"{OUTPUT_DIR}/team_ratings_scatter_week_{WEEK}.png" if WEEK < 18 else f"{OUTPUT_DIR}/team_ratings_scatter_{SEASON}_season.png")

figs = [
    f"{OUTPUT_DIR}/offense_rankings_week_{WEEK}.jpeg" if WEEK < 18 else f"{OUTPUT_DIR}/offense_rankings_{SEASON}_season.jpeg",
    f"{OUTPUT_DIR}/defense_rankings_week_{WEEK}.jpeg" if WEEK < 18 else f"{OUTPUT_DIR}/defense_rankings_{SEASON}_season.jpeg",
    f"{OUTPUT_DIR}/team_ratings_scatter_week_{WEEK}.jpeg" if WEEK < 18 else f"{OUTPUT_DIR}/team_ratings_scatter_{SEASON}_season.jpeg"
]

#Send Email
send_email(
    subject=f"NFL Rankings Week {WEEK}",
    body=f"Attached are the latest offense and defense rankings and tiers after week {WEEK}.",
    to_email=TO_EMAIL,
    attachments=figs
)
