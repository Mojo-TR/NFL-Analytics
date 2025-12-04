import base64
import nflreadpy as nfl
import polars as pl
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
import smtplib
import json
import os
from email.message import EmailMessage
from PIL import Image
import requests
from io import BytesIO

pbp = nfl.load_pbp(seasons=2025)
teams = nfl.load_teams()
week = max(pbp["week"])
season = pbp["season"].unique()[0]
OUTPUT_DIR = Path.home() / f"Documents/{season}_Metrics/Week_{week}_Metrics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EMAIL_ADDRESS = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASS")
TO_EMAIL = EMAIL_ADDRESS # Because I am sending it to myself

def get_offense_metrics (pbp, passing, rushing):
    off_passing = (
        passing.group_by("posteam")
        .agg([
            pl.sum("epa").alias("total_passing_epa"),
            pl.count("play_id").alias("total_pass_attempts")
        ])
        .with_columns(
            (pl.col("total_passing_epa") / pl.col("total_pass_attempts")).alias("epa_per_pass")
        )
    )

    off_rushing  = (
        rushing.group_by("posteam")
        .agg([
            pl.sum("epa").alias("total_rush_epa"),
            pl.count("play_id").alias("total_rushes")
        ])
        .with_columns([
            (pl.col("total_rush_epa") / pl.col("total_rushes")).alias("epa_per_rush")
        ])
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
    
    offense_metrics = (
        total_offense.select(["posteam", "off_yards_per_game"])
        .join(scoring_offense.select(["posteam", "scoring_offense_per_game"]), on="posteam")
        .join(scoring_pct.select(["posteam", "scoring_pct"]), on="posteam")
        .join(off_rushing.select(["posteam", "epa_per_rush"]), on="posteam")
        .join(off_passing.select(["posteam", "epa_per_pass"]), on="posteam")
    )
    
    return offense_metrics
    
def get_defense_metrics(pbp, passing, rushing):
    def_passing = (
        passing.group_by("defteam")
        .agg([
            pl.sum("epa").alias("total_passing_epa"),
            pl.count("play_id").alias("total_pass_attempts")
        ])
        .with_columns(
            (pl.col("total_passing_epa") / pl.col("total_pass_attempts")).alias("epa_per_pass")
        )
    )

    def_rushing = (
        rushing.group_by("defteam")
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

    defense_metrics = (
        total_defense.select(["defteam", "yards_allowed_per_game"])
        .join(scoring_defense.select(["defteam", "scoring_defense_per_game"]), on="defteam")
        .join(stop_rate.select(["defteam", "stop_rate"]), on="defteam")
        .join(def_rushing.select(["defteam", "epa_per_rush"]), on="defteam")
        .join(def_passing.select(["defteam", "epa_per_pass"]), on="defteam")
        .rename({"defteam": "posteam"})
    )
    
    return defense_metrics

def rank_metrics(df, category):
    df = df.with_columns(pl.col("posteam"))
    if category == "offense":
        df = (
            df.with_columns([
                pl.col("off_yards_per_game").rank(method="ordinal", descending=True).alias("rank_yards"),
                pl.col("scoring_offense_per_game").rank(method="ordinal", descending=True).alias("rank_scoring"),
                pl.col("scoring_pct").rank(method="ordinal", descending=True).alias("rank_pct"),
                pl.col("epa_per_rush").rank(method="ordinal", descending=True).alias("rank_epa_rush"),
                pl.col("epa_per_pass").rank(method="ordinal", descending=True).alias("rank_epa_pass"),
            ])
            .with_columns(pl.mean_horizontal([
                "rank_yards", "rank_scoring", "rank_pct", "rank_epa_rush", "rank_epa_pass"
            ]).alias("offense_power_score"))
            .sort("offense_power_score")
        )
    else:
        df = (
            df.with_columns([
                pl.col("epa_per_rush").rank(method="ordinal").alias("rank_epa_per_rush"),
                pl.col("epa_per_pass").rank(method="ordinal").alias("rank_epa_per_pass"),
                pl.col("yards_allowed_per_game").rank(method="ordinal").alias("rank_yards_allowed"),
                pl.col("scoring_defense_per_game").rank(method="ordinal").alias("rank_scoring_defense"),
                pl.col("stop_rate").rank(method="ordinal", descending=True).alias("rank_stop_rate"),
            ])
            .with_columns(pl.mean_horizontal([
                "rank_yards_allowed", "rank_scoring_defense", "rank_stop_rate",
                "rank_epa_per_rush", "rank_epa_per_pass"
            ]).alias("defense_power_score"))
            .sort("defense_power_score")
        )
        
    pdf = df.to_pandas()
    pdf["Team"] = pdf["posteam"]
    print
    return pdf

def ranking_to_csv(df, category, week, season):
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
   
    df = df.round(2).drop(columns=["posteam"])
    if "Rank" not in df.columns:
        df.insert(0, "Rank", range(1, len(df) + 1))

    cols = list(df.columns)
    if "Team" in cols:
        cols.remove("Team")
    if "Rank" in cols:
        cols.remove("Rank")
    cols = ["Rank", "Team"] + cols
    df = df[cols]

    # Save CSV
    output_path = f"{OUTPUT_DIR}/{category}_rankings_week_{week}.csv"
    df.to_csv(output_path, index=False)

def format_for_display(df, category):
    df = df.copy()  # ensure we're not modifying original outside

    if category == "offense":
        display_names = {
            "off_yards_per_game": "YDS/GM",
            "scoring_offense_per_game": "PTS/GM",
            "scoring_pct": "SC%",
            "epa_per_rush": "EPA/RUSH",
            "epa_per_pass": "EPA/PASS",
            "offense_power_score": "RATING",
            "rank_yards": "rank_yards",
            "rank_scoring": "rank_scoring",
            "rank_pct": "rank_pct",
            "rank_epa_rush": "rank_epa_rush",
            "rank_epa_pass": "rank_epa_pass"
        }
        column_order = ["RATING", "EPA/RUSH", "EPA/PASS", "PTS/GM", "YDS/GM", "SC%"]
    else:
        display_names = {
            "yards_allowed_per_game": "YDS/GM",
            "scoring_defense_per_game": "PTS/GM",
            "stop_rate": "STOP%",
            "epa_per_rush": "EPA/RUSH",
            "epa_per_pass": "EPA/PASS",
            "defense_power_score": "RATING",
            "rank_yards_allowed": "rank_yards_allowed",
            "rank_scoring_defense": "rank_scoring_defense",
            "rank_stop_rate": "rank_stop_rate",
            "rank_epa_per_rush": "rank_epa_per_rush",
            "rank_epa_per_pass": "rank_epa_per_pass"
        }
        column_order = ["RATING", "EPA/RUSH", "EPA/PASS", "PTS/GM", "YDS/GM", "STOP%"]

    stat_ranks = [c for c in display_names if c.startswith("rank")]

    # Rename columns
    df = df.rename(columns=display_names)
    
    # Reorder columns
    df = df[["Team"] + column_order + stat_ranks]
    
    # Add Rank column at the end
    df["Rank"] = range(1, len(df) + 1)
    
    return df, column_order, stat_ranks


def make_rank_table(df, week, category):

    df, column_order, stat_ranks = format_for_display(df, category)

    rating_note = (
        "Rating = Average rank of YDS/GM, PTS/GM, SC%, EPA/RUSH, EPA/PASS (nflverse)"
        if category=="offense" else
        "Rating = Average rank of YDS/GM, PTS/GM, STOP%, EPA/RUSH, EPA/PASS (nflverse)"
    )

    # Mapping from display column to rank column
    if category == "offense":
        rank_map = {
            "YDS/GM": "rank_yards",
            "PTS/GM": "rank_scoring",
            "SC%": "rank_pct",
            "EPA/RUSH": "rank_epa_rush",
            "EPA/PASS": "rank_epa_pass",
            "RATING": ""
        }
        high_is_good_cols = ["SC%", "EPA/RUSH", "EPA/PASS", "PTS/GM", "YDS/GM", "RATING"]
    else:
        rank_map = {
            "YDS/GM": "rank_yards_allowed",
            "PTS/GM": "rank_scoring_defense",
            "STOP%": "rank_stop_rate",
            "EPA/RUSH": "rank_epa_per_rush",
            "EPA/PASS": "rank_epa_per_pass",
            "RATING": ""
        }
        high_is_good_cols = ["STOP%"]

    # Colors helpers
    min_val = df["RATING"].min()
    max_val = df["RATING"].max()
    mid_val = (min_val + max_val) / 2

    def green_white_red_mid(v):
        ratio = (v - min_val)/(mid_val - min_val) if v <= mid_val else (v - mid_val)/(max_val - mid_val)
        if v <= mid_val:
            r, g, b = int(250*ratio), 250, int(250*ratio)
        else:
            r, g, b = 250, int(250*(1-ratio)), int(250*(1-ratio))
        return f"rgba({r},{g},{b},1)"

    def text_color_from_rgba(color):
        if "rgba(" in color:
            r, g, b, a = [float(x) for x in color.replace("rgba(","").replace(")","").split(",")]
            if g > r and g > b:
                return "black"
        r, g, b, a = [float(x) for x in color.replace("rgba(","").replace(")","").split(",")]
        brightness = 0.299*r + 0.587*g + 0.114*b
        return "white" if brightness < 150 else "black"


    # Build colors
    colors, text_colors = [], []

    # Rank & Team columns
    colors.append(["lightgrey"]*len(df))
    text_colors.append(["black"]*len(df))
    colors.append(["white"]*len(df))
    text_colors.append(["black"]*len(df))

    # Numeric columns
    for col in column_order:
        col_values = df[col].copy()
        
        if col == "RATING":
            col_colors = [green_white_red_mid(v) for v in col_values]
        else:
            numeric_vals = col_values.astype(float)  # ensure numeric
            norm = (numeric_vals - numeric_vals.min()) / (numeric_vals.max() - numeric_vals.min())
            if col in high_is_good_cols:
                norm = 1 - norm  # invert if high is good
            col_colors = [f"rgba({int(255*v)},{int(255*v)},255,1)" for v in norm]

        colors.append(col_colors)
        text_colors.append([text_color_from_rgba(c) for c in col_colors])

    # Build table values with (rank) and proper decimals
    table_values = [df["Rank"].tolist(), df["Team"].tolist()]

    for col in column_order:
        vals = []
        rank_col = rank_map.get(col, "")
        
        for i, v in enumerate(df[col]):
            # Determine rank string
            rank_str = f" ({int(df.iloc[i][rank_col])})" if rank_col and rank_col in df.columns else ""

            # Format values based on column type
            if col in ["SC%", "STOP%"]:
                v_str = f"{v:.1f}%"
            elif col in ["YDS/GM", "PTS/GM", "RATING"]:
                v_str = f"{v:.1f}"
            elif col in ["EPA/RUSH", "EPA/PASS"]:
                v_str = f"{v:.2f}"
            else:
                v_str = f"{v}"

            vals.append(f"{v_str}{rank_str}")
        
        table_values.append(vals)

    header_values = ["", "Team"] + column_order
    column_widths = [10] + [40] * 2 + [50]*(len(column_order) - 1)
    # Build figure
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=header_values,
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

    fig.data[0].columnwidth = column_widths
    fig.update_layout(
        title=dict(
            text=f"{category[:-1].capitalize()}ive Ratings (Weeks 1-{week}, 2025)",
            x=0.5,
            xanchor="center"
        ),
        annotations=[dict(
            text=rating_note,
            x=0.5,
            y=1.02,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=12, color="gray"),
            align="center"
        )],
        width=650, 
        height=730,
        margin=dict(l=0, r=0, t=60, b=0),
        paper_bgcolor="white"
    )

    return fig

def plot_ratings(odf, ddf, teams):
    off_rating = odf[['posteam', 'offense_power_score']].copy() 
    def_rating = ddf[['posteam', 'defense_power_score']].copy() 
    merged = off_rating.merge(def_rating, on='posteam', how='inner')

    team_meta = teams.select(["team_abbr", "team_color", "team_logo_espn"]).to_pandas()

    # join metadata
    merged = merged.merge(team_meta, left_on="posteam", right_on="team_abbr", how="inner")    

    def url_to_base64(url):
        try:
            r = requests.get(url)
            img = Image.open(BytesIO(r.content)).convert("RGBA")

            # resize for consistency
            img.thumbnail((80, 80), Image.LANCZOS)

            buf = BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except:
            return None

    merged["logo_b64"] = merged["team_logo_espn"].apply(url_to_base64)

    fig = go.Figure()

    # Invisible scatter for coordinate mapping
    fig.add_trace(go.Scatter(
        x=merged["offense_power_score"],
        y=merged["defense_power_score"],
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
                x=row["offense_power_score"],
                y=row["defense_power_score"],
                xref="x",
                yref="y",
                sizex=1.5, 
                sizey=1.53,
                xanchor="center",
                yanchor="middle",
                layer="above"
            )
        )

    # Add diagonal lines (y = -x + c)
    intercepts = np.arange(12, 60, 8)
    for c in intercepts:
        fig.add_trace(go.Scatter(
            x=[1, 32],
            y=[-1 + c, -32 + c],
            mode="lines",
            line=dict(color="lightgray", width=1, dash="solid"),
            showlegend=False
        ))

    # Axis + layout
    fig.update_layout(
        title=f"Team Ratings & Tiers — Week {week}",
        xaxis=dict(
            title="Offensive Rating",
            linecolor="black",
            gridcolor="lightgray",
            griddash="dash",
            dtick=4,
            range=[32, 1],
            zeroline=False
        ),
        yaxis=dict(
            title="Defensive Rating",
            linecolor="black",
            gridcolor="lightgray",
            griddash="dash",
            dtick=4,
            range=[32, 1],
            zeroline=False
        ),
        plot_bgcolor="#ffffff",
        paper_bgcolor="white",
        width=900,
        height=700
    )
    
    # Quadrant labels
    P = 4  # padding from edges

    top_y    = 1 + P
    bottom_y = 32 - P
    left_x   = 32 - P
    right_x  = 1 + P

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

def send_email(subject, body, to_email, attachments=[]):
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

passed = pbp.filter(pl.col("play_type") == "pass")
rushed = pbp.filter(pl.col("play_type") == "run")

offense = get_offense_metrics(pbp, passed, rushed)
defense = get_defense_metrics(pbp, passed, rushed)

offense_ranked = rank_metrics(offense, "offense")
defense_ranked = rank_metrics(defense, "defense")

offense_fig = make_rank_table(offense_ranked, week, "offense")
defense_fig = make_rank_table(defense_ranked, week, "defense")
rating_fig = plot_ratings(offense_ranked, defense_ranked, teams)


# # Create CSV
# ranking_to_csv(offense_ranked, "offense", week, season)
# ranking_to_csv(defense_ranked, "defense", week, season)

# Save images
offense_fig.write_image(f"{OUTPUT_DIR}/offense_rankings_week_{week}.png")
defense_fig.write_image(f"{OUTPUT_DIR}/defense_rankings_week_{week}.png")
rating_fig.write_image(f"{OUTPUT_DIR}/team_ratings_scatter_week_{week}.png")

attachments = [
    f"{OUTPUT_DIR}/offense_rankings_week_{week}.png",
    f"{OUTPUT_DIR}/defense_rankings_week_{week}.png",
    f"{OUTPUT_DIR}/team_ratings_scatter_week_{week}.png"
]

#Send Email
send_email(
    subject=f"NFL Rankings Week {week}",
    body=f"Attached are the latest offense and defense rankings and tiers after week {week}.",
    to_email=TO_EMAIL,
    attachments=attachments
)