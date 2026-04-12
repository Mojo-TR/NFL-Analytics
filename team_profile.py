
import warnings
from pathlib import Path
import nflreadpy as nfl
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
warnings.filterwarnings("ignore")

SEASON = 2025
OUT_DIR = Path("figures/images/team_profiles")
OUT_DIR.mkdir(exist_ok=True)

BG_DARK        = "#0D0F12"   # near black with cool tint
BG_PANEL       = "#141618"   # slightly lighter for subplot areas
BG_CARD        = "#1C1F24"   # card/table backgrounds
GRID_CLR       = "#1E2128"   # very subtle, barely there grid lines
ZERO_LINE      = "#4A5060"   # pronounced zero line, cool grey-blue

TEXT_PRI  = "#E8EAF0"        # soft white, easier on eyes than pure white
TEXT_SEC  = "#6B7485"        # muted label text
TEXT_ACC  = "#9AA3B2"        # slightly brighter for bar value labels

ACCENT_POS = "#00C48C"       # slightly desaturated teal
ACCENT_NEG = "#E8445A"       # slightly desaturated red
ACCENT_NEU = "#4A7FBD"       # muted steel blue for neutral bars

FONT_TITLE = "Avenir Next Condensed"
FONT_BODY  = "PT Mono"

def rank_suffix(r):
    if 11 <= r <= 13:
        return f"{r}th"
    s = {1: "st", 2: "nd", 3: "rd"}.get(r % 10, "th")
    return f"{r}{s}"
    
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

pbp = nfl.load_pbp(seasons=SEASON).to_pandas()
pbp = pbp[(pbp["play_type"] != "no_play") & (pbp["season_type"] == "REG")]
ftn = nfl.load_ftn_charting(seasons=SEASON).to_pandas()
participation = nfl.load_participation(seasons=SEASON).to_pandas()
teams = nfl.load_teams().to_pandas()

ftn_cols = ["nflverse_game_id" ,"nflverse_play_id", "qb_location", "is_motion", "is_no_huddle", "is_play_action", "is_screen_pass", "is_rpo", "n_blitzers", "n_pass_rushers"]
participation_cols = ["nflverse_game_id", "play_id", "offense_personnel", "defense_personnel", "was_pressure", "defense_coverage_type"]

pbp = (
    pbp
    .merge(
        ftn[ftn_cols],
        left_on=["play_id", "game_id"],
        right_on=["nflverse_play_id", "nflverse_game_id"],
        how="left"
    )
    .drop(columns=["nflverse_play_id", "nflverse_game_id"])
    .merge(
        participation[participation_cols],
        left_on=["play_id", "game_id"],
        right_on=["play_id", "nflverse_game_id"],
        how="left"
    )
    .drop(columns=["nflverse_game_id"])
)

bool_cols = ["was_pressure", "is_motion", "is_no_huddle", "is_play_action", "is_screen_pass", "is_rpo"]
for col in bool_cols:
    pbp[col] = pbp[col].fillna(0).astype(int)

team_off_ratings = pd.read_csv("csv/2025_season_metrics/offense_rankings.csv")
team_def_ratings = pd.read_csv("csv/2025_season_metrics/defense_rankings.csv")

home_off = (
    pbp[pbp["posteam"] == pbp["home_team"]]
    .groupby(["home_team", "game_id"])
    .agg(off_success_rate=("success", "mean"))
    .reset_index()
    .rename(columns={"home_team": "team"})
)

home_def = (
    pbp[pbp["defteam"] == pbp["home_team"]]
    .groupby(["home_team", "game_id"])
    .agg(success_rate_allowed=("success", "mean"))
    .reset_index()
    .rename(columns={"home_team": "team"})
)

away_off = (
    pbp[pbp["posteam"] == pbp["away_team"]]
    .groupby(["away_team", "game_id"])
    .agg(off_success_rate=("success", "mean"))
    .reset_index()
    .rename(columns={"away_team": "team"})
)

away_def = (
    pbp[pbp["defteam"] == pbp["away_team"]]
    .groupby(["away_team", "game_id"])
    .agg(success_rate_allowed=("success", "mean"))
    .reset_index()
    .rename(columns={"away_team": "team"})
)

game_score = (
    pbp.groupby("game_id")
    .agg(
        home_score = ("home_score", "max"),
        away_score = ("away_score", "max"),
        home_team=("home_team", "first"), 
        away_team=("away_team", "first")
    ).reset_index()
)

off_stats = pd.concat([home_off, away_off]).reset_index(drop=True)
def_stats = pd.concat([home_def, away_def]).reset_index(drop=True)

game_results = (
    off_stats
    .merge(def_stats, on=["team", "game_id"])
    .merge(game_score, on="game_id")
)

game_results["result"] = (
    game_results
    .apply(lambda row: "W" if (row["home_team"] == row["team"] and row["home_score"] > row["away_score"]) 
           or (row["away_team"] == row["team"] and row["away_score"] > row["home_score"]) 
           else "L", axis=1)
)

game_results["point_diff"] = game_results.apply(lambda row: row["home_score"] - row["away_score"] if row["home_team"] == row["team"] else row["away_score"] - row["home_score"], axis=1)

team_stats = (
    game_results
    .groupby("team")
    .agg(
        wins=("result", lambda x: (x == "W").sum()),
        losses=("result", lambda x: (x == "L").sum()),
        point_diff=("point_diff", "sum"),
        off_success_rate=("off_success_rate", "mean"),
        success_rate_allowed=("success_rate_allowed", "mean"),
    )
    .reset_index()
)

team_stats["point_diff"] = team_stats["point_diff"].astype(int)
team_stats["net_success_rate"] = team_stats["off_success_rate"] - team_stats["success_rate_allowed"]
team_stats = team_stats.drop(columns=["off_success_rate", "success_rate_allowed"])

team_stats = (
    team_stats
    .merge(
        team_off_ratings[["team", "overall_grade", "scoring_per_game", "yards_per_game", "scoring_pct", "epa_rush", "epa_pass"]],
        left_on="team",
        right_on="team",
        how="left"
    )
    .merge(
        team_def_ratings[["team", "overall_grade", "scoring_per_game", "yards_per_game", "scoring_pct", "epa_rush", "epa_pass"]],
        left_on="team",
        right_on="team",
        how="left",
        suffixes=("_off", "_def")
    )
)

# Offensive stats
off_stats = (
    pbp.groupby("posteam")
    .agg(
        pass_rate = ("play_type", lambda x: (x == "pass").mean()),
        rush_rate = ("play_type", lambda x: (x == "run").mean()),
        early_down_sr = ("success", lambda x: x[pbp.loc[x.index, "down"] <= 2].mean()),
        third_down_cr = ("first_down", lambda x: x[pbp.loc[x.index, "down"] == 3].mean())
    )
).reset_index()

off_pass_stats = (
    pbp[pbp["play_type"] == "pass"]
    .groupby("posteam")
    .agg(
        short_pass_rate = ("air_yards", lambda x: (x < 10).mean()),
        intermediate_pass_rate = ("air_yards", lambda x: ((x >= 10) & (x < 20)).mean()),
        deep_pass_rate = ("air_yards", lambda x: (x >= 20).mean()),
    ).reset_index()
)

runs = pbp[(pbp["play_type"] == "run") & (pbp["qb_dropback"] == 0)]
    
inside_mask = (runs["run_location"] == "middle") | (runs["run_gap"].isin(["guard"]))
outside_mask = (runs["run_location"].isin(["left", "right"])) & (runs["run_gap"].isin(["end", "tackle"]))

runs["run_type"] = None
runs.loc[inside_mask, "run_type"] = "inside"
runs.loc[outside_mask, "run_type"] = "outside"
    
runs = runs[runs["run_type"].notna()]
    
off_run_stats = (
    runs
    .groupby("posteam")
    .agg(
        inside_run_rate = ("run_type", lambda x: (x == "inside").mean()),
        outside_run_rate = ("run_type", lambda x: (x == "outside").mean()),
    ).reset_index()
)

off_red_zone = (
    pbp[pbp["yardline_100"] <= 20]
    .groupby(["posteam", "game_id", "drive"])
    .agg(
        scored_td = ("touchdown", "max"),
    ).reset_index()
)

off_red_zone_stats = (
    off_red_zone
    .groupby("posteam")
    .agg(
        red_zone_td_rate = ("scored_td", "mean"),
    ).reset_index()
)

off_stats = (
    off_stats
    .merge(off_pass_stats, on="posteam", how="left")
    .merge(off_run_stats, on="posteam", how="left")
    .merge(off_red_zone_stats, on="posteam", how="left")
)

team_stats = (
    team_stats
    .merge(off_stats, left_on="team", right_on="posteam", how="left")
    .drop(columns=["posteam"])
)

# Defensive stats
turnover_mask = (pbp["interception"] == 1) | (pbp["fumble_lost"] == 1)
blitz_mask = pbp["n_blitzers"] > 0

pbp["turnover"] = turnover_mask.astype(int)
pbp["is_blitz"] = blitz_mask.astype(int)
pbp["was_pressure"] = pbp["was_pressure"].fillna(0).astype(int)

def_turnover = (
    pbp.groupby(["defteam", "game_id", "drive"])
    .agg(
        turnover = ("turnover", "max"),
    ).reset_index()
)

def_turnover_stats = (
    def_turnover
    .groupby("defteam")
    .agg(
        turnovers_forced = ("turnover", "sum"),
    ).reset_index()
)

coverage_counts = (
    pbp[pbp["play_type"] == "pass"]
    .groupby(["defteam", "defense_coverage_type"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

coverage_cols = [c for c in coverage_counts.columns if c != "defteam"]
coverage_counts[coverage_cols] = coverage_counts[coverage_cols].div(
    coverage_counts[coverage_cols].sum(axis=1), axis=0
)

def_pass_stats = (
    pbp[pbp["play_type"] == "pass"]
    .groupby("defteam")
    .agg(
        blitz_rate = ("is_blitz", "mean"),
        pressure_rate = ("was_pressure", "mean"),
        sacks = ("sack", "sum"),
    )
    .reset_index()
    .merge(coverage_counts, on="defteam")
)

def_run_stats = (
    runs
    .groupby("defteam")
    .agg(
        run_stuffs = ("tackled_for_loss", "sum"),
    ).reset_index()
)

def_stats = (
    def_turnover_stats
    .merge(def_pass_stats, on="defteam", how="left")
    .merge(def_run_stats, on="defteam", how="left")
)

def_stats["sacks"] = def_stats["sacks"].astype(int)
def_stats["run_stuffs"] = def_stats["run_stuffs"].astype(int)

team_stats = (
    team_stats
    .merge(def_stats, left_on="team", right_on="defteam", how="left")
    .drop(columns=["defteam"])
)

# Special teams stats
special_teams_epa_off = (
    pbp[pbp["special"] == 1]
    .groupby("posteam")
    .agg(special_teams_epa_off = ("epa", "sum"))
    .reset_index()
)

special_teams_epa_def = (
    pbp[pbp["special"] == 1]
    .groupby("defteam")
    .agg(special_teams_epa_def = ("epa", lambda x: -x.sum()))
    .reset_index()
)

special_teams_epa = (
    special_teams_epa_off
    .merge(
        special_teams_epa_def,
        left_on="posteam",
        right_on="defteam",
        how="outer"
    )
    .drop(columns=["defteam"])
)

special_teams_epa = special_teams_epa.rename(columns={"posteam": "team"})

special_teams_epa["special_teams_epa"] = special_teams_epa["special_teams_epa_off"] + special_teams_epa["special_teams_epa_def"]

special_teams_epa = special_teams_epa.drop(columns=["special_teams_epa_off", "special_teams_epa_def"])

team_stats = (
    team_stats
    .merge(special_teams_epa, on="team", how="left")
)

team_stats["offensive_rank"] = team_stats["overall_grade_off"].rank(ascending=False).astype(int)
team_stats["defensive_rank"] = team_stats["overall_grade_def"].rank(ascending=False).astype(int)

team_stats["offensive_rank"] = team_stats["offensive_rank"].apply(rank_suffix)
team_stats["defensive_rank"] = team_stats["defensive_rank"].apply(rank_suffix)

team_stats = team_stats.merge(teams[["team_abbr", "team_nick", "team_logo_espn", "team_color", "team_color2"]], left_on="team", right_on="team_abbr", how="left").drop(columns=["team_abbr"])

def create_team_profile(league_df, team):
    
    row = league_df[league_df["team"] == team].iloc[0]
    
    fig = make_subplots(
        rows=4, cols=4,
        row_heights=[0.10, 0.30, 0.30, 0.30],
        column_widths=[0.25, 0.25, 0.25, 0.25],
        vertical_spacing=0.05,
        horizontal_spacing=0.04,
        specs=[
            [{"colspan": 4, "type": "domain"}, None, None, None],  # header row
            [{"type": "domain"}, {"type": "domain"}, {"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "domain"}, {"type": "bar"}, {"type": "bar"}],
            [{"colspan": 4, "type": "table"}, None, None, None],   # footer row
        ]
    )

    # ── Shared style helpers ─────────────────────────────────────────
    def get_header_bg(hex_color):
        r, g, b = hex_to_rgb(hex_color)
        luminance = (0.299 * r + 0.587 * g + 0.114 * b)  # perceived brightness 0-255
        if luminance < 60:        # very dark color like navy, forest green, black
            return "#3A3F4A"      # use lighter background
        else:
            return "#22262D"      # standard dark background
    
    def color_for_pct(pct):
        """Interpolate between red→neutral→team color based on percentile."""
        if pct >= 60:
            return ACCENT_POS
        elif pct >= 40:
            return TEXT_SEC
        else:
            return ACCENT_NEG

    def get_pct(col, ascending=True):
        return float(league_df[col].rank(pct=True, ascending=ascending)[league_df["team"] == team].values[0]) * 100

    def color_pie(vals, df):
        total = sum(vals)
        
        dark_teams = ["LV", "PIT", "SEA", "HOU", "CHI", "NE", "DEN", "WAS"]  # teams with very dark primary colors where we should use team_color2 for better visibility
        
        opacities = [v / total for v in vals]

        base = hex_to_rgb(df["team_color"]) if team not in dark_teams else hex_to_rgb(row["team_color2"])

        min_opacity = 0.10
        max_opacity = 0.95

        colors = [
            f"rgba({base[0]}, {base[1]}, {base[2]}, {min_opacity + (max_opacity - min_opacity) * (op / max(opacities))})"
            for op in opacities
        ]

        return colors
    
    def add_section_heading(text, y_pos):
        fig.add_annotation(
            text=text,
            xref="paper", yref="paper",
            x=0.0, y=y_pos,
            xanchor="left", yanchor="bottom",
            font=dict(size=16, color=TEXT_SEC, family=FONT_TITLE),
            showarrow=False,
    )
    
    def update_vert(row=None, col=None):
        fig.update_yaxes(
            showgrid=True,
            gridcolor=GRID_CLR,
            gridwidth=1,
            zeroline=True,
            zerolinecolor=ZERO_LINE,
            zerolinewidth=2,
            row=row, col=col,
        )
    
    def update_hor(row=None, col=None):
        fig.update_xaxes(
            showgrid=True,
            gridcolor=GRID_CLR,
            gridwidth=1,
            zeroline=True,
            zerolinecolor=ZERO_LINE,
            zerolinewidth=2,
            row=row, col=col,
        )

    # # ── ROW 1: Header (table used as text block) ─────────────────────
    pri = row["team_color"]
    bg_header_cell = get_header_bg(row["team_color"])
    fill_color = row["team_color"] if team not in ["LV", "PIT"] else row["team_color2"]

    nsr_color = ACCENT_POS if row["net_success_rate"] > 0 else ACCENT_NEG
    pd_color  = ACCENT_POS if row["point_diff"] > 0 else ACCENT_NEG

    header_cells = [
        [f"<b><span style='font-size:28px;color:{pri}'>{row['team_nick'].upper()}</span></b><br>"
            f"<span style='font-size:13px;color:{TEXT_SEC}'>{SEASON} TEAM PROFILE</span>"],
        [f"<span style='color:{TEXT_SEC}'>RECORD</span><br>"
            f"<b><span style='font-size:22px;color:{pri}'>{row['wins']} - {row['losses']}</span></b>"],
        [f"<span style='color:{TEXT_SEC}'>POINT DIFF</span><br>"
            f"<b><span style='font-size:22px;color:{pd_color}'>{row['point_diff']:+.0f}</span></b>"],
        [f"<span style='color:{TEXT_SEC}'>NET SUCCESS RATE</span><br>"
            f"<b><span style='font-size:22px;color:{nsr_color}'>{row['net_success_rate']:+.1%}</span></b>"],
        [f"<span style='color:{TEXT_SEC}'>OFF RANK</span><br>"
            f"<b><span style='font-size:22px;color:{pri}'>{row['offensive_rank']}</span></b>"],
        [f"<span style='color:{TEXT_SEC}'>DEF RANK</span><br>"
            f"<b><span style='font-size:22px;color:{pri}'>{row['defensive_rank']}</span></b>"],
    ]

    fig.add_trace(
        go.Table(
            header=dict(
                values=[""] * len(header_cells),
                fill_color=bg_header_cell,
                line_color=TEXT_SEC,
                height=0,
            ),
            cells=dict(
                values=[[h[0]] for h in header_cells],
                align="center",
                fill_color=bg_header_cell,
                line_color=TEXT_SEC,
                font=dict(size=13, color=TEXT_PRI, family=FONT_TITLE),
                height=60,
            ),
        ),
        row=1, col=1
    )

    # ── ROW 2: Offensive Overview ─────────────────────────────────
    add_section_heading("◈  OFFENSIVE OVERVIEW", y_pos=0.88)
    
    run_pass_vals = [row["pass_rate"], row["rush_rate"]]

    colors = color_pie(run_pass_vals, row)

    fig.add_trace(
        go.Pie(
            labels=["PASS %", "RUSH %"],
            values=run_pass_vals,
            hole=0.6,
            marker=dict(colors=colors),
            textinfo="label+percent",
            textfont=dict(size=9, color="white", family=FONT_BODY),
        ),
        row=2, col=1
    )

    passing_vals = [row["short_pass_rate"], row["intermediate_pass_rate"], row["deep_pass_rate"]]

    colors = color_pie(passing_vals, row)

    fig.add_trace(
        go.Pie(
            labels=["SHORT %", "INTER %", "DEEP %"],
            values=passing_vals,
            hole=0.6,
            marker=dict(colors=colors),
            textinfo="label+percent",
            textfont=dict(size=9, color="white", family=FONT_BODY),
        ),
        row=2, col=2
    )

    epa_vals = [row["epa_pass_off"], row["epa_rush_off"]]
    epa_pcts = [get_pct("epa_pass_off"), get_pct("epa_rush_off")]
    
    fig.add_trace(
        go.Bar(
            y=["Pass EPA", "Rush EPA"],
            x=epa_vals,
            orientation="h",
            marker_line_width=0,
            text=[f"{v:.2f}" for v in epa_vals],
            textposition="outside",
            textfont=dict(size=10, color=TEXT_ACC, family=FONT_BODY),
            textangle=-90,
            marker_color=[color_for_pct(p) for p in epa_pcts],
            name="EPA Split",
        ),
        row=2, col=3,
    )

    sr_vals = [row["early_down_sr"], row["third_down_cr"], row["red_zone_td_rate"]]
    sr_pcts = [get_pct("early_down_sr"), get_pct("third_down_cr"), get_pct("red_zone_td_rate")]
    
    fig.add_trace(
        go.Bar(
            x=["Early SR", "3rd CR", "RZ TD %"],
            y=sr_vals,
            marker_line_width=0,
            text=[f"{v:.1%}" for v in sr_vals],
            textposition="outside",
            textfont=dict(size=10, color=TEXT_ACC, family=FONT_BODY),
            name="Offensive Conversion",
            marker_color=[color_for_pct(p) for p in sr_pcts],
        ),
        row=2, col=4,
    )

    # ----- Row 3 Defensive Overview -----
    add_section_heading("◈  DEFENSIVE OVERVIEW", y_pos=0.57)
    
    pass_rush = [row["pressure_rate"], row["blitz_rate"]]
    fig.add_trace(
        go.Bar(
            x=["QBP %", "BLITZ %"],
            y=pass_rush,
            marker_line_width=0,
            text=[f"{v:.1%}" for v in pass_rush],
            textposition="outside",
            textfont=dict(size=10, color=TEXT_ACC, family=FONT_BODY),
            marker_color=fill_color,
            name="Pass Rush Split",
        ),
        row=3, col=1,
    )

    coverage_vals = [row['2_MAN'], row['COVER_0'], row['COVER_1'], row['COVER_2'], row['COVER_3'], row['COVER_4'], row['COVER_6'], row['COVER_9'], row['BLOWN'], row['COMBO']]

    labels_raw = ["2 MAN", "COV 0", "COV 1", "COV 2", "COV 3", "COV 4", "COV 6", "COV 9", "BLOWN", "COMBO"]

    # Pair labels with values and sort by value descending
    paired = sorted(zip(labels_raw, coverage_vals), key=lambda x: x[1], reverse=True)

    top_5 = paired[:5]
    others = paired[5:]

    final_labels = [p[0] for p in top_5] + ["OTHER"]
    final_vals   = [p[1] for p in top_5] + [sum(p[1] for p in others)]

    colors = color_pie(final_vals, row)

    fig.add_trace(
        go.Pie(
            labels=final_labels,
            values=final_vals,
            hole=0.6,
            sort=False,
            marker=dict(colors=colors),
            textinfo="label+percent",
            textfont=dict(size=9, color="white", family=FONT_BODY),
        ),
        row=3, col=2
    )

    epa_vals = [row["epa_pass_def"], row["epa_rush_def"]]
    epa_pcts = [get_pct("epa_pass_def", ascending=False), get_pct("epa_rush_def", ascending=False)]
    
    fig.add_trace(
        go.Bar(
            y=["Pass EPA", "Rush EPA"],
            x=epa_vals,
            orientation="h",
            marker_line_width=0,
            text=[f"{v:.2f}" for v in epa_vals],
            textposition="outside",
            textfont=dict(size=10, color=TEXT_ACC, family=FONT_BODY),
            textangle=-90,
            marker_color=[color_for_pct(p) for p in epa_pcts],
            name="EPA Split",
        ),
        row=3, col=3,
    )

    havoc_vals = [row["sacks"], row["run_stuffs"], row["turnovers_forced"]]
    havoc_pcts = [get_pct("sacks"), get_pct("run_stuffs"), get_pct("turnovers_forced")]

    fig.add_trace(
        go.Bar(
            x=["SACKS", "STUFFS", "TOVS"],
            y=havoc_vals,
            marker_line_width=0,
            text=havoc_vals,
            textposition="outside",
            textfont=dict(size=10, color=TEXT_ACC, family=FONT_BODY),
            name="Offensive Conversion",
            marker_color=[color_for_pct(p) for p in havoc_pcts],
        ),
        row=3, col=4,
    )

    #── ROW 4: Summary table ────────────────────────────────────
    add_section_heading("◈  SUMMARY", y_pos=0.26)
    
    stats_labels = [
        "Off EPA/Pass", "Off EPA/Rush", "3rd Down Conv %",
        "Early Down SR", "Red Zone TD %", "Def EPA/Pass",
        "Def EPA/Rush", "Pressure %", "Sacks",
        "Run Stuffs", "Turnovers Forced", "Special Teams EPA",
    ]
    stats_vals = [
        row["epa_pass_off"], row["epa_rush_off"], row["third_down_cr"],
        row["early_down_sr"], row["red_zone_td_rate"], row["epa_pass_def"],
        row["epa_rush_def"], row["pressure_rate"], row["sacks"],
        row["run_stuffs"], row["turnovers_forced"], row["special_teams_epa"],
    ]

    # Format
    def fmt(v, is_pct=False, is_epa=False, is_total=False):
        if is_pct:
            return f"{v:.1%}"
        if is_epa:
            return f"{v:.2f}"
        if is_total:
            return f"{v:.0f}"
        else:
            return f"{v:.1f}"

    # False = plain, True = pct, "epa" = epa format
    fmt_types = ["epa", "epa", "pct", "pct", "pct", "epa", "epa", "pct", "total", "total", "total", None]

    stats_formatted = [
        fmt(v, is_pct=(t == "pct"), is_epa=(t == "epa"), is_total=(t == "total")) for v, t in zip(stats_vals, fmt_types)
    ]

    # League rank for each
    rank_cols = [
        "epa_pass_off", "epa_rush_off", "third_down_cr",
        "early_down_sr", "red_zone_td_rate", "epa_pass_def",
        "epa_rush_def", "pressure_rate", "sacks",
        "run_stuffs", "turnovers_forced", "special_teams_epa",
    ]

    ascending_for_rank = [False, False, False, False, False, True, True, False, False, False, False, False]

    stats_ranks = []
    
    for col, asc in zip(rank_cols, ascending_for_rank):
        if col in league_df.columns:
            r = int(league_df[col].rank(ascending=asc)[league_df["team"] == team].values[0])
            stats_ranks.append(rank_suffix(r))
        else:
            stats_ranks.append("N/A")

    stats_pct_vals = []
    for col, asc in zip(rank_cols, ascending_for_rank):
        if col in league_df.columns:
            if col in ["epa_per_pass_def", "epa_per_rush_def"]:
                p = get_pct(col, ascending=False)
            else:
                p = get_pct(col)
            stats_pct_vals.append(p)
        else:
            stats_pct_vals.append(50.0)
        

    # Split into 3 groups of 4 for 3-column table layout
    def chunk(lst, n):
        return [lst[i:i+n] for i in range(0, len(lst), n)]

    lbl_chunks = chunk(stats_labels, 4)
    val_chunks = chunk(stats_formatted, 4)
    rnk_chunks = chunk(stats_ranks, 4)
    clr_chunks = chunk([color_for_pct(p) for p in stats_pct_vals], 4)

    fig.add_trace(
        go.Table(
            columnwidth=[2, 1.2, 0.8, 2, 1.2, 0.8, 2, 1.2, 0.8],
            header=dict(
                values=["<b>STAT</b>", "<b>VALUE</b>", "<b>RANK</b>"] * 3,
                fill_color=BG_CARD,
                line_color=GRID_CLR,
                font=dict(size=10, color=TEXT_SEC, family=FONT_TITLE),
                align="center",
                height=30,
            ),
            cells=dict(
                values=[
                    lbl_chunks[0], val_chunks[0], rnk_chunks[0],
                    lbl_chunks[1], val_chunks[1], rnk_chunks[1],
                    lbl_chunks[2] if len(lbl_chunks) > 2 else ["","","",""],
                    val_chunks[2] if len(val_chunks) > 2 else ["","","",""],
                    rnk_chunks[2] if len(rnk_chunks) > 2 else ["","","",""],
                ],
                align=["left", "center", "center"] * 3,
                fill_color=[
                    [BG_CARD]*4,
                    clr_chunks[0],
                    [BG_PANEL]*4,
                    [BG_CARD]*4,
                    clr_chunks[1],
                    [BG_PANEL]*4,
                    [BG_CARD]*4,
                    clr_chunks[2] if len(clr_chunks) > 2 else [BG_CARD]*4,
                    [BG_PANEL]*4,
                ],
                font=dict(size=10, color=TEXT_PRI, family=FONT_BODY),
                line_color=GRID_CLR,
                height=25,
            ),
        ),
        row=4, col=1,
    )
    
    fig.update_yaxes(tickformat=".0%", range=[0, 1], row=2, col=4)
    fig.update_yaxes(tickformat=".0%", range=[0, 1], row=3, col=1)
    fig.update_yaxes(range=[0, max(havoc_vals) * 1.1], row=3, col=4)
    
    update_vert(row=2, col=4)
    update_vert(row=3, col=1)
    update_vert(row=3, col=4)
    
    update_hor(row=2, col=3)
    update_hor(row=3, col=3)

    fig.update_layout(
        paper_bgcolor=BG_DARK,
        plot_bgcolor=BG_PANEL,
        width=1600,
        height=900,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )

    fig.write_image(f"{OUT_DIR}/{team}_{SEASON}_team_profile.png")
    
test_teams = team_stats["team"].unique()[:3]  # just generate for first 3 teams for testing

for team in test_teams:
    create_team_profile(team_stats, team)
    
print("All images generated in figures/images/team_profiles/ folder")
