from dash import Dash, dash_table, html
import pandas as pd
import dash_bootstrap_components as dbc

# Sample NFL-style data
df = pd.DataFrame({
    "Team": ["A", "B", "C", "D"],
    "Offense": [28, 10, 22, 34],
    "Defense": [14, 20, 30, 12],
    "SpecialTeams": [7, 3, 9, 5]
})

num_cols = ["Offense", "Defense", "SpecialTeams"]

# Normalize numeric columns for heatmap
norm_df = df[num_cols].apply(lambda col: (col - col.min()) / (col.max() - col.min()))

# Generate conditional formatting for heatmap
styles = []
for col in num_cols:
    for i, val in enumerate(norm_df[col]):
        if val < 0.5:
            t = val / 0.5
            r = int(215 + (255 - 215) * t)
            g = int(48 + (255 - 48) * t)
            b = int(39 + (191 - 39) * t)
        else:
            t = (val - 0.5) / 0.5
            r = int(255 + (26 - 255) * t)
            g = int(255 + (152 - 255) * t)
            b = int(191 + (80 - 191) * t)
        bg = f"rgb({r},{g},{b})"
        luminance = (0.299*r + 0.587*g + 0.114*b)/255
        color = "black" if luminance > 0.6 else "white"
        styles.append({
            "if": {"row_index": i, "column_id": col},
            "backgroundColor": bg,
            "color": color
        })

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H4("Sortable Heatmap Table"),
    dash_table.DataTable(
        df.to_dict('records'),
        columns=[{"name": c, "id": c} for c in df.columns],
        sort_action="native",
        style_data_conditional=styles,
        style_cell={'textAlign': 'center', 'padding': '6px', 'border': '1px solid #dee2e6'},
        style_header={'fontWeight': 'bold', 'backgroundColor': '#e9ecef', 'border': '1px solid #dee2e6'},
        style_table={'width': '100%', 'borderCollapse': 'collapse'},
        css=[{
            'selector': '.dash-table-container .dash-header .dash-table-sort-icon',
            'rule': 'display: none'
        }]
    )
], fluid=True)

if __name__ == "__main__":
    app.run(debug=True)
