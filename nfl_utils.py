from pathlib import Path

class Colors:
    def __init__(self):
        self.team_colors = {
            "ARI": "#97233F",
            "ATL": "#A71930",
            "BAL": "#241773",
            "BUF": "#00338D",
            "CAR": "#0085CA",
            "CHI": "#0B162A",
            "CIN": "#FB4F14",
            "CLE": "#311D00",
            "DAL": "#041E42",
            "DEN": "#FB4F14",
            "DET": "#0076B6",
            "GB": "#203731",
            "HOU": "#03202F",
            "IND": "#002C5F",
            "JAX": "#006778",
            "KC": "#E31837",
            "LV": "#000000",
            "LAC": "#0080C6",
            "LA": "#003594",
            "MIA": "#008E97",
            "MIN": "#4F2683",
            "NE": "#002244",
            "NO": "#D3BC8D",
            "NYG": "#0B2265",
            "NYJ": "#115740",
            "PHI": "#004C54",
            "PIT": "#FFB612",
            "SEA": "#69BE28",
            "SF": "#AA0000",
            "TB": "#D50A0A",
            "TEN": "#4B92DB",
            "WAS": "#5A1414"
        }

    def get_color(self, team):
        return self.team_colors.get(team.upper(), "#808080")

class Logos:
    def __init__(self, folder="NFL_Logos"):
        self.folder = Path(folder)
        self.team_logos = {
            "NE": self.folder / "new-england-patriots-logo.png",
            "DAL": self.folder / "dallas-cowboys-logo.png",
            "GB": self.folder / "green-bay-packers-logo.png",
            "KC": self.folder / "kansas-city-chiefs-logo.png",
            "SF": self.folder / "san-francisco-49ers-logo.png",
            "SEA": self.folder / "seattle-seahawks-logo.png",
            "PIT": self.folder / "pittsburgh-steelers-logo.png",
            "DEN": self.folder / "denver-broncos-logo.png",
            "CHI": self.folder / "chicago-bears-logo.png",
            "MIN": self.folder / "minnesota-vikings-logo.png",
            "NYG": self.folder / "new-york-giants-logo.png",
            "PHI": self.folder / "philadelphia-eagles-logo.png",
            "WAS": self.folder / "washington-commanders-logo.png",
            "TB": self.folder / "tampa-bay-buccaneers-logo.png",
            "ATL": self.folder / "atlanta-falcons-logo.png",
            "CAR": self.folder / "carolina-panthers-logo.png",
            "NO": self.folder / "new-orleans-saints-logo.png",
            "CIN": self.folder / "cincinnati-bengals-logo.png",
            "CLE": self.folder / "cleveland-browns-logo.png",
            "LV": self.folder / "vegas-raiders-logo.png",
            "LAC": self.folder / "los-angeles-chargers-logo.png",
            "LA": self.folder / "la-rams-logo.png",
            "JAX": self.folder / "jacksonville-jaguars-logo.png",
            "TEN": self.folder / "tennessee-titans-logo.png",
            "MIA": self.folder / "miami-dolphins-logo.png",
            "BAL": self.folder / "baltimore-ravens-logo.png",
            "IND": self.folder / "indianapolis-colts-logo.png",
            "HOU": self.folder / "houston-texans-logo.png",
            "DET": self.folder / "detroit-lions-logo.png",
            "NYJ": self.folder / "new-york-jets-logo.png",
            "ARI": self.folder / "arizona-cardinals-logo.png",
            "BUF": self.folder / "buffalo-bills-logo.png"
        }

    def get_logo(self, team):
        path = self.team_logos.get(team.upper())
        if path and path.exists():
            return path
        return None
