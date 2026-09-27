LEAGUES = {
    "E0": "English Premier League",
    "SP1": "Spain La Liga",
    "D1": "Germany Bundesliga",
    "I1": "Italy Serie A",
    "F1": "France Ligue 1",
    "N1": "Netherlands Eredivisie",
    "P1": "Portugal Primeira Liga",
}

DEFAULT_MODEL_LEAGUES = ["E0", "SP1", "D1", "I1", "F1"]

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
FIXTURES_URL = "https://www.football-data.co.uk/matches/resources/fixtures.csv"

ROLLING_WINDOW = 8
ELO_K = 24.0
ELO_HOME_ADVANTAGE = 65.0
INITIAL_ELO = 1500.0
MAX_GOALS_MATRIX = 10
RANDOM_STATE = 42
