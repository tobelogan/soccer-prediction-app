import warnings
from pandas.errors import PerformanceWarning

warnings.filterwarnings("error", category=PerformanceWarning)
