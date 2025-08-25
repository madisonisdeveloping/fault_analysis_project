# import pandas lib (data manipulation and CSV reading)
import pandas as pd

# function: loads power data from .csv
def load_power_data(filepath):
    return pd.read_csv(filepath)