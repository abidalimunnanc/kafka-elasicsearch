import pandas as pd

def load_data(path):
    return pd.read_csv(path)

# Load and inspect
df = load_data("data/raw/saas_churn.csv")
print(df.head())
