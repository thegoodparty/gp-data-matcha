# scripts/analysis/baseline_3source.py
"""Pull prematch data from Databricks, filter to 3 sources (no gp_api), write CSV."""
import pandas as pd
from scripts.databricks_io import read_table
from scripts.cli import _normalize_to_strings

TABLE = "goodparty_data_catalog.dbt.int__er_prematch_candidacy_stages"
BASELINE_CSV = "data/baseline_3source.csv"

df = read_table(TABLE)
df = _normalize_to_strings(df)

print(f"Total rows pulled: {len(df):,}")
print(f"Source distribution:\n{df['source_name'].value_counts().to_string()}")

df_filtered = df[df["source_name"] != "gp_api"]
df_filtered.to_csv(BASELINE_CSV, index=False)
print(f"\nWrote {len(df_filtered):,} rows (3-source, no gp_api) to {BASELINE_CSV}")
