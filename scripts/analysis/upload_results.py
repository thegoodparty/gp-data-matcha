# scripts/analysis/upload_results.py
"""Upload already-reviewed local CSV results to Databricks.

Reads the CSVs that were audited and approved, and uploads them
to the specified Databricks tables. Avoids re-running matching
(which would produce different results due to random sampling).
"""
import sys

import pandas as pd

from scripts.databricks_io import write_table

RESULTS_DIR = "results/candidacy_stage"
CLUSTERED_CSV = f"{RESULTS_DIR}/clustered_candidacies.csv"
PAIRWISE_CSV = f"{RESULTS_DIR}/pairwise_predictions.csv"


def upload(cluster_table: str, pairwise_table: str) -> None:
    clustered_df = pd.read_csv(CLUSTERED_CSV)
    pairwise_df = pd.read_csv(PAIRWISE_CSV)

    print(f"Uploading {len(clustered_df):,} clustered rows to {cluster_table}")
    write_table(clustered_df, cluster_table, overwrite=True)

    print(f"Uploading {len(pairwise_df):,} pairwise rows to {pairwise_table}")
    write_table(pairwise_df, pairwise_table, overwrite=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python -m scripts.analysis.upload_results"
            " <cluster_table> <pairwise_table>"
        )
        sys.exit(1)
    upload(sys.argv[1], sys.argv[2])
