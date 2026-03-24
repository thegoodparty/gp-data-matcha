"""
CLI entrypoint for entity resolution.

Usage:
    uv run python scripts/cli.py match --input data/input.csv
    uv run python scripts/cli.py match --input catalog.schema.table --output-cluster-table catalog.schema.output
"""

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd

from scripts.pipeline import run  # noqa: E402

_PROJECT_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_RESULTS = _PROJECT_DIR / "results"


def _load_input(input_value: str) -> pd.DataFrame:
    """Load input DataFrame from CSV path or Databricks FQN."""
    from scripts.databricks_io import is_databricks_fqn

    if is_databricks_fqn(input_value):
        from scripts.databricks_io import read_table

        print(f"Reading from Databricks: {input_value}")
        df = read_table(input_value)
        # Normalize to all-string dtypes to match pd.read_csv(dtype=str) behavior.
        # Databricks returns native types which can cause Splink EM training to
        # fail (e.g. "m values not fully trained") because it handles typed
        # columns differently during parameter estimation.
        for col in df.columns:
            sample = df[col].dropna().iloc[0] if df[col].notna().any() else None
            if isinstance(sample, (np.ndarray, list)):
                # Array columns (e.g. first_name_aliases) → JSON strings
                df[col] = df[col].apply(
                    lambda v: (
                        json.dumps(v.tolist() if isinstance(v, np.ndarray) else v)
                        if isinstance(v, (np.ndarray, list))
                        else v
                    )
                )
            else:
                df[col] = df[col].where(
                    df[col].isna(),
                    df[col].astype(str).str.replace(r"\.0$", "", regex=True),
                )
        return df
    else:
        path = Path(input_value)
        if not path.exists():
            raise click.BadParameter(f"File not found: {path}")
        print(f"Reading from CSV: {path}")
        return pd.read_csv(path, dtype=str)


@click.group()
def cli():
    """Entity resolution CLI."""


@cli.command()
@click.option(
    "--input",
    "input_value",
    required=True,
    type=str,
    help="Path to prematch CSV file or Databricks FQN (catalog.schema.table).",
)
@click.option(
    "--output-dir",
    "output_dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory for local results. Defaults to results/ in the project root.",
)
@click.option(
    "--output-cluster-table",
    "output_cluster_table",
    default=None,
    type=str,
    help="Databricks FQN to upload clustered results (catalog.schema.table).",
)
@click.option(
    "--output-pairwise-table",
    "output_pairwise_table",
    default=None,
    type=str,
    help="Databricks FQN to upload pairwise predictions (catalog.schema.table).",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite existing Databricks output tables.",
)
def match(
    input_value: str,
    output_dir: Path | None,
    output_cluster_table: str | None,
    output_pairwise_table: str | None,
    overwrite: bool,
) -> None:
    """Run Splink entity resolution on prematch data."""
    if output_dir is None:
        output_dir = _DEFAULT_RESULTS

    input_df = _load_input(input_value)
    pairwise_df, clustered_df = run(input_df=input_df, output_dir=output_dir)

    # Persist input so standalone audit commands can read it later
    input_df.to_parquet(output_dir / "input.parquet", index=False)

    # Upload to Databricks if requested
    if output_cluster_table or output_pairwise_table:
        from scripts.databricks_io import write_table

        if output_cluster_table:
            write_table(clustered_df, output_cluster_table, overwrite=overwrite)
        if output_pairwise_table:
            write_table(pairwise_df, output_pairwise_table, overwrite=overwrite)


if __name__ == "__main__":
    cli()
