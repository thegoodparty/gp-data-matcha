"""
CLI entrypoint for entity resolution.

Usage:
    uv run python scripts/cli.py match --input data/input.csv
"""

from pathlib import Path

import click
import pandas as pd

from scripts.pipeline import run  # noqa: E402

_PROJECT_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_RESULTS = _PROJECT_DIR / "results"


def _load_input(input_value: str) -> pd.DataFrame:
    """Load input DataFrame from a CSV path."""
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
    help="Path to prematch CSV file.",
)
@click.option(
    "--output-dir",
    "output_dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory for local results. Defaults to results/ in the project root.",
)
def match(
    input_value: str,
    output_dir: Path | None,
) -> None:
    """Run Splink entity resolution on prematch data."""
    if output_dir is None:
        output_dir = _DEFAULT_RESULTS

    input_df = _load_input(input_value)
    pairwise_df, clustered_df = run(input_df=input_df, output_dir=output_dir)

    # Persist input so standalone audit commands can read it later
    input_df.to_parquet(output_dir / "input.parquet", index=False)


if __name__ == "__main__":
    cli()
