"""Unit tests for the CLI entrypoint."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
from click.testing import CliRunner

from scripts.cli import cli

DUMMY_CSV = Path(__file__).parent / "dummy_data.csv"


def _fake_run(input_df, output_dir):
    """Return minimal pairwise + clustered DataFrames without running Splink."""
    pairwise = pd.DataFrame(
        {
            "unique_id_l": ["br_001"],
            "unique_id_r": ["ts_001"],
            "match_probability": [0.95],
        }
    )
    clustered = pd.DataFrame(
        {
            "unique_id": ["br_001", "ts_001"],
            "cluster_id": [1, 1],
            "source_name": ["ballotready", "techspeed"],
            "first_name": ["Jane", "Janet"],
            "last_name": ["Doe", "Doe"],
        }
    )
    # Write the CSVs that the real pipeline would produce
    output_dir.mkdir(parents=True, exist_ok=True)
    pairwise.to_csv(output_dir / "pairwise_predictions.csv", index=False)
    clustered.to_csv(output_dir / "clustered_candidacies.csv", index=False)
    return pairwise, clustered


def test_help():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Entity resolution CLI" in result.output


def test_match_help():
    result = CliRunner().invoke(cli, ["match", "--help"])
    assert result.exit_code == 0
    assert "--input" in result.output


@patch("scripts.cli.run", side_effect=_fake_run)
def test_match_with_csv(mock_run, tmp_path):
    """match subcommand reads a CSV, calls run(), and writes output."""
    result = CliRunner().invoke(
        cli,
        ["match", "--input", str(DUMMY_CSV), "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, f"CLI failed:\n{result.output}\n{result.exception}"
    mock_run.assert_called_once()

    # Verify output files were written
    assert (tmp_path / "pairwise_predictions.csv").exists()
    assert (tmp_path / "clustered_candidacies.csv").exists()
    assert (tmp_path / "input.parquet").exists()


@patch("scripts.cli.run", side_effect=_fake_run)
def test_match_missing_file(mock_run):
    """match fails gracefully when input file doesn't exist."""
    result = CliRunner().invoke(
        cli, ["match", "--input", "/nonexistent/file.csv"]
    )
    assert result.exit_code != 0


def test_match_requires_input():
    """match fails when --input is not provided."""
    result = CliRunner().invoke(cli, ["match"])
    assert result.exit_code != 0
    assert "Missing option" in result.output or "required" in result.output.lower()
