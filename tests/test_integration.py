"""Integration tests: CLI round-trip through Databricks.

Run with:  uv run pytest -m integration
"""

import pytest
from click.testing import CliRunner


@pytest.mark.integration
def test_match_databricks_round_trip(databricks_tables):
    """Full pipeline: read from Databricks, run Splink, write results back."""
    from scripts.cli import cli
    from scripts.databricks_io import read_table

    ctx = databricks_tables
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "match",
            "--input",
            ctx["input_fqn"],
            "--output-cluster-table",
            ctx["output_cluster_fqn"],
            "--output-pairwise-table",
            ctx["output_pairwise_fqn"],
            "--output-dir",
            str(ctx["output_dir"]),
            "--overwrite",
        ],
    )

    assert result.exit_code == 0, f"CLI failed:\n{result.output}\n{result.exception}"

    # ── Read back from Databricks ──
    clustered_df = read_table(ctx["output_cluster_fqn"])
    pairwise_df = read_table(ctx["output_pairwise_fqn"])

    # ── Clustered table assertions ──
    assert len(clustered_df) > 0, "Clustered table is empty"
    for col in ["unique_id", "cluster_id", "source_name", "first_name", "last_name"]:
        assert col in clustered_df.columns, f"Missing column: {col}"
    assert clustered_df["cluster_id"].notna().all(), "cluster_id has nulls"

    # ── Pairwise table assertions ──
    assert len(pairwise_df) > 0, "Pairwise table is empty"
    for col in ["unique_id_l", "unique_id_r", "match_probability"]:
        assert col in pairwise_df.columns, f"Missing column: {col}"

    # ── Local output files ──
    assert (ctx["output_dir"] / "pairwise_predictions.csv").exists()
    assert (ctx["output_dir"] / "clustered_candidacies.csv").exists()
    assert (ctx["output_dir"] / "input.parquet").exists()

    # ── At least 1 cross-source cluster (proves matching worked) ──
    multi_source = (
        clustered_df.groupby("cluster_id")["source_name"].nunique() > 1
    ).sum()
    assert (
        multi_source >= 1
    ), f"Expected at least 1 cross-source cluster, got {multi_source}"
