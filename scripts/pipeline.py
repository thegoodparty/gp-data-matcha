# scripts/pipeline.py
"""
Splink entity resolution: multi-source record matching.

Usage:
    uv run python -m scripts.cli match --entity-type candidacy --input data/input.csv
    uv run python -m scripts.cli match --entity-type elected_official --input catalog.schema.table
"""

import json
from pathlib import Path

import pandas as pd
from splink import Linker, SettingsCreator, block_on
from splink.internals.duckdb.database_api import DuckDBAPI

from scripts.entity_config import EntityConfig


def load_and_prepare(df: pd.DataFrame, config: EntityConfig) -> list[pd.DataFrame]:
    """Clean nulls, parse aliases, return one DataFrame per source (sorted by name)."""
    print(f"Preparing {len(df):,} rows for {config.display_name}")
    print(f"\nSource distribution:\n{df['source_name'].value_counts().to_string()}")

    for col in config.date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date.astype(str)
            df[col] = df[col].replace("NaT", None)

    # Parse first_name_aliases from JSON array string built by the dbt prematch model
    df["first_name_aliases"] = df["first_name_aliases"].apply(
        lambda v: json.loads(v) if isinstance(v, str) else None
    )

    # Normalize nulls so Splink treats missing data correctly
    df = df.where(df.notna(), None)
    df = df.replace({"": None, "nan": None, "null": None})

    sources = sorted(df["source_name"].unique())
    source_dfs = []
    for src in sources:
        src_df = df[df["source_name"] == src].copy()
        print(f"  {src}: {len(src_df):,} records")
        source_dfs.append(src_df)

    return source_dfs


def build_settings(config: EntityConfig) -> SettingsCreator:
    """Build Splink SettingsCreator from entity config."""
    return SettingsCreator(
        link_type="link_only",
        unique_id_column_name="unique_id",
        comparisons=config.comparisons,
        blocking_rules_to_generate_predictions=config.blocking_rules_for_prediction,
        retain_intermediate_calculation_columns=True,
        additional_columns_to_retain=config.additional_columns_to_retain,
    )


def train_model(linker: Linker, config: EntityConfig) -> None:
    """Estimate u via random sampling, then m via EM on each comparison column."""
    linker.training.estimate_u_using_random_sampling(max_pairs=5_000_000)
    for cols in config.em_training_blocks:
        try:
            linker.training.estimate_parameters_using_expectation_maximisation(
                block_on(*cols), fix_u_probabilities=True
            )
        except Exception as e:
            print(f"WARNING: EM training on {cols} failed: {e}")
            print("Continuing with remaining training blocks...")


def predict_and_cluster(
    linker: Linker, config: EntityConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict matches, apply post-prediction filters, cluster."""
    predictions = linker.inference.predict(
        threshold_match_probability=config.predict_threshold
    )

    pred_table = predictions.physical_name
    pre_count = linker._db_api._con.execute(
        f"SELECT count(*) FROM {pred_table}"
    ).fetchone()[0]
    print(f"Pairwise predictions: {pre_count:,} pairs above {config.predict_threshold}")

    if pre_count == 0:
        print("WARNING: No predictions found.")
        return pd.DataFrame(), pd.DataFrame()

    # Apply post-prediction filters from config
    if config.post_prediction_filters:
        combined_filter = " AND ".join(
            f"({f.strip()})" for f in config.post_prediction_filters
        )
        linker._db_api._con.execute(f"""
            CREATE OR REPLACE TABLE {pred_table} AS
            SELECT * FROM {pred_table}
            WHERE {combined_filter}
        """)

    pairwise_df = predictions.as_pandas_dataframe()
    dropped = pre_count - len(pairwise_df)
    if dropped > 0:
        print(f"Post-prediction filters: removed {dropped:,} pairs")

    clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
        predictions, threshold_match_probability=config.cluster_threshold
    )
    clustered_df = clusters.as_pandas_dataframe()

    n_matched = (clustered_df.groupby("cluster_id").size() > 1).sum()
    n_cross = (clustered_df.groupby("cluster_id")["source_dataset"].nunique() > 1).sum()
    print(f"Matched clusters: {n_matched:,}  |  Cross-source: {n_cross:,}")
    if (within := n_matched - n_cross) > 0:
        print(f"WARNING: {within} within-source duplicate clusters found")

    return pairwise_df, clustered_df


def save_results(
    linker: Linker,
    pairwise_df: pd.DataFrame,
    clustered_df: pd.DataFrame,
    output_dir: Path,
    config: EntityConfig,
) -> None:
    """Write CSVs and diagnostic charts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    def to_json(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "[]"
        if isinstance(v, str):
            return v
        if hasattr(v, "tolist"):
            return json.dumps(v.tolist())
        return json.dumps(list(v))

    if len(pairwise_df) > 0:
        for col in pairwise_df.columns:
            if col.endswith("_aliases_l") or col.endswith("_aliases_r"):
                pairwise_df[col] = pairwise_df[col].apply(to_json)

    pairwise_df.to_csv(output_dir / "pairwise_predictions.csv", index=False)
    if len(clustered_df) > 0:
        for col in clustered_df.columns:
            if "aliases" in col:
                clustered_df[col] = clustered_df[col].apply(to_json)
        clustered_df.to_csv(output_dir / config.clustered_output_name, index=False)

    for name, method in [
        ("match_weights", "match_weights_chart"),
        ("m_u_parameters", "m_u_parameters_chart"),
    ]:
        try:
            chart = getattr(linker.visualisations, method)()
            chart.save(str(output_dir / f"{name}_chart.html"))
            chart.save(str(output_dir / f"{name}_chart.png"), scale_factor=2)
        except Exception as e:
            print(f"Could not save {name} chart: {e}")

    print(f"\nResults saved to {output_dir}/")


def run(
    input_df: pd.DataFrame, output_dir: Path, config: EntityConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare data, train, predict, cluster, save. Returns (pairwise_df, clustered_df)."""
    source_dfs = load_and_prepare(input_df, config)
    settings = build_settings(config)
    linker = Linker(source_dfs, settings, DuckDBAPI())
    train_model(linker, config)
    pairwise_df, clustered_df = predict_and_cluster(linker, config)
    save_results(linker, pairwise_df, clustered_df, output_dir, config)
    return pairwise_df, clustered_df
