"""
Splink entity resolution: multi-source candidacy matching.

Usage:
    uv run python -m scripts.cli match --input data/input.csv
    uv run python -m scripts.cli match --input catalog.schema.table --output-cluster-table catalog.schema.output

Input:  CSV or Databricks table from int__er_prematch_candidacy_stages
Output: results/pairwise_predictions.csv
        results/clustered_candidacies.csv
        results/match_weights_chart.{html,png}
        results/m_u_parameters_chart.{html,png}
"""

import json
from pathlib import Path

import pandas as pd
import splink.comparison_level_library as cll
import splink.internals.comparison_library as cl
from splink import Linker, SettingsCreator, block_on
from splink.blocking_rule_library import CustomRule
from splink.comparison_library import CustomComparison
from splink.internals.duckdb.database_api import DuckDBAPI

PREDICT_THRESHOLD = 0.01
CLUSTER_THRESHOLD = 0.95


def load_and_prepare(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Clean nulls, parse aliases, return one DataFrame per source (sorted by name)."""
    print(f"Preparing {len(df):,} rows")
    print(f"\nSource distribution:\n{df['source_name'].value_counts().to_string()}")

    df["election_date"] = pd.to_datetime(
        df["election_date"], errors="coerce"
    ).dt.date.astype(str)
    df["election_date"] = df["election_date"].replace("NaT", None)

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


SETTINGS = SettingsCreator(
    link_type="link_only",
    unique_id_column_name="unique_id",
    # These comparisons configs define how Splink does pairwise calculations
    # between records, including match types (exact/fuzzy/array intersection).
    # Multiple score thresholds show up as separate m/u parameters.
    comparisons=[
        # ── Candidate-level ──
        cl.JaroWinklerAtThresholds(
            "last_name", score_threshold_or_thresholds=[0.95, 0.88]
        ).configure(term_frequency_adjustments=True),
        CustomComparison(
            output_column_name="first_name",
            comparison_levels=[
                cll.NullLevel("first_name"),
                cll.ExactMatchLevel("first_name").configure(
                    tf_adjustment_column="first_name",
                ),
                cll.ArrayIntersectLevel("first_name_aliases", min_intersection=1),
                cll.JaroWinklerLevel("first_name", distance_threshold=0.92),
                cll.ElseLevel(),
            ],
        ),
        cl.ExactMatch("party"),
        cl.ExactMatch("email"),
        cl.ExactMatch("phone"),
        # ── Race / election-level ──
        # These provide evidence that two candidates ran in the same race.
        # The person identity filter (below) prevents same-race,
        # different-candidate pairs from being false positives.
        cl.ExactMatch("state"),
        cl.ExactMatch("election_date"),
        # Office name with a 0.75 tier to catch cross-source formatting
        # differences (e.g. "durham school board" vs "durham county board of
        # education"). FPs from completely different offices (JW < 0.75) are
        # excluded; TPs with the same office reformatted (JW >= 0.76) are kept.
        cl.JaroWinklerAtThresholds(
            "official_office_name",
            score_threshold_or_thresholds=[0.95, 0.88, 0.75],
        ),
        cl.ExactMatch("district_identifier"),
    ],
    # All blocking rules are evaluated and candidate pairs are unioned together
    # before the pairwise matching. These are the comparisons used when
    # actually applying the estimates, not for training.
    blocking_rules_to_generate_predictions=[
        block_on("br_race_id"),
        CustomRule(
            "l.state = r.state"
            " AND l.election_date = r.election_date"
            " AND jaro_winkler_similarity(l.official_office_name,"
            " r.official_office_name) >= 0.88"
            " AND l.last_name = r.last_name",
            sql_dialect="duckdb",
        ),
        block_on("state", "last_name", "election_date"),
        CustomRule(
            "l.state = r.state"
            " AND l.election_date = r.election_date"
            " AND jaro_winkler_similarity(l.official_office_name,"
            " r.official_office_name) >= 0.88"
            " AND jaro_winkler_similarity(l.last_name,"
            " r.last_name) >= 0.88",
            sql_dialect="duckdb",
        ),
        block_on("phone"),
        block_on("email"),
    ],
    retain_intermediate_calculation_columns=True,
    additional_columns_to_retain=[
        "source_name",
        "source_id",
        "candidate_office",
        "office_level",
        "office_type",
        "district_raw",
        "seat_name",
        "br_race_id",
        "br_candidacy_id",
        "election_stage",
    ],
)

EM_TRAINING_BLOCKS = [
    # Block on last_name + state + election_date to train first_name (and
    # office/district) cleanly. Blocking on last_name alone produces too many
    # same-race different-person pairs, which inflates the first_name
    # non-agreement m probability and weakens its negative signal.
    #
    # These are the blocks used for parameter estimation (m probabilities), not
    # for evaluating all candidate pairs
    ("last_name", "state", "election_date"),
    ("first_name",),
    ("email",),
    ("state", "election_date"),
]


def train_model(linker: Linker) -> None:
    """Estimate u via random sampling, then m via EM on each comparison column."""
    linker.training.estimate_u_using_random_sampling(max_pairs=5_000_000)
    for cols in EM_TRAINING_BLOCKS:
        linker.training.estimate_parameters_using_expectation_maximisation(
            block_on(*cols), fix_u_probabilities=True
        )


def predict_and_cluster(linker: Linker) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict matches, apply person identity filter, cluster."""
    predictions = linker.inference.predict(
        threshold_match_probability=PREDICT_THRESHOLD
    )

    # Post-prediction filters applied in DuckDB (single source of truth for
    # both the pairwise CSV export and clustering).
    # Uses Splink's private _db_api._con because Splink 4 doesn't expose a
    # public method for executing raw SQL on the linker's DuckDB connection.
    pred_table = predictions.physical_name
    pre_count = linker._db_api._con.execute(
        f"SELECT count(*) FROM {pred_table}"
    ).fetchone()[0]
    print(f"Pairwise predictions: {pre_count:,} pairs above {PREDICT_THRESHOLD}")

    if pre_count == 0:
        print("WARNING: No predictions found.")
        return pd.DataFrame(), pd.DataFrame()

    # 1. Person identity: last name agreement + (first name OR contact info)
    # 2. Race-level: office name similarity — either JW >= 0.75 (gamma > 0)
    #    OR the office names share a meaningful locality token. This handles
    #    cross-source formatting differences like "mayor of brodhead" vs
    #    "brodhead city mayor" (JW 0.557) where the place name matches but
    #    the structural words differ.
    # 3. Race ID: exclude mismatched br_race_id unless office names match well
    #    (gamma >= 2, i.e. JW >= 0.88), since sources sometimes assign
    #    different race IDs to the same race.
    office_stop_words = (
        "'city','of','the','county','board','council','school','district',"
        "'mayor','alderperson','trustee','at','large','zone','ward','seat',"
        "'position','commission','precinct','town','village','member',"
        "'councilmember','supervisor','supervisors','commissioner','judge',"
        "'branch','education','unified','public','elementary','consolidated',"
        "'central','special','independent','office','clerk','treasurer',"
        "'coroner','sheriff','magistrate','property','value','administrator',"
        "'emergency','services','director','justice','peace','representative',"
        "'house','representatives','legislature','legislative','metro',"
        "'and','for','no.','odd','unexpired'"
    )
    linker._db_api._con.execute(f"""
        CREATE OR REPLACE TABLE {pred_table} AS
        SELECT * FROM {pred_table}
        WHERE gamma_last_name > 0
          AND (gamma_first_name > 0 OR gamma_email > 0 OR gamma_phone > 0)
          AND (
            gamma_official_office_name > 0
	    -- Catches offices with poor similarity due to word order, but
            -- overlapping tokens. A good candidate to move to embedding-
            -- based matching next
            OR list_has_any(
              list_filter(
                string_split(lower(official_office_name_l), ' '),
                x -> len(x) > 1
                  AND NOT list_contains([{office_stop_words}], x)
                  AND NOT regexp_matches(x, '^\\d+$')
              ),
              list_filter(
                string_split(lower(official_office_name_r), ' '),
                x -> len(x) > 1
                  AND NOT list_contains([{office_stop_words}], x)
                  AND NOT regexp_matches(x, '^\\d+$')
              )
            )
          )
          AND NOT (
            br_race_id_l IS NOT NULL
            AND br_race_id_r IS NOT NULL
            AND br_race_id_l != br_race_id_r
            AND gamma_official_office_name < 2
          )
    """)
    pairwise_df = predictions.as_pandas_dataframe()
    dropped = pre_count - len(pairwise_df)
    if dropped > 0:
        print(f"Person + race filter: removed {dropped:,} pairs")

    clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
        predictions, threshold_match_probability=CLUSTER_THRESHOLD
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
) -> None:
    """Write CSVs and diagnostic charts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert list/array columns to single-line JSON strings before CSV export.
    # numpy array repr wraps long arrays across multiple lines, which breaks
    # CSV parsers that don't support multiline quoted fields (e.g. Databricks).
    def to_json(v):
        # Databricks returns np.arrays; CSV paths return parsed Python lists.
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "[]"
        if isinstance(v, str):
            return v
        if hasattr(v, "tolist"):
            return json.dumps(v.tolist())
        return json.dumps(list(v))

    if len(pairwise_df) > 0:
        for col in ["first_name_aliases_l", "first_name_aliases_r"]:
            pairwise_df[col] = pairwise_df[col].apply(to_json)

    pairwise_df.to_csv(output_dir / "pairwise_predictions.csv", index=False)
    if len(clustered_df) > 0:
        clustered_df["first_name_aliases"] = clustered_df["first_name_aliases"].apply(
            to_json
        )
        clustered_df.to_csv(output_dir / "clustered_candidacies.csv", index=False)

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


def run(input_df: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare data, train, predict, cluster, save. Returns (pairwise_df, clustered_df)."""
    source_dfs = load_and_prepare(input_df)
    linker = Linker(source_dfs, SETTINGS, DuckDBAPI())
    train_model(linker)
    pairwise_df, clustered_df = predict_and_cluster(linker)
    save_results(linker, pairwise_df, clustered_df, output_dir)
    return pairwise_df, clustered_df


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    df = pd.read_csv(project_dir / "data" / "input.csv", dtype=str)
    run(input_df=df, output_dir=project_dir / "results")
