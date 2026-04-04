# scripts/audit_false_negatives.py
"""
Audit false negatives: find input records that didn't match but plausibly should have.

Usage:
    uv run python -m scripts.cli audit false-negatives --entity-type candidacy --results-dir results/candidacy/
"""

from pathlib import Path

import pandas as pd
from rapidfuzz.distance import JaroWinkler

from scripts.entity_config import EntityConfig


def _name_similar(a: str | None, b: str | None, threshold: float = 0.88) -> bool:
    """Check if two names are similar using JW similarity or exact match."""
    if pd.isna(a) or pd.isna(b):
        return False
    a, b = str(a).lower().strip(), str(b).lower().strip()
    if a == b:
        return True
    return JaroWinkler.similarity(a, b) >= threshold


def run_false_negatives(
    input_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    clustered_df: pd.DataFrame,
    results_dir: Path,
    config: EntityConfig,
    sample_n: int = 20,
) -> pd.DataFrame:
    """Find suspicious non-matches and write audit CSV."""
    # Find singletons — records in clusters of size 1 (unmatched)
    cluster_sizes = clustered_df.groupby("cluster_id").size()
    singleton_clusters = cluster_sizes[cluster_sizes == 1].index
    singletons = clustered_df[clustered_df["cluster_id"].isin(singleton_clusters)]
    print(f"Total clustered records: {len(clustered_df):,}")
    print(f"Singleton (unmatched) records: {len(singletons):,}")

    providers = sorted(input_df["source_name"].unique())
    print(f"Providers: {providers}")

    # Check if suspicious pairs were generated but fell below the clustering threshold
    pairwise_ids = set()
    if len(pairwise_df) > 0 and "unique_id_l" in pairwise_df.columns:
        pairwise_ids = set(zip(pairwise_df["unique_id_l"], pairwise_df["unique_id_r"]))

    # Pre-group input_df for O(1) candidate lookups
    group_cols = config.false_negative_group_cols
    candidate_groups = {k: v for k, v in input_df.groupby(group_cols)}

    # For each singleton, look for plausible matches in other providers
    suspicious_pairs = []
    singleton_sample = singletons.sample(
        n=min(sample_n * 5, len(singletons)), random_state=42
    )

    for _, singleton in singleton_sample.iterrows():
        s_provider = singleton["source_name"]
        s_last = singleton.get("last_name")

        if pd.isna(s_last):
            continue

        # Build the lookup key from the same group columns
        try:
            lookup_key_parts = []
            for col in group_cols:
                val = singleton.get(col)
                if pd.isna(val):
                    break
                lookup_key_parts.append(val)
            else:
                # All group cols were non-null
                for other_provider in providers:
                    if other_provider == s_provider:
                        continue

                    # Replace the source_name in the key with the other provider
                    other_key = tuple(
                        other_provider if col == "source_name" else part
                        for col, part in zip(group_cols, lookup_key_parts)
                    )
                    candidates = candidate_groups.get(other_key)
                    if candidates is None:
                        continue

                    for _, cand in candidates.iterrows():
                        if not _name_similar(s_last, cand.get("last_name")):
                            continue

                        first_ok = _name_similar(
                            singleton.get("first_name"),
                            cand.get("first_name"),
                            threshold=0.85,
                        )
                        email_ok = (
                            singleton.get("email")
                            and cand.get("email")
                            and singleton["email"] == cand["email"]
                        )
                        phone_ok = (
                            singleton.get("phone")
                            and cand.get("phone")
                            and singleton["phone"] == cand["phone"]
                        )

                        if not (first_ok or email_ok or phone_ok):
                            continue

                        pair_key = (singleton["unique_id"], cand["unique_id"])
                        was_generated = (
                            pair_key in pairwise_ids
                            or (pair_key[1], pair_key[0]) in pairwise_ids
                        )

                        row = {"was_in_pairwise_predictions": was_generated}
                        for col in config.audit_display_columns:
                            row[f"{col}_l"] = singleton.get(col)
                            row[f"{col}_r"] = cand.get(col)
                        suspicious_pairs.append(row)
        except (KeyError, TypeError):
            continue

        if len(suspicious_pairs) >= sample_n:
            break

    out_df = pd.DataFrame(suspicious_pairs[:sample_n])

    if len(out_df) == 0:
        print("\nNo suspicious non-matches found in sampled singletons.")
        return out_df

    out_path = results_dir / "audit_false_negatives.csv"
    out_df.to_csv(out_path, index=False)

    # Diagnostics
    in_pw = out_df["was_in_pairwise_predictions"].sum()
    not_in_pw = len(out_df) - in_pw
    print(f"\nSuspicious non-matches found: {len(out_df)}")
    print(
        f"  Were in pairwise predictions (model scored but below cluster threshold): {in_pw}"
    )
    print(f"  NOT in pairwise predictions (blocking rules missed them): {not_in_pw}")
    print(f"\nWritten to {out_path}")

    return out_df
