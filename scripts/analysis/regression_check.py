# scripts/analysis/regression_check.py
"""Compare 3-source baseline vs 4-source clustered results.

Uses co-cluster pair sets (not raw cluster_id, which renumbers across runs).
Baseline has no GP rows, so use as-is. Filter only 4-source output.
"""

from itertools import combinations
from pathlib import Path

import pandas as pd

BASELINE_DIR = Path("results/candidacy_stage_baseline")
FOUR_SRC_DIR = Path("results/candidacy_stage")
CLUSTERED_FILE = "clustered_candidacies.csv"


def co_cluster_pairs(df: pd.DataFrame) -> set[tuple[str, str]]:
    """Generate set of (uid_a, uid_b) pairs that share a cluster."""
    pairs: set[tuple[str, str]] = set()
    for _, group in df.groupby("cluster_id"):
        uids = sorted(group["unique_id"].tolist())
        for a, b in combinations(uids, 2):
            pairs.add((a, b))
    return pairs


def main() -> None:
    baseline = pd.read_csv(BASELINE_DIR / CLUSTERED_FILE)
    four_src = pd.read_csv(FOUR_SRC_DIR / CLUSTERED_FILE)

    # Baseline has no GP rows. Filter 4-source to non-GP only.
    non_gp_four = four_src[four_src["source_name"] != "gp_api"]

    print(f"Baseline records: {len(baseline):,}")
    print(f"4-source records (total): {len(four_src):,}")
    print(f"4-source records (non-GP): {len(non_gp_four):,}")

    baseline_pairs = co_cluster_pairs(baseline)
    four_src_pairs = co_cluster_pairs(non_gp_four)

    lost_pairs = baseline_pairs - four_src_pairs  # splits
    new_pairs = four_src_pairs - baseline_pairs  # merges among non-GP records

    print(f"\nBaseline co-cluster pairs: {len(baseline_pairs):,}")
    print(f"4-source co-cluster pairs (non-GP): {len(four_src_pairs):,}")
    print(f"\nLost pairs (splits): {len(lost_pairs)}")
    print(f"New pairs (merges): {len(new_pairs)}")

    # Detail lost pairs — these are regressions
    if lost_pairs:
        print("\n=== LOST PAIRS (REGRESSIONS) ===")
        for uid_a, uid_b in sorted(lost_pairs):
            row_a = baseline[baseline["unique_id"] == uid_a].iloc[0]
            row_b = baseline[baseline["unique_id"] == uid_b].iloc[0]
            print(
                f"  {uid_a} ({row_a['source_name']}) <-> {uid_b} ({row_b['source_name']})"
            )
            print(f"    baseline cluster: {row_a['cluster_id']}")

    # Detail new pairs — review for legitimacy (GP bridging is OK)
    if new_pairs:
        print("\n=== NEW PAIRS (REVIEW FOR GP BRIDGING) ===")
        for uid_a, uid_b in sorted(list(new_pairs)[:50]):  # cap at 50
            row_a = non_gp_four[non_gp_four["unique_id"] == uid_a].iloc[0]
            row_b = non_gp_four[non_gp_four["unique_id"] == uid_b].iloc[0]
            cluster = row_a["cluster_id"]
            # Check if a GP record is in the same cluster (bridging)
            gp_in_cluster = four_src[
                (four_src["cluster_id"] == cluster)
                & (four_src["source_name"] == "gp_api")
            ]
            bridged = "GP-BRIDGED" if len(gp_in_cluster) > 0 else "EM-SHIFT"
            print(
                f"  [{bridged}] {uid_a} ({row_a['source_name']}) <-> {uid_b} ({row_b['source_name']})"
            )
        if len(new_pairs) > 50:
            print(f"  ... and {len(new_pairs) - 50} more")

    # GP match stats
    gp_records = four_src[four_src["source_name"] == "gp_api"]
    gp_cluster_sizes = four_src.groupby("cluster_id").size()
    gp_multi = gp_cluster_sizes[gp_cluster_sizes > 1].index
    gp_matched = gp_records[gp_records["cluster_id"].isin(gp_multi)]
    print(f"\n=== GP API MATCH STATS ===")
    print(f"  Total GP records: {len(gp_records):,}")
    print(
        f"  GP records in multi-member clusters: {len(gp_matched):,} ({100*len(gp_matched)/len(gp_records):.1f}%)"
    )
    print(f"  GP singletons: {len(gp_records) - len(gp_matched):,}")


if __name__ == "__main__":
    main()
