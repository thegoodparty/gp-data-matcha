"""Check for cross-cycle false positives in GP API matching.

Identifies pairs where:
- gamma_election_date = 0 (different election dates)
- br_race_id_l != br_race_id_r (different races)
- gamma_email > 0 OR gamma_phone > 0 (contact-based match)

These are the risky cohort: same person, different election cycle,
matched only because of shared contact info.
"""

import pandas as pd

PAIRWISE_CSV = "results/candidacy_stage/pairwise_predictions.csv"


def main() -> None:
    df = pd.read_csv(PAIRWISE_CSV)

    # Filter to GP-involved pairs only
    gp_mask = (df["source_name_l"] == "gp_api") | (df["source_name_r"] == "gp_api")
    gp_pairs = df[gp_mask]
    print(f"Total GP-involved pairs: {len(gp_pairs):,}")

    # Cross-cycle cohort
    cohort = gp_pairs[
        (gp_pairs["gamma_election_date"] == 0)
        & (gp_pairs["br_race_id_l"].notna())
        & (gp_pairs["br_race_id_r"].notna())
        & (gp_pairs["br_race_id_l"] != gp_pairs["br_race_id_r"])
        & ((gp_pairs["gamma_email"] > 0) | (gp_pairs["gamma_phone"] > 0))
    ]

    print(f"\nCross-cycle risky cohort: {len(cohort)} pairs")

    if len(cohort) > 0:
        print("\n=== CROSS-CYCLE PAIRS (REVIEW EACH) ===")
        display_cols = [
            "unique_id_l",
            "unique_id_r",
            "source_name_l",
            "source_name_r",
            "first_name_l",
            "last_name_l",
            "first_name_r",
            "last_name_r",
            "election_date_l",
            "election_date_r",
            "official_office_name_l",
            "official_office_name_r",
            "br_race_id_l",
            "br_race_id_r",
            "gamma_email",
            "gamma_phone",
            "match_probability",
        ]
        available = [c for c in display_cols if c in cohort.columns]
        for _, row in cohort[available].iterrows():
            print("  ---")
            for col in available:
                print(f"    {col}: {row[col]}")
        print(f"\nACTION REQUIRED: {len(cohort)} cross-cycle pairs found.")
        print(
            "If these are real cross-cycle merges, add fallback filter (option C)"
            " to scripts/configs/candidacy.py post_prediction_filters:"
        )
        print("""
    '''
    NOT (
        gamma_election_date = 0
        AND br_race_id_l IS NOT NULL AND br_race_id_r IS NOT NULL
        AND br_race_id_l != br_race_id_r
        AND (gamma_email > 0 OR gamma_phone > 0)
    )
    '''""")
    else:
        print("\nNo cross-cycle false positives detected. Safe to proceed.")

    # Additional: GP pairs with gamma_party = 0 that still matched well
    print("\n=== GP PAIRS WITH PARTY MISMATCH ===")
    gp_party_mismatch = gp_pairs[
        (gp_pairs["gamma_party"] == 0) & (gp_pairs["match_probability"] >= 0.95)
    ]
    print(f"High-confidence GP pairs with gamma_party=0: {len(gp_party_mismatch):,}")
    if len(gp_party_mismatch) > 0:
        probs = gp_party_mismatch["match_probability"]
        print(
            f"  probability: mean={probs.mean():.3f},"
            f" min={probs.min():.3f}, max={probs.max():.3f}"
        )
        print(
            "  (Party disagreement is not blocking high-confidence matches"
            " — this is expected)"
        )


if __name__ == "__main__":
    main()
