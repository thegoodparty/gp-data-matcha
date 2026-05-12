# tests/test_candidacy_config.py
"""Tests for the candidacy ER config (DATA-1880 follow-up: office_level comparison)."""

from scripts.configs.candidacy import CANDIDACY_CONFIG


def test_office_level_in_comparisons():
    """office_level is added as an ExactMatch comparison."""
    comparison_columns = [
        c.get_comparison("duckdb").output_column_name
        for c in CANDIDACY_CONFIG.comparisons
    ]
    assert "office_level" in comparison_columns


def test_office_level_em_training_block():
    """office_level appears in at least one EM training block (mirrors EO pattern)."""
    assert any(
        "office_level" in cols for cols in CANDIDACY_CONFIG.em_training_blocks
    ), "Expected an EM training block including office_level"


def test_office_level_not_in_additional_columns_to_retain():
    """office_level is a comparison column and should NOT be in additional_columns_to_retain.

    Splink retains comparison columns automatically; listing it would duplicate
    the column and risk SQL errors. Mirrors EO convention.
    """
    retained = set(CANDIDACY_CONFIG.additional_columns_to_retain)
    assert "office_level" not in retained, (
        "office_level is a comparison column and should not be listed in "
        "additional_columns_to_retain — Splink retains comparison columns automatically."
    )


def test_gamma_office_level_in_audit_gamma_columns():
    """gamma_office_level is exposed in the audit gamma columns list."""
    assert "gamma_office_level" in CANDIDACY_CONFIG.audit_gamma_columns


def _first_name_comparison_sql():
    fn = next(
        c
        for c in CANDIDACY_CONFIG.comparisons
        if c.get_comparison("duckdb").output_column_name == "first_name"
    )
    return fn.get_comparison("duckdb").as_dict()


def test_first_name_comparison_has_normalized_token_level():
    """first_name comparison includes the normalized-token level between
    ArrayIntersectLevel and JaroWinklerLevel — catches compound names ("charles
    kirk" vs "charles") and period-separated initials ("r.j." vs "rj") that the
    alias array and JW (threshold 0.92) miss."""
    cmp = _first_name_comparison_sql()
    labels = [
        level.get("label_for_charts", "") for level in cmp["comparison_levels"]
    ]
    assert "normalized first name token overlap" in labels


def _eval_first_name_level(first_l: str, first_r: str) -> bool:
    """Evaluate ONLY the normalized-token level (NOT the full comparison cascade).

    The full Splink comparison evaluates levels top to bottom and returns the
    first match; for cases like exact matches, a higher-priority level would
    capture them. This helper isolates the new SQL clause for direct testing.
    """
    import duckdb

    cmp = _first_name_comparison_sql()
    level = next(
        lvl
        for lvl in cmp["comparison_levels"]
        if lvl.get("label_for_charts") == "normalized first name token overlap"
    )
    sql = level["sql_condition"]
    con = duckdb.connect()
    return con.execute(
        f"SELECT ({sql}) FROM (SELECT ? AS first_name_l, ? AS first_name_r)",
        [first_l, first_r],
    ).fetchone()[0]


def test_normalized_token_level_catches_period_initials():
    """'r.j.' vs 'rj' — period normalization makes them equal."""
    assert _eval_first_name_level("r.j.", "rj") is True
    assert _eval_first_name_level("c.j.", "cj") is True


def test_normalized_token_level_catches_compound_first_name():
    """'charles kirk' vs 'charles' — middle name added in one source."""
    assert _eval_first_name_level("charles kirk", "charles") is True
    assert _eval_first_name_level("jeremy paul", "jeremy") is True


def test_normalized_token_level_catches_honorific_prefix():
    """'dr. lori' vs 'lori' — honorific prefix in one source."""
    assert _eval_first_name_level("dr. lori", "lori") is True


def test_normalized_token_level_rejects_unrelated_names():
    """Different names with no shared tokens or normalized equality → no match."""
    assert _eval_first_name_level("michael", "denise") is False
    assert _eval_first_name_level("jeremy", "alexis") is False


def test_normalized_token_level_rejects_short_initial_only_overlap():
    """Single-char tokens (initials alone) should not trigger a match —
    overlap requires tokens of length >= 2."""
    # "j." vs "j. steven" share only the "j" initial; "j" is below the
    # length threshold so this should NOT match via the token-overlap branch.
    # The normalized-equality branch also fails ("j" != "jsteven").
    assert _eval_first_name_level("j.", "j. steven") is False
