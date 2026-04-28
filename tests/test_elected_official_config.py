# tests/test_elected_official_config.py
"""Tests for the elected_official ER config (DATA-1731 prematch rewrite)."""

from scripts.configs.elected_official import ELECTED_OFFICIAL_CONFIG


def test_ballotready_position_id_in_comparisons():
    """ballotready_position_id is added as an ExactMatch comparison."""
    comparison_columns = [
        c.get_comparison("duckdb").output_column_name
        for c in ELECTED_OFFICIAL_CONFIG.comparisons
    ]
    assert "ballotready_position_id" in comparison_columns


def _rule_text(rule) -> str:
    """Extract a searchable representation of a Splink blocking rule.

    Handles CustomRule (sql_condition), ExactMatchRule (col_expression),
    and And (blocking_rules list of nested rules).
    """
    if hasattr(rule, "sql_condition"):
        return rule.sql_condition or ""
    if hasattr(rule, "col_expression"):
        return getattr(rule.col_expression, "sql_expression", "") or ""
    if hasattr(rule, "blocking_rules"):
        return " AND ".join(_rule_text(r) for r in rule.blocking_rules)
    return ""


def test_ballotready_position_id_in_blocking_rules():
    """A blocking rule references ballotready_position_id (state + position_id)."""
    blocking_rule_strs = [
        _rule_text(r) for r in ELECTED_OFFICIAL_CONFIG.blocking_rules_for_prediction
    ]
    assert any(
        "ballotready_position_id" in s for s in blocking_rule_strs
    ), "Expected a blocking rule referencing ballotready_position_id"


def test_new_columns_in_additional_columns_to_retain():
    """The 8 new retained columns are present (position_id + 3 ICP + 4 gp_api IDs)."""
    expected = {
        "ballotready_position_id",
        "is_win_icp",
        "is_serve_icp",
        "is_win_supersize_icp",
        "gp_api_user_id",
        "gp_api_campaign_id",
        "gp_api_elected_office_id",
        "gp_api_organization_slug",
    }
    retained = set(ELECTED_OFFICIAL_CONFIG.additional_columns_to_retain)
    missing = expected - retained
    assert not missing, f"Missing retained columns: {missing}"
