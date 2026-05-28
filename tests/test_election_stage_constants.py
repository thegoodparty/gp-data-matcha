"""Tests for the election_stage post-prediction filter constant."""

from scripts.constants import ELECTION_STAGE_POST_PREDICTION_FILTER


def test_filter_requires_state_and_date():
    """Filter must require state match + election_date or position FK signal."""
    assert "gamma_state" in ELECTION_STAGE_POST_PREDICTION_FILTER
    assert "gamma_election_date" in ELECTION_STAGE_POST_PREDICTION_FILTER
    assert "gamma_ballotready_position_id" in ELECTION_STAGE_POST_PREDICTION_FILTER


def test_filter_requires_office_overlap():
    """Filter must require office name overlap or normalized office match."""
    assert "gamma_official_office_name" in ELECTION_STAGE_POST_PREDICTION_FILTER


def test_filter_does_not_require_person_signals():
    """Race-level filter should not reference person columns."""
    assert "gamma_last_name" not in ELECTION_STAGE_POST_PREDICTION_FILTER
    assert "gamma_first_name" not in ELECTION_STAGE_POST_PREDICTION_FILTER
    assert "gamma_email" not in ELECTION_STAGE_POST_PREDICTION_FILTER
