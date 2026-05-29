"""Tests for the election_stage post-prediction filter constant."""

from scripts.constants import ELECTION_STAGE_POST_PREDICTION_FILTER


def test_filter_requires_state_date_and_stage():
    """Filter must require matching state, election_date, and election_stage.

    These are the non-office identity keys. state and election_date are
    exact-equality blocking keys, so Splink never trains their m and drops the
    gamma_<col>; the filter references the retained raw _l/_r columns instead.
    """
    assert "state_l = state_r" in ELECTION_STAGE_POST_PREDICTION_FILTER
    assert "election_date_l = election_date_r" in ELECTION_STAGE_POST_PREDICTION_FILTER
    assert (
        "election_stage_l = election_stage_r" in ELECTION_STAGE_POST_PREDICTION_FILTER
    )
    # These gamma columns are dropped by Splink, so must not be referenced.
    assert "gamma_state" not in ELECTION_STAGE_POST_PREDICTION_FILTER
    assert "gamma_election_date" not in ELECTION_STAGE_POST_PREDICTION_FILTER


def test_filter_requires_district_and_seat_agreement():
    """When both sides expose district_identifier/seat_name they must agree."""
    assert (
        "district_identifier_l = district_identifier_r"
        in ELECTION_STAGE_POST_PREDICTION_FILTER
    )
    assert "seat_name_l = seat_name_r" in ELECTION_STAGE_POST_PREDICTION_FILTER


def test_filter_requires_office_type_with_locality():
    """Office match requires same candidate_office OR near-exact full name."""
    assert (
        "candidate_office_l = candidate_office_r"
        in ELECTION_STAGE_POST_PREDICTION_FILTER
    )


def test_filter_requires_office_overlap():
    """Filter must require office name overlap or normalized office match."""
    assert "gamma_official_office_name" in ELECTION_STAGE_POST_PREDICTION_FILTER


def test_filter_does_not_require_person_signals():
    """Race-level filter should not reference person columns."""
    assert "gamma_last_name" not in ELECTION_STAGE_POST_PREDICTION_FILTER
    assert "gamma_first_name" not in ELECTION_STAGE_POST_PREDICTION_FILTER
    assert "gamma_email" not in ELECTION_STAGE_POST_PREDICTION_FILTER
