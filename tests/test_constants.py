"""Tests for module-level constants in scripts.constants."""

from scripts.constants import EO_POST_PREDICTION_FILTER


def test_eo_post_filter_includes_position_id_clause():
    # The third clause should accept a pair when ballotready_position_id matches,
    # so a strong position match isn't filtered out by office_name/office_type.
    assert (
        "gamma_ballotready_position_id" in EO_POST_PREDICTION_FILTER
    ), "EO_POST_PREDICTION_FILTER missing gamma_ballotready_position_id clause"
