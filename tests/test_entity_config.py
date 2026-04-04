# tests/test_entity_config.py
"""Tests for EntityConfig and the config registry."""

from scripts.entity_config import EntityConfig, get_config


def test_entity_config_is_frozen():
    """EntityConfig instances are immutable."""
    config = EntityConfig(
        entity_type="test",
        display_name="Test",
        default_input_table="catalog.schema.table",
        comparisons=[],
        blocking_rules_for_prediction=[],
        additional_columns_to_retain=[],
        em_training_blocks=[],
    )
    try:
        config.entity_type = "changed"
        assert False, "Should have raised FrozenInstanceError"
    except AttributeError:
        pass


def test_entity_config_defaults():
    """Default values are set correctly."""
    config = EntityConfig(
        entity_type="test",
        display_name="Test",
        default_input_table="catalog.schema.table",
        comparisons=[],
        blocking_rules_for_prediction=[],
        additional_columns_to_retain=[],
        em_training_blocks=[],
    )
    assert config.predict_threshold == 0.01
    assert config.cluster_threshold == 0.95
    assert config.post_prediction_filters == []
    assert config.date_columns == []
    assert config.clustered_output_name == "clustered_records.csv"


def test_candidacy_config_comparisons():
    """Candidacy config has exactly 9 comparisons (regression guard)."""
    from scripts.configs.candidacy import CANDIDACY_CONFIG

    assert len(CANDIDACY_CONFIG.comparisons) == 9
    assert CANDIDACY_CONFIG.entity_type == "candidacy"
    assert CANDIDACY_CONFIG.clustered_output_name == "clustered_candidacies.csv"


def test_candidacy_config_has_election_date():
    """Candidacy config uses election_date for training and date parsing."""
    from scripts.configs.candidacy import CANDIDACY_CONFIG

    assert "election_date" in CANDIDACY_CONFIG.date_columns
    assert any("election_date" in cols for cols in CANDIDACY_CONFIG.em_training_blocks)


def test_candidacy_config_blocking_rules():
    """Candidacy config has 6 blocking rules including br_race_id."""
    from scripts.configs.candidacy import CANDIDACY_CONFIG

    assert len(CANDIDACY_CONFIG.blocking_rules_for_prediction) == 6
