from duckdb import DuckDBPyConnection

from arthur_common.aggregations.functions.shield_aggregations import (
    ShieldInferenceRuleCountAggregation,
)
from arthur_common.models.metrics import DatasetReference

from .helpers import *


def test_shield_inference_rule_count(
    get_shield_dataset_rule_based: tuple[DuckDBPyConnection, DatasetReference],
    monkeypatch,
):
    # Enable segmentation for this test
    monkeypatch.setenv("INFERENCE_USER_CONVERSATION_SEGMENTATION", "true")
    conn, dataset_ref = get_shield_dataset_rule_based

    rule_count_aggregator = ShieldInferenceRuleCountAggregation()
    metrics = rule_count_aggregator.aggregate(
        conn,
        dataset_ref,
        shield_response_column="shield_response",
    )
    validate_expected_metric_names(rule_count_aggregator, metrics)

    # validate there's a single rule count metric
    rule_count_metrics = [m for m in metrics if m.name == "rule_count"]
    assert len(rule_count_metrics) == 1

    # With user_id and conversation_id grouping, we expect more series
    # Each combination of dimensions creates a separate series
    assert len(rule_count_metrics[0].numeric_series) > 0

    # Test grouping by conversation_id and user_id
    # Find series for conversation_id_1 and user_id_1
    user1_series = []
    for series in rule_count_metrics[0].numeric_series:
        conversation_id_dim = next(
            (d for d in series.dimensions if d.name == "conversation_id"),
            None,
        )
        user_id_dim = next((d for d in series.dimensions if d.name == "user_id"), None)
        if conversation_id_dim and user_id_dim and user_id_dim.value == "user_id_1":
            user1_series.append(series)

    assert len(user1_series) > 0, "Expected to find series for user_id_1"

    # Find series for conversation_id_2 and user_id_2
    user2_series = []
    for series in rule_count_metrics[0].numeric_series:
        conversation_id_dim = next(
            (d for d in series.dimensions if d.name == "conversation_id"),
            None,
        )
        user_id_dim = next((d for d in series.dimensions if d.name == "user_id"), None)
        if conversation_id_dim and user_id_dim and user_id_dim.value == "user_id_2":
            user2_series.append(series)

    assert len(user2_series) > 0, "Expected to find series for user_id_2"

    # Verify that all series have the expected dimensions
    for series in rule_count_metrics[0].numeric_series:
        conversation_id_dim = next(
            (d for d in series.dimensions if d.name == "conversation_id"),
            None,
        )
        user_id_dim = next((d for d in series.dimensions if d.name == "user_id"), None)
        location_dim = next(
            (d for d in series.dimensions if d.name == "location"),
            None,
        )
        rule_type_dim = next(
            (d for d in series.dimensions if d.name == "rule_type"),
            None,
        )

        assert conversation_id_dim is not None, "Expected conversation_id dimension"
        assert user_id_dim is not None, "Expected user_id dimension"
        assert location_dim is not None, "Expected location dimension"
        assert rule_type_dim is not None, "Expected rule_type dimension"

        assert conversation_id_dim.value in [
            "conversation_id_1",
            "conversation_id_2",
            "conversation_id_3",
        ]
        assert user_id_dim.value in ["user_id_1", "user_id_2"]
        assert location_dim.value in ["prompt", "response"]
        assert rule_type_dim.value in [
            "ToxicityRule",
            "PIIDataRule",
            "ModelHallucinationRuleV2",
        ]
